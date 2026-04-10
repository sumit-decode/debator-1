"""Judge agent for claim-level and article-level verdicts."""
from __future__ import annotations

import logging

from tools.groq_client import groq_chat_json

logger = logging.getLogger(__name__)

CLAIM_VERDICTS = {"SUPPORTED", "REFUTED", "UNVERIFIABLE"}


def judge_debate(verifier_report: dict, falsifier_report: dict, progress_callback=None) -> dict:
    """Judge the debate between verifier and falsifier reports."""
    verifier_claims = verifier_report.get("claim_reports", [])
    falsifier_claims = falsifier_report.get("claim_reports", [])

    verdicts = []
    total = len(verifier_claims)

    for index, verifier_claim in enumerate(verifier_claims):
        falsifier_claim = falsifier_claims[index] if index < len(falsifier_claims) else None
        if progress_callback:
            progress_callback(index, total, f"Judging claim {index + 1}: {verifier_claim.get('claim', '')[:50]}...")
        verdicts.append(_judge_single(verifier_claim, falsifier_claim))

    if progress_callback:
        progress_callback(total, total, "Determining final verdict...")

    overall = _overall_verdict(
        verdicts,
        verifier_report.get("overall_assessment", ""),
        falsifier_report.get("overall_assessment", ""),
    )
    return {"claim_verdicts": verdicts, **overall}


def _judge_single(verifier_claim: dict, falsifier_claim: dict | None) -> dict:
    """Judge one claim by weighing both agents' evidence and arguments."""
    claim = str(verifier_claim.get("claim", "")).strip()
    verifier_argument = verifier_claim.get("argument", "")
    verifier_confidence = _clamp(verifier_claim.get("confidence", 0.0))
    verifier_count = int(verifier_claim.get("evidence_count", 0) or 0)
    verifier_evidence = _format_evidence_block(verifier_claim.get("supporting_evidence", []))

    falsifier_argument = falsifier_claim.get("argument", "No falsification attempted.") if falsifier_claim else "No falsification attempted."
    falsifier_confidence = _clamp(falsifier_claim.get("confidence", 0.0)) if falsifier_claim else 0.0
    falsifier_count = int(falsifier_claim.get("evidence_count", 0) or 0) if falsifier_claim else 0
    falsifier_evidence = _format_evidence_block(
        falsifier_claim.get("contradicting_evidence", []) if falsifier_claim else []
    )

    try:
        data = groq_chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an IMPARTIAL JUDGE in a fact-checking debate.\n"
                        "You must weigh evidence from both sides and deliver a verdict.\n\n"
                        "RULES:\n"
                        "1. Prioritize the raw evidence over each agent's rhetoric\n"
                        "2. Consider evidence quantity and quality\n"
                        "3. Official or institutional sources carry more weight\n"
                        "4. If evidence is inconclusive from both sides, return UNVERIFIABLE\n"
                        "5. Be specific about which evidence influenced your decision\n\n"
                        "Respond in JSON:\n"
                        "{\n"
                        '  "verdict": "SUPPORTED" | "REFUTED" | "UNVERIFIABLE",\n'
                        '  "confidence": 0.0 to 1.0,\n'
                        '  "reasoning": "Your detailed reasoning"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CLAIM: {claim}\n\n"
                        f"{'=' * 40}\n"
                        "VERIFIER CASE\n"
                        f"{'=' * 40}\n"
                        f"Supporting evidence found: {verifier_count}\n"
                        f"Average confidence: {verifier_confidence:.0%}\n"
                        f"Top supporting evidence:\n{verifier_evidence}\n\n"
                        f"Argument:\n{verifier_argument}\n\n"
                        f"{'=' * 40}\n"
                        "FALSIFIER CASE\n"
                        f"{'=' * 40}\n"
                        f"Contradicting evidence found: {falsifier_count}\n"
                        f"Average confidence: {falsifier_confidence:.0%}\n"
                        f"Top contradicting evidence:\n{falsifier_evidence}\n\n"
                        f"Argument:\n{falsifier_argument}\n\n"
                        "Deliver your verdict."
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=600,
        )

        return {
            "claim": claim,
            "verdict": _normalize_claim_verdict(data.get("verdict")),
            "confidence": _clamp(data.get("confidence", 0.5)),
            "reasoning": str(data.get("reasoning", "")).strip(),
        }
    except Exception as exc:
        logger.error("Judging failed for claim '%s': %s", claim[:80], exc)
        return {
            "claim": claim,
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "reasoning": "Error during judgment; could not reach a verdict.",
        }


def _overall_verdict(verdicts: list[dict], verifier_assessment: str, falsifier_assessment: str) -> dict:
    """Determine the overall article verdict from all per-claim results."""
    metrics = _score_overall_verdict(verdicts)
    overall = metrics["overall_verdict"]
    confidence = metrics["overall_confidence"]
    scores = metrics["confidence_metrics"]

    supported = sum(1 for verdict in verdicts if verdict.get("verdict") == "SUPPORTED")
    refuted = sum(1 for verdict in verdicts if verdict.get("verdict") == "REFUTED")
    unverifiable = sum(1 for verdict in verdicts if verdict.get("verdict") == "UNVERIFIABLE")

    reasoning = (
        "Overall verdict is based on claim-level outcomes rather than a final free-form model guess. "
        f"Supported claims contributed most to REAL ({scores['REAL']:.0%}), "
        f"refuted claims contributed most to FAKE ({scores['FAKE']:.0%}), and "
        f"mixed or unverifiable claims contributed most to MISLEADING ({scores['MISLEADING']:.0%}). "
        f"Claim counts: {supported} supported, {refuted} refuted, {unverifiable} unverifiable. "
        f"Verifier summary: {verifier_assessment[:180] or 'No verifier summary.'} "
        f"Falsifier summary: {falsifier_assessment[:180] or 'No falsifier summary.'}"
    )

    summary_map = {
        "REAL": "Most claims are supported strongly enough for the article to read as real.",
        "FAKE": "Refuted claims outweigh supported ones, so the article reads as fake.",
        "MISLEADING": "The article mixes weak, conflicting, or unverifiable claims, so it reads as misleading.",
    }

    return {
        "overall_verdict": overall,
        "overall_confidence": confidence,
        "reasoning": reasoning,
        "summary": summary_map[overall],
        "confidence_metrics": scores,
    }


def _score_overall_verdict(verdicts: list[dict]) -> dict:
    """
    Convert claim-level verdicts into stable article-level scores.

    This avoids the previous behavior where a final unconstrained LLM pass could
    overuse the MISLEADING label even when the claim outcomes leaned clearly real
    or fake.
    """
    if not verdicts:
        return {
            "overall_verdict": "MISLEADING",
            "overall_confidence": 0.0,
            "confidence_metrics": {"REAL": 0.0, "FAKE": 0.0, "MISLEADING": 1.0},
        }

    real_points = 0.0
    fake_points = 0.0
    misleading_points = 0.0

    for verdict in verdicts:
        label = _normalize_claim_verdict(verdict.get("verdict"))
        confidence = _clamp(verdict.get("confidence", 0.5))

        if label == "SUPPORTED":
            real_points += confidence
            misleading_points += (1.0 - confidence) * 0.25
        elif label == "REFUTED":
            fake_points += confidence
            misleading_points += (1.0 - confidence) * 0.25
        else:
            misleading_points += 0.45 + (confidence * 0.35)

    misleading_points += min(real_points, fake_points) * 0.9

    total = real_points + fake_points + misleading_points
    if total <= 0:
        return {
            "overall_verdict": "MISLEADING",
            "overall_confidence": 0.0,
            "confidence_metrics": {"REAL": 0.0, "FAKE": 0.0, "MISLEADING": 1.0},
        }

    scores = {
        "REAL": round(real_points / total, 3),
        "FAKE": round(fake_points / total, 3),
        "MISLEADING": round(misleading_points / total, 3),
    }

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    overall_verdict, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    if abs(scores["REAL"] - scores["FAKE"]) <= 0.08 and max(scores["REAL"], scores["FAKE"]) >= 0.3:
        overall_verdict = "MISLEADING"
        top_score = scores["MISLEADING"]
        runner_up = max(scores["REAL"], scores["FAKE"])

    confidence = round(max(top_score, top_score - (runner_up * 0.15)), 3)
    return {
        "overall_verdict": overall_verdict,
        "overall_confidence": confidence,
        "confidence_metrics": scores,
    }


def _fallback_verdict(verdicts: list[dict]) -> dict:
    """Heuristic fallback when judge verdict generation fails."""
    metrics = _score_overall_verdict(verdicts)
    supported = sum(1 for verdict in verdicts if verdict.get("verdict") == "SUPPORTED")
    refuted = sum(1 for verdict in verdicts if verdict.get("verdict") == "REFUTED")
    total = len(verdicts)

    return {
        "overall_verdict": metrics["overall_verdict"],
        "overall_confidence": metrics["overall_confidence"],
        "reasoning": (
            f"Fallback verdict: {supported} claims supported, "
            f"{refuted} refuted, {total - supported - refuted} unverifiable."
        ),
        "summary": f"Article appears to be {metrics['overall_verdict'].lower()} based on claim analysis.",
        "confidence_metrics": metrics["confidence_metrics"],
    }


def _normalize_claim_verdict(value: object) -> str:
    """Normalize claim verdict labels from the LLM."""
    label = str(value or "UNVERIFIABLE").strip().upper()
    if label not in CLAIM_VERDICTS:
        return "UNVERIFIABLE"
    return label


def _clamp(value: float | int) -> float:
    """Clamp a confidence-like value to the 0-1 range."""
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _format_evidence_block(evidence: list[dict], top_k: int = 3) -> str:
    """Format the strongest evidence snippets for the judge prompt."""
    if not evidence:
        return "No direct evidence provided."

    lines = []
    ranked = sorted(evidence, key=lambda item: item.get("confidence", 0), reverse=True)
    for item in ranked[:top_k]:
        excerpt = str(item.get("full_text") or item.get("snippet") or "").replace("\n", " ").strip()
        excerpt = excerpt[:280] if excerpt else "No excerpt available."
        lines.append(
            f"- {item.get('title', 'Untitled source')} | "
            f"{item.get('stance', 'UNKNOWN')} {_clamp(item.get('confidence', 0)):.0%} | "
            f"{item.get('url', 'No URL')}\n"
            f"  {excerpt}"
        )
    return "\n".join(lines)
