"""Verifier agent that searches for evidence supporting each claim."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from tools.groq_client import groq_chat, groq_chat_json
from tools.stance_detector import detect_stance
from tools.web_search import search_and_scrape

logger = logging.getLogger(__name__)


def verify_claims(claims: list[dict], progress_callback=None) -> dict:
    """Search for supporting evidence and build a verification case per claim."""
    reports = []

    for index, claim_data in enumerate(claims):
        claim = str(claim_data.get("claim", "")).strip()
        if not claim:
            logger.warning("[VERIFIER] Skipping empty claim payload at index %s", index)
            continue

        logger.info("[VERIFIER] Claim %s/%s: %s", index + 1, len(claims), claim[:80])

        if progress_callback:
            progress_callback(index, len(claims), f"Verifying: {claim[:60]}...")

        reports.append(_verify_single_claim(claim))

    overall = _build_overall(reports)
    return {"claim_reports": reports, "overall_assessment": overall}


def _verify_single_claim(claim: str) -> dict:
    """Search for supporting evidence for one claim and build an argument."""
    queries = _make_queries(claim)

    all_hits = []
    for query in queries:
        all_hits.extend(search_and_scrape(query, claim=claim, max_results=3, scrape_top=2))

    unique_hits = _dedupe_hits(all_hits)

    def _score(hit: dict) -> dict:
        evidence_text = str(hit.get("scraped_text") or hit.get("snippet") or "").strip()
        stance = detect_stance(claim, evidence_text)
        return {
            "title": str(hit.get("title", "Untitled source")),
            "snippet": str(hit.get("snippet", "")),
            "full_text": evidence_text,
            "url": str(hit.get("url", "")),
            "scraped": bool(hit.get("scraped", False)),
            "stance": stance["stance"],
            "confidence": stance["confidence"],
            "provider": stance.get("provider", "unknown"),
            "used_fallback": stance.get("used_fallback", False),
            "error": stance.get("error"),
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        evidence = list(pool.map(_score, unique_hits[:5]))

    supporting = sorted(
        [item for item in evidence if item["stance"] == "SUPPORT"],
        key=lambda item: item["confidence"],
        reverse=True,
    )

    argument = _build_argument(claim, supporting[:3], evidence)
    avg_confidence = (
        sum(item["confidence"] for item in supporting) / len(supporting)
        if supporting
        else 0.0
    )

    return {
        "claim": claim,
        "search_queries": queries,
        "evidence": evidence,
        "supporting_evidence": supporting[:3],
        "argument": argument,
        "confidence": round(avg_confidence, 3),
        "evidence_count": len(supporting),
        "classification_failures": sum(1 for item in evidence if item.get("used_fallback")),
        "scraped_count": sum(1 for item in evidence if item.get("scraped")),
        "classification_engine": "groq",
    }


def _make_queries(claim: str) -> list[str]:
    """Generate concise web queries that should support the claim."""
    try:
        data = groq_chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate 2 concise web search queries to find evidence that "
                        "SUPPORTS and CONFIRMS the following claim. "
                        "Target official sources, reputable news, and public records.\n"
                        '{"queries": ["q1", "q2"]}'
                    ),
                },
                {"role": "user", "content": f"Claim: {claim}"},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        return _normalize_queries(data.get("queries"), fallback=[claim])
    except Exception as exc:
        logger.warning("Query generation failed: %s", exc)
        return [claim]


def _build_argument(claim: str, supporting: list[dict], all_evidence: list[dict]) -> str:
    """Build a concise verifier argument using the strongest evidence."""
    if supporting:
        evidence_block = "\n".join(
            f"- [{item['title']}]: {item.get('full_text', item['snippet'])[:300]} "
            f"(stance: {item['stance']}, score: {item['confidence']:.0%}, scraped: {item.get('scraped', False)})"
            for item in supporting[:3]
        )
    else:
        evidence_block = "No strong supporting evidence was found."

    try:
        return groq_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a VERIFIER agent in a fact-checking debate. "
                        "Your role is to argue FOR the truth of the given claim using the evidence provided.\n"
                        "- Cite specific evidence and sources\n"
                        "- Be persuasive but honest\n"
                        "- If evidence is weak, acknowledge it\n"
                        "- Keep your argument to 2-3 paragraphs"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CLAIM: {claim}\n\n"
                        f"SUPPORTING EVIDENCE:\n{evidence_block}\n\n"
                        f"Total snippets found: {len(all_evidence)}, supporting: {len(supporting)}\n\n"
                        "Build your verification argument."
                    ),
                },
            ],
            temperature=0.4,
            max_tokens=800,
        )
    except Exception as exc:
        logger.error("Argument generation failed: %s", exc)
        return f"[Argument generation failed] Found {len(supporting)} supporting evidence snippets."


def _build_overall(reports: list[dict]) -> str:
    """Summarize all per-claim verification results."""
    summary = "\n".join(
        f"- Claim {index + 1}: {report['claim'][:80]} -> "
        f"{report['evidence_count']} supporting, confidence {report['confidence']:.0%}"
        for index, report in enumerate(reports)
    )

    try:
        return groq_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a verification agent. Summarize your overall "
                        "assessment of the article's truthfulness in 1-2 paragraphs."
                    ),
                },
                {"role": "user", "content": f"VERIFICATION RESULTS:\n{summary or 'No claims analyzed.'}"},
            ],
            temperature=0.3,
            max_tokens=500,
        )
    except Exception as exc:
        logger.error("Overall assessment failed: %s", exc)
        return "Unable to generate overall assessment."


def _normalize_queries(raw_queries: object, fallback: list[str]) -> list[str]:
    """Normalize LLM query output into a clean, non-empty list."""
    if not isinstance(raw_queries, list):
        return fallback

    queries = []
    seen = set()
    for item in raw_queries:
        query = " ".join(str(item).split()).strip()
        if not query:
            continue
        lowered = query.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        queries.append(query)
        if len(queries) >= 2:
            break

    return queries or fallback


def _dedupe_hits(hits: list[dict]) -> list[dict]:
    """Deduplicate search hits by URL while preserving order."""
    unique_hits = []
    seen_urls = set()

    for hit in hits:
        url = str(hit.get("url", "")).strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        unique_hits.append(hit)

    return unique_hits
