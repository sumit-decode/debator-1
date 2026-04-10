"""Stance detection utilities backed by Groq."""
from __future__ import annotations

import logging
import time
from functools import lru_cache

from tools.groq_client import groq_chat_json

logger = logging.getLogger(__name__)

STANCE_LABEL_MAP = {
    "SUPPORT": "SUPPORT",
    "SUPPORTED": "SUPPORT",
    "ENTAILMENT": "SUPPORT",
    "CONTRADICT": "CONTRADICT",
    "CONTRADICTION": "CONTRADICT",
    "REFUTE": "CONTRADICT",
    "REFUTED": "CONTRADICT",
    "NEUTRAL": "NEUTRAL",
    "UNCLEAR": "NEUTRAL",
    "UNVERIFIABLE": "NEUTRAL",
}

SYSTEM_PROMPT = """You are a stance classification system for fact-checking.
Given a factual claim and an evidence passage, decide whether the evidence:
- SUPPORTS the claim
- CONTRADICTS the claim
- is NEUTRAL or insufficient

Rules:
1. Use SUPPORT only when the evidence directly backs the claim.
2. Use CONTRADICT only when the evidence directly disputes the claim.
3. Use NEUTRAL when the evidence is unrelated, ambiguous, or insufficient.
4. Confidence must be a number from 0.0 to 1.0.

Respond in JSON only:
{
  "stance": "SUPPORT" | "CONTRADICT" | "NEUTRAL",
  "confidence": 0.0,
  "reasoning": "short explanation"
}"""


def detect_stance(claim: str, evidence: str, max_retries: int = 3) -> dict:
    """Classify the stance of an evidence passage against a claim."""
    clean_claim = str(claim or "").strip()
    clean_evidence = str(evidence or "").strip()
    if not clean_claim or not clean_evidence:
        return _neutral_result("empty_input")

    return _detect_stance_cached(clean_claim, clean_evidence, max_retries)


def batch_detect_stance(claim: str, evidence_list: list[str]) -> list[dict]:
    """Score multiple evidence passages against a single claim."""
    results = []
    for evidence in evidence_list:
        results.append(detect_stance(claim, evidence))
        time.sleep(0.1)
    return results


@lru_cache(maxsize=512)
def _detect_stance_cached(claim: str, evidence: str, max_retries: int) -> dict:
    """Cache identical stance checks to reduce duplicate Groq calls."""
    for attempt in range(max_retries):
        try:
            data = groq_chat_json(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"CLAIM:\n{claim[:500]}\n\n"
                            f"EVIDENCE:\n{evidence[:1400]}\n\n"
                            "Return the stance classification."
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=150,
                max_retries=2,
                raise_on_parse_error=True,
            )

            parsed = _parse_result(data)
            if parsed["confidence"] == 0.0 and not str(data.get("stance", "")).strip():
                raise ValueError(f"Missing stance label in response: {data}")

            parsed["provider"] = "groq"
            parsed["used_fallback"] = False
            parsed["error"] = None
            return parsed
        except Exception as exc:
            logger.warning("Stance detection attempt %s failed: %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                time.sleep(1.5)

    logger.error("All stance detection attempts failed; returning neutral fallback")
    return _neutral_result("classification_failed")


def _parse_result(result: object) -> dict:
    """
    Normalize stance outputs from Groq.

    This parser also tolerates the old Hugging Face-style list output so the
    rest of the project remains stable during the transition.
    """
    try:
        if isinstance(result, dict):
            raw_label = str(result.get("stance") or result.get("label") or "NEUTRAL").strip().upper()
            confidence = _clamp(result.get("confidence", result.get("score", 0.0)))
            return {
                "stance": STANCE_LABEL_MAP.get(raw_label, "NEUTRAL"),
                "confidence": round(confidence, 3),
            }

        scores = result
        if isinstance(result, list) and result:
            scores = result[0] if isinstance(result[0], list) else result

        if isinstance(scores, list) and scores:
            best = max(scores, key=lambda item: item.get("score", 0))
            raw_label = str(best.get("label", "NEUTRAL")).strip().upper()
            return {
                "stance": STANCE_LABEL_MAP.get(raw_label, "NEUTRAL"),
                "confidence": round(_clamp(best.get("score", 0.0)), 3),
            }
    except Exception as exc:
        logger.warning("Failed to parse stance result: %s; raw=%s", exc, result)

    return {"stance": "NEUTRAL", "confidence": 0.0}


def _neutral_result(reason: str) -> dict:
    """Return a neutral fallback result with provider metadata."""
    return {
        "stance": "NEUTRAL",
        "confidence": 0.0,
        "provider": "groq",
        "used_fallback": True,
        "error": reason,
    }


def _clamp(value: float | int) -> float:
    """Clamp a confidence-like value to the 0-1 range."""
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0
