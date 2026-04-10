"""
Claim Extractor Agent.

Uses spaCy NER and Groq to extract verifiable factual claims from articles.
"""
import logging

import spacy

from config import MAX_CLAIMS, SPACY_MODEL
from tools.groq_client import groq_chat_json

logger = logging.getLogger(__name__)

try:
    nlp = spacy.load(SPACY_MODEL)
    logger.info(f"spaCy model loaded: {SPACY_MODEL}")
except OSError:
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.warning("Fell back to en_core_web_sm")
    except OSError:
        nlp = None
        logger.error("No spaCy model found. Run: python -m spacy download en_core_web_sm")


LABEL_NAMES = {
    "PERSON": "People",
    "ORG": "Organizations",
    "GPE": "Locations",
    "LOC": "Locations",
    "DATE": "Dates",
    "TIME": "Times",
    "MONEY": "Money",
    "PERCENT": "Percentages",
    "EVENT": "Events",
    "NORP": "Groups/Nationalities",
    "PRODUCT": "Products",
    "WORK_OF_ART": "Works",
    "LAW": "Laws/Regulations",
    "QUANTITY": "Quantities",
}

SYSTEM_PROMPT = """You are an expert fact-checker and claim extraction specialist.
Your task is to extract specific, verifiable factual claims from news articles.

Rules:
1. Extract exactly 3 to 5 claims that can be independently verified via web search.
2. Each claim must be a factual statement, not an opinion.
3. Claims should reference specific people, organizations, places, dates, or numbers.
4. Claims must be self-contained.
5. Prioritize the most important and checkable claims.

Respond in this exact JSON format only:
{
  "claims": [
    {
      "claim": "The specific factual claim statement",
      "entities": ["Entity1", "Entity2"],
      "importance": "high"
    }
  ]
}"""


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract named entities from text using spaCy."""
    if not nlp:
        return {}

    doc = nlp(text[:10000])
    entities: dict[str, list[str]] = {}

    for ent in doc.ents:
        label = ent.label_
        entities.setdefault(label, [])
        value = ent.text.strip()
        if value and value not in entities[label]:
            entities[label].append(value)

    logger.info(
        "NER extracted %s entities across %s categories",
        sum(len(values) for values in entities.values()),
        len(entities),
    )
    return entities


def extract_claims(article_text: str) -> list[dict]:
    """Extract 3 to 5 verifiable factual claims from an article."""
    entities = extract_entities(article_text)
    entity_block = _format_entities(entities)

    user_prompt = f"""ARTICLE TEXT:
{article_text[:4000]}

DETECTED ENTITIES:
{entity_block}

Extract 3 to 5 verifiable factual claims from this article."""

    try:
        data = groq_chat_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
            raise_on_parse_error=True,
        )

        claims = _sanitize_claims(data.get("claims", []))
        if claims:
            logger.info("Extracted %s claims via LLM", len(claims))
            return claims

        logger.warning("LLM returned no valid claims; using heuristic fallback")
        return _fallback_extraction(article_text)

    except Exception as exc:
        logger.error("Claim extraction failed: %s", exc)
        return _fallback_extraction(article_text)


def _format_entities(entities: dict[str, list[str]]) -> str:
    """Format NER output into a readable block for the prompt."""
    if not entities:
        return "  (no entities detected)"

    lines = []
    for label, items in entities.items():
        display = LABEL_NAMES.get(label, label)
        lines.append(f"  {display}: {', '.join(items[:6])}")
    return "\n".join(lines)


def _sanitize_claims(raw_claims: object) -> list[dict]:
    """Normalize the LLM response into a stable claim list."""
    if not isinstance(raw_claims, list):
        return []

    claims = []
    seen = set()

    for item in raw_claims:
        if not isinstance(item, dict):
            continue

        claim = " ".join(str(item.get("claim", "")).split()).strip()
        if len(claim) < 20:
            continue

        key = claim.lower()
        if key in seen:
            continue
        seen.add(key)

        raw_entities = item.get("entities", [])
        if not isinstance(raw_entities, list):
            raw_entities = []

        entities = []
        for entity in raw_entities[:6]:
            value = str(entity).strip()
            if value:
                entities.append(value)

        importance = str(item.get("importance", "medium")).lower().strip()
        if importance not in {"low", "medium", "high"}:
            importance = "medium"

        claims.append(
            {
                "claim": claim if claim.endswith(".") else f"{claim}.",
                "entities": entities,
                "importance": importance,
            }
        )

        if len(claims) >= MAX_CLAIMS:
            break

    return claims


def _fallback_extraction(article_text: str) -> list[dict]:
    """Simple heuristic fallback when LLM extraction fails."""
    logger.info("Using heuristic fallback for claim extraction")

    sentences = [sentence.strip() for sentence in article_text.split(".") if sentence.strip()]
    claims = []

    for sentence in sentences[:20]:
        if len(sentence) > 40 and any(char.isdigit() for char in sentence):
            claims.append(
                {
                    "claim": sentence.rstrip(".") + ".",
                    "entities": [],
                    "importance": "medium",
                }
            )
        if len(claims) >= 3:
            break

    if not claims and sentences:
        claims.append(
            {
                "claim": sentences[0].rstrip(".") + ".",
                "entities": [],
                "importance": "medium",
            }
        )

    return claims
