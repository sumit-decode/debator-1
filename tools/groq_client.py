"""Shared Groq client helpers with retry and JSON parsing support."""
from __future__ import annotations

import json
import logging
import threading
import time
from functools import lru_cache

from groq import Groq

from config import API_CALL_DELAY, GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

_client_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    """Create the Groq client lazily as a thread-safe singleton."""
    with _client_lock:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured.")
        return Groq(api_key=GROQ_API_KEY)


def groq_chat(
    messages: list[dict],
    *,
    temperature: float = 0.3,
    max_tokens: int = 800,
    json_mode: bool = False,
    max_retries: int = 3,
    model: str | None = None,
) -> str:
    """Call Groq chat completion with basic retry logic."""
    client = get_groq_client()
    target_model = model or GROQ_MODEL
    prepared_messages = [
        {"role": message.get("role", "user"), "content": str(message.get("content", ""))}
        for message in messages
    ]

    kwargs = {
        "model": target_model,
        "messages": prepared_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
        if not any("json" in message["content"].lower() for message in prepared_messages):
            for message in prepared_messages:
                if message["role"] == "system":
                    message["content"] += "\nRespond in JSON format."
                    break
            else:
                prepared_messages.insert(
                    0,
                    {"role": "system", "content": "Respond in JSON format."},
                )

    last_error = None

    for attempt in range(max_retries):
        try:
            time.sleep(API_CALL_DELAY)
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
            error_text = str(exc).lower()
            retryable = any(
                token in error_text
                for token in [
                    "429",
                    "rate_limit",
                    "too many requests",
                    "500",
                    "502",
                    "503",
                    "timeout",
                ]
            )

            if retryable and attempt < max_retries - 1:
                wait_seconds = min(2 ** (attempt + 1), 30)
                logger.warning(
                    "Groq API error (attempt %s/%s): %s; retrying in %ss",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            if retryable:
                logger.error("Groq API failed after %s attempts: %s", max_retries, exc)
            raise

    raise last_error  # type: ignore[misc]


def groq_chat_json(
    messages: list[dict],
    *,
    temperature: float = 0.3,
    max_tokens: int = 800,
    max_retries: int = 3,
    raise_on_parse_error: bool = False,
) -> dict:
    """Call Groq and parse the response as JSON."""
    text = groq_chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
        max_retries=max_retries,
    )

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("Groq response was not valid JSON: %s", exc)
        if raise_on_parse_error:
            raise
        return {}

    if isinstance(parsed, dict):
        return parsed

    logger.error("Groq JSON response was not an object: %s", type(parsed).__name__)
    if raise_on_parse_error:
        raise ValueError("Groq JSON response was not an object.")
    return {}
