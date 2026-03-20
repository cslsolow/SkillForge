"""
Thin LLM wrapper around litellm with retry logic.

Credentials are read from environment variables (OPENAI_API_KEY, OPENAI_BASE_URL)
or passed explicitly as keyword arguments to the call site.
"""

from __future__ import annotations

import os

import litellm
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


def query(
    prompt: str,
    model: str | None = None,
    system_prompt: str | None = None,
    **kwargs,
) -> str:
    """Send a single-turn prompt and return the response string."""
    model = model or os.getenv("SWE_LEARNER_MODEL")
    if not model:
        raise ValueError(
            "No model specified. Set the SWE_LEARNER_MODEL env var or pass model=."
        )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = _query_with_retry(model, messages, **kwargs)
    return response.choices[0].message.content or ""


def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    **kwargs,
) -> str:
    """Send a multi-turn message list and return the response string."""
    model = model or os.getenv("SWE_LEARNER_MODEL")
    if not model:
        raise ValueError(
            "No model specified. Set the SWE_LEARNER_MODEL env var or pass model=."
        )
    response = _query_with_retry(model, messages, **kwargs)
    return response.choices[0].message.content or ""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def _query_with_retry(model: str, messages: list[dict], **kwargs):
    return litellm.completion(model=model, messages=messages, **kwargs)
