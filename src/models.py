import os
import time
from typing import Callable

import anthropic
from openai import OpenAI
from xai_sdk import Client
from xai_sdk.chat import user


OPUS_MODEL = "claude-opus-4-7"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
GPT_MODEL = "gpt-5.5"
GROK_MODEL = "grok-4.3"


def retry_call(func: Callable[[], str], retries: int = 3, delay_seconds: int = 30) -> str:
    """
    Run an API call with simple retry logic.

    If a provider API has a temporary failure, wait briefly and try again.
    After the final attempt, raise the error so the notebook shows what went wrong.
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as error:
            last_error = error
            print(f"Attempt {attempt} failed: {error}")

            if attempt < retries:
                print(f"Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)

    raise last_error


def extract_claude_text(response) -> str:
    """
    Extract text from an Anthropic response safely.

    Claude responses usually contain one or more content blocks. We collect all text
    blocks rather than assuming response.content[0].text always exists.
    """
    text_parts = []

    for block in response.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)

    if text_parts:
        return "\n".join(text_parts).strip()

    stop_reason = getattr(response, "stop_reason", "unknown")
    raise ValueError(f"Claude returned no text content. stop_reason={stop_reason}")


def call_claude(prompt: str, model: str, max_tokens: int = 800) -> str:
    """
    Call an Anthropic Claude model and return the text response.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Missing ANTHROPIC_API_KEY in .env")

    client = anthropic.Anthropic(api_key=api_key)

    def _call() -> str:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return extract_claude_text(response)

    return retry_call(_call)


def call_opus(prompt: str, max_tokens: int = 800) -> str:
    """
    Call Claude Opus 4.7.
    """
    return call_claude(
        prompt=prompt,
        model=OPUS_MODEL,
        max_tokens=max_tokens,
    )


def call_haiku(prompt: str, max_tokens: int = 800) -> str:
    """
    Call Claude Haiku 4.5.
    """
    return call_claude(
        prompt=prompt,
        model=HAIKU_MODEL,
        max_tokens=max_tokens,
    )


def call_gpt5(prompt: str, max_tokens: int = 800) -> str:
    """
    Call GPT-5.5 via the OpenAI Responses API.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in .env")

    client = OpenAI(api_key=api_key)

    def _call() -> str:
        response = client.responses.create(
            model=GPT_MODEL,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        return response.output_text

    return retry_call(_call)


def call_grok(prompt: str, max_tokens: int = 800) -> str:
    """
    Call Grok 4.3 via the xAI SDK.

    max_tokens is accepted for interface consistency with the other wrappers,
    but is not passed into xAI yet because the installed SDK's chat.create()
    method rejected max_completion_tokens during smoke testing.
    """
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("Missing XAI_API_KEY in .env")

    client = Client(
        api_key=api_key,
        timeout=3600,
    )

    def _call() -> str:
        chat = client.chat.create(
            model=GROK_MODEL,
        )
        chat.append(user(prompt))
        response = chat.sample()
        return response.content

    return retry_call(_call)