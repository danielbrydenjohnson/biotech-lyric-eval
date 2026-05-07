import os
import time
from typing import Callable

import anthropic
from openai import OpenAI
from google import genai


OPUS_MODEL = "claude-opus-4-7"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
GPT_MODEL = "gpt-5.5"
GEMINI_MODEL = "gemini-3.1-pro-preview"


def retry_call(func: Callable[[], str], retries: int = 3, delay_seconds: int = 5) -> str:
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


def call_claude(prompt: str, model: str, max_tokens: int = 800, temperature: float = 0.8) -> str:
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
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return response.content[0].text

    return retry_call(_call)


def call_opus(prompt: str, max_tokens: int = 800, temperature: float = 0.8) -> str:
    """
    Call Claude Opus 4.7.
    """
    return call_claude(
        prompt=prompt,
        model=OPUS_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def call_haiku(prompt: str, max_tokens: int = 800, temperature: float = 0.8) -> str:
    """
    Call Claude Haiku 4.5.
    """
    return call_claude(
        prompt=prompt,
        model=HAIKU_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def call_gpt5(prompt: str, max_tokens: int = 800, temperature: float = 0.8) -> str:
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
            temperature=temperature,
        )
        return response.output_text

    return retry_call(_call)


def call_gemini(prompt: str, max_tokens: int = 800, temperature: float = 0.8) -> str:
    """
    Call Gemini 3.1 Pro via the Google GenAI SDK.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY in .env")

    client = genai.Client(api_key=api_key)

    def _call() -> str:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        return response.text

    return retry_call(_call)