import os
import json
import re
import asyncio
from typing import Any, Optional, Callable
from langchain_groq import ChatGroq
from huggingface_hub import AsyncInferenceClient
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


def groq(model: str, temperature: float = 0.1, max_tokens: int = 2048, reasoning_effort: Optional[str] = None) -> ChatGroq:
    kwargs = {
        "api_key": os.environ.get("GROQ_API_KEY"),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatGroq(**kwargs)


note_llm = groq("qwen/qwen3.8-27b", temperature=0.2,
                max_tokens=512, reasoning_effort="none")


hf = AsyncInferenceClient(token=os.environ.get("HF_TOKEN"))
MEDGEMMA_MODEL = "google/medgemma-27b-text-it"


async def medgemma_stream(
    system: str,
    user: str,
    on_token: Optional[Callable[[str], Any]] = None,
    max_tokens: int = 1536,
    temperature: float = 0.2,
    attempts: int = 3
) -> str:
    last_err = None

    for attempt in range(1, attempts + 1):
        try:
            full = ""
            stream = await hf.chat_completion(
                model=MEDGEMMA_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full += delta
                    if on_token:
                        if asyncio.iscoroutinefunction(on_token):
                            await on_token(delta)
                        else:
                            on_token(delta)
            return full
        except Exception as err:
            last_err = err
            msg = str(err).lower()

            transient = any(x in msg for x in [
                            "busy", "rate limit", "429", "503", "timeout", "overloaded"])
            if not transient or attempt == attempts:
                raise err

            wait_ms = 1000 * (2 ** (attempt - 1))
            print(
                f"[HealthPilot] MedGemma busy (attempt {attempt}/{attempts}), retrying in {wait_ms}ms")
            await asyncio.sleep(wait_ms / 1000.0)

    raise last_err


async def medgemma_complete(system: str, user: str, max_tokens: int = 1536, temperature: float = 0.2) -> str:
    return await medgemma_stream(system=system, user=user, max_tokens=max_tokens, temperature=temperature)


def parse_json_block(raw: str) -> Any:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    body = match.group(1).strip() if match else raw.strip()

    start = re.search(r"[{[]", body)
    if start is None:
        raise ValueError("No JSON object found in model output")

    return json.loads(body[start.start():])
