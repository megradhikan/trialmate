"""
llm/client.py
LLM API wrapper supporting Anthropic Claude, OpenAI, and local Ollama models.
Includes prompt caching via functools.lru_cache + disk cache.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from loguru import logger

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_CACHE_DIR = os.getenv("LLM_CACHE_DIR", ".llm_cache")
Path(LLM_CACHE_DIR).mkdir(exist_ok=True)


def _cache_key(prompt: str, system: str = "") -> str:
    h = hashlib.sha256(f"{system}||{prompt}".encode()).hexdigest()[:16]
    return h


def _read_cache(key: str) -> Optional[str]:
    path = Path(LLM_CACHE_DIR) / f"{key}.txt"
    if path.exists():
        return path.read_text()
    return None


def _write_cache(key: str, content: str):
    path = Path(LLM_CACHE_DIR) / f"{key}.txt"
    path.write_text(content)


class LLMClient:
    def __init__(self):
        self.provider = LLM_PROVIDER
        self._setup_client()

    def _setup_client(self):
        if self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        elif self.provider == "openai":
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif self.provider == "local":
            # Ollama via HTTP
            self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            self.model = os.getenv("LOCAL_MODEL", "mistral:7b-instruct")
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    async def complete(
        self,
        prompt: str,
        system: str = "You are a clinical research assistant. Return only valid JSON.",
        max_tokens: int = 1000,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> str:
        """
        Send a prompt and return the text response.
        Uses disk cache to avoid redundant LLM calls.
        """
        cache_key = _cache_key(prompt, system)
        if use_cache:
            cached = _read_cache(cache_key)
            if cached:
                logger.debug(f"LLM cache hit: {cache_key}")
                return cached

        # Log prompt (no PHI)
        logger.debug(f"LLM call [{self.provider}] model={self.model} prompt_len={len(prompt)}")

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = response.content[0].text

            elif self.provider == "openai":
                import openai
                response = await self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                result = response.choices[0].message.content

            elif self.provider == "local":
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{self.base_url}/api/generate",
                        json={"model": self.model, "prompt": f"{system}\n\n{prompt}", "stream": False},
                        timeout=60,
                    )
                    result = resp.json()["response"]
            else:
                result = ""

            if use_cache:
                _write_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
