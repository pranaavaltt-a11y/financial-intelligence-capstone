"""LLM access (Google Gemini free tier) behind a small interface.

If GEMINI_API_KEY is unset or a call fails (quota, network), callers get None and fall back to the
deterministic template path, so the system always produces a verified answer.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

log = logging.getLogger(__name__)


class LLM(Protocol):
    name: str

    def generate(self, prompt: str, as_json: bool = False) -> str | None: ...


# Tried in order when a model is retired (HTTP 404). Google's error message usually names the replacement,
# which is tried first; FININTEL_LLM_MODEL overrides the default.
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash"]


def suggested_model(message: str) -> str | None:
    """Pull 'models/<name>' recommended in a 'model is no longer available' error."""
    m = re.search(r"(?:use|to use)\s+models/([\w.\-]+)", message)
    return m.group(1) if m else None


class GeminiLLM:
    def __init__(self, api_key: str, model: str = FALLBACK_MODELS[0], client=None):
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self.client = client
        self.model = model
        self.name = model
        self._tried: set[str] = set()

    def _candidates(self, failed: str, message: str) -> list[str]:
        out = [suggested_model(message)] + FALLBACK_MODELS
        return [m for m in out if m and m != failed and m not in self._tried]

    def generate(self, prompt: str, as_json: bool = False) -> str | None:
        from google.genai import types

        cfg = types.GenerateContentConfig(temperature=0.0,
                                          response_mime_type="application/json" if as_json else "text/plain")
        while True:
            try:
                res = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                return res.text
            except Exception as exc:  # quota, network, safety block, retired model
                msg = str(exc)
                if "404" in msg or "NOT_FOUND" in msg:
                    self._tried.add(self.model)
                    nxt = self._candidates(self.model, msg)
                    if nxt:
                        log.warning("Model %s unavailable; switching to %s", self.model, nxt[0])
                        self.model = self.name = nxt[0]
                        continue
                log.warning("LLM call failed: %s", exc)
                return None


def parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def make_llm(api_key: str | None, model: str = FALLBACK_MODELS[0]) -> LLM | None:
    if not api_key:
        return None
    try:
        return GeminiLLM(api_key, model)
    except Exception as exc:
        log.warning("Could not start Gemini client: %s", exc)
        return None


PLAN_PROMPT = """You plan financial questions. Known companies (ticker: name): {companies}
Known metrics: {metrics}. Years with data: {years}.
Return JSON only: {{"companies": [tickers], "metrics": [metric names from the list], "years": [ints],
"operation": "lookup"|"change"|"compare"|"cagr", "needs_explanation": true|false}}
Question: {question}"""

ANSWER_PROMPT = """You are a financial analyst assistant. Answer the question using ONLY the evidence below.
Rules:
- Copy numbers exactly as written in FACTS or CALCULATIONS. Never compute or estimate new numbers.
- Put a citation after every sentence: [F#] for facts, [C#] for calculations, [S#] for passages.
- Use passages only for explanations (the "why"); a sentence citing [S#] must restate what that passage says.
- If the evidence does not answer part of the question, say so plainly.
- At most 120 words, plain prose, no headings.
{feedback}
QUESTION: {question}

FACTS:
{facts}

CALCULATIONS:
{calcs}

PASSAGES:
{passages}
"""
