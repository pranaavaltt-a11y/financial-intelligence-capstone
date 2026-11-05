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
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3-flash-preview", "gemini-3.8-flash", "gemini-flash-latest"]


def suggested_model(message: str) -> str | None:
    """Pull 'models/<name>' recommended in a 'model is no longer available' error."""
    m = re.search(r"(?:use|to use)\s+models/([\w.\-]+)", message)
    return m.group(1) if m else None


class GeminiLLM:
    """Gemini text generation that survives the free tier's rough edges.

    * 404 (model retired)        -> switch to the model Google recommends, then the fallback list
    * 503 / 500 (overloaded)     -> wait 2s and retry once; if still failing, try the next model
    * 429 per-minute rate limit  -> wait the delay Google asks for (max 10s) and retry
    * no reply within 15s        -> treated as overloaded
    * 429 daily quota, bad key,
      or 25s spent on one call   -> give up (caller falls back to the verified template answer)
    * after a failure Gemini is skipped for 5 minutes, so the app stays fast while Google is down
    """

    def __init__(self, api_key: str, model: str = FALLBACK_MODELS[0], client=None, sleep=None,
                 retries: int = 1, max_wait: float = 10.0, timeout_s: float = 15.0, budget_s: float = 25.0,
                 clock=None, cooldown_s: float = 300.0):
        if client is None:
            from google import genai
            from google.genai import types

            # A request that never answers is treated like any other failure (it used to hang forever).
            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        import time

        self.client = client
        self.clock = clock or time.monotonic
        self.budget_s = budget_s
        self.model = model
        self.name = model
        self.sleep = sleep or time.sleep
        self.retries = retries
        self.max_wait = max_wait
        self._tried: set[str] = set()
        self.last_error: str | None = None
        self.cooldown_s = cooldown_s
        self.down_until = 0.0

    def _next_model(self, message: str) -> bool:
        self._tried.add(self.model)
        for m in [suggested_model(message)] + FALLBACK_MODELS:
            if m and m not in self._tried:
                log.warning("Model %s unavailable; switching to %s", self.model, m)
                self.model = self.name = m
                return True
        return False

    def generate(self, prompt: str, as_json: bool = False) -> str | None:
        from google.genai import types

        cfg = types.GenerateContentConfig(temperature=0.0,
                                          response_mime_type="application/json" if as_json else "text/plain")
        if self.clock() < self.down_until:
            self.last_error = "skipped: Gemini failed in the last few minutes (" + (self.last_error or "") + ")"
            return None
        attempt = 0
        start = self.clock()
        while True:
            if self.clock() - start > self.budget_s:
                self.last_error = (self.last_error or "") + " (gave up after the time limit)"
                break
            try:
                res = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                self.last_error = None
                return res.text
            except Exception as exc:
                msg = str(exc)
                self.last_error = msg[:300]
                if "404" in msg or "NOT_FOUND" in msg:
                    if self._next_model(msg):
                        attempt = 0
                        continue
                    break
                transient = any(k in msg for k in ("503", "UNAVAILABLE", "500", "INTERNAL", "overloaded",
                                                   "timed out", "Timeout", "timeout"))
                per_minute = ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and not re.search(r"PerDay|per day", msg)
                if transient or per_minute:
                    if attempt < self.retries:
                        m = re.search(r"retry in ([\d.]+)s", msg)
                        wait = min(float(m.group(1)) + 1 if m else 2.0 * 2 ** attempt, self.max_wait)
                        attempt += 1
                        log.info("Gemini busy (%s) - retrying in %.0fs (%d/%d)",
                                 "overloaded" if transient else "rate limit", wait, attempt, self.retries)
                        self.sleep(wait)
                        continue
                    if transient and self._next_model(msg):     # this model stays overloaded: try another
                        attempt = 0
                        continue
                break
        log.warning("LLM call failed: %s", self.last_error)
        self.down_until = self.clock() + self.cooldown_s
        if self.cooldown_s:
            log.warning("Skipping Gemini for %.0f minutes; answers use the verified template meanwhile",
                        self.cooldown_s / 60)
        return None


def check_models(api_key: str, timeout_s: float = 15.0) -> list[dict]:
    """List the text models this key can use and try each with a tiny prompt (for `finintel check-llm`)."""
    import time

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        name = (m.name or "").removeprefix("models/")
        if "generateContent" in actions and "gemini" in name and not any(
                k in name for k in ("embedding", "image", "tts", "audio", "live", "vision")):
            names.append(name)
    flash_first = sorted(names, key=lambda n: (0 if "flash" in n else 1, "lite" in n, n))
    out = []
    for name in flash_first[:8]:
        t0 = time.monotonic()
        try:
            res = client.models.generate_content(model=name, contents="Reply with the single word OK.",
                                                 config=types.GenerateContentConfig(temperature=0.0))
            out.append({"model": name, "ok": True, "seconds": round(time.monotonic() - t0, 1),
                        "reply": (res.text or "").strip()[:20]})
        except Exception as exc:
            out.append({"model": name, "ok": False, "seconds": round(time.monotonic() - t0, 1),
                        "error": str(exc)[:140]})
    return out


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
- Use passages only for explanations (the "why"). Look for sentences stating causes - "primarily due to",
  "driven by", "as a result of", "partially offset by" - and summarise them close to the passage's own wording,
  citing [S#]. A sentence citing [S#] must restate what that passage says.
- Only say the evidence does not explain something if no passage discusses it.
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
