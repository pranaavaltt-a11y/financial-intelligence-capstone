"""Gemini model fallback, with a fake client (no network)."""

from types import SimpleNamespace

from google.genai import errors

from finintel.agent.llm import GeminiLLM, suggested_model

RETIRED = ("This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use "
           "models/gemini-3.8-flash for the latest features and improvements.")


def not_found(msg=RETIRED):
    return errors.ClientError(404, {"error": {"code": 404, "message": msg, "status": "NOT_FOUND"}})


class FakeModels:
    def __init__(self, available):
        self.available = available
        self.seen = []

    def generate_content(self, model, contents, config):
        self.seen.append(model)
        if model not in self.available:
            raise not_found()
        return SimpleNamespace(text=f"answer from {model}")


def test_suggested_model_is_parsed_from_error():
    assert suggested_model(RETIRED) == "gemini-3.8-flash"
    assert suggested_model("some other error") is None


def test_switches_to_the_model_google_recommends():
    models = FakeModels({"gemini-3.8-flash"})
    llm = GeminiLLM("k", model="gemini-2.5-flash", client=SimpleNamespace(models=models))
    assert llm.generate("q") == "answer from gemini-3.8-flash"
    assert llm.model == "gemini-3.8-flash" and models.seen == ["gemini-2.5-flash", "gemini-3.8-flash"]
    assert llm.generate("q2") == "answer from gemini-3.8-flash"      # sticks with the working model


def test_gives_up_cleanly_when_no_model_works():
    llm = GeminiLLM("k", model="gemini-2.5-flash", client=SimpleNamespace(models=FakeModels(set())))
    assert llm.generate("q") is None


def overloaded():
    return errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE",
                                              "message": "This model is currently experiencing high demand."}})


class FlakyModels:
    """Fails with 503 for the first `fails` calls to `bad` model; other models always work."""

    def __init__(self, fails, bad="gemini-3.8-flash"):
        self.fails, self.bad, self.seen = fails, bad, []

    def generate_content(self, model, contents, config):
        self.seen.append(model)
        if model == self.bad and self.fails > 0:
            self.fails -= 1
            raise overloaded()
        return SimpleNamespace(text=f"answer from {model}")


def test_retries_when_google_is_overloaded():
    slept = []
    llm = GeminiLLM("k", client=SimpleNamespace(models=FlakyModels(fails=2)), sleep=slept.append)
    assert llm.generate("q") == "answer from gemini-3.8-flash"
    assert slept == [2.0, 4.0]


def test_switches_model_when_one_stays_overloaded():
    slept = []
    models = FlakyModels(fails=99)
    llm = GeminiLLM("k", client=SimpleNamespace(models=models), sleep=slept.append)
    assert llm.generate("q") == "answer from gemini-flash-latest"
    assert len(slept) == 3 and llm.model == "gemini-flash-latest"


def test_daily_quota_is_not_retried():
    class Daily:
        def generate_content(self, model, contents, config):
            raise errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                                     "message": "Quota exceeded: GenerateRequestsPerDayPerProject"}})
    slept = []
    llm = GeminiLLM("k", client=SimpleNamespace(models=Daily()), sleep=slept.append)
    assert llm.generate("q") is None and slept == [] and "429" in llm.last_error


class HangingModels:
    """Simulates a request that times out (the httpx error the SDK raises)."""

    def __init__(self, bad):
        self.bad, self.seen = bad, []

    def generate_content(self, model, contents, config):
        self.seen.append(model)
        if model in self.bad:
            raise TimeoutError("The read operation timed out")
        return SimpleNamespace(text=f"answer from {model}")


def test_timeout_is_retried_then_switches_model():
    llm = GeminiLLM("k", client=SimpleNamespace(models=HangingModels({"gemini-3.8-flash"})), sleep=lambda s: None)
    assert llm.generate("q") == "answer from gemini-flash-latest"


def test_overall_time_budget_stops_retrying():
    t = [0.0]

    def sleep(s):
        t[0] += s

    llm = GeminiLLM("k", client=SimpleNamespace(models=HangingModels({"gemini-3.8-flash", "gemini-flash-latest",
                                                                       "gemini-2.5-flash"})),
                    sleep=sleep, clock=lambda: t[0], budget_s=10)
    assert llm.generate("q") is None and "time limit" in llm.last_error
    assert t[0] <= 10 + 8
