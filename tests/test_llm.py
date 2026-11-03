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
