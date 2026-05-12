from doge.models import BreakdownSuggestion
from doge.ontologist import _base_url, _model_name


def test_model_name_preserves_provider_specific_names() -> None:
    assert _model_name("gpt-5.2") == "gpt-5.2"
    assert _model_name("openai:gpt-5.2") == "gpt-5.2"
    assert _model_name("GPT-5.5") == "GPT-5.5"


def test_base_url_accepts_openai_compatible_endpoint_env(monkeypatch) -> None:
    monkeypatch.delenv("DOGE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("BASE_URL", "https://openrouter.ai/api/v1")

    assert _base_url() == "https://openrouter.ai/api/v1"


def test_breakdown_accepts_predicate_alias_from_llm() -> None:
    suggestion = BreakdownSuggestion.model_validate(
        {
            "logic_gate": "AND",
            "conditions": [
                {
                    "predicate": "no_specific_asset",
                    "description": "No named financial instrument or asset class.",
                }
            ],
        }
    )

    assert suggestion.conditions[0].name == "no_specific_asset"
    assert suggestion.conditions[0].condition == "No named financial instrument or asset class."
