import pytest

from ragkit.observability import cost


def test_cost_recording_and_budget_cap(settings):
    settings.llm_provider = "openai"  # budget checks only apply to paid providers
    settings.daily_budget_usd = 0.001
    usd = cost.record(settings, "llm", "gpt-5-nano", input_tokens=10_000, output_tokens=1_000)
    assert usd == pytest.approx((10_000 * 0.05 + 1_000 * 0.40) / 1e6)
    cost.check_budget(settings)  # 0.0009 < 0.001
    cost.record(settings, "embedding", "text-embedding-3-small", input_tokens=10_000)
    with pytest.raises(cost.BudgetExceededError):
        cost.check_budget(settings)
    assert cost.spent_total(settings) == pytest.approx(cost.spent_today(settings))


def test_fake_providers_skip_budget(settings):
    settings.daily_budget_usd = 0.0
    cost.check_budget(settings)  # fake LLM + fake embeddings: free, never blocked
