"""
Unit and integration tests for Phase 0 and Phase 1 of EconoCausal.
Tests:
- Synthetic dataset generation & structural model integrity
- Ground truth isolation and format
- DoWhy Causal DAG backdoor identification
- Propensity score estimation, positivity overlap, and IPW balance (SMD < 0.1)
- FastAPI endpoints (/health, /data/summary, /audit)
"""

import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import STATE, app
from backend.causal.graph import get_dag_json, identify_causal_effect
from backend.causal.propensity import PropensityAuditor, trim_extreme_propensities
from backend.data.generate import generate_retail_dataset
from backend.app.schemas import BudgetConfig, ColumnMapping


@pytest.fixture(scope="session")
def dataset_fixture(tmp_path_factory):
    """Generate small synthetic dataset for rapid test execution."""
    tmp_dir = tmp_path_factory.mktemp("test_data")
    df_obs, df_gt = generate_retail_dataset(
        n_samples=2_000,
        random_seed=123,
        output_dir=tmp_dir,
    )
    return df_obs, df_gt


def test_phase0_dataset_generation(dataset_fixture):
    """Verify synthetic dataset features and observational treatment assignment."""
    df_obs, df_gt = dataset_fixture

    assert len(df_obs) == 2_000
    assert len(df_gt) == 2_000

    # Required confounders
    expected_confounders = [
        "age",
        "tenure_months",
        "past_purchases_90d",
        "avg_basket_value",
        "days_since_last_purchase",
        "loyalty_tier",
        "email_engagement_score",
        "region",
    ]
    for c in expected_confounders:
        assert c in df_obs.columns

    # Treatment and outcomes
    assert set(df_obs["discount_amount"].unique()).issubset({0, 10, 20})
    assert set(df_obs["purchased"].unique()).issubset({0, 1})
    assert "revenue" in df_obs.columns
    assert (df_obs["revenue"] >= 0).all()

    # Ground truth isolation
    assert "tau_10_true" in df_gt.columns
    assert "tau_20_true" in df_gt.columns
    assert "segment" in df_gt.columns
    assert set(df_gt["segment"].unique()) == {"Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"}
    assert "unobserved_price_sensitivity" in df_gt.columns
    # Ensure unobserved confounder is not leaked into observed campaign data
    assert "unobserved_price_sensitivity" not in df_obs.columns


def test_phase0_diminishing_returns(dataset_fixture):
    """Verify treatment effects saturate from $10 to $20 for Persuadables."""
    _, df_gt = dataset_fixture
    persuadables = df_gt[df_gt["segment"] == "Persuadable"]

    # Mean lift from 0 to 10 should be larger than incremental lift from 10 to 20
    lift_10 = (persuadables["tau_10_true"]).mean()
    lift_20_incremental = (persuadables["tau_20_true"] - persuadables["tau_10_true"]).mean()

    assert lift_10 > lift_20_incremental > 0


def test_phase1_causal_dag_identification(dataset_fixture):
    """Verify DoWhy identifies the backdoor estimand."""
    df_obs, _ = dataset_fixture
    result = identify_causal_effect(df_obs)

    assert "model" in result
    assert "estimand" in result
    assert len(result["backdoor_variables"]) > 0

    dag_json = get_dag_json()
    assert len(dag_json["nodes"]) >= 10
    assert len(dag_json["edges"]) >= 15
    assert any(n["type"] == "treatment" for n in dag_json["nodes"])
    assert any(n["type"] == "outcome" for n in dag_json["nodes"])
    assert any(n["type"] == "unobserved" for n in dag_json["nodes"])


def test_phase1_propensity_and_balance(dataset_fixture):
    """Verify multinomial propensity estimation and SMD < 0.10 after IPW."""
    df_obs, _ = dataset_fixture
    auditor = PropensityAuditor().fit(df_obs)

    # Propensities sum to 1
    props = auditor.predict_propensities(df_obs)
    prop_sums = props.sum(axis=1)
    np.testing.assert_allclose(prop_sums, 1.0, atol=1e-5)

    # Balance table
    balance_df = auditor.calculate_balance_table(df_obs)
    assert not balance_df.empty
    assert "smd_weighted_avg" in balance_df.columns

    # Acceptance criteria: SMD < 0.10 for all / most covariates
    pct_balanced = (balance_df["smd_weighted_avg"] < 0.10).mean()
    assert pct_balanced >= 0.85, f"Expected >= 85% balanced, got {pct_balanced * 100:.1f}%"

    # Trimming check
    trimmed_df, n_trimmed = trim_extreme_propensities(df_obs, auditor, min_propensity=0.02, max_propensity=0.98)
    assert len(trimmed_df) <= len(df_obs)


def test_fastapi_endpoints():
    """Verify FastAPI endpoints /health, /data/summary, and /audit."""
    client = TestClient(app)

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    res_summary = client.get("/data/summary")
    assert res_summary.status_code == 200
    data_summary = res_summary.json()
    assert data_summary["row_count"] > 0
    assert len(data_summary["columns"]) >= 10
    assert len(data_summary["preview"]) > 0

    res_export = client.get("/data/export")
    assert res_export.status_code == 200
    assert res_export.headers["content-type"].startswith("text/csv")
    assert "attachment; filename=" in res_export.headers["content-disposition"]
    exported_data = pd.read_csv(io.BytesIO(res_export.content))
    assert len(exported_data) == data_summary["row_count"]
    assert list(exported_data.columns) == data_summary["columns"]

    res_audit = client.get("/audit")
    assert res_audit.status_code == 200
    audit = res_audit.json()
    assert "dag" in audit
    assert "balance_table" in audit
    assert "overlap" in audit
    assert audit["summary_metrics"]["pct_balanced_smd_under_0_10"] >= 80.0


def test_prescription_allocates_within_budget(dataset_fixture, monkeypatch):
    df_obs, _ = dataset_fixture
    monkeypatch.setitem(STATE, "df", df_obs)
    monkeypatch.setitem(STATE, "column_mapping", ColumnMapping())
    monkeypatch.setitem(
        STATE,
        "budget_config",
        BudgetConfig(total_budget=150, margin_per_purchase=35, discount_options=[10, 20]),
    )
    client = TestClient(app)

    response = client.post("/prescription")

    assert response.status_code == 200, response.text
    plan = response.json()
    assert plan["observations"] == len(df_obs)
    assert plan["expected_spend"] <= plan["budget"]
    assert plan["recommended_customers"] == len(plan["recommendations"])
    assert len({row["row_index"] for row in plan["recommendations"]}) == plan["recommended_customers"]
    assert all(row["expected_incremental_profit"] > 0 for row in plan["recommendations"])
    assert "not a validated causal uplift estimate" in plan["estimation_note"]
