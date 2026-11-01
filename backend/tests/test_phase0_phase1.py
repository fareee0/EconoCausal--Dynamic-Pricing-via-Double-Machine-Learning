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
from backend.data.generate_churn_campaign import generate_churn_campaign_dataset
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

    overlap = auditor.get_overlap_diagnostics(df_obs)
    weight_diagnostics = overlap["weight_diagnostics"]
    assert set(weight_diagnostics) == {"arm_0", "arm_10", "arm_20"}
    for arm, metrics in weight_diagnostics.items():
        assert metrics["observations"] > 0
        assert 0 < metrics["effective_sample_size"] <= metrics["observations"]
        assert 0 < metrics["effective_sample_size_fraction"] <= 1
        assert metrics["max_weight"] > 0

    support = overlap["assignment_support"]
    for arm, metrics in support.items():
        treatment = int(arm.removeprefix("arm_"))
        received_mask = df_obs["discount_amount"] == treatment
        received_propensity = props.loc[received_mask, f"propensity_{treatment}"]
        assert metrics["observations"] == int(received_mask.sum())
        assert metrics["median_received_propensity"] == pytest.approx(
            received_propensity.median(), abs=0.0001
        )
        assert metrics["low_propensity_count"] == int((received_propensity < 0.05).sum())
        assert 0 <= metrics["low_propensity_fraction"] <= 1


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
    assert audit["overlap"]["weight_diagnostics"]["arm_0"]["effective_sample_size"] > 0
    assert "arm_0" in audit["overlap"]["assignment_support"]


def test_uplift_endpoint_returns_summary():
    client = TestClient(app)
    response = client.get("/uplift")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["observations"] > 0
    assert "summary_metrics" in payload
    assert "by_segment" in payload
    assert "qini_curve" in payload


def test_churn_endpoint_uses_recency_proxy_without_target_leakage(dataset_fixture, monkeypatch):
    df_obs, _ = dataset_fixture
    monkeypatch.setitem(STATE, "df", df_obs)
    monkeypatch.setitem(STATE, "column_mapping", ColumnMapping())

    response = TestClient(app).get("/churn")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["label_source"] == "recency_proxy"
    assert "90 days" in payload["target_definition"]
    assert "days_since_last_purchase" not in payload["feature_columns"]
    assert len(payload["high_risk_customers"]) == 25
    assert all(0 <= row["churn_probability"] <= 1 for row in payload["high_risk_customers"])
    assert all(0 <= value <= 1 for value in payload["validation"]["metrics"].values())
    assert payload["model"] in {"Logistic Regression", "Random Forest"}
    assert set(payload["validation"]["candidate_roc_auc"]) == {
        "Logistic Regression",
        "Random Forest",
    }
    selected_auc = payload["validation"]["candidate_roc_auc"][payload["model"]]
    assert selected_auc >= max(payload["validation"]["candidate_roc_auc"].values()) - 0.0001
    assert 0 <= payload["validation"]["decision_threshold"] <= 1
    assert payload["top_churn_drivers"]
    assert all(
        driver["feature"] in payload["feature_columns"]
        for driver in payload["top_churn_drivers"]
    )


def test_churn_endpoint_prefers_explicit_label(dataset_fixture, monkeypatch):
    df_obs, _ = dataset_fixture
    df_with_churn = df_obs.copy()
    df_with_churn["churn"] = np.where(df_with_churn["days_since_last_purchase"] >= 90, "Yes", "No")
    monkeypatch.setitem(STATE, "df", df_with_churn)
    monkeypatch.setitem(STATE, "column_mapping", ColumnMapping())

    response = TestClient(app).get("/churn")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["label_source"] == "explicit"
    assert "Observed churn label" in payload["target_definition"]
    assert "churn" not in payload["feature_columns"]
    assert "days_since_last_purchase" in payload["feature_columns"]


def test_uploading_different_csv_remaps_columns_and_enables_predictions(monkeypatch):
    row_count = 120
    uploaded = pd.DataFrame({
        "customer_id": [f"new-{index}" for index in range(row_count)],
        "discount": np.tile([0, 10], row_count // 2),
        "conversion": np.tile([0, 1, 1, 0], row_count // 4),
        "tenure": np.arange(row_count) % 36,
        "segment": np.tile(["new", "returning", "loyal"], row_count // 3),
        "churn": np.tile([0, 1], row_count // 2),
    })
    monkeypatch.setitem(STATE, "column_mapping", ColumnMapping())
    monkeypatch.setitem(STATE, "budget_config", BudgetConfig())
    client = TestClient(app)

    upload_response = client.post(
        "/upload",
        files={"file": ("different_campaign.csv", uploaded.to_csv(index=False), "text/csv")},
    )

    assert upload_response.status_code == 200, upload_response.text
    mapping = upload_response.json()["column_mapping"]
    assert mapping["treatment"] == "discount"
    assert mapping["outcome"] == "conversion"
    assert mapping["confounders"] == ["tenure", "segment"]

    config_response = client.post(
        "/config",
        json={
            "column_mapping": mapping,
            "budget_config": BudgetConfig().model_dump(),
        },
    )
    assert config_response.status_code == 200, config_response.text

    churn_response = client.get("/churn")
    assert churn_response.status_code == 200, churn_response.text
    assert churn_response.json()["label_source"] == "explicit"

    uplift_response = client.get("/uplift")
    assert uplift_response.status_code == 200, uplift_response.text
    assert uplift_response.json()["available_ground_truth"] is False


def test_churn_csv_without_campaign_columns_works_with_selected_target(monkeypatch):
    row_count = 120
    customer_data = pd.DataFrame({
        "CustomerId": [f"account-{index}" for index in range(row_count)],
        "CreditScore": np.arange(row_count) + 600,
        "Age": np.arange(row_count) % 50 + 20,
        "Balance": np.linspace(100, 10_000, row_count),
        "Geography": np.tile(["North", "South", "West"], row_count // 3),
        "Account Status": np.tile(["Exited", "Stayed"], row_count // 2),
    })
    monkeypatch.setitem(STATE, "column_mapping", ColumnMapping())
    client = TestClient(app)

    upload_response = client.post(
        "/upload",
        files={"file": ("bank_customers.csv", customer_data.to_csv(index=False), "text/csv")},
    )

    assert upload_response.status_code == 200, upload_response.text
    mapping = upload_response.json()["column_mapping"]
    assert mapping["treatment"] == ""
    assert mapping["outcome"] == ""
    assert "Account Status" not in mapping["confounders"]

    churn_response = client.get("/churn")
    assert churn_response.status_code == 200, churn_response.text
    churn_result = churn_response.json()
    assert churn_result["label_source"] == "explicit"
    assert churn_result["target_column"] == "Account Status"
    assert "Account Status" not in churn_result["feature_columns"]
    assert any(
        customer["customer_id"] == "account-0"
        for customer in churn_result["high_risk_customers"]
    )

    audit_response = client.get("/audit")
    assert audit_response.status_code == 400
    assert "treatment and purchase-outcome columns" in audit_response.json()["detail"]

    uplift_response = client.get("/uplift")
    assert uplift_response.status_code == 400
    assert "treatment and purchase-outcome columns" in uplift_response.json()["detail"]

    prescription_response = client.post("/prescription")
    assert prescription_response.status_code == 400
    assert "treatment and purchase-outcome columns" in prescription_response.json()["detail"]


def test_churn_endpoint_accepts_user_selected_positive_class(dataset_fixture, monkeypatch):
    df_obs, _ = dataset_fixture
    df_custom = df_obs.copy()
    df_custom["renewal_result"] = np.where(
        np.arange(len(df_custom)) % 2 == 0,
        "Needs follow-up",
        "Retained",
    )
    monkeypatch.setitem(STATE, "df", df_custom)
    monkeypatch.setitem(STATE, "column_mapping", ColumnMapping())

    response = TestClient(app).get(
        "/churn",
        params={"target": "renewal_result", "positive_value": "Needs follow-up"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["target_column"] == "renewal_result"
    assert payload["positive_value"] == "Needs follow-up"
    assert "renewal_result" not in payload["feature_columns"]


def test_generated_churn_campaign_dataset_runs_prediction_endpoints(monkeypatch):
    df = generate_churn_campaign_dataset(n_samples=1_200, random_seed=111)
    monkeypatch.setitem(STATE, "df", None)
    monkeypatch.setitem(STATE, "column_mapping", ColumnMapping())
    monkeypatch.setitem(STATE, "budget_config", BudgetConfig())
    client = TestClient(app)
    upload_response = client.post(
        "/upload",
        files={"file": ("retail_churn_campaign.csv", df.to_csv(index=False), "text/csv")},
    )

    assert upload_response.status_code == 200, upload_response.text
    assert upload_response.json()["column_mapping"]["treatment"] == "discount_amount"
    assert upload_response.json()["column_mapping"]["outcome"] == "purchased"
    assert client.get("/churn").status_code == 200
    assert client.get("/uplift").status_code == 200
    assert client.get("/audit").status_code == 200
    assert client.post("/prescription").status_code == 200


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
