"""
EconoCausal - FastAPI Application Entry Point.
Provides REST endpoints for:
- Data upload & preview
- Column mapping & budget configuration
- Causal audit (DAG, Backdoor Estimand, Propensity Overlap, Covariate Balance Table)
- Double ML training & Uplift metrics (Phase 2)
- Prescriptive Knapsack Optimization (Phase 3)
- Model Scoring & Monitoring (Phase 4)
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
import io
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.app.schemas import (
    AuditResponse,
    BudgetConfig,
    ColumnMapping,
    DataSummaryResponse,
    UpliftSummaryResponse,
)
from backend.causal.graph import get_dag_json, identify_causal_effect
from backend.causal.propensity import PropensityAuditor, trim_extreme_propensities

# Global in-memory state
STATE: Dict[str, Any] = {
    "df": None,
    "filename": "retail_campaign.csv",
    "column_mapping": ColumnMapping(),
    "budget_config": BudgetConfig(),
    "audit_cache": None,
}


def load_default_dataset() -> pd.DataFrame:
    """Load default synthetic campaign dataset if available."""
    paths = [
        Path("backend/data/retail_campaign.csv"),
        Path("data/retail_campaign.csv"),
        Path("../data/retail_campaign.csv"),
    ]
    for p in paths:
        if p.exists():
            return pd.read_csv(p)
    # If not generated yet, generate on the fly
    from backend.data.generate import generate_retail_dataset
    df_obs, _ = generate_retail_dataset(n_samples=20_000)
    return df_obs


def load_ground_truth_dataset() -> Optional[pd.DataFrame]:
    """Load ground-truth uplift labels when they are available for validation."""
    paths = [
        Path("backend/data/ground_truth.csv"),
        Path("data/ground_truth.csv"),
        Path("../data/ground_truth.csv"),
    ]
    for p in paths:
        if p.exists():
            return pd.read_csv(p)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize default state on startup."""
    try:
        STATE["df"] = load_default_dataset()
        print(f"Loaded campaign data with {len(STATE['df']):,} rows.")
    except Exception as e:
        print(f"Startup data loading notice: {e}")
    yield


app = FastAPI(
    title="EconoCausal API",
    version="1.0.0",
    description="Double Machine Learning & Prescriptive Promotion Targeting Engine",
    lifespan=lifespan,
)

# Enable CORS for frontend Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "EconoCausal",
        "dataset_loaded": STATE["df"] is not None,
        "rows": len(STATE["df"]) if STATE["df"] is not None else 0,
    }


@app.get("/data/summary", response_model=DataSummaryResponse)
def get_data_summary():
    """Return dataset overview, column mapping, budget config, and first 50 rows."""
    if STATE["df"] is None:
        STATE["df"] = load_default_dataset()

    df: pd.DataFrame = STATE["df"]
    preview_records = df.head(50).fillna("").to_dict(orient="records")

    return DataSummaryResponse(
        filename=STATE["filename"],
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        preview=preview_records,
        column_mapping=STATE["column_mapping"],
        budget_config=STATE["budget_config"],
    )


@app.get("/data/export")
def export_dataset():
    """Download the complete active dataset as a CSV file."""
    if STATE["df"] is None:
        STATE["df"] = load_default_dataset()

    filename = Path(STATE["filename"]).name
    filename = "".join(
        char if char.isascii() and (char.isalnum() or char in "._-") else "_"
        for char in filename
    ).strip("._")
    filename = filename or "campaign"
    if not filename.lower().endswith(".csv"):
        filename = f"{filename}.csv"

    return Response(
        content=STATE["df"].to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/upload", response_model=DataSummaryResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload CSV dataset, update active state, invalidate audit cache."""
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

        STATE["df"] = df
        STATE["filename"] = file.filename or "uploaded_campaign.csv"
        STATE["audit_cache"] = None  # invalidate cache

        # Auto-detect or adapt column mapping
        mapping = STATE["column_mapping"]
        available_cols = set(df.columns)
        confounders = [c for c in mapping.confounders if c in available_cols]
        if not confounders:
            confounders = [
                c for c in df.columns
                if c not in {mapping.treatment, mapping.outcome, mapping.revenue, "customer_id", "id"}
            ]
        mapping.confounders = confounders

        preview_records = df.head(50).fillna("").to_dict(orient="records")
        return DataSummaryResponse(
            filename=STATE["filename"],
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            preview=preview_records,
            column_mapping=STATE["column_mapping"],
            budget_config=STATE["budget_config"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")


@app.post("/config")
def update_configuration(
    column_mapping: Optional[ColumnMapping] = None,
    budget_config: Optional[BudgetConfig] = None,
):
    """Update column mapping and budget settings."""
    if column_mapping is not None:
        STATE["column_mapping"] = column_mapping
        STATE["audit_cache"] = None  # invalidate cache
    if budget_config is not None:
        STATE["budget_config"] = budget_config

    return {
        "status": "success",
        "column_mapping": STATE["column_mapping"],
        "budget_config": STATE["budget_config"],
    }


@app.get("/audit", response_model=AuditResponse)
def get_causal_audit():
    """Compute and return Causal DAG, identified backdoor estimand, propensity overlap, and balance table."""
    if STATE["audit_cache"] is not None:
        return STATE["audit_cache"]

    if STATE["df"] is None:
        STATE["df"] = load_default_dataset()

    df: pd.DataFrame = STATE["df"]
    mapping: ColumnMapping = STATE["column_mapping"]

    # 1. Identify causal effect & DAG
    dag_result = identify_causal_effect(
        df=df,
        treatment=mapping.treatment,
        outcome=mapping.outcome,
        confounders=mapping.confounders,
    )

    # 2. Fit propensity auditor
    num_cols = [c for c in mapping.confounders if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in mapping.confounders if c not in num_cols]

    auditor = PropensityAuditor(
        treatment_col=mapping.treatment,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
    ).fit(df)

    # 3. Covariate balance table
    balance_df = auditor.calculate_balance_table(df)
    balance_records = balance_df.to_dict(orient="records")

    # 4. Overlap diagnostics
    overlap_diag = auditor.get_overlap_diagnostics(df)

    # 5. Summary metrics
    pct_balanced = float((balance_df["smd_weighted_avg"] < 0.10).mean() * 100)
    max_unw_smd = float(balance_df["smd_unweighted_avg"].max())
    max_wt_smd = float(balance_df["smd_weighted_avg"].max())

    _, n_trimmed = trim_extreme_propensities(df, auditor)

    response_data = AuditResponse(
        dag=dag_result["dag_json"],
        balance_table=balance_records,
        overlap=overlap_diag,
        summary_metrics={
            "total_covariates": len(balance_records),
            "pct_balanced_smd_under_0_10": round(pct_balanced, 1),
            "max_unweighted_smd": round(max_unw_smd, 4),
            "max_weighted_smd": round(max_wt_smd, 4),
            "units_with_positivity_violations": n_trimmed,
            "estimand_summary": dag_result["dag_json"].get("estimand_summary", ""),
        },
    )

    STATE["audit_cache"] = response_data
    return response_data


@app.get("/uplift", response_model=UpliftSummaryResponse)
def get_uplift_summary():
    """Estimate heterogeneous treatment effects and summarize uplift by segment."""
    if STATE["df"] is None:
        STATE["df"] = load_default_dataset()

    df: pd.DataFrame = STATE["df"]
    mapping: ColumnMapping = STATE["column_mapping"]
    required_columns = [mapping.treatment, mapping.outcome, *mapping.confounders]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Uplift model columns are missing from the dataset: {', '.join(missing_columns)}",
        )

    if not mapping.confounders:
        raise HTTPException(status_code=400, detail="Select at least one confounder before fitting uplift models.")

    treatment = pd.to_numeric(df[mapping.treatment], errors="coerce")
    outcome = pd.to_numeric(df[mapping.outcome], errors="coerce")
    valid_mask = treatment.notna() & outcome.isin([0, 1])
    valid_indices = np.flatnonzero(valid_mask.to_numpy())
    if len(valid_indices) == 0:
        raise HTTPException(status_code=400, detail="No rows are valid for uplift estimation.")

    feature_frame = df.loc[valid_mask, mapping.confounders]
    response_probabilities = {}
    arms = sorted({0.0, *[float(value) for value in STATE["budget_config"].discount_options]})
    for arm in arms:
        if arm not in treatment[valid_mask].unique():
            continue
        arm_mask = valid_mask & (treatment == arm)
        arm_outcomes = outcome.loc[arm_mask].astype(int)
        if arm_outcomes.nunique() != 2:
            continue

        numeric_columns = [
            column for column in mapping.confounders if pd.api.types.is_numeric_dtype(feature_frame[column])
        ]
        categorical_columns = [column for column in mapping.confounders if column not in numeric_columns]
        transformers = []
        if numeric_columns:
            transformers.append((
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric_columns,
            ))
        if categorical_columns:
            transformers.append((
                "categorical",
                make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")),
                categorical_columns,
            ))
        preprocessor = ColumnTransformer(transformers)
        estimator = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
        estimator.fit(feature_frame.loc[arm_mask], arm_outcomes)
        response_probabilities[arm] = estimator.predict_proba(df.loc[:, mapping.confounders])[:, 1]

    if 0.0 not in response_probabilities:
        raise HTTPException(status_code=400, detail="The control arm is required to estimate uplift.")

    control_prob = response_probabilities[0.0]
    uplift_10 = response_probabilities.get(10.0, control_prob) - control_prob
    uplift_20 = response_probabilities.get(20.0, control_prob) - control_prob

    combined = df.copy()
    combined["estimated_uplift_10"] = uplift_10
    combined["estimated_uplift_20"] = uplift_20

    summary_metrics = {
        "estimated_avg_uplift_10": round(float(np.mean(uplift_10)), 4),
        "estimated_avg_uplift_20": round(float(np.mean(uplift_20)), 4),
        "max_estimated_uplift_10": round(float(np.max(uplift_10)), 4),
        "max_estimated_uplift_20": round(float(np.max(uplift_20)), 4),
    }

    ground_truth = load_ground_truth_dataset()
    available_ground_truth = ground_truth is not None and not ground_truth.empty
    by_segment = []
    qini_curve = []

    if available_ground_truth and {"customer_id", "segment", "tau_10_true", "tau_20_true"}.issubset(ground_truth.columns):
        combined = combined.merge(
            ground_truth[["customer_id", "segment", "tau_10_true", "tau_20_true"]],
            on="customer_id",
            how="left",
        )
        for arm, true_col in [(10, "tau_10_true"), (20, "tau_20_true")]:
            estimate_col = f"estimated_uplift_{arm}"
            valid_truth = combined[true_col].notna()
            if valid_truth.any():
                true_values = combined.loc[valid_truth, true_col].to_numpy(dtype=float)
                est_values = combined.loc[valid_truth, estimate_col].to_numpy(dtype=float)
                summary_metrics[f"rmse_{arm}"] = round(float(np.sqrt(np.mean((est_values - true_values) ** 2))), 4)
                summary_metrics[f"mae_{arm}"] = round(float(np.mean(np.abs(est_values - true_values))), 4)

        for segment_name, group in combined.groupby("segment", sort=True):
            if group.empty:
                continue
            by_segment.append({
                "segment": segment_name,
                "customers": int(len(group)),
                "avg_true_uplift_10": round(float(group["tau_10_true"].mean()), 4) if group["tau_10_true"].notna().any() else None,
                "avg_estimated_uplift_10": round(float(group["estimated_uplift_10"].mean()), 4),
                "avg_true_uplift_20": round(float(group["tau_20_true"].mean()), 4) if group["tau_20_true"].notna().any() else None,
                "avg_estimated_uplift_20": round(float(group["estimated_uplift_20"].mean()), 4),
            })

        for arm in (10, 20):
            true_col = f"tau_{arm}_true"
            estimate_col = f"estimated_uplift_{arm}"
            valid_truth = combined[true_col].notna()
            if not valid_truth.any():
                continue
            ranked = combined.loc[valid_truth, [estimate_col, true_col]].copy()
            ranked = ranked.sort_values(by=estimate_col, ascending=False)
            cumulative_true = ranked[true_col].cumsum()
            step_values = []
            for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                idx = max(1, int(np.ceil(len(ranked) * p)))
                step_values.append({
                    "fraction_targeted": round(float(p), 4),
                    "cumulative_true_uplift": round(float(cumulative_true.iloc[idx - 1]), 4),
                })
            qini_curve.append({"discount": arm, "curve": step_values})

    elif "segment" in df.columns:
        for segment_name, group in df.groupby("segment", sort=True):
            by_segment.append({
                "segment": segment_name,
                "customers": int(len(group)),
                "avg_true_uplift_10": None,
                "avg_estimated_uplift_10": round(float(group["estimated_uplift_10"].mean()), 4),
                "avg_true_uplift_20": None,
                "avg_estimated_uplift_20": round(float(group["estimated_uplift_20"].mean()), 4),
            })

    return {
        "status": "ready",
        "observations": int(len(df)),
        "available_ground_truth": available_ground_truth,
        "summary_metrics": summary_metrics,
        "by_segment": by_segment,
        "qini_curve": qini_curve,
        "estimation_note": (
            "Estimated uplift is derived from separate response models by treatment arm. "
            "When ground-truth uplift labels are available, the summary compares the model's estimated "
            "ITE to the synthetic truth, but it remains a benchmark rather than a validated causal estimate."
        ),
    }


@app.post("/prescription")
def create_prescription():
    """Estimate response by discount arm and optimize customer-level allocation."""
    if STATE["df"] is None:
        STATE["df"] = load_default_dataset()

    df: pd.DataFrame = STATE["df"]
    mapping: ColumnMapping = STATE["column_mapping"]
    budget: BudgetConfig = STATE["budget_config"]
    required_columns = [mapping.treatment, mapping.outcome, *mapping.confounders]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Prescription columns are missing from the dataset: {', '.join(missing_columns)}",
        )
    if not mapping.confounders:
        raise HTTPException(status_code=400, detail="Select at least one confounder before optimizing.")
    if mapping.treatment in mapping.confounders or mapping.outcome in mapping.confounders:
        raise HTTPException(
            status_code=400,
            detail="The treatment and outcome columns cannot also be used as confounders.",
        )

    discounts = [float(value) for value in budget.discount_options]
    if not discounts or any(not np.isfinite(value) or value <= 0 for value in discounts):
        raise HTTPException(status_code=400, detail="Configure at least one positive, finite discount option.")
    if len(set(discounts)) != len(discounts):
        raise HTTPException(status_code=400, detail="Discount options must not contain duplicates.")
    if not np.isfinite(budget.total_budget) or budget.total_budget <= 0:
        raise HTTPException(status_code=400, detail="The total budget must be positive and finite.")
    if not np.isfinite(budget.margin_per_purchase) or budget.margin_per_purchase <= 0:
        raise HTTPException(status_code=400, detail="The margin per purchase must be positive and finite.")

    treatment = pd.to_numeric(df[mapping.treatment], errors="coerce")
    outcome = pd.to_numeric(df[mapping.outcome], errors="coerce")
    complete_rows = treatment.notna() & outcome.isin([0, 1])
    if not complete_rows.any():
        raise HTTPException(status_code=400, detail="No rows have valid treatment and binary outcome values.")

    arms = [0.0, *discounts]
    missing_arms = [arm for arm in arms if not (treatment[complete_rows] == arm).any()]
    if missing_arms:
        arms_text = ", ".join(f"${arm:g}" for arm in missing_arms)
        raise HTTPException(
            status_code=400,
            detail=f"The dataset has no observations for these treatment arms: {arms_text}.",
        )

    features = df.loc[:, mapping.confounders]
    numeric_columns = [
        column for column in mapping.confounders
        if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_columns = [column for column in mapping.confounders if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical_columns,
            )
        )
    preprocessor = ColumnTransformer(transformers)

    response_probabilities = {}
    for arm in arms:
        arm_rows = complete_rows & (treatment == arm)
        arm_outcomes = outcome.loc[arm_rows].astype(int)
        if arm_outcomes.nunique() != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Both purchase outcomes (0 and 1) are required for the ${arm:g} treatment arm.",
            )
        estimator = make_pipeline(
            preprocessor,
            LogisticRegression(max_iter=500),
        )
        estimator.fit(features.loc[arm_rows], arm_outcomes)
        response_probabilities[arm] = estimator.predict_proba(features)[:, 1]

    customer_indices = []
    selected_discounts = []
    expected_spends = []
    incremental_profits = []
    for discount in discounts:
        purchase_probability = response_probabilities[discount]
        spend = purchase_probability * discount
        incremental_profit = (
            (purchase_probability - response_probabilities[0]) * budget.margin_per_purchase
            - spend
        )
        eligible_indices = np.flatnonzero((incremental_profit > 0) & (spend > 0))
        customer_indices.extend(eligible_indices.tolist())
        selected_discounts.extend([discount] * len(eligible_indices))
        expected_spends.extend(spend[eligible_indices].tolist())
        incremental_profits.extend(incremental_profit[eligible_indices].tolist())

    selected_rows = []
    if customer_indices:
        variable_count = len(customer_indices)
        variable_indices = np.arange(variable_count)
        constraints = coo_matrix(
            (
                np.concatenate((np.ones(variable_count), expected_spends)),
                (
                    np.concatenate((customer_indices, np.full(variable_count, len(df)))),
                    np.concatenate((variable_indices, variable_indices)),
                ),
            ),
            shape=(len(df) + 1, variable_count),
        ).tocsc()
        upper_bounds = np.full(len(df) + 1, np.inf)
        upper_bounds[:len(df)] = 1
        upper_bounds[-1] = budget.total_budget
        result = milp(
            c=-np.asarray(incremental_profits),
            integrality=np.ones(variable_count),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(constraints, 0, upper_bounds),
        )
        if not result.success or result.x is None:
            raise HTTPException(
                status_code=503,
                detail=f"Budget optimization did not complete: {result.message}",
            )
        selected_rows = np.flatnonzero(result.x > 0.5).tolist()

    recommendations = []
    for variable_index in selected_rows:
        row_index = customer_indices[variable_index]
        discount = selected_discounts[variable_index]
        recommendations.append(
            {
                "row_index": int(row_index),
                "discount": discount,
                "predicted_purchase_without_discount": round(
                    float(response_probabilities[0][row_index]), 4
                ),
                "predicted_purchase_with_discount": round(
                    float(response_probabilities[discount][row_index]), 4
                ),
                "expected_spend": round(expected_spends[variable_index], 2),
                "expected_incremental_profit": round(incremental_profits[variable_index], 2),
            }
        )

    total_spend = sum(expected_spends[index] for index in selected_rows)
    total_profit = sum(incremental_profits[index] for index in selected_rows)
    return {
        "observations": int(len(df)),
        "recommended_customers": len(recommendations),
        "budget": round(float(budget.total_budget), 2),
        "expected_spend": round(total_spend, 2),
        "expected_incremental_profit": round(total_profit, 2),
        "expected_incremental_purchases": round(
            sum(
                row["predicted_purchase_with_discount"]
                - row["predicted_purchase_without_discount"]
                for row in recommendations
            ),
            2,
        ),
        "estimation_note": (
            "Response probabilities are fitted separately within each observed discount arm "
            "using the configured confounders. This is an observational response-model "
            "benchmark, not a validated causal uplift estimate."
        ),
        "recommendations": recommendations,
    }
