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
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
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


def _normalized_column_name(column: str) -> str:
    return "".join(character for character in str(column).lower() if character.isalnum())


def _find_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    normalized = {_normalized_column_name(column): column for column in columns}
    return next((normalized[alias] for alias in aliases if alias in normalized), None)


def _churn_target_candidates(df: pd.DataFrame, excluded: set[str]) -> List[str]:
    candidates = []
    for column in df.columns:
        normalized = _normalized_column_name(column)
        if normalized in excluded or normalized in {
            "customerid", "id", "rowid", "revenue", "sales", "balance", "amount",
            "discount", "discountamount", "treatment", "treatmentarm", "purchase",
            "purchased", "conversion", "converted",
        }:
            continue
        values = df[column].dropna().unique()
        if len(values) == 2:
            candidates.append(column)
    return candidates


def _best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Choose a classification threshold using validation data only."""
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    f1_scores = [
        f1_score(labels, scores >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    best_f1 = max(f1_scores)
    return float(min(
        (threshold for threshold, score in zip(thresholds, f1_scores) if score == best_f1),
        key=lambda threshold: abs(threshold - 0.5),
    ))


def _validate_campaign_mapping(df: pd.DataFrame, mapping: ColumnMapping) -> None:
    """Fail clearly when the active file cannot support campaign causal analysis."""
    missing_columns = [
        column
        for column in (mapping.treatment, mapping.outcome)
        if not column or column not in df.columns
    ]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "This CSV does not have mapped treatment and purchase-outcome columns. "
                "Choose a campaign treatment column and binary purchase/conversion outcome "
                "in Data & Budget to use causal audit, uplift, and prescription."
            ),
        )


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

        # Re-map common column names when the uploaded dataset differs from the active one.
        mapping = STATE["column_mapping"]
        columns = list(df.columns)
        treatment = _find_column(
            columns,
            ["discountamount", "discount", "treatment", "treatmentarm", "treatmentgroup", "offeramount", "promotionamount"],
        )
        outcome = _find_column(
            columns,
            ["purchased", "purchase", "purchaseflag", "converted", "conversion", "conversionflag", "response", "outcome"],
        )
        revenue = _find_column(
            columns,
            ["revenue", "sales", "salesamount", "ordervalue", "transactionvalue"],
        )
        mapping.treatment = treatment or (mapping.treatment if mapping.treatment in df.columns else "")
        mapping.outcome = outcome or (mapping.outcome if mapping.outcome in df.columns else "")
        mapping.revenue = revenue or (mapping.revenue if mapping.revenue in df.columns else None)

        excluded_names = {
            "customerid", "id", "rowid", "churn", "churned", "ischurn", "ischurned",
            "churnstatus", "churnlabel", "attrition", "attritionflag", "attrited", "isattrited",
            "exited", "exit", "defaulted", "default", "target", "label", "accountstatus",
            _normalized_column_name(mapping.treatment),
            _normalized_column_name(mapping.outcome),
            _normalized_column_name(mapping.revenue) if mapping.revenue else "",
        }
        mapping.confounders = [
            column
            for column in columns
            if _normalized_column_name(column) not in excluded_names
            and (
                pd.api.types.is_numeric_dtype(df[column])
                or df[column].nunique(dropna=True) <= 50
            )
        ]

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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")


@app.post("/config")
def update_configuration(
    column_mapping: Optional[ColumnMapping] = None,
    budget_config: Optional[BudgetConfig] = None,
):
    """Update column mapping and budget settings."""
    if column_mapping is not None:
        if STATE["df"] is not None:
            missing_columns = sorted(
                column
                for column in [
                    column_mapping.treatment,
                    column_mapping.outcome,
                    *column_mapping.confounders,
                ]
                if column and column not in STATE["df"].columns
            )
            if missing_columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Mapped columns are missing from the active dataset: {', '.join(missing_columns)}",
                )
            if (
                column_mapping.treatment
                and column_mapping.outcome
                and column_mapping.treatment == column_mapping.outcome
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Treatment and outcome must be different columns.",
                )
            if any(
                column and column in column_mapping.confounders
                for column in (column_mapping.treatment, column_mapping.outcome)
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Treatment and outcome columns cannot also be selected as confounders.",
                )
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
    _validate_campaign_mapping(df, mapping)

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
    _validate_campaign_mapping(df, mapping)
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
    available_ground_truth = False
    by_segment = []
    qini_curve = []

    ground_truth_columns = {"customer_id", "segment", "tau_10_true", "tau_20_true"}
    if (
        ground_truth is not None
        and not ground_truth.empty
        and ground_truth_columns.issubset(ground_truth.columns)
        and "customer_id" in combined.columns
    ):
        known_customer_ids = set(ground_truth["customer_id"].dropna())
        matched_ids = combined["customer_id"].notna() & combined["customer_id"].isin(known_customer_ids)
        if matched_ids.any() and matched_ids.all():
            validation_truth = ground_truth[
                ["customer_id", "segment", "tau_10_true", "tau_20_true"]
            ].rename(columns={
                "segment": "ground_truth_segment",
                "tau_10_true": "ground_truth_tau_10_true",
                "tau_20_true": "ground_truth_tau_20_true",
            })
            combined = combined.merge(validation_truth, on="customer_id", how="left")
            available_ground_truth = True

    if available_ground_truth:
        for arm, true_col in [(10, "tau_10_true"), (20, "tau_20_true")]:
            estimate_col = f"estimated_uplift_{arm}"
            validation_column = f"ground_truth_{true_col}"
            valid_truth = combined[validation_column].notna()
            if valid_truth.any():
                true_values = combined.loc[valid_truth, validation_column].to_numpy(dtype=float)
                est_values = combined.loc[valid_truth, estimate_col].to_numpy(dtype=float)
                summary_metrics[f"rmse_{arm}"] = round(float(np.sqrt(np.mean((est_values - true_values) ** 2))), 4)
                summary_metrics[f"mae_{arm}"] = round(float(np.mean(np.abs(est_values - true_values))), 4)

        for segment_name, group in combined.groupby("ground_truth_segment", sort=True):
            if group.empty:
                continue
            by_segment.append({
                "segment": segment_name,
                "customers": int(len(group)),
                "avg_true_uplift_10": round(float(group["ground_truth_tau_10_true"].mean()), 4) if group["ground_truth_tau_10_true"].notna().any() else None,
                "avg_estimated_uplift_10": round(float(group["estimated_uplift_10"].mean()), 4),
                "avg_true_uplift_20": round(float(group["ground_truth_tau_20_true"].mean()), 4) if group["ground_truth_tau_20_true"].notna().any() else None,
                "avg_estimated_uplift_20": round(float(group["estimated_uplift_20"].mean()), 4),
            })

        for arm in (10, 20):
            true_col = f"ground_truth_tau_{arm}_true"
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
        for segment_name, group in combined.groupby("segment", sort=True):
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


@app.get("/churn")
def get_churn_predictions(
    target: Optional[str] = None,
    positive_value: Optional[str] = None,
):
    """Train a churn classifier, report holdout metrics, and score active customers."""
    if STATE["df"] is None:
        STATE["df"] = load_default_dataset()

    df: pd.DataFrame = STATE["df"]
    mapping: ColumnMapping = STATE["column_mapping"]
    normalized_columns = {
        _normalized_column_name(column): column
        for column in df.columns
    }
    churn_names = {
        "churn",
        "churned",
        "ischurned",
        "ischurn",
        "churnstatus",
        "churnlabel",
        "attrition",
        "attritionflag",
        "attrited",
        "isattrited",
        "exited",
    }
    if target is not None and target not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Selected churn target '{target}' is not a column in the active dataset.",
        )
    churn_column = target or next(
        (column for name, column in normalized_columns.items() if name in churn_names),
        None,
    )
    recency_names = {
        "dayssincelastpurchase",
        "dayssincelastorder",
        "dayssincelastvisit",
        "daysinactive",
        "recencydays",
        "lastpurchasedaysago",
    }

    explicit_target_selection = target is not None
    if churn_column is None:
        excluded_target_names = {
            _normalized_column_name(column)
            for column in (mapping.treatment, mapping.outcome, mapping.revenue)
            if column
        }
        candidates = _churn_target_candidates(df, excluded_target_names)
        named_candidates = [
            column for column in candidates
            if any(token in _normalized_column_name(column) for token in (
                "churn", "attrition", "exit", "cancel", "target", "label", "default", "status",
            ))
        ]
        if len(named_candidates) == 1:
            churn_column = named_candidates[0]
        elif not named_candidates and len(candidates) == 1:
            churn_column = candidates[0]
        elif len(named_candidates) > 1 or len(candidates) > 1:
            ambiguous = named_candidates or candidates
            raise HTTPException(
                status_code=400,
                detail=(
                    "Multiple binary columns could be the churn label. Select the correct target "
                    f"in Churn Prediction: {', '.join(ambiguous[:12])}."
                ),
            )

    if churn_column is not None:
        raw_target = df[churn_column]
        numeric_target = pd.to_numeric(raw_target, errors="coerce")
        numeric_classes = set(numeric_target.dropna().unique())
        if positive_value is not None:
            normalized_target = raw_target.astype("string").str.strip().str.lower()
            observed_values = normalized_target.dropna().unique()
            if len(observed_values) != 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Selected churn target '{churn_column}' must contain exactly two non-empty classes.",
                )
            labels_by_row = (normalized_target == positive_value.strip().lower()).astype("Int64")
            valid_mask = raw_target.notna()
            if not labels_by_row.loc[valid_mask].any():
                raise HTTPException(
                    status_code=400,
                    detail=f"Positive churn value '{positive_value}' is not present in '{churn_column}'.",
                )
            labels = labels_by_row.loc[valid_mask].astype(int).to_numpy()
        elif numeric_classes and numeric_classes.issubset({0, 1}):
            valid_mask = numeric_target.notna()
            labels = numeric_target.loc[valid_mask].astype(int).to_numpy()
        else:
            normalized_target = raw_target.astype("string").str.strip().str.lower()
            positive_labels = {
                "1", "true", "yes", "y", "churn", "churned", "left", "inactive", "attrited", "exited",
            }
            negative_labels = {
                "0", "false", "no", "n", "not churned", "no churn", "retained",
                "active", "stay", "stayed", "current", "existing", "not exited",
            }
            label_map = {
                **{value: 1 for value in positive_labels},
                **{value: 0 for value in negative_labels},
            }
            labels_by_row = normalized_target.map(label_map)
            valid_mask = labels_by_row.notna()
            if not valid_mask.any():
                observed_values = sorted(normalized_target.dropna().unique().tolist())
                if explicit_target_selection and len(observed_values) == 2:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Target '{churn_column}' has values {', '.join(observed_values)}. "
                            "Choose which value means churn in the Positive churn value field."
                        ),
                    )
                raise HTTPException(
                    status_code=400,
                    detail=f"Target '{churn_column}' must have binary 0/1, Yes/No, churn/retained values.",
                )
            labels = labels_by_row.loc[valid_mask].astype(int).to_numpy()
        target_definition = f"Observed churn label from '{churn_column}'."
        label_source = "explicit"
    else:
        recency_column = next(
            (column for name, column in normalized_columns.items() if name in recency_names),
            None,
        )
        if recency_column is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Churn prediction needs a binary churn label (for example, churn or attrition) "
                    "or a supported recency column to create a disclosed proxy label."
                ),
            )
        recency = pd.to_numeric(df[recency_column], errors="coerce")
        valid_mask = recency.notna()
        labels = (recency.loc[valid_mask] >= 90).astype(int).to_numpy()
        churn_column = None
        target_definition = (
            f"Proxy label: no purchase for at least 90 days, derived from '{recency_column}'. "
            "This is not a confirmed cancellation or account-closure label."
        )
        label_source = "recency_proxy"

    excluded_columns = {
        column
        for column in df.columns
        if _normalized_column_name(column) in churn_names
    }
    excluded_columns.update(
        column
        for column in df.columns
        if str(column).lower() in {"customer_id", "id", "row_id"}
    )
    excluded_columns.update(
        column
        for column in (mapping.treatment, mapping.outcome, mapping.revenue)
        if column in df.columns
    )
    if churn_column is not None:
        excluded_columns.add(churn_column)
    if label_source == "recency_proxy":
        excluded_columns.update(
            column
            for column in df.columns
            if _normalized_column_name(column) in recency_names
        )

    feature_columns = [
        column
        for column in mapping.confounders
        if column in df.columns and column not in excluded_columns
    ]
    if not feature_columns:
        feature_columns = [
            column for column in df.columns if column not in excluded_columns
        ]
    if not feature_columns:
        raise HTTPException(status_code=400, detail="No eligible customer features are available for churn prediction.")

    features = df.loc[valid_mask, feature_columns].replace([np.inf, -np.inf], np.nan)
    features = features.loc[:, features.notna().any(axis=0)]
    feature_columns = list(features.columns)
    if not feature_columns:
        raise HTTPException(status_code=400, detail="All available churn prediction features are empty.")

    class_counts = np.bincount(labels, minlength=2)
    if len(labels) < 20 or class_counts.min() < 4:
        raise HTTPException(
            status_code=400,
            detail="Churn prediction requires at least four examples in each class and 20 valid rows.",
        )

    numeric_columns = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    train_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    fit_indices, validation_indices = train_test_split(
        train_indices,
        test_size=0.2,
        random_state=43,
        stratify=labels[train_indices],
    )

    def build_model(classifier: Any):
        transformers = []
        if numeric_columns:
            transformers.append(
                (
                    "numeric",
                    make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                    numeric_columns,
                )
            )
        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")),
                    categorical_columns,
                )
            )
        return make_pipeline(ColumnTransformer(transformers), classifier)

    candidate_classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }
    candidate_models = {}
    candidate_scores = {}
    selection_auc = {}
    for model_name, classifier in candidate_classifiers.items():
        candidate = build_model(classifier)
        candidate.fit(features.iloc[fit_indices], labels[fit_indices])
        validation_scores = candidate.predict_proba(features.iloc[validation_indices])[:, 1]
        candidate_models[model_name] = candidate
        candidate_scores[model_name] = validation_scores
        selection_auc[model_name] = float(
            roc_auc_score(labels[validation_indices], validation_scores)
        )

    selected_model_name = max(selection_auc, key=selection_auc.get)
    selected_model = candidate_models[selected_model_name]
    decision_threshold = _best_f1_threshold(
        labels[validation_indices],
        candidate_scores[selected_model_name],
    )
    driver_importances = permutation_importance(
        selected_model,
        features.iloc[validation_indices],
        labels[validation_indices],
        scoring="roc_auc",
        n_repeats=3,
        random_state=42,
    )
    top_churn_drivers = [
        {
            "feature": feature_columns[index],
            "importance": round(float(importance), 4),
        }
        for index, importance in sorted(
            enumerate(driver_importances.importances_mean),
            key=lambda item: item[1],
            reverse=True,
        )
        if importance > 0
    ][:5]

    model = build_model(candidate_classifiers[selected_model_name])
    model.fit(features.iloc[train_indices], labels[train_indices])
    test_scores = model.predict_proba(features.iloc[test_indices])[:, 1]
    test_predictions = (test_scores >= decision_threshold).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(labels[test_indices], test_predictions)), 4),
        "precision": round(float(precision_score(labels[test_indices], test_predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels[test_indices], test_predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(labels[test_indices], test_predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(labels[test_indices], test_scores)), 4),
    }

    model.fit(features, labels)
    scored_features = df.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan)
    churn_scores = model.predict_proba(scored_features)[:, 1]
    risk_levels = np.where(churn_scores >= 0.7, "High", np.where(churn_scores >= 0.4, "Medium", "Low"))
    customer_id_column = next(
        (column for column in df.columns if _normalized_column_name(column) == "customerid"),
        None,
    )
    sorted_indices = np.argsort(-churn_scores)[:25]
    high_risk_customers = []
    for row_index in sorted_indices:
        score = float(churn_scores[row_index])
        if score >= 0.7:
            action = "Prioritize a personal retention conversation and investigate service friction."
        elif score >= 0.4:
            action = "Use a relevant re-engagement message and monitor response before offering incentives."
        else:
            action = "Continue useful loyalty engagement; avoid unnecessary discounting."
        high_risk_customers.append(
            {
                "row_index": int(row_index),
                "customer_id": str(df.iloc[row_index][customer_id_column]) if customer_id_column else None,
                "churn_probability": round(score, 4),
                "risk_level": str(risk_levels[row_index]),
                "suggested_action": action,
            }
        )

    return {
        "status": "ready",
        "observations": int(len(labels)),
        "label_source": label_source,
        "target_column": churn_column,
        "positive_value": positive_value or "Inferred from binary label encoding",
        "target_definition": target_definition,
        "model": selected_model_name,
        "validation": {
            "method": "Stratified 75/25 holdout; model selection and threshold tuning use training data only",
            "metrics": metrics,
            "decision_threshold": round(decision_threshold, 4),
            "candidate_roc_auc": {
                name: round(auc, 4) for name, auc in selection_auc.items()
            },
        },
        "top_churn_drivers": top_churn_drivers,
        "risk_distribution": {
            "high": int(np.sum(risk_levels == "High")),
            "medium": int(np.sum(risk_levels == "Medium")),
            "low": int(np.sum(risk_levels == "Low")),
        },
        "feature_columns": feature_columns,
        "high_risk_customers": high_risk_customers,
        "scoring_note": (
            "Scores rank modeled risk; they are not causal estimates. Validate retention actions "
            "with controlled experiments and use uplift estimates to target incentives."
        ),
    }


@app.post("/prescription")
def create_prescription():
    """Estimate response by discount arm and optimize customer-level allocation."""
    if STATE["df"] is None:
        STATE["df"] = load_default_dataset()

    df: pd.DataFrame = STATE["df"]
    mapping: ColumnMapping = STATE["column_mapping"]
    _validate_campaign_mapping(df, mapping)
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
