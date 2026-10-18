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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.app.schemas import (
    AuditResponse,
    BudgetConfig,
    ColumnMapping,
    DataSummaryResponse,
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
