"""
EconoCausal - FastAPI Pydantic Schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ColumnMapping(BaseModel):
    treatment: str = Field(default="discount_amount", description="Treatment column name")
    outcome: str = Field(default="purchased", description="Binary purchase outcome column name")
    revenue: Optional[str] = Field(default="revenue", description="Revenue column name")
    confounders: List[str] = Field(
        default=[
            "age",
            "tenure_months",
            "past_purchases_90d",
            "avg_basket_value",
            "days_since_last_purchase",
            "loyalty_tier",
            "email_engagement_score",
            "region",
        ],
        description="Observed confounders for adjustment",
    )


class BudgetConfig(BaseModel):
    total_budget: float = Field(default=5000.0, ge=100.0, description="Total budget in USD")
    margin_per_purchase: float = Field(
        default=35.0, ge=1.0, description="Gross margin earned per purchase before discount in USD"
    )
    discount_options: List[float] = Field(
        default=[10.0, 20.0], description="Available discount amounts (excluding $0 baseline)"
    )


class AuditResponse(BaseModel):
    dag: Dict[str, Any]
    balance_table: List[Dict[str, Any]]
    overlap: Dict[str, Any]
    summary_metrics: Dict[str, Any]


class DataSummaryResponse(BaseModel):
    filename: str
    row_count: int
    column_count: int
    columns: List[str]
    preview: List[Dict[str, Any]]
    column_mapping: ColumnMapping
    budget_config: BudgetConfig


class UpliftSummaryResponse(BaseModel):
    status: str
    observations: int
    available_ground_truth: bool
    summary_metrics: Dict[str, float]
    by_segment: List[Dict[str, Any]]
    qini_curve: List[Dict[str, Any]]
    estimation_note: str
