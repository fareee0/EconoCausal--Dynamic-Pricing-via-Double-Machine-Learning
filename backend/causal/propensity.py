"""
EconoCausal - Propensity Modeling & Balance Diagnostics (Phase 1)
Fits a multinomial propensity model for discount levels {0, 10, 20}.
Calculates:
- Propensity scores e_d(X) = P(T = d | X)
- Positivity / Overlap distributions per arm
- Inverse Propensity Weights (IPW with stabilization)
- Covariate balance table with Standardized Mean Differences (SMD) before & after weighting
- Extreme propensity trimming for positivity enforcement
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_CONFOUNDERS = [
    "age",
    "tenure_months",
    "past_purchases_90d",
    "avg_basket_value",
    "days_since_last_purchase",
    "email_engagement_score",
]

CATEGORICAL_CONFOUNDERS = [
    "loyalty_tier",
    "region",
]


class PropensityAuditor:
    """Estimates propensity scores and audits covariate balance and overlap."""

    def __init__(
        self,
        treatment_col: str = "discount_amount",
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.treatment_col = treatment_col
        self.numeric_cols = numeric_cols or NUMERIC_CONFOUNDERS
        self.categorical_cols = categorical_cols or CATEGORICAL_CONFOUNDERS
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
            random_state=self.random_state,
        )
        self.feature_names_: List[str] = []
        self.classes_: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Preprocess continuous and categorical confounders."""
        num_cols = [c for c in self.numeric_cols if c in df.columns]
        cat_cols = [c for c in self.categorical_cols if c in df.columns]

        if fit:
            X_num = self.scaler.fit_transform(df[num_cols].values)
            if cat_cols:
                X_cat = self.encoder.fit_transform(df[cat_cols].values)
                cat_feature_names = self.encoder.get_feature_names_out(cat_cols).tolist()
            else:
                X_cat = np.empty((len(df), 0))
                cat_feature_names = []
            self.feature_names_ = num_cols + cat_feature_names
        else:
            X_num = self.scaler.transform(df[num_cols].values)
            if cat_cols:
                X_cat = self.encoder.transform(df[cat_cols].values)
            else:
                X_cat = np.empty((len(df), 0))

        if X_cat.shape[1] > 0:
            return np.hstack([X_num, X_cat])
        return X_num

    def fit(self, df: pd.DataFrame) -> "PropensityAuditor":
        """Fit multinomial propensity model."""
        X = self._prepare_features(df, fit=True)
        y = df[self.treatment_col].values
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self

    def predict_propensities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return N x K DataFrame of estimated propensity scores."""
        if not self.is_fitted:
            raise RuntimeError("PropensityAuditor must be fitted before predict_propensities.")
        X = self._prepare_features(df, fit=False)
        probs = self.model.predict_proba(X)
        col_names = [f"propensity_{c}" for c in self.classes_]
        return pd.DataFrame(probs, columns=col_names, index=df.index)

    def compute_weights(self, df: pd.DataFrame, stabilized: bool = True) -> np.ndarray:
        """Compute stabilized inverse probability weights (IPW)."""
        prop_df = self.predict_propensities(df)
        t_vals = df[self.treatment_col].values
        weights = np.zeros(len(df))

        # Marginal treatment probabilities for stabilization
        marginal_probs = {c: np.mean(t_vals == c) for c in self.classes_}

        for idx, c in enumerate(self.classes_):
            mask = t_vals == c
            prop_c = np.clip(prop_df[f"propensity_{c}"].values[mask], 0.01, 0.99)
            if stabilized:
                weights[mask] = marginal_probs[c] / prop_c
            else:
                weights[mask] = 1.0 / prop_c

        # Clip extreme weights at 99th percentile to prevent high-variance outliers
        p99 = np.percentile(weights, 99)
        return np.clip(weights, 0.05, p99)

    @staticmethod
    def _compute_smd(
        x_treat: np.ndarray,
        x_ctrl: np.ndarray,
        w_treat: Optional[np.ndarray] = None,
        w_ctrl: Optional[np.ndarray] = None,
    ) -> float:
        """Compute Standardized Mean Difference (SMD) between two groups."""
        if w_treat is None or w_ctrl is None:
            mean_t = np.mean(x_treat)
            mean_c = np.mean(x_ctrl)
            var_t = np.var(x_treat, ddof=1)
            var_c = np.var(x_ctrl, ddof=1)
        else:
            sum_wt = np.sum(w_treat)
            sum_wc = np.sum(w_ctrl)
            if sum_wt == 0 or sum_wc == 0:
                return 0.0
            mean_t = np.sum(w_treat * x_treat) / sum_wt
            mean_c = np.sum(w_ctrl * x_ctrl) / sum_wc
            var_t = np.sum(w_treat * (x_treat - mean_t) ** 2) / sum_wt
            var_c = np.sum(w_ctrl * (x_ctrl - mean_c) ** 2) / sum_wc

        pooled_sd = np.sqrt(0.5 * (var_t + var_c))
        if pooled_sd <= 1e-9:
            return 0.0
        return float(abs(mean_t - mean_c) / pooled_sd)

    def calculate_balance_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate unweighted vs IPW-weighted SMD for each covariate and treatment pair."""
        weights = self.compute_weights(df, stabilized=True)
        t_vals = df[self.treatment_col].values

        # Build design matrix of confounders
        X = self._prepare_features(df, fit=False)
        feature_names = self.feature_names_

        mask_0 = t_vals == 0
        mask_10 = t_vals == 10
        mask_20 = t_vals == 20

        rows = []
        for j, feat in enumerate(feature_names):
            x_col = X[:, j]

            # 10 vs 0
            smd_unw_10 = self._compute_smd(x_col[mask_10], x_col[mask_0])
            smd_wt_10 = self._compute_smd(
                x_col[mask_10], x_col[mask_0], weights[mask_10], weights[mask_0]
            )

            # 20 vs 0
            smd_unw_20 = self._compute_smd(x_col[mask_20], x_col[mask_0])
            smd_wt_20 = self._compute_smd(
                x_col[mask_20], x_col[mask_0], weights[mask_20], weights[mask_0]
            )

            # Average SMD
            avg_unw = np.round((smd_unw_10 + smd_unw_20) / 2.0, 4)
            avg_wt = np.round((smd_wt_10 + smd_wt_20) / 2.0, 4)

            rows.append({
                "covariate": feat,
                "smd_unweighted_10": np.round(smd_unw_10, 4),
                "smd_weighted_10": np.round(smd_wt_10, 4),
                "smd_unweighted_20": np.round(smd_unw_20, 4),
                "smd_weighted_20": np.round(smd_wt_20, 4),
                "smd_unweighted_avg": avg_unw,
                "smd_weighted_avg": avg_wt,
                "balanced": bool(avg_wt < 0.10),
            })

        return pd.DataFrame(rows)

    def get_overlap_diagnostics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute overlap distribution data suitable for frontend Plotly histograms."""
        prop_df = self.predict_propensities(df)
        t_vals = df[self.treatment_col].values

        diagnostics: Dict[str, Any] = {
            "summary": {},
            "histograms": {},
            "weight_diagnostics": {},
            "assignment_support": {},
        }

        # Histogram bins [0, 1]
        bins = np.linspace(0, 1, 21).tolist()
        weights = self.compute_weights(df, stabilized=True)

        for c in self.classes_:
            col_name = f"propensity_{c}"
            props = prop_df[col_name].values
            arm_mask = t_vals == c
            arm_weights = weights[arm_mask]
            received_arm_props = props[arm_mask]
            weight_sum = float(np.sum(arm_weights))
            squared_weight_sum = float(np.sum(arm_weights ** 2))
            effective_sample_size = (
                weight_sum ** 2 / squared_weight_sum if squared_weight_sum > 0 else 0.0
            )
            low_propensity_count = int(np.sum(received_arm_props < 0.05))

            diagnostics["summary"][f"arm_{c}"] = {
                "min": float(np.min(props)),
                "p25": float(np.percentile(props, 25)),
                "median": float(np.median(props)),
                "p75": float(np.percentile(props, 75)),
                "max": float(np.max(props)),
            }
            diagnostics["weight_diagnostics"][f"arm_{c}"] = {
                "observations": int(np.sum(arm_mask)),
                "effective_sample_size": round(effective_sample_size, 2),
                "effective_sample_size_fraction": round(
                    effective_sample_size / max(int(np.sum(arm_mask)), 1), 4
                ),
                "max_weight": round(float(np.max(arm_weights)), 4) if len(arm_weights) else 0.0,
            }
            diagnostics["assignment_support"][f"arm_{c}"] = {
                "observations": int(np.sum(arm_mask)),
                "median_received_propensity": (
                    round(float(np.median(received_arm_props)), 4)
                    if len(received_arm_props) else 0.0
                ),
                "low_propensity_threshold": 0.05,
                "low_propensity_count": low_propensity_count,
                "low_propensity_fraction": round(
                    low_propensity_count / max(int(np.sum(arm_mask)), 1), 4
                ),
            }

            # Histograms conditioned on actual received treatment
            arm_hist: Dict[str, List[int]] = {}
            for actual_t in self.classes_:
                counts, _ = np.histogram(props[t_vals == actual_t], bins=bins)
                arm_hist[f"received_{actual_t}"] = counts.tolist()

            diagnostics["histograms"][f"propensity_for_{c}"] = {
                "bins": bins,
                "data_by_received_arm": arm_hist,
            }

        return diagnostics


def trim_extreme_propensities(
    df: pd.DataFrame,
    auditor: PropensityAuditor,
    min_propensity: float = 0.05,
    max_propensity: float = 0.95,
) -> Tuple[pd.DataFrame, int]:
    """Trim units violating common support/positivity."""
    prop_df = auditor.predict_propensities(df)
    valid_mask = np.ones(len(df), dtype=bool)

    for c in auditor.classes_:
        col = f"propensity_{c}"
        valid_mask &= (prop_df[col] >= min_propensity) & (prop_df[col] <= max_propensity)

    trimmed_df = df[valid_mask].copy()
    num_trimmed = int(len(df) - len(trimmed_df))
    return trimmed_df, num_trimmed


if __name__ == "__main__":
    df = pd.read_csv("backend/data/retail_campaign.csv")
    auditor = PropensityAuditor().fit(df)
    balance_df = auditor.calculate_balance_table(df)
    print("=== Covariate Balance Table (SMD Before & After IPW) ===")
    print(balance_df[["covariate", "smd_unweighted_avg", "smd_weighted_avg", "balanced"]].to_string())
    
    pct_balanced = (balance_df["smd_weighted_avg"] < 0.10).mean() * 100
    print(f"\nPercentage of covariates balanced (SMD < 0.10): {pct_balanced:.1f}%")

    overlap = auditor.get_overlap_diagnostics(df)
    print("\nOverlap Summary:")
    for arm, stats in overlap["summary"].items():
        print(f"  {arm}: min={stats['min']:.3f}, median={stats['median']:.3f}, max={stats['max']:.3f}")

    trimmed_df, n_trimmed = trim_extreme_propensities(df, auditor)
    print(f"\nTrimmed {n_trimmed:,} records with extreme propensities out of {len(df):,}.")
