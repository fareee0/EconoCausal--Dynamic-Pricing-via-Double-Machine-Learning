"""
EconoCausal - Synthetic Retail Dataset Generator (Phase 0)
Generates ~20,000 customers from a known structural causal model with:
- Confounders: age, tenure_months, past_purchases_90d, avg_basket_value,
  days_since_last_purchase, loyalty_tier, email_engagement_score, region.
- Observational Treatment Assignment: discount_amount in {0, 10, 20}.
  Assignment depends on confounders (e.g. lapsing customers and high email engagement
  targeted historically), introducing observational selection bias.
- Heterogeneous Treatment Effects (ITE) across 4 segments:
  * Persuadables (large positive effect, saturating returns from 10 to 20)
  * Sure Things (high baseline, ~0 effect)
  * Lost Causes (low baseline, ~0 effect)
  * Sleeping Dogs (slightly negative effect)
- Unobserved variable (unobserved_price_sensitivity) affecting treatment and outcome weakly
  for refutation sensitivity testing.
- Outputs:
  * data/retail_campaign.csv (observed data for training)
  * data/ground_truth.csv (true ITE and latent segments for validation ONLY)
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


def generate_retail_dataset(
    n_samples: int = 20_000,
    random_seed: int = 42,
    output_dir: Path | str = "data",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic retail dataset and corresponding ground truth.

    Parameters
    ----------
    n_samples : int
        Number of customer records to generate (default: 20,000).
    random_seed : int
        RNG seed for strict reproducibility.
    output_dir : Path or str
        Directory where CSV files will be saved.

    Returns
    -------
    df_observed : pd.DataFrame
        Observed campaign dataset (treatment, outcome, observed confounders).
    df_ground_truth : pd.DataFrame
        Ground truth validation dataset (true ITE, true probabilities, segment).
    """
    rng = np.random.default_rng(random_seed)

    # 1. Customer IDs
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_samples + 1)]

    # 2. Confounders Generation
    # Continuous / Discrete demographics & history
    age = rng.integers(18, 75, size=n_samples)
    tenure_months = rng.integers(1, 61, size=n_samples)
    past_purchases_90d = rng.poisson(lam=2.8, size=n_samples)
    avg_basket_value = np.round(rng.lognormal(mean=4.2, sigma=0.45, size=n_samples), 2)
    avg_basket_value = np.clip(avg_basket_value, 20.0, 350.0)

    # Days since last purchase (recency): mixture of active and lapsing
    is_lapsing = rng.random(size=n_samples) < 0.35
    days_since_last_purchase = np.where(
        is_lapsing,
        rng.integers(60, 180, size=n_samples),
        rng.integers(1, 60, size=n_samples),
    )

    # Loyalty tier: Bronze (45%), Silver (30%), Gold (18%), Platinum (7%)
    loyalty_tiers = ["Bronze", "Silver", "Gold", "Platinum"]
    tier_probs = [0.45, 0.30, 0.18, 0.07]
    loyalty_tier = rng.choice(loyalty_tiers, p=tier_probs, size=n_samples)

    # Email engagement score: beta distribution [0, 1]
    email_engagement_score = np.round(rng.beta(a=2.0, b=4.0, size=n_samples), 4)

    # Region: North, South, East, West
    regions = ["North", "South", "East", "West"]
    region_probs = [0.28, 0.26, 0.24, 0.22]
    region = rng.choice(regions, p=region_probs, size=n_samples)

    # 3. Unobserved Confounder (weakly affects both treatment & outcome)
    # Documented for refutation testing (sensitivity to unobserved confounding)
    unobserved_price_sensitivity = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    # 4. Latent Segment Assignment
    # Determine natural customer behavior segments:
    # - Sure Things (~30%): high frequency, low recency, higher tier
    # - Lost Causes (~35%): dormant, low purchases, low engagement
    # - Sleeping Dogs (~10%): long tenure, sensitive/annoyed by promos
    # - Persuadables (~25%): fence-sitters with positive lift
    
    score_sure_thing = (
        0.35 * (past_purchases_90d >= 4)
        + 0.30 * (days_since_last_purchase <= 30)
        + 0.20 * np.isin(loyalty_tier, ["Gold", "Platinum"])
        + 0.15 * (email_engagement_score > 0.35)
    )
    score_lost_cause = (
        0.40 * (days_since_last_purchase > 65)
        + 0.35 * (past_purchases_90d <= 1)
        + 0.25 * (email_engagement_score < 0.25)
    )
    score_sleeping_dog = (
        0.45 * (tenure_months > 30)
        + 0.30 * (past_purchases_90d <= 3)
        + 0.25 * (email_engagement_score < 0.20)
    )

    segment = np.full(n_samples, "Persuadable", dtype=object)
    segment[score_sure_thing >= 0.50] = "Sure Thing"
    segment[(score_lost_cause >= 0.52) & (segment != "Sure Thing")] = "Lost Cause"
    segment[(score_sleeping_dog >= 0.65) & (segment == "Persuadable")] = "Sleeping Dog"

    # 5. Baseline Purchase Probability (d = 0) and True Treatment Effects (ITE)
    p_baseline = np.zeros(n_samples)
    tau_10_true = np.zeros(n_samples)
    tau_20_true = np.zeros(n_samples)

    for i in range(n_samples):
        seg = segment[i]
        eng = email_engagement_score[i]
        rec_factor = np.exp(-days_since_last_purchase[i] / 60.0)

        if seg == "Sure Thing":
            p_baseline[i] = 0.72 + 0.18 * rec_factor + 0.05 * (loyalty_tier[i] == "Platinum")
            tau_10_true[i] = 0.012 + 0.008 * eng
            tau_20_true[i] = 0.020 + 0.010 * eng
        elif seg == "Lost Cause":
            p_baseline[i] = 0.02 + 0.03 * rec_factor
            tau_10_true[i] = 0.010 + 0.015 * eng
            tau_20_true[i] = 0.018 + 0.020 * eng
        elif seg == "Sleeping Dog":
            p_baseline[i] = 0.28 + 0.10 * rec_factor
            tau_10_true[i] = -0.040 - 0.020 * (1.0 - eng)
            tau_20_true[i] = -0.065 - 0.030 * (1.0 - eng)
        else:  # Persuadable
            p_baseline[i] = 0.18 + 0.12 * rec_factor + 0.08 * eng
            # Saturating returns: discount 10 gives big lift, discount 20 gives modest extra lift
            lift_base = 0.22 + 0.10 * eng + 0.05 * (avg_basket_value[i] > 80.0)
            tau_10_true[i] = lift_base
            tau_20_true[i] = lift_base + 0.10  # diminishing returns from $10 to $20

    # Weak influence of unobserved confounder on outcome
    p_baseline = np.clip(p_baseline + 0.02 * unobserved_price_sensitivity, 0.01, 0.98)
    p_10_true = np.clip(p_baseline + tau_10_true, 0.005, 0.995)
    p_20_true = np.clip(p_baseline + tau_20_true, 0.005, 0.995)

    # 6. Observational Treatment Assignment
    # Treatment in {0, 10, 20} assigned based on historical marketing policy:
    # Target high-engagement users and lapsing users, plus loyalty tier and unobserved sensitivity
    tier_numeric = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
    tier_val = np.array([tier_numeric[t] for t in loyalty_tier])

    z_0 = 0.3 * np.ones(n_samples)
    z_10 = (
        -0.8
        + 1.2 * email_engagement_score
        + 0.9 * (days_since_last_purchase / 100.0)
        + 0.2 * tier_val
        + 0.15 * unobserved_price_sensitivity
    )
    z_20 = (
        -1.4
        + 1.5 * email_engagement_score
        + 1.3 * (days_since_last_purchase / 100.0)
        + 0.35 * tier_val
        + 0.25 * unobserved_price_sensitivity
    )

    # Multinomial logit probabilities (softmax)
    logits = np.column_stack([z_0, z_10, z_20])
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    propensities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Sample treatment according to observational propensities
    discount_amounts = np.array([0, 10, 20])
    chosen_arm_idx = np.array([
        rng.choice(3, p=propensities[i]) for i in range(n_samples)
    ])
    discount_amount = discount_amounts[chosen_arm_idx]

    # 7. Realized Outcome Generation
    realized_prob = np.where(
        discount_amount == 0,
        p_baseline,
        np.where(discount_amount == 10, p_10_true, p_20_true),
    )
    purchased = rng.binomial(n=1, p=realized_prob)

    # Realized revenue: only if purchased, basket value minus discount
    basket_noise = rng.normal(1.0, 0.08, size=n_samples)
    revenue = np.where(
        purchased == 1,
        np.maximum(0.0, np.round(avg_basket_value * basket_noise - discount_amount, 2)),
        0.0,
    )

    # 8. Construct DataFrames
    df_observed = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "tenure_months": tenure_months,
        "past_purchases_90d": past_purchases_90d,
        "avg_basket_value": avg_basket_value,
        "days_since_last_purchase": days_since_last_purchase,
        "loyalty_tier": loyalty_tier,
        "email_engagement_score": email_engagement_score,
        "region": region,
        "discount_amount": discount_amount,
        "purchased": purchased,
        "revenue": revenue,
    })

    df_ground_truth = pd.DataFrame({
        "customer_id": customer_ids,
        "segment": segment,
        "p_baseline": np.round(p_baseline, 5),
        "tau_10_true": np.round(tau_10_true, 5),
        "tau_20_true": np.round(tau_20_true, 5),
        "p_10_true": np.round(p_10_true, 5),
        "p_20_true": np.round(p_20_true, 5),
        "propensity_0": np.round(propensities[:, 0], 5),
        "propensity_10": np.round(propensities[:, 1], 5),
        "propensity_20": np.round(propensities[:, 2], 5),
        "unobserved_price_sensitivity": np.round(unobserved_price_sensitivity, 5),
    })

    # Save CSVs to output directory and backend/data
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df_observed.to_csv(out_path / "retail_campaign.csv", index=False)
    df_ground_truth.to_csv(out_path / "ground_truth.csv", index=False)

    # Ensure backend/data also has a copy if output_dir is data/
    backend_data_path = Path("backend/data")
    if backend_data_path.resolve() != out_path.resolve():
        backend_data_path.mkdir(parents=True, exist_ok=True)
        df_observed.to_csv(backend_data_path / "retail_campaign.csv", index=False)
        df_ground_truth.to_csv(backend_data_path / "ground_truth.csv", index=False)

    print(f"Generated {n_samples:,} records successfully.")
    print(f"Segments distribution:\n{pd.Series(segment).value_counts(normalize=True).round(3)}")
    print(f"Treatment distribution:\n{df_observed['discount_amount'].value_counts(normalize=True).round(3)}")
    print(f"Purchase rate by treatment:\n{df_observed.groupby('discount_amount')['purchased'].mean().round(3)}")

    return df_observed, df_ground_truth


if __name__ == "__main__":
    generate_retail_dataset()
