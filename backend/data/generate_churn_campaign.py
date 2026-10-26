"""Generate a large synthetic customer campaign dataset for EconoCausal."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_churn_campaign_dataset(
    n_samples: int = 25_000,
    random_seed: int = 2026,
) -> pd.DataFrame:
    """Return customer, campaign, purchase, and churn fields for the dashboard."""
    if n_samples < 300:
        raise ValueError("n_samples must be at least 300 for reliable model validation.")

    rng = np.random.default_rng(random_seed)
    age = rng.integers(18, 76, size=n_samples)
    tenure_months = rng.integers(1, 121, size=n_samples)
    past_purchases_90d = np.clip(rng.poisson(3.2, size=n_samples), 0, 20)
    avg_basket_value = np.clip(
        rng.lognormal(mean=4.25, sigma=0.48, size=n_samples),
        15,
        500,
    ).round(2)
    loyalty_tier = rng.choice(
        ["Bronze", "Silver", "Gold", "Platinum"],
        size=n_samples,
        p=[0.44, 0.31, 0.18, 0.07],
    )
    email_engagement_score = rng.beta(2.2, 3.8, size=n_samples).round(4)
    region = rng.choice(
        ["North", "South", "East", "West"],
        size=n_samples,
        p=[0.27, 0.25, 0.24, 0.24],
    )
    support_tickets_90d = np.clip(rng.poisson(0.45, size=n_samples), 0, 8)
    app_sessions_30d = np.clip(
        rng.poisson(5.0 + 7.0 * email_engagement_score, size=n_samples),
        0,
        40,
    )
    days_since_last_purchase = np.clip(
        rng.gamma(shape=2.1, scale=18, size=n_samples)
        + 8 * (past_purchases_90d == 0)
        + 9 * (email_engagement_score < 0.2),
        1,
        180,
    ).round().astype(int)

    churn_logit = (
        -2.1
        + 0.018 * (days_since_last_purchase - 30)
        - 0.11 * past_purchases_90d
        - 0.018 * (tenure_months - 24)
        + 0.38 * support_tickets_90d
        - 1.15 * email_engagement_score
        - 0.42 * (loyalty_tier == "Platinum")
        - 0.20 * (loyalty_tier == "Gold")
    )
    churn_probability = 1 / (1 + np.exp(-churn_logit))
    churn = rng.binomial(1, churn_probability)

    treatment_probabilities = np.full((n_samples, 3), 1 / 3)
    treatment_probabilities[:, 1] += 0.08 * (days_since_last_purchase > 45)
    treatment_probabilities[:, 2] += 0.12 * (days_since_last_purchase > 75)
    treatment_probabilities[:, 0] -= (
        treatment_probabilities[:, 1] - 1 / 3
    ) + (treatment_probabilities[:, 2] - 1 / 3)
    treatment_probabilities = np.clip(treatment_probabilities, 0.05, None)
    treatment_probabilities /= treatment_probabilities.sum(axis=1, keepdims=True)
    treatment_arm = np.array([
        rng.choice([0, 10, 20], p=probabilities)
        for probabilities in treatment_probabilities
    ])

    baseline_purchase_logit = (
        -0.55
        + 0.13 * past_purchases_90d
        - 0.014 * days_since_last_purchase
        + 0.72 * email_engagement_score
        + 0.10 * (loyalty_tier == "Gold")
        + 0.16 * (loyalty_tier == "Platinum")
        + 0.035 * app_sessions_30d
        - 0.20 * support_tickets_90d
    )
    discount_lift = np.select(
        [treatment_arm == 10, treatment_arm == 20],
        [0.38, 0.62],
        default=0,
    )
    purchase_probability = 1 / (1 + np.exp(-(baseline_purchase_logit + discount_lift)))
    purchased = rng.binomial(1, purchase_probability)
    revenue = np.where(
        purchased == 1,
        np.maximum(avg_basket_value - treatment_arm, 0),
        0,
    ).round(2)

    return pd.DataFrame({
        "customer_id": [f"DEMO_{index:07d}" for index in range(1, n_samples + 1)],
        "age": age,
        "tenure_months": tenure_months,
        "past_purchases_90d": past_purchases_90d,
        "avg_basket_value": avg_basket_value,
        "days_since_last_purchase": days_since_last_purchase,
        "loyalty_tier": loyalty_tier,
        "email_engagement_score": email_engagement_score,
        "region": region,
        "support_tickets_90d": support_tickets_90d,
        "app_sessions_30d": app_sessions_30d,
        "discount_amount": treatment_arm,
        "purchased": purchased,
        "revenue": revenue,
        "churn": churn,
    })


def main() -> None:
    output_path = Path(__file__).with_name("retail_churn_campaign.csv")
    dataset = generate_churn_campaign_dataset()
    dataset.to_csv(output_path, index=False)
    print(f"Generated {len(dataset):,} customers at {output_path}")
    print(f"Churn rate: {dataset['churn'].mean():.1%}")
    print(f"Purchase rate by discount:\n{dataset.groupby('discount_amount')['purchased'].mean().round(3)}")


if __name__ == "__main__":
    main()
