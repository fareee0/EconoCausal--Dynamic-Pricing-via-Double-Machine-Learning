"""
EconoCausal - Causal Graph & DAG Definition (Phase 1)
Defines the causal Directed Acyclic Graph (DAG) using DoWhy and DOT representation.
Identifies the backdoor causal estimand and exports the graph schema for UI rendering.
"""

from typing import Any, Dict, List, Optional
import json
import pandas as pd
import networkx as nx

# Compatibility bridge for NetworkX >= 3.2
try:
    import networkx.algorithms.d_separation as ds
    if not hasattr(nx.algorithms, "d_separated"):
        nx.algorithms.d_separated = ds.is_d_separator
    if not hasattr(nx, "d_separated"):
        nx.d_separated = ds.is_d_separator
except Exception:
    pass

from dowhy import CausalModel


# Standard confounders observed in the campaign
DEFAULT_CONFOUNDERS = [
    "age",
    "tenure_months",
    "past_purchases_90d",
    "avg_basket_value",
    "days_since_last_purchase",
    "loyalty_tier",
    "email_engagement_score",
    "region",
]

UNOBSERVED_CONFOUNDER = "unobserved_price_sensitivity"


def build_dot_graph(
    confounders: List[str],
    treatment: str = "discount_amount",
    outcome: str = "purchased",
    include_unobserved: bool = True,
) -> str:
    """Construct DOT representation of the causal DAG."""
    lines = ["digraph {"]

    # Confounder -> Treatment and Confounder -> Outcome
    for c in confounders:
        lines.append(f'  "{c}" -> "{treatment}";')
        lines.append(f'  "{c}" -> "{outcome}";')

    # Unobserved confounder if requested
    if include_unobserved:
        lines.append(f'  "{UNOBSERVED_CONFOUNDER}" -> "{treatment}";')
        lines.append(f'  "{UNOBSERVED_CONFOUNDER}" -> "{outcome}";')

    # Direct treatment effect: Treatment -> Outcome
    lines.append(f'  "{treatment}" -> "{outcome}";')
    lines.append("}")
    return "\n".join(lines)


def get_dag_json(
    confounders: Optional[List[str]] = None,
    treatment: str = "discount_amount",
    outcome: str = "purchased",
    include_unobserved: bool = True,
) -> Dict[str, Any]:
    """Export graph structure as JSON for frontend visualization (Plotly / React Flow / SVG)."""
    if confounders is None:
        confounders = DEFAULT_CONFOUNDERS

    nodes = [
        {
            "id": treatment,
            "label": f"Treatment: {treatment}",
            "type": "treatment",
            "description": "Discrete promotional discount ($0, $10, $20)",
        },
        {
            "id": outcome,
            "label": f"Outcome: {outcome}",
            "type": "outcome",
            "description": "Customer purchase decision (0 or 1)",
        },
    ]

    for c in confounders:
        nodes.append({
            "id": c,
            "label": c.replace("_", " ").title(),
            "type": "confounder",
            "description": f"Observed pre-treatment covariate ({c})",
        })

    if include_unobserved:
        nodes.append({
            "id": UNOBSERVED_CONFOUNDER,
            "label": "Unobserved Confounder (U)",
            "type": "unobserved",
            "description": "Latent customer price sensitivity (tested in sensitivity analysis)",
        })

    edges = []
    for c in confounders:
        edges.append({"source": c, "target": treatment, "type": "confounding"})
        edges.append({"source": c, "target": outcome, "type": "confounding"})

    if include_unobserved:
        edges.append({"source": UNOBSERVED_CONFOUNDER, "target": treatment, "type": "unobserved"})
        edges.append({"source": UNOBSERVED_CONFOUNDER, "target": outcome, "type": "unobserved"})

    edges.append({"source": treatment, "target": outcome, "type": "causal_effect"})

    return {
        "nodes": nodes,
        "edges": edges,
        "treatment": treatment,
        "outcome": outcome,
        "confounders": confounders,
        "unobserved": UNOBSERVED_CONFOUNDER if include_unobserved else None,
    }


def identify_causal_effect(
    df: pd.DataFrame,
    treatment: str = "discount_amount",
    outcome: str = "purchased",
    confounders: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Instantiate a DoWhy CausalModel, build DAG, and identify the causal estimand via backdoor."""
    if confounders is None:
        confounders = [c for c in DEFAULT_CONFOUNDERS if c in df.columns]

    dot_graph = build_dot_graph(
        confounders=confounders,
        treatment=treatment,
        outcome=outcome,
        include_unobserved=False,  # identify on observed data
    )

    # Prepare DataFrame subset
    cols = [treatment, outcome] + confounders
    df_sub = df[cols].copy()

    # DoWhy causal model
    model = CausalModel(
        data=df_sub,
        treatment=treatment,
        outcome=outcome,
        graph=dot_graph,
    )

    estimand = model.identify_effect(proceed_when_unidentifiable=True)

    backdoor_variables = estimand.get_backdoor_variables()
    estimand_str = str(estimand)

    dag_json = get_dag_json(confounders=confounders, treatment=treatment, outcome=outcome)
    dag_json["identified_backdoor_variables"] = backdoor_variables
    dag_json["estimand_summary"] = (
        f"Identified Backdoor Adjustment Set: {', '.join(backdoor_variables) if backdoor_variables else 'None'}"
    )
    dag_json["dot_graph"] = dot_graph

    return {
        "model": model,
        "estimand": estimand,
        "backdoor_variables": backdoor_variables,
        "estimand_text": estimand_str,
        "dag_json": dag_json,
    }


if __name__ == "__main__":
    df = pd.read_csv("backend/data/retail_campaign.csv")
    result = identify_causal_effect(df)
    print("=== Identified Causal Estimand ===")
    print(result["estimand_text"])
    print("\n=== Backdoor Adjustment Set ===")
    print(result["backdoor_variables"])
    print("\nDAG JSON Node Count:", len(result["dag_json"]["nodes"]))
    print("DAG JSON Edge Count:", len(result["dag_json"]["edges"]))
