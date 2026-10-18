# EconoCausal: Double ML & Prescriptive Promotion Targeting

**EconoCausal** is a dynamic-pricing and promotion-targeting decision engine that uses **Double Machine Learning (DML)** to estimate heterogeneous Individual Treatment Effects (ITE) of discounts and allocates promotional budgets optimally using **Mixed-Integer Linear Programming (MILP)**.

---

## Architecture & Technology Stack

- **Backend**: Python 3.11+ / 3.15, FastAPI, Uvicorn, Pydantic, DoWhy, EconML, Scikit-learn, LightGBM, SciPy (`milp`), Pandas, NumPy, Pytest.
- **Frontend**: React (Vite), Plotly.js (`react-plotly.js`), Axios, Lucide Icons, Custom Modern CSS design system.
- **Project Structure**:
  ```
  Econocasual/
  ├── backend/
  │   ├── app/                 # FastAPI application, CORS, schemas, routes
  │   │   ├── main.py
  │   │   └── schemas.py
  │   ├── causal/              # Causal inference core
  │   │   ├── __init__.py      # NetworkX 3.x / DoWhy compatibility bridge
  │   │   ├── graph.py         # Causal DAG & DoWhy Backdoor identification
  │   │   └── propensity.py    # Multinomial propensity, IPW balance, overlap
  │   ├── data/                # Data generator & CSV storage
  │   │   ├── generate.py      # Structural causal DGP (20,000 customers)
  │   │   ├── retail_campaign.csv  # Observed training dataset
  │   │   └── ground_truth.csv     # True ITE & latent segments (validation ONLY)
  │   ├── optimization/        # Phase 3 Prescriptive Knapsack
  │   ├── tests/               # Pytest test suite
  │   │   └── test_phase0_phase1.py
  │   └── requirements.txt
  ├── frontend/
  │   ├── src/
  │   │   ├── views/           # Data & Budget, Causal Audit, Uplift, Prescription
  │   │   │   ├── DataBudgetView.jsx
  │   │   │   ├── CausalAuditView.jsx
  │   │   │   ├── UpliftView.jsx
  │   │   │   └── PrescriptionView.jsx
  │   │   ├── components/      # Sidebar, Glass cards, KPI blocks
  │   │   ├── api.js           # REST API client
  │   │   ├── App.jsx
  │   │   └── index.css        # Curated Dark Obsidian tech theme
  │   ├── package.json
  │   └── vite.config.js
  ├── data/                    # Root symlink / mirrors for retail campaign CSVs
  ├── pytest.ini
  └── README.md
  ```

---

## Phase 0: Synthetic Retail Dataset with Known Ground Truth

Generated via `backend/data/generate.py` ($N = 20,000$ customers) from a structural model:
1. **Observed Confounders**:
   - `age` (18–75)
   - `tenure_months` (1–60)
   - `past_purchases_90d` (Poisson count, 0–15)
   - `avg_basket_value` ($20–$350, lognormal)
   - `days_since_last_purchase` (Recency: active 1–60d vs. lapsing 60–180d)
   - `loyalty_tier` (Bronze, Silver, Gold, Platinum)
   - `email_engagement_score` (Beta distribution, 0.0–1.0)
   - `region` (North, South, East, West)
2. **Observational Treatment Assignment**:
   - Treatment discount levels: $T \in \{0, 10, 20\}$
   - Assignment mechanism: Multinomial softmax probability based on email engagement, recency, loyalty tier, and unobserved sensitivity.
   - Result: Strong selection bias where marketing historically targeted lapsing customers (lower baseline purchase probability) with higher discounts.
3. **Four Latent Segments & Heterogeneous Treatment Effects (ITE)**:
   - **Persuadables (~27%)**: Modest baseline ($P \approx 0.18–0.32$), large positive treatment effect ($\tau_{10} \approx +0.24$, $\tau_{20} \approx +0.34$, demonstrating diminishing returns from $10 to $20).
   - **Sure Things (~28%)**: High baseline ($P \approx 0.75–0.90$), negligible discount effect ($\tau_{10} \approx +0.01$, $\tau_{20} \approx +0.02$).
   - **Lost Causes (~21%)**: Low baseline ($P \approx 0.02–0.06$), negligible discount effect ($\tau_{10} \approx +0.01$, $\tau_{20} \approx +0.02$).
   - **Sleeping Dogs (~24%)**: Moderate baseline ($P \approx 0.28–0.38$), negative discount effect ($\tau_{10} \approx -0.04$, $\tau_{20} \approx -0.07$) due to promo fatigue/unsubscribes.
4. **Unobserved Confounder**:
   - `unobserved_price_sensitivity` ($U \sim \mathcal{N}(0, 1)$) weakly influences treatment choice and purchase outcome; documented for refutation sensitivity testing and strictly excluded from observed campaign data.
5. **Data Files**:
   - `backend/data/retail_campaign.csv`: Observed data ($N = 20,000$) for model training.
   - `backend/data/ground_truth.csv`: Latent segments and true ITEs, used **only for evaluation and validation**.

---

## Phase 1: Causal Graph + Audit Dashboard

1. **Causal DAG & Backdoor Estimand (`backend/causal/graph.py`)**:
   - DoWhy `CausalModel` using a formal DOT graph.
   - Identifies non-parametric backdoor adjustment set:
     `{age, tenure_months, past_purchases_90d, avg_basket_value, days_since_last_purchase, loyalty_tier, email_engagement_score, region}`.
   - Exports graph JSON (nodes, edges, node types, estimand) for rendering.
2. **Propensity Modeling & Balance Diagnostics (`backend/causal/propensity.py`)**:
   - Multinomial Logistic Regression model for $P(T = d \mid X)$.
   - Positivity / overlap check across all three arms ($d \in \{0, 10, 20\}$).
   - Stabilized Inverse Propensity Weighting (IPW).
   - **Balance Result**: **100% of covariates achieved Standardized Mean Difference (SMD) < 0.10** after weighting (max unweighted SMD was $0.4891$, reduced to $0.0362$).
   - Extreme propensity trimming: 4 out of 20,000 units trimmed (< 0.02%).
3. **Frontend Dashboard (`frontend/`)**:
   - Sidebar navigation: **Data & Budget**, **Causal Audit**, **Uplift & DML**, **Prescription**.
   - **Data & Budget View**: Drag-and-drop CSV upload, dynamic column mapping, promotional budget ($5,000) and unit gross margin ($35.00) inputs, discount option tags, and 50-row paginated preview.
   - **Causal Audit View**:
     - Interactive SVG Causal DAG visualization.
     - Plotly.js propensity overlap histograms with interactive arm toggling.
     - Covariate balance table with unweighted vs. IPW-weighted SMDs and green badges.

---

## Quickstart & Commands

### 1. Backend Setup & Run
```bash
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
pip install httpx httpx2

# Generate synthetic dataset (if not already generated)
python backend/data/generate.py

# Start FastAPI server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Causal Audit: [http://127.0.0.1:8000/audit](http://127.0.0.1:8000/audit)

### 2. Frontend Setup & Run
```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```
- Web Application: [http://127.0.0.1:5173](http://127.0.0.1:5173)

### 3. Run Automated Tests
```bash
python -m pytest backend/tests -v
```
All 5 unit & integration tests pass cleanly.
