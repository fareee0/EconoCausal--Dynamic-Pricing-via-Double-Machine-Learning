import React from 'react';
import { TrendingUp, Layers, CheckCircle2 } from 'lucide-react';

export default function UpliftView() {
  return (
    <div>
      <div className="view-header">
        <div>
          <h2>Uplift Modeling & Double Machine Learning</h2>
          <div className="view-subtitle">
            Causal Forest DML vs T-Learner vs Naive Supervised Churn Baselines
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          <TrendingUp size={20} color="#818cf8" />
          <span>Phase 2 Engine Staging: Double ML (EconML)</span>
        </div>
        <p style={{ color: '#94a3b8', fontSize: '0.92rem', lineHeight: '1.6', marginBottom: '20px' }}>
          This view will train EconML's <code>CausalForestDML</code> and <code>LinearDML</code> with cross-fitting (K = 5) and LightGBM / Random Forest nuisance models to estimate individual treatment effects:
          &nbsp;<code style={{ color: '#a5b4fc' }}>&tau;<sub>10</sub>(X)</code> and <code style={{ color: '#a5b4fc' }}>&tau;<sub>20</sub>(X)</code>.
        </p>

        <div className="kpi-grid">
          <div className="kpi-card">
            <span className="kpi-label">Target Estimator</span>
            <span className="kpi-value" style={{ fontSize: '1.3rem' }}>CausalForestDML</span>
            <span className="kpi-subtext">Discrete treatment cross-fitted</span>
          </div>
          <div className="kpi-card">
            <span className="kpi-label">Uplift Metric Target</span>
            <span className="kpi-value" style={{ fontSize: '1.3rem', color: '#38bdf8' }}>Qini / AUUC</span>
            <span className="kpi-subtext">Area Under Uplift Curve</span>
          </div>
          <div className="kpi-card">
            <span className="kpi-label">Ground Truth Validation</span>
            <span className="kpi-value" style={{ fontSize: '1.3rem', color: '#34d399' }}>PEHE / RMSE</span>
            <span className="kpi-subtext">Evaluated against synthetic truth</span>
          </div>
          <div className="kpi-card">
            <span className="kpi-label">Customer Segments</span>
            <span className="kpi-value" style={{ fontSize: '1.3rem' }}>4 Latent Types</span>
            <span className="kpi-subtext">Persuadables, Sure Things, etc.</span>
          </div>
        </div>

        <div className="info-box" style={{ marginTop: '16px' }}>
          Phase 1 acceptance completed! Ready to proceed to Phase 2 (Double ML training, Qini curves, AUUC, segment scatter).
        </div>
      </div>
    </div>
  );
}
