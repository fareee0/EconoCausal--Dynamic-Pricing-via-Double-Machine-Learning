import React from 'react';
import { Sliders, DollarSign, Target, Award } from 'lucide-react';

export default function PrescriptionView() {
  return (
    <div>
      <div className="view-header">
        <div>
          <h2>Prescriptive Budget Optimization</h2>
          <div className="view-subtitle">
            Mixed-Integer Linear Programming (MILP) Multiple-Choice Knapsack
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          <Sliders size={20} color="#10b981" />
          <span>Phase 3 Engine Staging: Prescriptive Allocation</span>
        </div>
        <p style={{ color: '#94a3b8', fontSize: '0.92rem', lineHeight: '1.6', marginBottom: '20px' }}>
          This view solves the multiple-choice knapsack problem with <code>scipy.optimize.milp</code> to maximize total expected incremental profit subject to at most one discount per customer and total expected spend &le; B:
          <br />
          <code style={{ display: 'inline-block', marginTop: '6px', color: '#a5b4fc', background: 'rgba(99,102,241,0.1)', padding: '2px 8px', borderRadius: '4px' }}>
            &Sigma;<sub>d</sub> x<sub>i,d</sub> &le; 1, &nbsp;&nbsp; &Sigma;<sub>i,d</sub> x<sub>i,d</sub> &times; P(buy | d) &times; d &le; B
          </code>
        </p>

        <div className="kpi-grid">
          <div className="kpi-card">
            <span className="kpi-label">Optimizer Solver</span>
            <span className="kpi-value" style={{ fontSize: '1.3rem' }}>SciPy MILP</span>
            <span className="kpi-subtext">Binary decision variables x_&#123;i,d&#125;</span>
          </div>
          <div className="kpi-card">
            <span className="kpi-label">Fast Fallback</span>
            <span className="kpi-value" style={{ fontSize: '1.3rem', color: '#f59e0b' }}>Greedy Marginal ROI</span>
            <span className="kpi-subtext">Sub-second large scale solver</span>
          </div>
          <div className="kpi-card">
            <span className="kpi-label">Benchmark Strategies</span>
            <span className="kpi-value" style={{ fontSize: '1.3rem', color: '#6366f1' }}>4 Policies</span>
            <span className="kpi-subtext">Blanket, churn risk, high prob</span>
          </div>
          <div className="kpi-card">
            <span className="kpi-label">Customer Output</span>
            <span className="kpi-value" style={{ fontSize: '1.3rem', color: '#34d399' }}>Exportable CSV</span>
            <span className="kpi-subtext">Targeting action list with ROI</span>
          </div>
        </div>

        <div className="info-box" style={{ marginTop: '16px' }}>
          Phase 1 acceptance completed! Prescriptive optimization will be integrated in Phase 3.
        </div>
      </div>
    </div>
  );
}
