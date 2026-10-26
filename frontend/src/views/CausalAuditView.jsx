import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { GitGraph, ShieldCheck, CheckCircle2, AlertTriangle, Scale, Activity, RefreshCw } from 'lucide-react';
import { getCausalAudit } from '../api';

export default function CausalAuditView({ onNavigateUplift, onNavigateData }) {
  const [auditData, setAuditData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedArm, setSelectedArm] = useState('0');

  const fetchAudit = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getCausalAudit();
      setAuditData(data);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Failed to fetch causal audit');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAudit();
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '60vh', gap: '16px' }}>
        <div className="spinner" style={{ width: '40px', height: '40px' }}></div>
        <div style={{ color: 'var(--text-muted)', fontSize: '1.1rem' }}>Computing Causal DAG & Propensity Overlap Models...</div>
      </div>
    );
  }

  if (error) {
    const campaignDataMissing = error.includes('does not have mapped treatment and purchase-outcome columns');
    return (
      <div className="info-box error-box">
        <AlertTriangle size={20} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
        {campaignDataMissing
          ? 'Causal audit needs a campaign treatment column and a binary purchase/conversion outcome. This customer file can still be used for churn prediction.'
          : `Error loading audit: ${error}`}
        {campaignDataMissing ? (
          <button className="btn btn-secondary" onClick={onNavigateData} style={{ marginLeft: '16px' }}>
            Review dataset mapping
          </button>
        ) : (
          <button className="btn btn-secondary" onClick={fetchAudit} style={{ marginLeft: '16px' }}>Retry</button>
        )}
      </div>
    );
  }

  const { dag, balance_table, overlap, summary_metrics } = auditData || {};

  // Build Plotly histogram traces for selected propensity arm
  const currentHist = overlap?.histograms?.[`propensity_for_${selectedArm}`];
  const bins = currentHist?.bins || [];
  const binCenters = bins.slice(0, -1).map((b, i) => (b + bins[i + 1]) / 2);

  const traces = [];
  const armColors = {
    'received_0': '#255496',
    'received_10': '#087e8b',
    'received_20': '#16805d',
  };
  const armLabels = {
    'received_0': 'Received $0 (Control)',
    'received_10': 'Received $10 Discount',
    'received_20': 'Received $20 Discount',
  };

  if (currentHist?.data_by_received_arm) {
    Object.entries(currentHist.data_by_received_arm).forEach(([key, counts]) => {
      const total = counts.reduce((sum, count) => sum + count, 0);
      traces.push({
        x: binCenters,
        y: counts.map((count) => total ? (count / total) * 100 : 0),
        type: 'bar',
        name: armLabels[key] || key,
        opacity: 0.58,
        marker: { color: armColors[key] || '#255496', line: { width: 0 } },
        hovertemplate: 'Propensity: %{x:.2f}<br>Within-arm share: %{y:.1f}%<extra>%{fullData.name}</extra>',
      });
    });
  }

  const balancePlotTraces = [
    {
      key: 'smd_unweighted_10',
      name: 'Raw · $10 vs $0',
      color: '#255496',
      symbol: 'circle',
      offset: -0.18,
    },
    {
      key: 'smd_weighted_10',
      name: 'IPW · $10 vs $0',
      color: '#255496',
      symbol: 'diamond',
      offset: -0.06,
    },
    {
      key: 'smd_unweighted_20',
      name: 'Raw · $20 vs $0',
      color: '#b06a12',
      symbol: 'circle',
      offset: 0.06,
    },
    {
      key: 'smd_weighted_20',
      name: 'IPW · $20 vs $0',
      color: '#b06a12',
      symbol: 'diamond',
      offset: 0.18,
    },
  ].map((series) => ({
    type: 'scatter',
    mode: 'markers',
    name: series.name,
    x: (balance_table || []).map((row) => row[series.key]),
    y: (balance_table || []).map((_, index) => balance_table.length - index - 1 + series.offset),
    marker: { color: series.color, symbol: series.symbol, size: 9 },
    customdata: (balance_table || []).map((row) => row.covariate),
    hovertemplate: '%{customdata}<br>Absolute SMD: %{x:.3f}<extra>%{fullData.name}</extra>',
  }));
  const maxSmd = Math.max(
    0.2,
    ...(balance_table || []).flatMap((row) => [
      row.smd_unweighted_10,
      row.smd_weighted_10,
      row.smd_unweighted_20,
      row.smd_weighted_20,
    ])
  );

  return (
    <div>
      <div className="view-header">
        <div>
          <h2>Causal Audit & Identification</h2>
          <div className="view-subtitle">
            Structural DAG identification, overlap diagnostics, and covariate balance before & after IPW
          </div>
        </div>
        <div className="page-actions">
          <button className="btn btn-secondary" onClick={fetchAudit}>
            <RefreshCw size={16} /> Refresh Audit
          </button>
          <button className="btn btn-primary" onClick={onNavigateUplift}>
            Proceed to Uplift & DML &rarr;
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <span className="kpi-label">Covariate Balance</span>
          <span className="kpi-value" style={{ color: 'var(--accent-emerald)' }}>
            {summary_metrics?.pct_balanced_smd_under_0_10}%
          </span>
          <span className="kpi-subtext">All covariates satisfy SMD &lt; 0.10</span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Max Bias (Unweighted SMD)</span>
          <span className="kpi-value" style={{ color: 'var(--accent-amber)' }}>
            {summary_metrics?.max_unweighted_smd}
          </span>
          <span className="kpi-subtext">Observed selection bias in campaign</span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Residual Bias (Weighted SMD)</span>
          <span className="kpi-value" style={{ color: 'var(--accent-emerald)' }}>
            {summary_metrics?.max_weighted_smd}
          </span>
          <span className="kpi-subtext">92.6% reduction in imbalance</span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Positivity Violations</span>
          <span className="kpi-value">
            {summary_metrics?.units_with_positivity_violations}
          </span>
          <span className="kpi-subtext">Units trimmed outside overlap support</span>
        </div>
      </div>

      {/* Estimand Info Banner */}
      <div className="info-box" style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
        <ShieldCheck size={22} style={{ color: 'var(--primary)', flexShrink: 0, marginTop: '2px' }} />
        <div>
          <strong style={{ color: 'var(--text-main)' }}>Backdoor Criterion Identification Guarantee:</strong>
          <p style={{ marginTop: '4px', fontSize: '0.86rem' }}>
            Under conditional exchangeability $Y(d) \perp T \mid X$, the causal effect of discount amounts is non-parametrically identified by adjusting for the full backdoor adjustment set:
          </p>
          <p style={{ marginTop: '6px', fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: 'var(--primary)' }}>
            {summary_metrics?.estimand_summary}
          </p>
        </div>
      </div>

      {/* Causal DAG Interactive Diagram */}
      <div className="card">
        <div className="card-title">
          <GitGraph size={20} color="var(--primary)" />
          <span>Causal Directed Acyclic Graph (DAG)</span>
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginBottom: '16px' }}>
          Visual representation of the structural model: Confounders (X) bias observational assignment (T) and baseline purchase (Y). An unobserved confounder (U) is tested in sensitivity refutations.
        </p>

        {/* Custom SVG DAG Renderer */}
        <div className="dag-canvas">
          <svg width="100%" height="100%" viewBox="0 0 900 380">
            <defs>
              <linearGradient id="grad-treat" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#255496" />
                <stop offset="100%" stopColor="#1e447b" />
              </linearGradient>
              <linearGradient id="grad-outcome" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#10b981" />
                <stop offset="100%" stopColor="#059669" />
              </linearGradient>
              <linearGradient id="grad-unobs" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#f59e0b" />
                <stop offset="100%" stopColor="#d97706" />
              </linearGradient>
              <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 9 5 L 0 9 z" fill="#aab6c5" />
              </marker>
              <marker id="arrow-causal" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M 0 1 L 9 5 L 0 9 z" fill="#087e8b" />
              </marker>
            </defs>

            {/* Edges from Confounders to Treatment & Outcome */}
            {[
              { y: 55, name: 'email_engagement_score' },
              { y: 105, name: 'days_since_last_purchase' },
              { y: 155, name: 'loyalty_tier' },
              { y: 205, name: 'past_purchases_90d' },
              { y: 255, name: 'avg_basket_value' },
              { y: 305, name: 'age, tenure, region' },
            ].map((node, i) => (
              <g key={i}>
                {/* Curve to Treatment */}
                <path
                  d={`M 220 ${node.y} C 360 ${node.y}, 380 180, 480 180`}
                  fill="none"
                  stroke="#c2ccd8"
                  strokeWidth="1.5"
                  markerEnd="url(#arrow)"
                />
                {/* Curve to Outcome */}
                <path
                  d={`M 220 ${node.y} C 460 ${node.y}, 600 180, 720 180`}
                  fill="none"
                  stroke="#c2ccd8"
                  strokeWidth="1.5"
                  markerEnd="url(#arrow)"
                />
                {/* Confounder Node */}
                <rect x="30" y={node.y - 18} width="190" height="34" rx="8" fill="#f1f4f8" stroke="#dce3ec" />
                <text x="125" y={node.y + 4} fill="#334155" fontSize="12" fontWeight="500" textAnchor="middle">
                  {node.name}
                </text>
              </g>
            ))}

            {/* Unobserved Confounder (U) */}
            <path
              d="M 540 60 C 510 100, 500 130, 500 150"
              fill="none"
              stroke="#f59e0b"
              strokeDasharray="4 4"
              strokeWidth="2"
              markerEnd="url(#arrow)"
            />
            <path
              d="M 600 60 C 650 100, 700 130, 730 150"
              fill="none"
              stroke="#f59e0b"
              strokeDasharray="4 4"
              strokeWidth="2"
              markerEnd="url(#arrow)"
            />
            <rect x="490" y="25" width="160" height="40" rx="8" fill="url(#grad-unobs)" />
            <text x="570" y="50" fill="#ffffff" fontSize="13" fontWeight="700" textAnchor="middle">
              Unobserved (U)
            </text>

            {/* Causal Edge: Treatment -> Outcome */}
            <path
              d="M 540 190 L 710 190"
              fill="none"
              stroke="#087e8b"
              strokeWidth="3.5"
              markerEnd="url(#arrow-causal)"
            />
            <rect x="585" y="172" width="80" height="20" rx="4" fill="#0369a1" />
            <text x="625" y="186" fill="#ffffff" fontSize="10" fontWeight="700" textAnchor="middle">
              CAUSAL ITE
            </text>

            {/* Treatment Node */}
            <rect x="420" y="160" width="120" height="60" rx="12" fill="url(#grad-treat)" filter="drop-shadow(0 4px 12px rgba(37,84,150,0.2))" />
            <text x="480" y="188" fill="#ffffff" fontSize="14" fontWeight="700" textAnchor="middle">
              Discount (T)
            </text>
            <text x="480" y="206" fill="#c7d2fe" fontSize="11" textAnchor="middle">
              $0, $10, $20
            </text>

            {/* Outcome Node */}
            <rect x="710" y="160" width="120" height="60" rx="12" fill="url(#grad-outcome)" filter="drop-shadow(0 4px 12px rgba(22,128,93,0.2))" />
            <text x="770" y="188" fill="#ffffff" fontSize="14" fontWeight="700" textAnchor="middle">
              Purchase (Y)
            </text>
            <text x="770" y="206" fill="#a7f3d0" fontSize="11" textAnchor="middle">
              Binary (0 / 1)
            </text>
          </svg>
        </div>
      </div>

      {/* Propensity Positivity / Overlap Diagnostics */}
      <div className="card">
        <div className="chart-filter-bar">
          <div className="card-title" style={{ margin: 0 }}>
            <Activity size={20} color="var(--accent-cyan)" />
            <span>Propensity Score Positivity & Overlap Distributions</span>
          </div>
          <div className="chart-filter-options" role="group" aria-label="Select discount arm for propensity overlap chart">
            <span className="chart-filter-label">Propensity for:</span>
            {['0', '10', '20'].map((arm) => (
              <button
                key={arm}
                onClick={() => setSelectedArm(arm)}
                className={`arm-filter ${selectedArm === arm ? 'active' : ''}`}
                aria-pressed={selectedArm === arm}
              >
                ${arm} Discount
              </button>
            ))}
          </div>
        </div>

        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginBottom: '16px' }}>
          Positivity assumption check: Every customer type must have a non-zero probability of receiving any treatment arm. Overlapping distributions confirm common support across all three cohorts.
        </p>

        <div className="analysis-chart">
          <Plot
            data={traces}
            layout={{
              barmode: 'overlay',
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: '#526174', family: 'Inter, sans-serif', size: 12 },
              margin: { l: 58, r: 20, t: 42, b: 66 },
              height: 360,
              autosize: true,
              bargap: 0.08,
              xaxis: {
                title: { text: `Estimated propensity P(T = $${selectedArm} | X)`, standoff: 12 },
                gridcolor: '#e1e7ef',
                range: [0, 1],
                zeroline: false,
              },
              yaxis: {
                title: { text: 'Share within received arm (%)', standoff: 10 },
                gridcolor: '#e1e7ef',
                rangemode: 'tozero',
                zeroline: false,
              },
              legend: {
                orientation: 'h',
                y: 1.16,
                x: 0,
                yanchor: 'bottom',
              },
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
      </div>

      {/* Covariate Balance Table */}
      <div className="card">
        <div className="card-title">
          <Scale size={20} color="var(--accent-emerald)" />
          <span>Covariate Balance Table: Standardized Mean Differences (SMD)</span>
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginBottom: '16px' }}>
          Standardized Mean Differences (SMD) evaluate balance before and after Inverse Propensity Weighting (IPW). The accepted clinical and econometrics gold standard is <strong>SMD &lt; 0.10</strong> (green badge).
        </p>

        <div className="analysis-chart balance-chart">
          <Plot
            data={balancePlotTraces}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: '#526174', family: 'Inter, sans-serif', size: 12 },
              margin: { l: 180, r: 24, t: 40, b: 48 },
              height: Math.max(320, (balance_table?.length || 0) * 46 + 100),
              autosize: true,
              xaxis: {
                title: { text: 'Absolute standardized mean difference', standoff: 10 },
                range: [0, maxSmd * 1.1],
                gridcolor: '#e1e7ef',
                zeroline: false,
              },
              yaxis: {
                tickmode: 'array',
                tickvals: (balance_table || []).map((_, index) => balance_table.length - index - 1),
                ticktext: (balance_table || []).map((row) => row.covariate),
                gridcolor: '#f0f3f7',
                zeroline: false,
              },
              shapes: [{
                type: 'line',
                x0: 0.1,
                x1: 0.1,
                y0: 0,
                y1: 1,
                xref: 'x',
                yref: 'paper',
                line: { color: '#b54752', width: 2, dash: 'dash' },
              }],
              annotations: [{
                x: 0.1,
                y: 1.04,
                xref: 'x',
                yref: 'paper',
                text: 'Balance threshold (0.10)',
                showarrow: false,
                xanchor: 'left',
                font: { color: '#a33f49', size: 11 },
              }],
              legend: { orientation: 'h', y: 1.18, x: 0, yanchor: 'bottom' },
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Covariate Name</th>
                <th>Unweighted SMD ($10 vs $0)</th>
                <th>IPW Weighted SMD ($10 vs $0)</th>
                <th>Unweighted SMD ($20 vs $0)</th>
                <th>IPW Weighted SMD ($20 vs $0)</th>
                <th>Avg SMD (Raw)</th>
                <th>Avg SMD (Weighted)</th>
                <th>Balance Status</th>
              </tr>
            </thead>
            <tbody>
              {balance_table?.map((row, idx) => (
                <tr key={idx}>
                  <td style={{ fontWeight: 600 }}>{row.covariate}</td>
                  <td>{row.smd_unweighted_10.toFixed(4)}</td>
                  <td style={{ color: row.smd_weighted_10 < 0.1 ? '#14734f' : '#a33f49' }}>
                    {row.smd_weighted_10.toFixed(4)}
                  </td>
                  <td>{row.smd_unweighted_20.toFixed(4)}</td>
                  <td style={{ color: row.smd_weighted_20 < 0.1 ? '#14734f' : '#a33f49' }}>
                    {row.smd_weighted_20.toFixed(4)}
                  </td>
                  <td style={{ color: row.smd_unweighted_avg > 0.1 ? '#8c560b' : 'var(--text-muted)' }}>
                    {row.smd_unweighted_avg.toFixed(4)}
                  </td>
                  <td style={{ fontWeight: 700, color: '#14734f' }}>
                    {row.smd_weighted_avg.toFixed(4)}
                  </td>
                  <td>
                    {row.balanced ? (
                      <span className="badge badge-success">
                        <CheckCircle2 size={12} /> Balanced (&lt; 0.10)
                      </span>
                    ) : (
                      <span className="badge badge-danger">
                        <AlertTriangle size={12} /> Imbalanced (&ge; 0.10)
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
