import React, { useCallback, useEffect, useState } from 'react';
import { Activity, BrainCircuit, Download, HeartHandshake, RefreshCw, ShieldAlert, Sparkles, Target } from 'lucide-react';
import { getChurnPredictions } from '../api';

const formatPercent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
const churnLabelNames = new Set([
  'churn', 'churned', 'ischurn', 'ischurned', 'churnstatus', 'churnlabel',
  'churnflag', 'attrition', 'attritionflag', 'attrited', 'isattrited', 'exited',
  'exit', 'defaulted', 'default', 'target', 'label', 'accountstatus',
]);
const normalizeColumn = (column) => column.toLowerCase().replace(/[^a-z0-9]/g, '');
const positiveValues = new Set([
  '1', 'true', 'yes', 'y', 'churn', 'churned', 'left', 'inactive', 'attrited',
  'exited', 'exit', 'default', 'defaulted', 'cancelled', 'canceled',
]);
const valuesForColumn = (column, preview) => [...new Set(
  preview
    .map((row) => row[column])
    .filter((value) => value !== '' && value !== null && value !== undefined)
    .map(String)
)];

const exportHighRiskCustomers = (customers) => {
  const headers = ['row_index', 'customer_id', 'churn_probability', 'risk_level', 'suggested_action'];
  const escapeCsv = (value) => {
    const text = String(value ?? '');
    const safeText = /^[\t\r\n ]*[=+\-@]/.test(text) ? `'${text}` : text;
    return `"${safeText.replace(/"/g, '""')}"`;
  };
  const rows = customers.map((customer) => [
    customer.row_index,
    customer.customer_id,
    customer.churn_probability,
    customer.risk_level,
    customer.suggested_action,
  ]);
  const csv = [headers, ...rows].map((row) => row.map(escapeCsv).join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob(['\uFEFF', csv], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'high-risk-customers.csv';
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
};

const solutionTracks = [
  {
    icon: Activity,
    title: 'Practical ML',
    label: 'Start here',
    methods: 'Logistic regression, Random Forest, LightGBM',
    actions: 'Rank customers by risk, identify churn drivers, and trigger tested service recovery or re-engagement.',
    when: 'Best for structured customer history and teams that need explainable, fast-to-refresh scores.',
  },
  {
    icon: BrainCircuit,
    title: 'Deep learning',
    label: 'When data supports it',
    methods: 'LSTM/GRU or Transformers for event sequences; NLP for support conversations',
    actions: 'Use purchase and contact sequences to detect changing behavior, or surface themes in customer feedback.',
    when: 'Consider when you have large, timestamped interaction histories or substantial text—not just a small tabular dataset.',
  },
  {
    icon: HeartHandshake,
    title: 'Retention actioning',
    label: 'Turn risk into action',
    methods: 'Next-best action, uplift modeling, survival analysis, controlled experiments',
    actions: 'Match the response to the reason for risk; reserve discounts for customers whose incremental response justifies the cost.',
    when: 'Measure save rate, incremental margin, customer experience, and unsubscribe/complaint guardrails.',
  },
];

export default function ChurnPredictionView({ columns = [], preview = [] }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [target, setTarget] = useState(
    () => columns.find((column) => churnLabelNames.has(normalizeColumn(column))) || ''
  );
  const [positiveValue, setPositiveValue] = useState(() => {
    const inferredTarget = columns.find((column) => churnLabelNames.has(normalizeColumn(column)));
    return inferredTarget
      ? valuesForColumn(inferredTarget, preview).find((value) => positiveValues.has(value.toLowerCase())) || ''
      : '';
  });

  const loadPredictions = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setResult(await getChurnPredictions(target, positiveValue));
    } catch (err) {
      setResult(null);
      setError(err?.response?.data?.detail || err.message || 'Unable to load churn predictions.');
    } finally {
      setLoading(false);
    }
  }, [positiveValue, target]);

  useEffect(() => {
    let active = true;
    getChurnPredictions(target, positiveValue)
      .then((data) => {
        if (active) setResult(data);
      })
      .catch((err) => {
        if (active) {
          setError(err?.response?.data?.detail || err.message || 'Unable to load churn predictions.');
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [positiveValue, target]);

  const metrics = result?.validation?.metrics || {};
  const risk = result?.risk_distribution || {};
  const targetValues = target ? valuesForColumn(target, preview) : [];

  return (
    <div>
      <div className="view-header">
        <div>
          <h2>Customer Churn Prediction</h2>
          <div className="view-subtitle">
            Identify customers at risk, validate the model, and plan measured retention actions
          </div>
        </div>
        <button className="btn btn-primary" onClick={loadPredictions} disabled={loading}>
          <RefreshCw size={16} className={loading ? 'spin' : ''} />
          {loading ? 'Scoring…' : 'Refresh scores'}
        </button>
      </div>

      <div className="card">
        <div className="card-title">
          <ShieldAlert size={20} color="var(--accent-amber)" />
          <span>Risk scoring & validation</span>
        </div>

        <div className="churn-target-picker">
          <label className="form-label" htmlFor="churn-target">
            <Target size={15} /> Churn / target label
          </label>
          <select
            id="churn-target"
            className="form-select"
            value={target}
            onChange={(event) => {
              const selectedTarget = event.target.value;
              const values = selectedTarget ? valuesForColumn(selectedTarget, preview) : [];
              setTarget(selectedTarget);
              setPositiveValue(values.find((value) => positiveValues.has(value.toLowerCase())) || '');
              setLoading(true);
              setError('');
            }}
          >
            <option value="">Auto-detect churn label or recency</option>
            {columns.map((column) => <option key={column} value={column}>{column}</option>)}
          </select>
          <span className="churn-target-help">
            Choose the binary column that indicates churn. The target column is excluded from model features.
          </span>
          {target && targetValues.length === 2 && (
            <>
              <label className="form-label" htmlFor="churn-positive-value">Which value means churn?</label>
              <select
                id="churn-positive-value"
                className="form-select"
                value={positiveValue}
                onChange={(event) => {
                  setPositiveValue(event.target.value);
                  setLoading(true);
                  setError('');
                }}
              >
                <option value="">Use inferred positive class</option>
                {targetValues.map((value) => <option key={value} value={value}>{value}</option>)}
              </select>
            </>
          )}
        </div>

        {error && <div className="info-box error-box" role="alert">{error}</div>}
        {loading && <div className="churn-loading">Training the churn model and scoring customers…</div>}

        {result && !loading && (
          <>
            <div className="info-box churn-definition">
              <strong>Target definition:</strong> {result.target_definition}
              <span>
                Model: {result.model}. Holdout: {result.validation.method}. Risk scores help prioritize
                attention; they do not prove that an intervention will prevent churn.
              </span>
            </div>

            <div className="kpi-grid">
              <div className="kpi-card">
                <span className="kpi-label">Customers scored</span>
                <span className="kpi-value">{Number(result.observations).toLocaleString()}</span>
                <span className="kpi-subtext">{result.label_source === 'explicit' ? 'Observed churn labels' : 'Recency-proxy labels'}</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">High risk</span>
                <span className="kpi-value" style={{ color: 'var(--accent-rose)' }}>{Number(risk.high || 0).toLocaleString()}</span>
                <span className="kpi-subtext">Predicted risk ≥ 70%</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Holdout ROC AUC</span>
                <span className="kpi-value">{formatPercent(metrics.roc_auc)}</span>
                <span className="kpi-subtext">Ranking performance on held-out data</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Holdout recall</span>
                <span className="kpi-value">{formatPercent(metrics.recall)}</span>
                <span className="kpi-subtext">At the tuned {formatPercent(result.validation.decision_threshold)} threshold</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Selected model</span>
                <span className="kpi-value" style={{ fontSize: '1.1rem' }}>{result.model}</span>
                <span className="kpi-subtext">Chosen on training-only validation data</span>
              </div>
            </div>

            <div className="churn-metric-strip">
              <span><b>Precision</b> {formatPercent(metrics.precision)}</span>
              <span><b>F1</b> {formatPercent(metrics.f1)}</span>
              <span><b>Accuracy</b> {formatPercent(metrics.accuracy)}</span>
              <span><b>Medium risk</b> {Number(risk.medium || 0).toLocaleString()}</span>
              <span><b>Low risk</b> {Number(risk.low || 0).toLocaleString()}</span>
            </div>

            <div className="analysis-panel churn-driver-panel">
              <div className="analysis-panel-heading">
                <div>
                  <h3>Top churn drivers</h3>
                  <p>Validation ROC AUC change when each feature is shuffled; this is predictive, not causal.</p>
                </div>
              </div>
              <div className="churn-driver-list">
                {result.top_churn_drivers.length === 0 && (
                  <div className="churn-driver-row">No stable predictive feature signal was found.</div>
                )}
                {result.top_churn_drivers.map((driver) => (
                  <div className="churn-driver-row" key={driver.feature}>
                    <span>{driver.feature}</span>
                    <span>{Number(driver.importance).toFixed(3)} AUC impact</span>
                  </div>
                ))}
              </div>
              <div className="churn-target-help">
                Model selection AUC: {Object.entries(result.validation.candidate_roc_auc)
                  .map(([name, auc]) => `${name} ${formatPercent(auc)}`).join(' · ')}
              </div>
            </div>

            <div className="analysis-panel churn-customer-panel">
              <div className="analysis-panel-heading">
                <div>
                  <h3>Highest-risk customers</h3>
                  <p>Top 25 by predicted risk; suggested action is a starting point, not an automated offer.</p>
                </div>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => exportHighRiskCustomers(result.high_risk_customers)}
                  disabled={!result.high_risk_customers?.length}
                >
                  <Download size={16} />
                  Export CSV
                </button>
              </div>
              <div className="prescription-table-wrap">
                <table className="prescription-table churn-table">
                  <thead>
                    <tr><th>Customer</th><th>Risk</th><th>Level</th><th>Suggested next step</th></tr>
                  </thead>
                  <tbody>
                    {result.high_risk_customers.map((customer) => (
                      <tr key={customer.row_index}>
                        <td>{customer.customer_id || `Row ${customer.row_index + 1}`}</td>
                        <td>{formatPercent(customer.churn_probability)}</td>
                        <td><span className={`badge ${customer.risk_level === 'High' ? 'badge-danger' : customer.risk_level === 'Medium' ? 'badge-warning' : 'badge-success'}`}>{customer.risk_level}</span></td>
                        <td className="churn-action-cell">{customer.suggested_action}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="info-box churn-caveat">
              <Sparkles size={17} />
              <span>{result.scoring_note}</span>
            </div>
          </>
        )}

        {!result && !loading && !error && (
          <div className="info-box">No churn prediction results are available for this dataset.</div>
        )}
      </div>

      <section className="churn-solutions">
        <div className="churn-solutions-heading">
          <div>
            <span className="workspace-eyebrow">AFTER IDENTIFYING RISK</span>
            <h3>From prediction to retention</h3>
            <p>Choose the method based on the data you have, then test whether the intervention creates incremental value.</p>
          </div>
        </div>
        <div className="churn-solution-grid">
          {solutionTracks.map((track) => {
            const Icon = track.icon;
            return (
              <article className="churn-solution-card" key={track.title}>
                <div className="churn-solution-top">
                  <span className="churn-solution-icon"><Icon size={19} /></span>
                  <span className="badge badge-info">{track.label}</span>
                </div>
                <h4>{track.title}</h4>
                <p className="churn-methods">{track.methods}</p>
                <p>{track.actions}</p>
                <div className="churn-when"><b>Use when:</b> {track.when}</div>
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}
