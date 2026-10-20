import React, { useState } from 'react';
import { Download, DollarSign, LoaderCircle, Sliders, Target, TrendingUp } from 'lucide-react';
import { runPrescription } from '../api';

const formatCurrency = (value) => `$${Number(value).toLocaleString(undefined, {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})}`;

function downloadRecommendations(recommendations) {
  const columns = [
    'row_index',
    'discount',
    'predicted_purchase_without_discount',
    'predicted_purchase_with_discount',
    'expected_spend',
    'expected_incremental_profit',
  ];
  const csv = [
    columns.join(','),
    ...recommendations.map((recommendation) =>
      columns.map((column) => recommendation[column]).join(',')
    ),
  ].join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'prescription-recommendations.csv';
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

export default function PrescriptionView() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleOptimize = async () => {
    setLoading(true);
    setError('');
    try {
      setResult(await runPrescription());
    } catch (err) {
      setResult(null);
      setError(err?.response?.data?.detail || err.message || 'Optimization failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="view-header">
        <div>
          <h2>Prescriptive Budget Optimization</h2>
          <div className="view-subtitle">
            Customer-level discount recommendations under a hard promotional budget
          </div>
        </div>
        <button className="btn btn-primary" onClick={handleOptimize} disabled={loading}>
          {loading ? <LoaderCircle size={17} className="spin" /> : <Sliders size={17} />}
          {loading ? 'Optimizing…' : 'Optimize Budget'}
        </button>
      </div>

      <div className="card">
        <div className="card-title">
          <Sliders size={20} color="#10b981" />
          <span>Multiple-Choice Budget Allocation</span>
        </div>
        <p style={{ color: '#94a3b8', fontSize: '0.92rem', lineHeight: '1.6', marginBottom: '20px' }}>
          Fit purchase-response models for each configured discount arm, then use a mixed-integer
          optimizer to select at most one promotion per customer while keeping expected discount
          spend within the saved campaign budget.
        </p>

        {error && (
          <div className="info-box" role="alert" style={{
            borderColor: '#f43f5e',
            background: 'rgba(244,63,94,0.1)',
            color: '#fda4af',
            marginBottom: '16px',
          }}>
            {error}
          </div>
        )}

        {result && (
          <>
            <div className="kpi-grid">
              <div className="kpi-card">
                <span className="kpi-label">Recommended Customers</span>
                <span className="kpi-value">{result.recommended_customers.toLocaleString()}</span>
                <span className="kpi-subtext">of {result.observations.toLocaleString()} records</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Expected Budget Used</span>
                <span className="kpi-value" style={{ fontSize: '1.3rem' }}>
                  {formatCurrency(result.expected_spend)}
                </span>
                <span className="kpi-subtext">of {formatCurrency(result.budget)} limit</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Estimated Incremental Profit</span>
                <span className="kpi-value" style={{ fontSize: '1.3rem', color: '#34d399' }}>
                  {formatCurrency(result.expected_incremental_profit)}
                </span>
                <span className="kpi-subtext">after expected promotional cost</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Estimated Incremental Purchases</span>
                <span className="kpi-value" style={{ fontSize: '1.3rem', color: '#38bdf8' }}>
                  {result.expected_incremental_purchases}
                </span>
                <span className="kpi-subtext">response-model estimate</span>
              </div>
            </div>

            <div className="prescription-toolbar">
              <p className="prescription-note">{result.estimation_note}</p>
              <button
                className="btn btn-secondary"
                onClick={() => downloadRecommendations(result.recommendations)}
                disabled={!result.recommendations.length}
              >
                <Download size={16} />
                Export plan
              </button>
            </div>

            <div className="prescription-table-wrap">
              <table className="prescription-table">
                <thead>
                  <tr>
                    <th>Dataset row</th>
                    <th>Recommended discount</th>
                    <th>Purchase probability</th>
                    <th>Expected spend</th>
                    <th>Expected incremental profit</th>
                  </tr>
                </thead>
                <tbody>
                  {result.recommendations.slice(0, 25).map((row) => (
                    <tr key={row.row_index}>
                      <td>{row.row_index + 1}</td>
                      <td>{formatCurrency(row.discount)}</td>
                      <td>
                        {(row.predicted_purchase_without_discount * 100).toFixed(1)}% →{' '}
                        {(row.predicted_purchase_with_discount * 100).toFixed(1)}%
                      </td>
                      <td>{formatCurrency(row.expected_spend)}</td>
                      <td>{formatCurrency(row.expected_incremental_profit)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {result.recommendations.length === 0 && (
                <p className="prescription-empty">No positive-profit allocations fit the current budget.</p>
              )}
              {result.recommendations.length > 25 && (
                <p className="prescription-empty">
                  Showing 25 of {result.recommendations.length.toLocaleString()} recommendations. Export
                  the full plan to see every row.
                </p>
              )}
            </div>
          </>
        )}

        {!result && !loading && !error && (
          <>
            <div className="kpi-grid">
              <div className="kpi-card">
                <span className="kpi-label">Optimization</span>
                <span className="kpi-value" style={{ fontSize: '1.3rem' }}>SciPy MILP</span>
                <span className="kpi-subtext">Binary customer/discount decisions</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Budget constraint</span>
                <span className="kpi-value" style={{ fontSize: '1.3rem', color: '#f59e0b' }}>
                  <DollarSign size={18} style={{ verticalAlign: 'middle' }} /> Hard cap
                </span>
                <span className="kpi-subtext">Expected spend cannot exceed budget</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Allocation objective</span>
                <span className="kpi-value" style={{ fontSize: '1.3rem', color: '#6366f1' }}>
                  <TrendingUp size={18} style={{ verticalAlign: 'middle' }} /> Profit
                </span>
                <span className="kpi-subtext">Positive modeled incremental value</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Customer output</span>
                <span className="kpi-value" style={{ fontSize: '1.3rem', color: '#34d399' }}>
                  <Target size={18} style={{ verticalAlign: 'middle' }} /> Exportable
                </span>
                <span className="kpi-subtext">Download a full targeting CSV</span>
              </div>
            </div>
            <div className="info-box" style={{ marginTop: '16px' }}>
              Save your column mapping, discount options, margin, and budget in Data &amp; Budget,
              then run optimization. Estimates use observed treatment-arm response models and are
              not validated causal uplift estimates.
            </div>
          </>
        )}
      </div>
    </div>
  );
}
