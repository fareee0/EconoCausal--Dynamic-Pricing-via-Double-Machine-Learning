import React, { useEffect, useState } from 'react';
import Plot from '../components/ThemedPlot';
import { Activity, Layers, Target, TrendingUp } from 'lucide-react';
import { getUpliftSummary } from '../api';

const formatNumber = (value, digits = 3) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(digits);
};

export default function UpliftView() {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        setLoading(true);
        setError('');
        const data = await getUpliftSummary();
        setSummary(data);
      } catch (err) {
        setError(err?.response?.data?.detail || err.message || 'Unable to load uplift summary.');
      } finally {
        setLoading(false);
      }
    };

    fetchSummary();
  }, []);

  const metric = summary?.summary_metrics || {};
  const segmentRows = summary?.by_segment || [];
  const qiniData = summary?.qini_curve || [];
  const qiniTrace = qiniData.map((series) => ({
    x: series.curve.map((point) => Number(point.fraction_targeted)),
    y: series.curve.map((point) => Number(point.cumulative_true_uplift)),
    mode: 'lines+markers',
    name: `$${series.discount}`,
  }));

  return (
    <div>
      <div className="view-header">
        <div>
          <h2>Uplift Modeling & Double Machine Learning</h2>
          <div className="view-subtitle">
            Response-model uplift estimates with segment diagnostics and validation against ground truth
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          <TrendingUp size={20} color="var(--primary)" />
          <span>Phase 2 Uplift Summary</span>
        </div>

        {error && (
          <div className="info-box error-box" role="alert" style={{ marginBottom: '16px' }}>
            {error}
          </div>
        )}

        {loading ? (
          <div style={{ padding: '10px 0', color: 'var(--text-muted)' }}>Loading uplift diagnostics…</div>
        ) : (
          <>
            <div className="kpi-grid">
              <div className="kpi-card">
                <span className="kpi-label">Observations</span>
                <span className="kpi-value" style={{ fontSize: '1.3rem' }}>{summary?.observations ?? 0}</span>
                <span className="kpi-subtext">Customers scored</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Avg uplift @ $10</span>
                <span className="kpi-value" style={{ fontSize: '1.3rem', color: 'var(--accent-cyan)' }}>
                  {formatNumber(metric.estimated_avg_uplift_10)}
                </span>
                <span className="kpi-subtext">Probability lift</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Avg uplift @ $20</span>
                <span className="kpi-value" style={{ fontSize: '1.3rem', color: 'var(--accent-emerald)' }}>
                  {formatNumber(metric.estimated_avg_uplift_20)}
                </span>
                <span className="kpi-subtext">Probability lift</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Ground-truth check</span>
                <span className="kpi-value" style={{ fontSize: '1.1rem' }}>
                  {summary?.available_ground_truth ? 'Available' : 'Not available'}
                </span>
                <span className="kpi-subtext">{summary?.available_ground_truth ? 'Synthetic truth merged' : 'Observed-only mode'}</span>
              </div>
            </div>

            {summary?.available_ground_truth && (
              <div className="kpi-grid" style={{ marginTop: '16px' }}>
                <div className="kpi-card">
                  <span className="kpi-label">RMSE @ $10</span>
                  <span className="kpi-value" style={{ fontSize: '1.2rem' }}>{formatNumber(metric.rmse_10)}</span>
                  <span className="kpi-subtext">Estimated vs true lift</span>
                </div>
                <div className="kpi-card">
                  <span className="kpi-label">RMSE @ $20</span>
                  <span className="kpi-value" style={{ fontSize: '1.2rem' }}>{formatNumber(metric.rmse_20)}</span>
                  <span className="kpi-subtext">Estimated vs true lift</span>
                </div>
                <div className="kpi-card">
                  <span className="kpi-label">MAE @ $10</span>
                  <span className="kpi-value" style={{ fontSize: '1.2rem' }}>{formatNumber(metric.mae_10)}</span>
                  <span className="kpi-subtext">Mean absolute error</span>
                </div>
                <div className="kpi-card">
                  <span className="kpi-label">MAE @ $20</span>
                  <span className="kpi-value" style={{ fontSize: '1.2rem' }}>{formatNumber(metric.mae_20)}</span>
                  <span className="kpi-subtext">Mean absolute error</span>
                </div>
              </div>
            )}

            <div className="info-box" style={{ marginTop: '16px' }}>
              {summary?.estimation_note || 'Uplift diagnostics are ready.'}
            </div>

            {qiniTrace.length > 0 && (
              <div className="analysis-panel" style={{ marginTop: '20px' }}>
                <div className="analysis-panel-heading">
                  <div>
                    <h3>Qini-style uplift curve</h3>
                    <p>Sorted customers by model-estimated lift vs cumulative true uplift</p>
                  </div>
                </div>
                <div className="analysis-chart">
                  <Plot
                    data={qiniTrace}
                    layout={{
                      paper_bgcolor: 'transparent',
                      plot_bgcolor: 'transparent',
                      font: { color: '#526174', family: 'Inter, sans-serif', size: 12 },
                      margin: { l: 52, r: 16, t: 20, b: 48 },
                      height: 280,
                      autosize: true,
                      xaxis: { title: { text: 'Fraction targeted' }, range: [0, 1], zeroline: false },
                      yaxis: { title: { text: 'Cumulative true uplift' }, zeroline: false },
                      legend: { orientation: 'h', y: 1.12 },
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%' }}
                  />
                </div>
              </div>
            )}

            {segmentRows.length > 0 && (
              <div className="analysis-panel" style={{ marginTop: '20px' }}>
                <div className="analysis-panel-heading">
                  <div>
                    <h3>Segment uplift diagnostics</h3>
                    <p>Estimated vs true average uplift by latent customer type</p>
                  </div>
                </div>
                <div className="analysis-chart">
                  <Plot
                    data={[
                      {
                        type: 'bar',
                        x: segmentRows.map((row) => row.segment),
                        y: segmentRows.map((row) => Number(row.avg_estimated_uplift_10 ?? 0)),
                        name: 'Est. @ $10',
                        marker: { color: '#b6a2ff' },
                      },
                      {
                        type: 'bar',
                        x: segmentRows.map((row) => row.segment),
                        y: segmentRows.map((row) => Number(row.avg_estimated_uplift_20 ?? 0)),
                        name: 'Est. @ $20',
                        marker: { color: '#55d4c5' },
                      },
                    ]}
                    layout={{
                      barmode: 'group',
                      paper_bgcolor: 'transparent',
                      plot_bgcolor: 'transparent',
                      font: { color: '#526174', family: 'Inter, sans-serif', size: 12 },
                      margin: { l: 52, r: 16, t: 20, b: 72 },
                      height: 300,
                      autosize: true,
                      xaxis: { title: { text: 'Segment' }, type: 'category' },
                      yaxis: { title: { text: 'Avg uplift' }, zeroline: false },
                      legend: { orientation: 'h', y: 1.12 },
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%' }}
                  />
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
