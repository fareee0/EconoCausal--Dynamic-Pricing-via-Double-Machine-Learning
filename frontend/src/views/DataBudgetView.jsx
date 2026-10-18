import React, { useState, useRef } from 'react';
import { UploadCloud, CheckCircle, RefreshCw, DollarSign, Database, SlidersHorizontal, Table } from 'lucide-react';
import { uploadDataset, updateConfig } from '../api';

export default function DataBudgetView({ dataSummary, onDataRefresh, onNavigateAudit }) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [savingConfig, setSavingConfig] = useState(false);
  const [successMsg, setSuccessMsg] = useState('');
  const fileInputRef = useRef(null);

  // Form states initialized from dataSummary
  const [treatment, setTreatment] = useState(dataSummary?.column_mapping?.treatment || 'discount_amount');
  const [outcome, setOutcome] = useState(dataSummary?.column_mapping?.outcome || 'purchased');
  const [revenue, setRevenue] = useState(dataSummary?.column_mapping?.revenue || 'revenue');
  const [confounders, setConfounders] = useState(dataSummary?.column_mapping?.confounders || []);

  const [budget, setBudget] = useState(dataSummary?.budget_config?.total_budget || 5000);
  const [margin, setMargin] = useState(dataSummary?.budget_config?.margin_per_purchase || 35.0);
  const [discountOptions, setDiscountOptions] = useState(
    dataSummary?.budget_config?.discount_options ? dataSummary.budget_config.discount_options.join(', ') : '10, 20'
  );

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileUpload = async (file) => {
    try {
      setUploading(true);
      setSuccessMsg('');
      const updatedSummary = await uploadDataset(file);
      onDataRefresh(updatedSummary);
      setSuccessMsg(`Successfully uploaded ${file.name} (${updatedSummary.row_count.toLocaleString()} rows)`);
      setTimeout(() => setSuccessMsg(''), 4000);
    } catch (err) {
      alert(`Upload error: ${err?.response?.data?.detail || err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const handleSaveConfiguration = async (e) => {
    e.preventDefault();
    try {
      setSavingConfig(true);
      const parsedDiscounts = discountOptions
        .split(',')
        .map((x) => parseFloat(x.trim()))
        .filter((x) => !isNaN(x) && x > 0);

      const mapping = {
        treatment,
        outcome,
        revenue,
        confounders,
      };
      const bConfig = {
        total_budget: parseFloat(budget),
        margin_per_purchase: parseFloat(margin),
        discount_options: parsedDiscounts,
      };

      await updateConfig(mapping, bConfig);
      setSuccessMsg('Configuration updated! Causal DAG and audit models ready.');
      setTimeout(() => setSuccessMsg(''), 4000);
    } catch (err) {
      alert(`Config update error: ${err?.response?.data?.detail || err.message}`);
    } finally {
      setSavingConfig(false);
    }
  };

  const toggleConfounder = (col) => {
    if (confounders.includes(col)) {
      setConfounders(confounders.filter((c) => c !== col));
    } else {
      setConfounders([...confounders, col]);
    }
  };

  const columns = dataSummary?.columns || [];
  const preview = dataSummary?.preview || [];

  return (
    <div>
      <div className="view-header">
        <div>
          <h2>Data & Budget Configuration</h2>
          <div className="view-subtitle">
            Upload campaign transaction data, map causal variables, and set promotional budget constraints
          </div>
        </div>
        <button className="btn btn-primary" onClick={onNavigateAudit}>
          Proceed to Causal Audit &rarr;
        </button>
      </div>

      {successMsg && (
        <div className="info-box" style={{ borderColor: '#10b981', background: 'rgba(16,185,129,0.1)', color: '#34d399' }}>
          <CheckCircle size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
          {successMsg}
        </div>
      )}

      {/* KPI Overview */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <span className="kpi-label">Active Dataset</span>
          <span className="kpi-value" style={{ fontSize: '1.25rem' }}>{dataSummary?.filename || 'retail_campaign.csv'}</span>
          <span className="kpi-subtext">Known structural ground truth available</span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Total Observations</span>
          <span className="kpi-value">{dataSummary?.row_count?.toLocaleString() || '20,000'}</span>
          <span className="kpi-subtext">Customer interaction records</span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Campaign Budget</span>
          <span className="kpi-value">${parseFloat(budget).toLocaleString()}</span>
          <span className="kpi-subtext">Allocated to promotional discounts</span>
        </div>
        <div className="kpi-card">
          <span className="kpi-label">Unit Margin</span>
          <span className="kpi-value">${parseFloat(margin).toFixed(2)}</span>
          <span className="kpi-subtext">Gross margin per incremental sale</span>
        </div>
      </div>

      {/* CSV Upload Dropzone */}
      <div
        className={`dropzone ${isDragging ? 'dragover' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept=".csv"
          onChange={(e) => {
            if (e.target.files?.[0]) handleFileUpload(e.target.files[0]);
          }}
        />
        <UploadCloud className="dropzone-icon" size={48} />
        <h3 style={{ marginBottom: '6px' }}>
          {uploading ? 'Processing Dataset...' : 'Drag & drop campaign CSV, or browse'}
        </h3>
        <p style={{ color: '#94a3b8', fontSize: '0.88rem' }}>
          Supports transaction and customer features. Synthetic 20k retail benchmark loaded by default.
        </p>
      </div>

      {/* Column Mapping & Budget Form */}
      <div className="card">
        <div className="card-title">
          <SlidersHorizontal size={20} color="#818cf8" />
          <span>Causal Variable Mapping & Economic Parameters</span>
        </div>

        <form onSubmit={handleSaveConfiguration}>
          <div className="form-grid" style={{ marginBottom: '24px' }}>
            <div className="form-group">
              <label className="form-label">Treatment Column (T)</label>
              <select
                className="form-select"
                value={treatment}
                onChange={(e) => setTreatment(e.target.value)}
              >
                {columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
              <span style={{ fontSize: '0.76rem', color: '#64748b' }}>
                Discrete discount levels (e.g. 0, 10, 20)
              </span>
            </div>

            <div className="form-group">
              <label className="form-label">Outcome Column (Y)</label>
              <select
                className="form-select"
                value={outcome}
                onChange={(e) => setOutcome(e.target.value)}
              >
                {columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
              <span style={{ fontSize: '0.76rem', color: '#64748b' }}>
                Primary binary purchase decision (0 or 1)
              </span>
            </div>

            <div className="form-group">
              <label className="form-label">Total Optimization Budget ($)</label>
              <input
                type="number"
                step="100"
                className="form-input"
                value={budget}
                onChange={(e) => setBudget(e.target.value)}
                required
              />
              <span style={{ fontSize: '0.76rem', color: '#64748b' }}>
                Hard spend ceiling for MILP knapsack
              </span>
            </div>

            <div className="form-group">
              <label className="form-label">Gross Margin per Purchase ($)</label>
              <input
                type="number"
                step="0.5"
                className="form-input"
                value={margin}
                onChange={(e) => setMargin(e.target.value)}
                required
              />
              <span style={{ fontSize: '0.76rem', color: '#64748b' }}>
                Contribution margin before promotional cost
              </span>
            </div>

            <div className="form-group">
              <label className="form-label">Discount Options ($)</label>
              <input
                type="text"
                className="form-input"
                value={discountOptions}
                onChange={(e) => setDiscountOptions(e.target.value)}
                placeholder="10, 20"
                required
              />
              <span style={{ fontSize: '0.76rem', color: '#64748b' }}>
                Comma-separated active promotion arms
              </span>
            </div>
          </div>

          <div className="form-group" style={{ marginBottom: '24px' }}>
            <label className="form-label">
              Observed Confounders for Backdoor Adjustment ({confounders.length} selected):
            </label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '6px' }}>
              {columns
                .filter((col) => col !== treatment && col !== outcome && col !== 'customer_id')
                .map((col) => {
                  const isSelected = confounders.includes(col);
                  return (
                    <button
                      type="button"
                      key={col}
                      onClick={() => toggleConfounder(col)}
                      style={{
                        background: isSelected ? 'rgba(99, 102, 241, 0.25)' : 'rgba(255, 255, 255, 0.04)',
                        border: isSelected ? '1px solid #6366f1' : '1px solid rgba(255, 255, 255, 0.1)',
                        color: isSelected ? '#ffffff' : '#94a3b8',
                        padding: '6px 12px',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontSize: '0.84rem',
                        transition: 'all 0.15s ease',
                      }}
                    >
                      {isSelected ? '✓ ' : '+ '}
                      {col}
                    </button>
                  );
                })}
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
            <button type="submit" className="btn btn-primary" disabled={savingConfig}>
              {savingConfig ? 'Saving...' : 'Save Configuration & Update DAG'}
            </button>
          </div>
        </form>
      </div>

      {/* Dataset Preview Table */}
      <div className="card">
        <div className="card-title">
          <Table size={20} color="#38bdf8" />
          <span>Dataset Preview (First 50 Rows)</span>
        </div>
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                {columns.map((col) => (
                  <th key={col}>
                    {col}
                    {col === treatment && <span className="badge badge-warning" style={{ marginLeft: '4px' }}>T</span>}
                    {col === outcome && <span className="badge badge-success" style={{ marginLeft: '4px' }}>Y</span>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.map((row, idx) => (
                <tr key={idx}>
                  {columns.map((col) => (
                    <td key={col}>{row[col] !== undefined ? String(row[col]) : ''}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
