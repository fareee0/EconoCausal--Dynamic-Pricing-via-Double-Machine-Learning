import React, { useMemo, useState, useRef } from 'react';
import { UploadCloud, CheckCircle, SlidersHorizontal, Table, Search, ChevronLeft, ChevronRight, Download, ArrowUpDown } from 'lucide-react';
import { exportDataset, uploadDataset, updateConfig } from '../api';

const EMPTY_ARRAY = [];

export default function DataBudgetView({ dataSummary, onDataRefresh, onNavigateAudit }) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [savingConfig, setSavingConfig] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [successMsg, setSuccessMsg] = useState('');
  const [uploadError, setUploadError] = useState('');
  const [previewSearch, setPreviewSearch] = useState('');
  const [sortColumn, setSortColumn] = useState('');
  const [sortDirection, setSortDirection] = useState('asc');
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const fileInputRef = useRef(null);

  // Form states initialized from dataSummary
  const [treatment, setTreatment] = useState(dataSummary?.column_mapping?.treatment || 'discount_amount');
  const [outcome, setOutcome] = useState(dataSummary?.column_mapping?.outcome || 'purchased');
  const [revenue, setRevenue] = useState(dataSummary?.column_mapping?.revenue || '');
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
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setUploadError('Choose a file with the .csv extension.');
      setSuccessMsg('');
      return;
    }
    if (file.size === 0) {
      setUploadError('The selected CSV is empty. Choose a file that contains data.');
      setSuccessMsg('');
      return;
    }

    try {
      setUploading(true);
      setSuccessMsg('');
      setUploadError('');
      const updatedSummary = await uploadDataset(file);
      onDataRefresh(updatedSummary);
      setTreatment(updatedSummary.column_mapping.treatment);
      setOutcome(updatedSummary.column_mapping.outcome);
      setRevenue(updatedSummary.column_mapping.revenue || '');
      setConfounders(updatedSummary.column_mapping.confounders);
      setPreviewSearch('');
      setSortColumn('');
      setSortDirection('asc');
      setCurrentPage(1);
      setSuccessMsg(`Successfully uploaded ${file.name} (${updatedSummary.row_count.toLocaleString()} rows)`);
      setTimeout(() => setSuccessMsg(''), 4000);
    } catch (err) {
      setUploadError(err?.response?.data?.detail || err.message || 'The CSV could not be uploaded.');
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

  const handleExportDataset = async () => {
    try {
      setExporting(true);
      const csv = await exportDataset();
      const url = URL.createObjectURL(csv);
      const link = document.createElement('a');
      const filename = (dataSummary?.filename || 'campaign.csv').replace(/\.csv$/i, '');
      link.href = url;
      link.download = `${filename}-export.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (err) {
      alert(`Export error: ${err?.response?.data?.detail || err.message}`);
    } finally {
      setExporting(false);
    }
  };

  const handleSort = (column) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
    setCurrentPage(1);
  };

  const toggleConfounder = (col) => {
    if (confounders.includes(col)) {
      setConfounders(confounders.filter((c) => c !== col));
    } else {
      setConfounders([...confounders, col]);
    }
  };

  const columns = dataSummary?.columns || EMPTY_ARRAY;
  const preview = dataSummary?.preview || EMPTY_ARRAY;
  const campaignMappingReady = Boolean(
    dataSummary?.column_mapping?.treatment
    && dataSummary?.column_mapping?.outcome
    && columns.includes(dataSummary.column_mapping.treatment)
    && columns.includes(dataSummary.column_mapping.outcome)
  );
  const filteredPreview = useMemo(() => {
    const query = previewSearch.trim().toLocaleLowerCase();
    const filtered = query
      ? preview.filter((row) => columns.some((column) =>
        String(row[column] ?? '').toLocaleLowerCase().includes(query)))
      : preview;

    if (!sortColumn) return filtered;

    return [...filtered].sort((a, b) => {
      const left = a[sortColumn];
      const right = b[sortColumn];
      const leftNumber = Number(left);
      const rightNumber = Number(right);
      const bothNumeric = left !== '' && right !== '' && Number.isFinite(leftNumber) && Number.isFinite(rightNumber);
      const comparison = bothNumeric
        ? leftNumber - rightNumber
        : String(left ?? '').localeCompare(String(right ?? ''), undefined, { numeric: true, sensitivity: 'base' });
      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [columns, preview, previewSearch, sortColumn, sortDirection]);
  const pageCount = Math.max(1, Math.ceil(filteredPreview.length / pageSize));
  const pageRows = filteredPreview.slice((currentPage - 1) * pageSize, currentPage * pageSize);

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
        <div className="info-box" style={{ borderColor: '#b9e1cd', background: 'var(--accent-emerald-light)', color: '#14734f' }}>
          <CheckCircle size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
          {successMsg}
        </div>
      )}
      {uploadError && (
        <div className="info-box" role="alert" style={{ borderColor: '#efb7b7', background: '#fff1f1', color: '#9b2525' }}>
          {uploadError}
        </div>
      )}

      {/* KPI Overview */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <span className="kpi-label">Active Dataset</span>
          <span className="kpi-value" style={{ fontSize: '1.25rem' }}>{dataSummary?.filename || 'retail_campaign.csv'}</span>
          <span className="kpi-subtext">Current uploaded customer dataset</span>
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
        aria-busy={uploading}
      >
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept=".csv"
          onChange={(e) => {
            if (e.target.files?.[0]) handleFileUpload(e.target.files[0]);
            e.target.value = '';
          }}
        />
        <UploadCloud className="dropzone-icon" size={48} />
        <h3 style={{ marginBottom: '6px' }}>
          {uploading ? 'Processing Dataset...' : 'Drag & drop campaign CSV, or browse'}
        </h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Supports transaction and customer features. Synthetic 20k retail benchmark loaded by default.
        </p>
      </div>

      {/* Column Mapping & Budget Form */}
      <div className="card">
        <div className="card-title">
          <SlidersHorizontal size={20} color="var(--primary)" />
          <span>Causal Variable Mapping & Economic Parameters</span>
        </div>

        {!campaignMappingReady && (
          <div className="info-box" role="status">
            This file does not appear to contain mapped campaign treatment and purchase-outcome columns.
            Churn prediction can still run from a churn label; causal audit, uplift, and promotion
            recommendations need valid campaign treatment and binary outcome data.
          </div>
        )}

        <form onSubmit={handleSaveConfiguration}>
          <div className="form-grid" style={{ marginBottom: '24px' }}>
            <div className="form-group">
              <label className="form-label">Treatment Column (T, campaign data only)</label>
              <select
                className="form-select"
                value={treatment}
                onChange={(e) => setTreatment(e.target.value)}
              >
                <option value="">Select a treatment column</option>
                {columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
              <span style={{ fontSize: '0.76rem', color: 'var(--text-dim)' }}>
                Optional for churn-only datasets; required for causal campaign analysis.
              </span>
            </div>

            <div className="form-group">
              <label className="form-label">Purchase / Conversion Outcome (Y)</label>
              <select
                className="form-select"
                value={outcome}
                onChange={(e) => setOutcome(e.target.value)}
              >
                <option value="">Select an outcome column</option>
                {columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
              <span style={{ fontSize: '0.76rem', color: 'var(--text-dim)' }}>
                Optional for churn-only datasets; use a binary 0/1 purchase or conversion field for campaign analysis.
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
              <span style={{ fontSize: '0.76rem', color: 'var(--text-dim)' }}>
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
              <span style={{ fontSize: '0.76rem', color: 'var(--text-dim)' }}>
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
              <span style={{ fontSize: '0.76rem', color: 'var(--text-dim)' }}>
                Comma-separated active promotion arms
              </span>
            </div>
          </div>

          <div className="form-group" style={{ marginBottom: '24px' }}>
            <label className="form-label">
              Observed Confounders for Backdoor Adjustment ({confounders.length} selected):
            </label>
            <div className="confounder-list">
              {columns
                .filter((col) => col !== treatment && col !== outcome && col !== revenue && !['customer_id', 'id', 'row_id'].includes(col.toLowerCase()))
                .map((col) => {
                  const isSelected = confounders.includes(col);
                  return (
                    <button
                      type="button"
                      key={col}
                      onClick={() => toggleConfounder(col)}
                      className={`confounder-chip ${isSelected ? 'selected' : ''}`}
                      aria-pressed={isSelected}
                    >
                      <span aria-hidden="true">{isSelected ? '✓' : '+'}</span>
                      {col}
                    </button>
                  );
                })}
            </div>
          </div>

          <div className="page-actions">
            <button type="submit" className="btn btn-primary" disabled={savingConfig}>
              {savingConfig ? 'Saving...' : 'Save Configuration & Update DAG'}
            </button>
          </div>
        </form>
      </div>

      {/* Dataset Preview Table */}
      <div className="card">
        <div className="preview-heading">
          <div className="card-title" style={{ marginBottom: 0 }}>
            <Table size={20} color="var(--accent-cyan)" />
            <span>Dataset Preview (First 50 Rows)</span>
          </div>
          <button className="btn btn-secondary" onClick={handleExportDataset} disabled={exporting || !dataSummary?.row_count}>
            <Download size={16} />
            {exporting ? 'Exporting...' : 'Export full CSV'}
          </button>
        </div>
        <div className="preview-toolbar">
          <label className="preview-search">
            <Search size={16} aria-hidden="true" />
            <input
              className="form-input"
              type="search"
              value={previewSearch}
              onChange={(event) => {
                setPreviewSearch(event.target.value);
                setCurrentPage(1);
              }}
              placeholder="Search preview rows..."
              aria-label="Search dataset preview"
            />
          </label>
          <span className="preview-count">
            {filteredPreview.length} of {preview.length} preview rows
          </span>
        </div>
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                {columns.map((col) => (
                  <th key={col} aria-sort={sortColumn === col ? (sortDirection === 'asc' ? 'ascending' : 'descending') : 'none'}>
                    <button className="table-sort-button" onClick={() => handleSort(col)}>
                      {col}
                      {sortColumn === col
                        ? <span aria-hidden="true">{sortDirection === 'asc' ? ' ↑' : ' ↓'}</span>
                        : <ArrowUpDown size={12} aria-hidden="true" />}
                    </button>
                    {col === treatment && <span className="badge badge-warning" style={{ marginLeft: '4px' }}>T</span>}
                    {col === outcome && <span className="badge badge-success" style={{ marginLeft: '4px' }}>Y</span>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pageRows.map((row, idx) => (
                <tr key={row.customer_id ?? row.id ?? (currentPage - 1) * pageSize + idx}>
                  {columns.map((col) => (
                    <td key={col}>{row[col] !== undefined ? String(row[col]) : ''}</td>
                  ))}
                </tr>
              ))}
              {pageRows.length === 0 && (
                <tr>
                  <td colSpan={Math.max(columns.length, 1)} className="preview-empty">No preview rows match that search.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="preview-pagination">
          <label className="preview-page-size">
            Rows per page
            <select
              className="form-select"
              value={pageSize}
              onChange={(event) => {
                setPageSize(Number(event.target.value));
                setCurrentPage(1);
              }}
              aria-label="Rows per page"
            >
              {[10, 25, 50].map((size) => <option key={size} value={size}>{size}</option>)}
            </select>
          </label>
          <span className="preview-count">Page {Math.min(currentPage, pageCount)} of {pageCount}</span>
          <div className="preview-page-buttons">
            <button
              className="btn btn-secondary"
              onClick={() => setCurrentPage((page) => Math.max(1, page - 1))}
              disabled={currentPage <= 1}
              aria-label="Previous page"
            >
              <ChevronLeft size={16} /> Previous
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => setCurrentPage((page) => Math.min(pageCount, page + 1))}
              disabled={currentPage >= pageCount}
              aria-label="Next page"
            >
              Next <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
