import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import DataBudgetView from './views/DataBudgetView';
import CausalAuditView from './views/CausalAuditView';
import UpliftView from './views/UpliftView';
import PrescriptionView from './views/PrescriptionView';
import { getDataSummary } from './api';

const viewLabels = {
  data_budget: 'Data & Budget',
  causal_audit: 'Causal Audit',
  uplift: 'Uplift & DML',
  prescription: 'Prescription',
};

export default function App() {
  const [activeView, setActiveView] = useState('data_budget');
  const [dataSummary, setDataSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isConnected, setIsConnected] = useState(false);

  const fetchSummary = async () => {
    try {
      setLoading(true);
      const summary = await getDataSummary();
      setDataSummary(summary);
      setIsConnected(true);
    } catch (err) {
      console.error('Failed to connect to EconoCausal backend:', err);
      setIsConnected(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSummary();
  }, []);

  return (
    <div className="app-layout">
      <Sidebar
        activeView={activeView}
        setActiveView={setActiveView}
        isConnected={isConnected}
      />
      <main className="main-content">
        <div className="workspace-bar">
          <div className="workspace-context">
            <span className="workspace-eyebrow">CAMPAIGN INTELLIGENCE</span>
            <span className="workspace-divider" aria-hidden="true">/</span>
            <span className="workspace-current">{viewLabels[activeView]}</span>
          </div>
          <div className={`engine-status ${isConnected ? 'online' : 'offline'}`}>
            <span className="status-dot" aria-hidden="true"></span>
            {isConnected ? 'Engine connected' : 'Engine offline'}
          </div>
        </div>
        {loading && !dataSummary ? (
          <div className="loading-state">
            <div className="spinner"></div>
            <div className="loading-copy">
              <strong>Preparing your workspace</strong>
              <span>Connecting to the EconoCausal engine…</span>
            </div>
          </div>
        ) : (
          <>
            {activeView === 'data_budget' && (
              <DataBudgetView
                dataSummary={dataSummary}
                onDataRefresh={(newSummary) => setDataSummary(newSummary)}
                onNavigateAudit={() => setActiveView('causal_audit')}
              />
            )}
            {activeView === 'causal_audit' && (
              <CausalAuditView
                onNavigateUplift={() => setActiveView('uplift')}
              />
            )}
            {activeView === 'uplift' && <UpliftView />}
            {activeView === 'prescription' && <PrescriptionView />}
          </>
        )}
      </main>
    </div>
  );
}
