import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import DataBudgetView from './views/DataBudgetView';
import CausalAuditView from './views/CausalAuditView';
import UpliftView from './views/UpliftView';
import PrescriptionView from './views/PrescriptionView';
import { getDataSummary } from './api';

export default function App() {
  const [activeView, setActiveView] = useState('data_budget');
  const [dataSummary, setDataSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isConnected, setIsConnected] = useState(true);

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
        {loading && !dataSummary ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '60vh', gap: '16px' }}>
            <div className="spinner" style={{ width: '40px', height: '40px' }}></div>
            <div style={{ color: '#94a3b8' }}>Connecting to EconoCausal Engine...</div>
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
