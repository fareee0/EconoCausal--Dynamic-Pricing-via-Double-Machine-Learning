import React from 'react';
import { Database, GitGraph, TrendingUp, Sliders, CheckCircle2, ShieldCheck } from 'lucide-react';

export default function Sidebar({ activeView, setActiveView, isConnected = true }) {
  const navItems = [
    { id: 'data_budget', label: 'Data & Budget', icon: Database, badge: null },
    { id: 'causal_audit', label: 'Causal Audit', icon: GitGraph, badge: 'Phase 1' },
    { id: 'uplift', label: 'Uplift & DML', icon: TrendingUp, badge: 'Phase 2' },
    { id: 'prescription', label: 'Prescription', icon: Sliders, badge: 'Phase 3' },
  ];

  return (
    <aside className="sidebar">
      <div className="brand-header">
        <div className="brand-icon">⚡</div>
        <div>
          <div className="brand-title">EconoCausal</div>
          <div style={{ display: 'flex', alignItems: 'center', marginTop: '2px' }}>
            <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Double ML Optimizer</span>
            <span className="brand-badge">v1.0</span>
          </div>
        </div>
      </div>

      <ul className="nav-menu">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeView === item.id;
          return (
            <li key={item.id}>
              <button
                className={`nav-item-btn ${isActive ? 'active' : ''}`}
                onClick={() => setActiveView(item.id)}
              >
                <span className="nav-icon"><Icon size={18} /></span>
                <span style={{ flex: 1 }}>{item.label}</span>
                {item.badge && (
                  <span style={{
                    fontSize: '0.65rem',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    background: isActive ? 'rgba(99,102,241,0.3)' : 'rgba(255,255,255,0.06)',
                    color: isActive ? '#c7d2fe' : '#64748b'
                  }}>
                    {item.badge}
                  </span>
                )}
              </button>
            </li>
          );
        })}
      </ul>

      <div className="sidebar-footer">
        <div className="system-status">
          <div className="status-dot" style={{ background: isConnected ? '#10b981' : '#f43f5e' }}></div>
          <span>{isConnected ? 'FastAPI Backend Online' : 'Backend Connecting...'}</span>
        </div>
        <div style={{ fontSize: '0.74rem', color: '#64748b', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <ShieldCheck size={14} color="#6366f1" />
          <span>Unconfoundedness Verified</span>
        </div>
      </div>
    </aside>
  );
}
