import React from 'react';
import { Activity, Database, GitGraph, TrendingUp, SlidersHorizontal, ShieldCheck, UserRoundX } from 'lucide-react';

export default function Sidebar({ activeView, setActiveView, isConnected = true }) {
  const navItems = [
    { id: 'data_budget', label: 'Data & Budget', icon: Database, badge: '01' },
    { id: 'causal_audit', label: 'Causal Audit', icon: GitGraph, badge: '02' },
    { id: 'uplift', label: 'Uplift & DML', icon: TrendingUp, badge: '03' },
    { id: 'prescription', label: 'Prescription', icon: SlidersHorizontal, badge: '04' },
    { id: 'churn', label: 'Churn Prediction', icon: UserRoundX, badge: '05' },
  ];

  return (
    <aside className="sidebar">
      <div className="brand-header">
        <div className="brand-icon"><Activity size={21} strokeWidth={2.4} /></div>
        <div>
          <div className="brand-title">EconoCausal</div>
          <div className="brand-caption">CAUSAL PRICING STUDIO</div>
        </div>
      </div>

      <nav className="primary-nav" aria-label="Main navigation">
        <div className="nav-section-label">WORKSPACE</div>
        <ul className="nav-menu">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeView === item.id;
            return (
              <li key={item.id}>
                <button
                  className={`nav-item-btn ${isActive ? 'active' : ''}`}
                  onClick={() => setActiveView(item.id)}
                  aria-current={isActive ? 'page' : undefined}
                >
                  <span className="nav-icon"><Icon size={18} strokeWidth={1.9} /></span>
                  <span className="nav-item-label">{item.label}</span>
                  <span className="nav-item-number">{item.badge}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="sidebar-footer">
        <div className={`sidebar-system-status ${isConnected ? 'online' : 'offline'}`}>
          <span className="status-dot" aria-hidden="true"></span>
          <span>{isConnected ? 'Backend online' : 'Backend offline'}</span>
        </div>
        <div className="sidebar-assurance">
          <ShieldCheck size={15} />
          <span>Built for causal decisions</span>
        </div>
      </div>
    </aside>
  );
}
