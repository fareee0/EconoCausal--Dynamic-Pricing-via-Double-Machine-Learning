import React from 'react';
import {
  ArrowRight,
  Check,
  CircleHelp,
  Database,
  GitGraph,
  SlidersHorizontal,
  TrendingUp,
  UserRoundX,
} from 'lucide-react';
import './CampaignTimelineView.css';

const campaignStages = [
  {
    id: 'data_budget',
    title: 'Load campaign data',
    description: 'Review the customer records and confirm the dataset is ready for analysis.',
    detail: (summary) => summary?.filename || 'No campaign dataset is available yet.',
    icon: Database,
    requirement: 'dataset',
  },
  {
    id: 'data_budget',
    title: 'Map campaign variables',
    description: 'Choose the treatment and purchase outcome, then set the promotion budget.',
    detail: (summary) => {
      const treatment = summary?.column_mapping?.treatment;
      const outcome = summary?.column_mapping?.outcome;
      return treatment && outcome
        ? `Treatment: ${treatment} · Outcome: ${outcome}`
        : 'Select valid treatment and purchase-outcome columns in Data & Budget.';
    },
    icon: SlidersHorizontal,
    requirement: 'mapping',
  },
  {
    id: 'causal_audit',
    title: 'Audit causal assumptions',
    description: 'Check the adjustment graph, treatment overlap, and covariate balance.',
    detail: () => 'Review propensity diagnostics before interpreting treatment effects.',
    icon: GitGraph,
    requirement: 'mapping',
  },
  {
    id: 'uplift',
    title: 'Estimate customer uplift',
    description: 'Fit the Double Machine Learning workflow and inspect heterogeneous effects.',
    detail: () => 'Use estimated uplift to distinguish persuadable customers from others.',
    icon: TrendingUp,
    requirement: 'mapping',
  },
  {
    id: 'prescription',
    title: 'Build a budget-aware offer plan',
    description: 'Allocate discounts under the configured promotional budget.',
    detail: () => 'Recommendations are observational benchmarks, not validated causal effects.',
    icon: SlidersHorizontal,
    requirement: 'mapping',
  },
  {
    id: 'churn',
    title: 'Review retention risk',
    description: 'Score customers for churn risk and explore practical retention actions.',
    detail: () => 'Churn scoring is a prioritization signal; validate actions with experiments.',
    icon: UserRoundX,
    requirement: 'dataset',
  },
];

function getStageStatus(stage, hasDataset, hasMapping) {
  if (stage.requirement === 'dataset' && hasDataset) return 'available';
  if (stage.requirement === 'mapping' && hasMapping) return 'available';
  return 'needs-setup';
}

export default function CampaignTimelineView({ dataSummary, onNavigate }) {
  const columns = dataSummary?.columns || [];
  const hasDataset = Boolean(dataSummary?.row_count && columns.length);
  const mapping = dataSummary?.column_mapping;
  const hasMapping = Boolean(
    hasDataset
    && mapping?.treatment
    && mapping?.outcome
    && columns.includes(mapping.treatment)
    && columns.includes(mapping.outcome)
  );
  const availableCount = campaignStages.filter(
    (stage) => getStageStatus(stage, hasDataset, hasMapping) === 'available',
  ).length;

  return (
    <section className="campaign-timeline">
      <header className="view-header timeline-header">
        <div>
          <p className="timeline-eyebrow">YOUR ANALYSIS WORKFLOW</p>
          <h2>Campaign Timeline</h2>
          <p className="view-subtitle">
            Follow the campaign from raw data to retention decisions. Availability updates
            from your active dataset and saved variable mapping.
          </p>
        </div>
        <div className="timeline-progress" aria-label={`${availableCount} of ${campaignStages.length} stages available`}>
          <span className="timeline-progress-value">{availableCount}<small>/{campaignStages.length}</small></span>
          <span className="timeline-progress-label">stages available</span>
        </div>
      </header>

      <div className="timeline-summary">
        <div className="timeline-summary-icon"><Database size={19} /></div>
        <div className="timeline-summary-copy">
          <strong>{hasDataset ? dataSummary.filename : 'Waiting for a campaign dataset'}</strong>
          <span>
            {hasDataset
              ? `${Number(dataSummary.row_count).toLocaleString()} customer records · ${columns.length} columns`
              : 'Upload a CSV in Data & Budget to begin this workflow.'}
          </span>
        </div>
        <button className="btn btn-secondary" onClick={() => onNavigate('data_budget')}>
          {hasDataset ? 'Review data' : 'Set up data'} <ArrowRight size={15} />
        </button>
      </div>

      <ol className="timeline-list">
        {campaignStages.map((stage, index) => {
          const Icon = stage.icon;
          const status = getStageStatus(stage, hasDataset, hasMapping);
          const isAvailable = status === 'available';

          return (
            <li className={`timeline-item ${status}`} key={`${stage.title}-${index}`}>
              <div className="timeline-rail" aria-hidden="true">
                <span className="timeline-marker">
                  {isAvailable ? <Check size={16} /> : <Icon size={16} />}
                </span>
                {index < campaignStages.length - 1 && <span className="timeline-connector" />}
              </div>
              <article className="timeline-stage-card">
                <div className="timeline-stage-content">
                  <div className="timeline-stage-heading">
                    <span className="timeline-stage-icon"><Icon size={17} /></span>
                    <div>
                      <span className="timeline-step-label">STAGE {String(index + 1).padStart(2, '0')}</span>
                      <h3>{stage.title}</h3>
                    </div>
                  </div>
                  <p className="timeline-description">{stage.description}</p>
                  <p className="timeline-detail">{stage.detail(dataSummary)}</p>
                </div>
                <div className="timeline-stage-action">
                  <span className={`timeline-status ${status}`}>
                    {isAvailable ? <Check size={13} /> : <CircleHelp size={13} />}
                    {isAvailable ? 'Available' : 'Needs setup'}
                  </span>
                  <button
                    className="timeline-open-button"
                    onClick={() => onNavigate(stage.id)}
                    aria-label={`Open ${stage.title}`}
                  >
                    Open <ArrowRight size={15} />
                  </button>
                </div>
              </article>
            </li>
          );
        })}
      </ol>

      <footer className="timeline-note">
        <CircleHelp size={16} />
        <span>
          “Available” means the current dataset meets the stage’s basic input requirements;
          it does not mean that analysis has already been run.
        </span>
      </footer>
    </section>
  );
}
