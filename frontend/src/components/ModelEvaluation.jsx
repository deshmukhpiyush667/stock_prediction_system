import React from 'react';

export default function ModelEvaluation({ metricsData, selectedModel }) {
  if (!metricsData || !metricsData.models) {
    return null;
  }

  const { baseline, random_forest, xgboost } = metricsData.models;
  const activeModelKey = selectedModel.includes("forest") || selectedModel.includes("rf") ? "random_forest" : "xgboost";
  const activeMetrics = metricsData.models[activeModelKey];

  return (
    <div className="eval-card">
      <div className="eval-header">
        <div>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 600 }}>Model Benchmark & Out-of-Sample Performance</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: 4 }}>
            Evaluated on {metricsData.meta?.test_samples || 524} unseen forward trading days ({metricsData.meta?.test_start_date} to {metricsData.meta?.test_end_date})
          </p>
        </div>

        <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
          Active Model: <span style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>{activeMetrics?.model_name}</span>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="eval-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Test Accuracy</th>
              <th>Precision (UP)</th>
              <th>Recall (UP)</th>
              <th>F1-Score</th>
              <th>Strategy Type</th>
            </tr>
          </thead>
          <tbody>
            {baseline && (
              <tr>
                <td style={{ fontWeight: 500 }}>{baseline.model_name}</td>
                <td className="mono">{(baseline.accuracy * 100).toFixed(2)}%</td>
                <td className="mono">{baseline.precision.toFixed(4)}</td>
                <td className="mono">{baseline.recall.toFixed(4)}</td>
                <td className="mono">{baseline.f1.toFixed(4)}</td>
                <td style={{ color: 'var(--text-muted)' }}>Naive Majority Benchmark</td>
              </tr>
            )}

            {random_forest && (
              <tr style={{ background: activeModelKey === 'random_forest' ? 'rgba(59, 130, 246, 0.08)' : 'transparent' }}>
                <td style={{ fontWeight: 600, color: activeModelKey === 'random_forest' ? 'var(--accent-primary)' : 'inherit' }}>
                  {random_forest.model_name} {activeModelKey === 'random_forest' && '★'}
                </td>
                <td className="mono" style={{ fontWeight: 600, color: 'var(--color-up)' }}>
                  {(random_forest.accuracy * 100).toFixed(2)}%
                </td>
                <td className="mono">{random_forest.precision.toFixed(4)}</td>
                <td className="mono">{random_forest.recall.toFixed(4)}</td>
                <td className="mono">{random_forest.f1.toFixed(4)}</td>
                <td>Bagged Ensemble (Depth 6)</td>
              </tr>
            )}

            {xgboost && (
              <tr style={{ background: activeModelKey === 'xgboost' ? 'rgba(59, 130, 246, 0.08)' : 'transparent' }}>
                <td style={{ fontWeight: 600, color: activeModelKey === 'xgboost' ? 'var(--accent-primary)' : 'inherit' }}>
                  {xgboost.model_name} {activeModelKey === 'xgboost' && '★'}
                </td>
                <td className="mono" style={{ fontWeight: 600, color: 'var(--color-up)' }}>
                  {(xgboost.accuracy * 100).toFixed(2)}%
                </td>
                <td className="mono">{xgboost.precision.toFixed(4)}</td>
                <td className="mono">{xgboost.recall.toFixed(4)}</td>
                <td className="mono">{xgboost.f1.toFixed(4)}</td>
                <td>Gradient Boosted Trees</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="eval-note">
        <strong>Model Formulation & Objective:</strong> The model uses historical market features (intraday price range, daily return, volume surges, moving averages, and 14-day RSI) to classify whether the next trading day's closing price will be higher (UP) or lower (DOWN) than today's closing price.
      </div>
    </div>
  );
}
