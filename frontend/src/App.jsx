import React, { useState, useEffect } from 'react';
import StockChart from './components/StockChart';
import ModelEvaluation from './components/ModelEvaluation';
import { TrendingUp, TrendingDown, Activity, AlertCircle, RefreshCw, Layers } from 'lucide-react';

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [stocksList, setStocksList] = useState([]);
  const [selectedStock, setSelectedStock] = useState("TCS");
  const [selectedModel, setSelectedModel] = useState("xgboost");
  
  const [stockDetails, setStockDetails] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [metrics, setMetrics] = useState(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [apiOnline, setApiOnline] = useState(false);

  // 1. Initial Load: Stocks list, health, and metrics
  useEffect(() => {
    async function initDashboard() {
      try {
        const [healthRes, stocksRes, metricsRes] = await Promise.all([
          fetch(`${API_BASE}/health`).catch(() => null),
          fetch(`${API_BASE}/stocks`).catch(() => null),
          fetch(`${API_BASE}/metrics`).catch(() => null)
        ]);

        if (healthRes && healthRes.ok) {
          setApiOnline(true);
        } else {
          setApiOnline(false);
        }

        if (stocksRes && stocksRes.ok) {
          const stocksData = await stocksRes.json();
          setStocksList(stocksData.stocks || ["TCS"]);
          if (stocksData.default_stock) setSelectedStock(stocksData.default_stock);
        }

        if (metricsRes && metricsRes.ok) {
          const metricsData = await metricsRes.json();
          setMetrics(metricsData);
        }
      } catch (err) {
        console.error("Initialization error:", err);
      }
    }
    initDashboard();
  }, []);

  // 2. Fetch stock details and prediction whenever selectedStock or selectedModel changes
  useEffect(() => {
    if (!selectedStock) return;
    loadStockData(selectedStock, selectedModel);
  }, [selectedStock, selectedModel]);

  async function loadStockData(ticker, model) {
    setLoading(true);
    setError(null);
    try {
      const [detailRes, predRes] = await Promise.all([
        fetch(`${API_BASE}/stock/${ticker}?days=120`),
        fetch(`${API_BASE}/predict/${ticker}?model=${model}`)
      ]);

      if (!detailRes.ok) {
        const errDetail = await detailRes.json().catch(() => ({}));
        throw new Error(errDetail.detail || `Failed to fetch data for ${ticker}`);
      }

      if (!predRes.ok) {
        const errPred = await predRes.json().catch(() => ({}));
        throw new Error(errPred.detail || `Failed to generate prediction for ${ticker}`);
      }

      const details = await detailRes.json();
      const pred = await predRes.json();

      setStockDetails(details);
      setPrediction(pred);
      setApiOnline(true);
    } catch (err) {
      console.error("Data load failed:", err);
      setError(err.message || "Could not connect to FastAPI backend at " + API_BASE);
      setApiOnline(false);
    } finally {
      setLoading(false);
    }
  }

  const isUp = prediction?.prediction === "UP";
  const prob = prediction ? prediction.probability : 0.5;

  return (
    <div className="dashboard-container">
      {/* Header */}
      <header className="dashboard-header">
        <div className="brand-section">
          <h1>
            <span>📈</span> Stock Prediction System
            <span className="brand-badge">AI / ML Quantitative System</span>
          </h1>
          <p>Next-day market direction classification trained on historical National Stock Exchange (NSE) data</p>
        </div>

        <div className="header-controls">
          {/* API Health indicator */}
          <div className="status-indicator">
            <span className={`status-dot ${apiOnline ? '' : 'error'}`}></span>
            <span style={{ fontSize: '0.82rem', color: apiOnline ? 'var(--text-primary)' : 'var(--color-down)' }}>
              {apiOnline ? 'FastAPI Online' : 'API Offline'}
            </span>
          </div>

          {/* Model Selector */}
          <div className="control-group">
            <label>Model Architecture</label>
            <select
              className="custom-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <option value="xgboost">XGBoost Classifier</option>
              <option value="random_forest">Random Forest</option>
            </select>
          </div>

          {/* Stock Selector */}
          <div className="control-group">
            <label>Stock Ticker</label>
            <select
              className="custom-select"
              value={selectedStock}
              onChange={(e) => setSelectedStock(e.target.value)}
            >
              {stocksList.map(ticker => (
                <option key={ticker} value={ticker}>
                  {ticker}
                </option>
              ))}
            </select>
          </div>
        </div>
      </header>

      {/* Error state */}
      {error && (
        <div className="error-box">
          <AlertCircle size={32} />
          <p>{error}</p>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Ensure your backend is running: <code>uvicorn backend.main:app --reload</code>
          </p>
          <button className="retry-btn" onClick={() => loadStockData(selectedStock, selectedModel)}>
            <RefreshCw size={14} style={{ display: 'inline', marginRight: 6 }} />
            Retry Connection
          </button>
        </div>
      )}

      {/* Loading state */}
      {loading && !error && (
        <div className="loading-box">
          <RefreshCw size={28} className="animate-spin" />
          <p>Processing time-series & calculating momentum features for {selectedStock}...</p>
        </div>
      )}

      {/* Main Content */}
      {!loading && !error && stockDetails && (
        <>
          {/* Key Metrics Strip */}
          <div className="metrics-grid">
            <div className="metric-card">
              <span className="metric-title">Latest Close</span>
              <span className="metric-value mono">₹{stockDetails.latest_close.toFixed(2)}</span>
              <div className="metric-sub" style={{ color: stockDetails.day_change >= 0 ? 'var(--color-up)' : 'var(--color-down)' }}>
                {stockDetails.day_change >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                <span className="mono">{stockDetails.day_return_pct} ({stockDetails.day_change >= 0 ? '+' : ''}{stockDetails.day_change})</span>
              </div>
            </div>

            <div className="metric-card">
              <span className="metric-title">Session Open</span>
              <span className="metric-value mono">₹{stockDetails.open.toFixed(2)}</span>
              <span className="metric-sub" style={{ color: 'var(--text-muted)' }}>Session Open</span>
            </div>

            <div className="metric-card">
              <span className="metric-title">Day High</span>
              <span className="metric-value mono">₹{stockDetails.high.toFixed(2)}</span>
              <span className="metric-sub" style={{ color: 'var(--text-muted)' }}>Session Peak</span>
            </div>

            <div className="metric-card">
              <span className="metric-title">Day Low</span>
              <span className="metric-value mono">₹{stockDetails.low.toFixed(2)}</span>
              <span className="metric-sub" style={{ color: 'var(--text-muted)' }}>Session Floor</span>
            </div>

            <div className="metric-card">
              <span className="metric-title">Volume</span>
              <span className="metric-value mono">{stockDetails.volume.toLocaleString()}</span>
              <span className="metric-sub" style={{ color: 'var(--text-muted)' }}>Shares Traded</span>
            </div>
          </div>

          {/* Main Grid: Prediction Card (Left) + Interactive Chart (Right) */}
          <div className="main-grid">
            {/* Prediction Card */}
            <div className={`prediction-card ${isUp ? 'up' : 'down'}`}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="card-header-label">Tomorrow's Forecast</span>
                  <span className="mono" style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    {stockDetails.latest_date}
                  </span>
                </div>

                <div className="prediction-badge-wrapper">
                  <div className={`prediction-badge ${isUp ? 'up' : 'down'}`}>
                    {isUp ? <TrendingUp size={30} /> : <TrendingDown size={30} />}
                    <span>{prediction.prediction}</span>
                  </div>
                </div>

                {/* Model Confidence Bar */}
                <div className="confidence-bar-container">
                  <div className="confidence-header">
                    <span>Model Probability</span>
                    <span className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                      {prediction.confidence_pct}
                    </span>
                  </div>
                  <div className="progress-track">
                    <div
                      className={`progress-fill ${isUp ? 'up' : 'down'}`}
                      style={{ width: `${Math.round(prob * 100)}%` }}
                    ></div>
                  </div>
                </div>

                {/* Details Breakdown */}
                <div className="prediction-details mono">
                  <div className="prediction-details-row">
                    <span>Target Class:</span>
                    <span style={{ fontWeight: 600 }}>{isUp ? "1 (Price UP)" : "0 (Price DOWN)"}</span>
                  </div>
                  <div className="prediction-details-row">
                    <span>Model In Service:</span>
                    <span>{prediction.model_used}</span>
                  </div>
                  <div className="prediction-details-row">
                    <span>Raw Probability:</span>
                    <span>{prob.toFixed(4)}</span>
                  </div>
                  <div className="prediction-details-row">
                    <span>Stock:</span>
                    <span>{prediction.stock}</span>
                  </div>
                </div>
              </div>

              <div className="disclaimer-text">
                ⚠️ <strong>Financial Disclaimer:</strong> Directional classification is generated from historical market technical indicators. It is intended for educational and algorithmic analysis and does not constitute financial advice.
              </div>
            </div>

            {/* Historical Price Chart */}
            <StockChart
              history={stockDetails.history}
              ticker={stockDetails.stock}
            />
          </div>

          {/* Model Information & Benchmark Section */}
          <ModelEvaluation
            metricsData={metrics}
            selectedModel={selectedModel}
          />
        </>
      )}
    </div>
  );
}
