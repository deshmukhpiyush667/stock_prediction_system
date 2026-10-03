import React, { useState, useMemo } from 'react';

export default function StockChart({ history, ticker }) {
  const [showMA20, setShowMA20] = useState(true);
  const [showMA50, setShowMA50] = useState(true);
  const [hoverIndex, setHoverIndex] = useState(null);

  // Filter valid data points
  const data = useMemo(() => {
    if (!history || history.length === 0) return [];
    return history;
  }, [history]);

  // Compute price extents
  const { minPrice, maxPrice } = useMemo(() => {
    if (data.length === 0) return { minPrice: 0, maxPrice: 100 };
    let min = Infinity;
    let max = -Infinity;

    data.forEach(d => {
      if (d.close < min) min = d.close;
      if (d.close > max) max = d.close;
      if (showMA20 && d.ma_20 !== null) {
        if (d.ma_20 < min) min = d.ma_20;
        if (d.ma_20 > max) max = d.ma_20;
      }
      if (showMA50 && d.ma_50 !== null) {
        if (d.ma_50 < min) min = d.ma_50;
        if (d.ma_50 > max) max = d.ma_50;
      }
    });

    const padding = (max - min) * 0.08;
    return {
      minPrice: Math.max(0, min - padding),
      maxPrice: max + padding
    };
  }, [data, showMA20, showMA50]);

  const width = 850;
  const height = 320;
  const paddingLeft = 55;
  const paddingRight = 20;
  const paddingTop = 20;
  const paddingBottom = 30;

  const chartW = width - paddingLeft - paddingRight;
  const chartH = height - paddingTop - paddingBottom;

  const getX = (index) => paddingLeft + (index / (data.length - 1 || 1)) * chartW;
  const getY = (val) => paddingTop + chartH - ((val - minPrice) / (maxPrice - minPrice || 1)) * chartH;

  // Generate SVG path strings
  const closePath = useMemo(() => {
    if (data.length < 2) return '';
    return data.map((d, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(d.close)}`).join(' ');
  }, [data, minPrice, maxPrice]);

  const ma20Path = useMemo(() => {
    if (!showMA20 || data.length < 2) return '';
    const points = [];
    data.forEach((d, i) => {
      if (d.ma_20 !== null) {
        points.push(`${points.length === 0 ? 'M' : 'L'} ${getX(i)} ${getY(d.ma_20)}`);
      }
    });
    return points.join(' ');
  }, [data, showMA20, minPrice, maxPrice]);

  const ma50Path = useMemo(() => {
    if (!showMA50 || data.length < 2) return '';
    const points = [];
    data.forEach((d, i) => {
      if (d.ma_50 !== null) {
        points.push(`${points.length === 0 ? 'M' : 'L'} ${getX(i)} ${getY(d.ma_50)}`);
      }
    });
    return points.join(' ');
  }, [data, showMA50, minPrice, maxPrice]);

  const hoveredItem = hoverIndex !== null && data[hoverIndex] ? data[hoverIndex] : null;

  // Y-axis ticks
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map(ratio => {
    const val = minPrice + ratio * (maxPrice - minPrice);
    return { val, y: getY(val) };
  });

  return (
    <div className="chart-card">
      <div className="chart-header">
        <div className="chart-title">
          <h3>{ticker} Historical Price & Moving Averages</h3>
        </div>

        <div className="chart-legend">
          <div className="legend-item">
            <span className="legend-color" style={{ background: '#3b82f6' }}></span>
            <span>Close</span>
          </div>

          <label className="legend-item" style={{ cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={showMA20}
              onChange={() => setShowMA20(!showMA20)}
              style={{ marginRight: 4 }}
            />
            <span className="legend-color" style={{ background: 'var(--ma20-color)' }}></span>
            <span>MA 20</span>
          </label>

          <label className="legend-item" style={{ cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={showMA50}
              onChange={() => setShowMA50(!showMA50)}
              style={{ marginRight: 4 }}
            />
            <span className="legend-color" style={{ background: 'var(--ma50-color)' }}></span>
            <span>MA 50</span>
          </label>
        </div>
      </div>

      <div
        className="chart-svg-container"
        onMouseLeave={() => setHoverIndex(null)}
      >
        <svg
          viewBox={`0 0 ${width} ${height}`}
          style={{ width: '100%', height: '100%', display: 'block' }}
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const relX = ((e.clientX - rect.left) / rect.width) * width;
            if (relX >= paddingLeft && relX <= width - paddingRight) {
              const idx = Math.round(((relX - paddingLeft) / chartW) * (data.length - 1));
              if (idx >= 0 && idx < data.length) setHoverIndex(idx);
            }
          }}
        >
          {/* Grid lines and Y axis ticks */}
          {yTicks.map((tick, i) => (
            <g key={i}>
              <line
                x1={paddingLeft}
                y1={tick.y}
                x2={width - paddingRight}
                y2={tick.y}
                stroke="#1f2937"
                strokeDasharray="4 4"
              />
              <text
                x={paddingLeft - 8}
                y={tick.y + 4}
                fill="#6b7280"
                fontSize="11"
                textAnchor="end"
                className="mono"
              >
                ₹{tick.val.toFixed(0)}
              </text>
            </g>
          ))}

          {/* Date Ticks on X-Axis */}
          {data.length > 0 && [0, Math.floor(data.length / 2), data.length - 1].map((idx) => (
            <text
              key={idx}
              x={getX(idx)}
              y={height - 8}
              fill="#6b7280"
              fontSize="11"
              textAnchor={idx === 0 ? "start" : idx === data.length - 1 ? "end" : "middle"}
              className="mono"
            >
              {data[idx].date}
            </text>
          ))}

          {/* MA 50 Line */}
          {showMA50 && ma50Path && (
            <path
              d={ma50Path}
              fill="none"
              stroke="var(--ma50-color)"
              strokeWidth="2"
              strokeLinecap="round"
            />
          )}

          {/* MA 20 Line */}
          {showMA20 && ma20Path && (
            <path
              d={ma20Path}
              fill="none"
              stroke="var(--ma20-color)"
              strokeWidth="2"
              strokeLinecap="round"
            />
          )}

          {/* Close Price Line */}
          {closePath && (
            <path
              d={closePath}
              fill="none"
              stroke="#3b82f6"
              strokeWidth="2.5"
              strokeLinecap="round"
            />
          )}

          {/* Hover Crosshair & Indicator */}
          {hoveredItem && (
            <g>
              <line
                x1={getX(hoverIndex)}
                y1={paddingTop}
                x2={getX(hoverIndex)}
                y2={paddingTop + chartH}
                stroke="#9ca3af"
                strokeDasharray="3 3"
                strokeWidth="1.2"
              />
              <circle
                cx={getX(hoverIndex)}
                cy={getY(hoveredItem.close)}
                r="5"
                fill="#3b82f6"
                stroke="#fff"
                strokeWidth="2"
              />
            </g>
          )}
        </svg>

        {/* Hover Tooltip Box */}
        {hoveredItem && (
          <div
            style={{
              position: 'absolute',
              top: 15,
              right: 25,
              background: 'rgba(17, 24, 39, 0.95)',
              border: '1px solid #374151',
              borderRadius: 8,
              padding: '8px 12px',
              fontSize: '0.8rem',
              boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
              pointerEvents: 'none'
            }}
            className="mono"
          >
            <div style={{ color: '#9ca3af', marginBottom: 4 }}>Date: {hoveredItem.date}</div>
            <div style={{ color: '#3b82f6', fontWeight: 600 }}>Close: ₹{hoveredItem.close.toFixed(2)}</div>
            {hoveredItem.ma_20 !== null && (
              <div style={{ color: 'var(--ma20-color)' }}>MA 20: ₹{hoveredItem.ma_20.toFixed(2)}</div>
            )}
            {hoveredItem.ma_50 !== null && (
              <div style={{ color: 'var(--ma50-color)' }}>MA 50: ₹{hoveredItem.ma_50.toFixed(2)}</div>
            )}
            <div style={{ color: '#6b7280', marginTop: 2 }}>Volume: {hoveredItem.volume.toLocaleString()}</div>
          </div>
        )}
      </div>
    </div>
  );
}
