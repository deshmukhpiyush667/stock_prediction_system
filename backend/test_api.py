"""
Automated unit & integration test suite for FastAPI backend endpoints.
"""
import sys
import os

# Ensure backend root is in python path
backend_dir = os.path.abspath(os.path.dirname(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_endpoints():
    print("==================================================")
    print("PHASE 12: TESTING FASTAPI BACKEND ENDPOINTS")
    print("==================================================")

    # 1. Test Root
    res = client.get("/")
    print(f"\n1. GET / -> Status {res.status_code}")
    assert res.status_code == 200
    print(f"   System: {res.json()['system']}")

    # 2. Test Health
    res = client.get("/health")
    print(f"\n2. GET /health -> Status {res.status_code}")
    assert res.status_code == 200
    print(f"   Status: {res.json()['status']}, Components: {res.json()['components']}")

    # 3. Test Stocks list
    res = client.get("/stocks")
    print(f"\n3. GET /stocks -> Status {res.status_code}")
    assert res.status_code == 200
    stocks = res.json()["stocks"]
    print(f"   Total available stocks: {res.json()['total_stocks']} (Sample: {stocks[:6]}...)")
    assert "TCS" in stocks

    # 4. Test Stock details (OHLCV & Chart history)
    res = client.get("/stock/TCS?days=60")
    print(f"\n4. GET /stock/TCS?days=60 -> Status {res.status_code}")
    assert res.status_code == 200
    stock_data = res.json()
    print(f"   Latest Date: {stock_data['latest_date']}, Latest Close: INR {stock_data['latest_close']}")
    print(f"   History bars returned: {len(stock_data['history'])} (Chart ready with MA_20 and MA_50)")

    # 5. Test Prediction with XGBoost
    res = client.get("/predict/TCS?model=xgboost")
    print(f"\n5. GET /predict/TCS?model=xgboost -> Status {res.status_code}")
    assert res.status_code == 200
    pred = res.json()
    print(f"   Stock: {pred['stock']}, Prediction: {pred['prediction']}, Probability: {pred['probability']}")
    assert pred["prediction"] in ["UP", "DOWN"]
    assert 0.0 <= pred["probability"] <= 1.0

    # 6. Test Prediction with Random Forest
    res = client.get("/predict/TCS?model=random_forest")
    print(f"\n6. GET /predict/TCS?model=random_forest -> Status {res.status_code}")
    assert res.status_code == 200
    pred_rf = res.json()
    print(f"   Stock: {pred_rf['stock']}, Prediction: {pred_rf['prediction']}, Probability: {pred_rf['probability']}")

    # 7. Test Metrics endpoint
    res = client.get("/metrics")
    print(f"\n7. GET /metrics -> Status {res.status_code}")
    assert res.status_code == 200
    metrics = res.json()
    print(f"   RF Accuracy: {metrics['models']['random_forest']['accuracy'] * 100:.2f}%")
    print(f"   XGB Accuracy: {metrics['models']['xgboost']['accuracy'] * 100:.2f}%")

    # 8. Test 404 on Invalid Ticker
    res = client.get("/stock/NONEXISTENT_XYZ")
    print(f"\n8. GET /stock/NONEXISTENT_XYZ -> Status {res.status_code} (Expected 404)")
    assert res.status_code == 404
    print(f"   Error detail: {res.json()['detail']}")

    print("\n[SUCCESS] All 8 API endpoint tests PASSED.")
    print("==================================================")

if __name__ == "__main__":
    test_endpoints()
