"""
CLI script to test Phase 11: Prediction Service
Usage:
    python backend/test_prediction.py
    python backend/test_prediction.py --ticker TCS --model xgboost
    python backend/test_prediction.py --ticker RELIANCE --model random_forest
"""
import argparse
import json
from services.prediction_service import predict_stock_direction

def main():
    parser = argparse.ArgumentParser(description="Test Stock Prediction Service")
    parser.add_argument("--ticker", type=str, default="TCS", help="Stock ticker symbol (default: TCS)")
    parser.add_argument("--model", type=str, default="random_forest", choices=["random_forest", "xgboost"], help="ML model to use")
    args = parser.parse_args()

    print(f"Running prediction service for {args.ticker} using {args.model}...")
    result = predict_stock_direction(ticker=args.ticker, model_name=args.model)

    print("\n==================================================")
    print("PHASE 11: PREDICTION SERVICE RESULT")
    print("==================================================")
    print(f"Stock Ticker:       {result['stock']}")
    print(f"Latest Market Date: {result['latest_date']}")
    print(f"Latest Close Price: INR {result['latest_close']:,.2f}")
    print(f"Day's Range (OHLC): O: INR {result['open']} | H: INR {result['high']} | L: INR {result['low']}")
    print(f"Trading Volume:     {result['volume']:,} shares")
    print("-" * 50)
    print(f"Model In Service:   {result['model_used']}")
    direction_label = "[UP]" if result['prediction'] == "UP" else "[DOWN]"
    print(f"Tomorrow Direction: {direction_label} {result['prediction']}")
    print(f"Model Probability:  {result['probability']:.4f} ({result['confidence_pct']})")
    print("-" * 50)
    print(f"Full JSON Payload:")
    print(json.dumps(result, indent=2))
    print("==================================================")

if __name__ == "__main__":
    main()
