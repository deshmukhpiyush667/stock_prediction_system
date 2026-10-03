import os
import glob
from typing import List

def get_available_tickers(raw_dir: str) -> List[str]:
    """
    Scans the actual data/raw/ directory and extracts all available stock ticker symbols.
    Never invents fake stock symbols.
    """
    tickers = []
    if not os.path.exists(raw_dir):
        return tickers

    files = os.listdir(raw_dir)
    for f in sorted(files):
        if f.startswith("Stock_") and (f.endswith(".xlsx") or f.endswith(".csv")):
            ticker = f.replace("Stock_", "").replace(".xlsx", "").replace(".csv", "")
            tickers.append(ticker)
        elif not f.startswith("Index_") and (f.endswith(".xlsx") or f.endswith(".csv")):
            ticker = os.path.splitext(f)[0]
            tickers.append(ticker)

    return sorted(list(set(tickers)))
