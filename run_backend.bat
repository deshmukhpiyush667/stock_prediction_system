@echo off
title Stock Prediction System - Backend API
echo =========================================================
echo   Stock Market Direction Prediction System - Backend
echo =========================================================
echo.
echo Installing / checking Python dependencies...
python -m pip install -r backend\requirements.txt
echo.
echo Starting FastAPI Backend on http://127.0.0.1:8000 ...
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
