@echo off
title Stock Prediction System - Frontend Dashboard
echo =========================================================
echo   Stock Market Direction Prediction System - Frontend
echo =========================================================
echo.
cd frontend
echo Installing / checking npm dependencies...
call npm install
echo.
echo Starting React + Vite Dev Server on http://localhost:5173 ...
call npm run dev
pause
