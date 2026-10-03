@echo off
title Launch Stock Prediction System
echo =========================================================
echo   Launching Stock Prediction System (Backend + Frontend)
echo =========================================================
echo.
start "Stock Prediction Backend" cmd /k "run_backend.bat"
timeout /t 3 /nobreak >nul
start "Stock Prediction Frontend" cmd /k "run_frontend.bat"
echo.
echo Both servers have been launched!
echo Open your browser at: http://localhost:5173
echo =========================================================
pause
