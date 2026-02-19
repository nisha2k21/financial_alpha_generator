@echo off
echo ============================================================
echo   Financial Alpha Generator - Institutional Trading Suite
echo ============================================================
echo.
echo [1/2] Checking dependencies...
pip install -r requirements.txt
echo.
echo [2/2] Launching Streamlit Dashboard...
streamlit run app.py
pause
