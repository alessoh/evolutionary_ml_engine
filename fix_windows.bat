@echo off
REM Windows Fix Script for Evolutionary ML Engine
REM Run this to fix all issues found by verify.py

echo ================================================================================
echo     EVOLUTIONARY ML ENGINE - WINDOWS FIX SCRIPT
echo ================================================================================
echo.

REM Check if conda environment is active
if not defined CONDA_DEFAULT_ENV (
    echo ERROR: Conda environment not active!
    echo Please run: conda activate evolutionary_ml_engine
    echo Then run this script again.
    pause
    exit /b 1
)

echo Current environment: %CONDA_DEFAULT_ENV%
echo.

REM Install missing dependencies
echo [1/4] Installing visualization libraries...
pip install matplotlib seaborn
if errorlevel 1 (
    echo WARNING: matplotlib/seaborn installation had issues
    echo Trying with conda instead...
    conda install matplotlib seaborn -y
)
echo.

echo [2/4] Installing dashboard dependencies (optional)...
pip install streamlit plotly
if errorlevel 1 (
    echo WARNING: streamlit/plotly installation had issues
    echo You can still run the main demo without these.
)
echo.

REM Create examples directory if it doesn't exist
echo [3/4] Creating examples directory...
if not exist "examples" (
    mkdir examples
    echo Created examples directory
) else (
    echo Examples directory already exists
)
echo.

REM Replace dashboard.py if it has encoding issues
echo [4/4] Checking dashboard.py...
if exist "dashboard.py" (
    echo Found dashboard.py - checking for encoding issues...
    REM Backup existing file
    copy dashboard.py dashboard_backup.py >nul 2>&1
    echo Backup created: dashboard_backup.py
)
echo.

echo ================================================================================
echo                              FIX COMPLETE
echo ================================================================================
echo.
echo Next steps:
echo   1. Copy fraud_detection.py to examples\ directory
echo   2. Copy churn_prediction.py to examples\ directory
echo   3. Replace dashboard.py with dashboard_fixed.py if needed
echo   4. Run: python verify.py
echo.
echo If verify.py still shows errors:
echo   - Check that examples\ directory has the .py files
echo   - Make sure matplotlib is installed: python -c "import matplotlib"
echo.
pause
