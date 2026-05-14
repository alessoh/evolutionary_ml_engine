# Windows Fix Script for Evolutionary ML Engine (PowerShell)
# Run this to fix all issues found by verify.py

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "    EVOLUTIONARY ML ENGINE - WINDOWS FIX SCRIPT" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if conda environment is active
if (-not $env:CONDA_DEFAULT_ENV) {
    Write-Host "ERROR: Conda environment not active!" -ForegroundColor Red
    Write-Host "Please run: conda activate evolutionary_ml_engine" -ForegroundColor Yellow
    Write-Host "Then run this script again." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Current environment: $env:CONDA_DEFAULT_ENV" -ForegroundColor Green
Write-Host ""

# Install missing dependencies
Write-Host "[1/4] Installing visualization libraries..." -ForegroundColor Yellow
try {
    pip install matplotlib seaborn
    Write-Host "Matplotlib and seaborn installed successfully" -ForegroundColor Green
}
catch {
    Write-Host "WARNING: pip installation had issues" -ForegroundColor Yellow
    Write-Host "Trying with conda instead..." -ForegroundColor Yellow
    conda install matplotlib seaborn -y
}
Write-Host ""

Write-Host "[2/4] Installing dashboard dependencies (optional)..." -ForegroundColor Yellow
try {
    pip install streamlit plotly
    Write-Host "Streamlit and plotly installed successfully" -ForegroundColor Green
}
catch {
    Write-Host "WARNING: Dashboard dependencies had issues" -ForegroundColor Yellow
    Write-Host "You can still run the main demo without these." -ForegroundColor Yellow
}
Write-Host ""

# Create examples directory if it doesn't exist
Write-Host "[3/4] Creating examples directory..." -ForegroundColor Yellow
if (-not (Test-Path "examples")) {
    New-Item -ItemType Directory -Path "examples" | Out-Null
    Write-Host "Created examples directory" -ForegroundColor Green
}
else {
    Write-Host "Examples directory already exists" -ForegroundColor Green
}
Write-Host ""

# Check dashboard.py
Write-Host "[4/4] Checking dashboard.py..." -ForegroundColor Yellow
if (Test-Path "dashboard.py") {
    Write-Host "Found dashboard.py - creating backup..." -ForegroundColor Green
    Copy-Item "dashboard.py" "dashboard_backup.py" -Force
    Write-Host "Backup created: dashboard_backup.py" -ForegroundColor Green
}
Write-Host ""

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "                             FIX COMPLETE" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Copy fraud_detection.py to examples\ directory"
Write-Host "  2. Copy churn_prediction.py to examples\ directory"
Write-Host "  3. Replace dashboard.py with dashboard_fixed.py if needed"
Write-Host "  4. Run: python verify.py"
Write-Host ""
Write-Host "Quick commands to fix remaining issues:" -ForegroundColor Green
Write-Host "  Move-Item fraud_detection.py examples\"
Write-Host "  Move-Item churn_prediction.py examples\"
Write-Host "  Move-Item dashboard_fixed.py dashboard.py -Force"
Write-Host ""

Read-Host "Press Enter to continue"
