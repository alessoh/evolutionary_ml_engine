# Quick Start Guide

**Get the Evolutionary ML Engine running in under 5 minutes**

## Prerequisites

- Python 3.10 or higher
- 5 minutes of time
- A terminal/command prompt

## Step 1: Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/evolutionary_ml_engine.git
cd evolutionary_ml_engine

# Create and activate virtual environment
python -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# On Windows Command Prompt:
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Run the Demo (30 seconds)

```bash
python evolutionary_ml_engine.py
```

You should see output showing:
1. Initial population seeding with 3 generations
2. A table of 10 streaming cycles with drift simulation
3. The system detecting drift and evolving automatically

## Step 3: Run the Interactive Dashboard (Optional)

If you want a visual interface:

```bash
streamlit run dashboard.py
```

This opens a web browser with:
- Real-time population evolution charts
- Control charts showing detection events
- Performance metrics over time
- Interactive parameter controls

## What You're Seeing

The demo simulates a production ML system experiencing data drift:

- **Cycles 1-6:** Stable performance, system monitors but doesn't intervene
- **Cycle 7:** Drift crosses threshold, conformal coverage drops
- **Action:** System automatically evolves new hyperparameters
- **Cycles 8-10:** Continued adaptation to sustained drift

## Key Metrics Explained

| Column | Meaning |
|--------|---------|
| **cycle** | Streaming batch number |
| **drift** | Simulated drift strength (0 = none, 1.4 = severe) |
| **auc** | Model accuracy (ROC AUC, higher is better) |
| **cov** | Conformal coverage (should stay near 0.90 for α=0.10) |
| **avg\|set\|** | Average prediction set size (larger = more uncertainty) |
| **PH** | Page-Hinkley detector fired (Y = drift detected) |
| **ooc** | Out of control (OOC = retraining needed) |
| **action** | What the system did (stable or evolved) |

## Troubleshooting

**"python: command not found"**
- Try `python3` instead of `python`
- Ensure Python 3.10+ is installed

**"ModuleNotFoundError"**
- Confirm your virtual environment is activated (you should see `(.venv)` in your prompt)
- Re-run `pip install -r requirements.txt`

**Warnings about sklearn**
- These are harmless deprecation warnings, ignore them

**Demo runs but shows all "ok" rows**
- The run succeeded! Random seed variation may produce slightly different trajectories
- Check the EVENT logs for detailed cycle information

## Next Steps

Once the demo runs successfully:

1. **Explore the code:** Read `evolutionary_ml_engine.py` - it's documented inline
2. **Modify parameters:** Edit the `drift_schedule` in `main()` to try different scenarios
3. **Run visualizations:** Execute `python visualize.py` to generate performance charts
4. **Try different models:** See examples in `examples/` directory

## Understanding the Output

### Seeding Phase
```
seed gen 1/3: best worker=w_a1b2c3d4 auc=0.8234 diversity=0.423
```
- The population is evolving initial hyperparameters
- AUC should improve or stabilize
- Diversity decreases as population converges

### Streaming Phase
```
cycle  drift    auc    cov  avg|set|  PH  ooc action
    1   0.00  0.823  0.905      1.12   .   ok stable
    7   0.90  0.741  0.821      1.89   Y  OOC evolved; new_best=w_e5f6g7h8 auc=0.798
```
- Cycle 1: Everything normal
- Cycle 7: Drift detected, system evolved, found better config

### Event Logs
```json
EVENT {"event_type": "cycle", "cycle_id": "c_xyz", "auc": 0.823, ...}
```
- Structured JSON for monitoring integration
- Contains full details of each cycle
- Ready for production logging systems

## Questions?

- Check the main README.md for detailed documentation
- Review the architecture diagram
- Explore the configuration options

**You're now ready to explore the Evolutionary ML Engine!**
