# evolutionary_ml_engine
evolutionary ml engine
# evolutionary_ml_engine

An autonomous machine learning engine that adapts itself to drifting data using a population-based hyperparameter autopilot, split conformal prediction for distribution-free uncertainty, and sequential statistical process control for diagnostics. Built as a single self-contained Python file that runs end to end on a laptop with no GPU, no internet connection beyond the initial install, and no proprietary dependencies.

## Table Of Contents

1. What This Is
2. What It Does
3. How It Works
4. Requirements
5. Installation
6. Quick Start
7. Expected Output
8. Configuration
9. Extending The Engine
10. File Structure
11. Troubleshooting
12. Honest Limits
13. References

## What This Is

The file `evolutionary_ml_engine.py` is a working reference implementation of an autonomous machine learning system. It is intended to be small enough to read in one sitting, complete enough to demonstrate every interesting part of the design, and runnable without setup beyond installing four standard Python libraries. The program creates synthetic data, trains a small population of models, simulates a stream of new data whose distribution slowly drifts, and shows the engine detecting that drift and retraining itself in response. The whole demonstration takes about thirty seconds on a normal laptop.

The engine is not a research preview, a toy, or a placeholder. Every component it ships has a real implementation, including the population dynamics, the conformal calibration, the Page-Hinkley change-point detector, and the Shewhart control charts. The code is written so that a working engineer can read the entire file in an afternoon and the design choices are documented inline.

What this is not, to be clear, is a production platform. The engine is a teaching artifact and an architectural skeleton that a team could grow into a production deployment by adding persistent storage, label-delay handling, traffic management, and the other plumbing that real systems need. The honest limits section at the end of this README lists exactly what is missing.

## What It Does

The engine performs two jobs that would normally require a human machine learning operator. The first job is hyperparameter optimization. The second job is health monitoring with automatic response. Together, those two jobs cover the routine, repetitive, time-sensitive work of keeping a deployed model healthy as the world around it changes.

For hyperparameter optimization the engine maintains a small population of model configurations, each one a different combination of choices for things like the number of trees in a gradient booster, the learning rate, the maximum tree depth, and so on. Every time the engine retrains, it scores every member of the population on a fresh slice of data, keeps the best performers untouched, and replaces the worst performers with mutated children of the best. Over many cycles the population converges on configurations that work well on whatever the current data looks like. When the data later drifts and old configurations stop working, the same evolutionary pressure pushes the population toward configurations that work on the new distribution, without the engine ever needing to be told that drift has happened.

For health monitoring the engine watches three signals continuously and reacts when any of them moves out of its normal range. The first signal is the model's predictive accuracy, monitored with a Shewhart control chart that knows what the normal range looks like because it observed the system during a baseline phase. The second signal is the empirical coverage of the conformal prediction sets, which should hover near the target coverage as long as the data distribution has not changed; when coverage drops, the data has shifted, and the same control-chart machinery raises a flag. The third signal is the cycle-to-cycle log loss, watched by a Page-Hinkley cumulative sum test that detects gradual upward drift in error rates earlier than a single-point threshold would. When any of the three trips, the engine evolves one generation, recalibrates the conformal predictor on the new data, and resets the change-point detector.

The result is a loop that, once started, requires no human intervention to maintain its own quality across changing conditions. The human role moves from operating the engine to designing it, auditing its decisions, and intervening only when something happens that the engine's automatic response is not equipped to handle.

## How It Works

The implementation is organized into seven cooperating components, each one a small class that takes responsibility for one of the questions an autonomous machine learning system needs to answer.

The `HPGenome` class declares the search space and the genetic operators that act on it. A genome spec is a Python dictionary that names each hyperparameter, gives its type and range, and optionally specifies how aggressively it should mutate. The genome's `sample` method draws a fresh random configuration, the `mutate` method perturbs an existing configuration by Gaussian noise on continuous dimensions and bounded random walks on integer dimensions, and the `crossover` method combines two configurations into a child that inherits each hyperparameter from one parent at random. This design lets evolution work uniformly across heterogeneous spaces of floats, integers, and discrete choices.

The `PopulationAutopilot` class runs the evolutionary loop. Each generation fits every worker on a training slice, scores every worker on a validation slice by ROC AUC, sorts by fitness, preserves the top two performers with elitism, leaves the middle of the pack untouched to carry forward, and replaces the bottom quarter with mutated children of the top quarter. The autopilot also exposes a `diversity` measure, defined as the mean pairwise distance over normalized numeric hyperparameters, so the engine can report whether the population has collapsed onto a single configuration or remains spread out enough to keep exploring.

The `ConformalCalibrator` class implements split conformal prediction for classifiers. Given a held-out calibration set, it computes one minus the predicted probability of the true class for each calibration point, takes the appropriately adjusted empirical quantile of those scores, and uses that quantile to decide which labels to include in any future prediction set. The result is a finite-sample distribution-free guarantee that the true label lands inside the predicted set with probability at least one minus alpha, regardless of what the underlying model is or what distribution the data comes from, provided the calibration and test data are exchangeable.

The `PageHinkley` class is a sequential change-point detector that consumes one number at a time and maintains a running mean, a cumulative sum of deviations from that mean minus a small tolerance, and the historical minimum of that cumulative sum. When the gap between the current cumulative sum and its historical minimum exceeds a threshold, the detector declares a positive shift at the current index. Its parameters have direct interpretations. The tolerance is the size of innocent fluctuation that should be ignored and the threshold is the strength of evidence that must accumulate before an alarm.

The `ControlChart` class implements a Shewhart x-bar chart with a subset of the Western Electric runs rules. After a baseline phase that estimates the mean and standard deviation from the first several observations, every subsequent observation is evaluated against four conditions. A single point more than three standard deviations from the mean is an out-of-control signal. Eight consecutive points on the same side of the mean indicates a shift. Six consecutive monotone points indicates a trend. Two of three consecutive points beyond two standard deviations on the same side indicates an early warning.

The `EvolutionaryMLSystem` class is the top-level orchestrator that wires the other components together. It exposes two methods, `seed` and `step`. The `seed` method trains the initial population on a stationary dataset, evolves it for a few generations to converge on a starting basin, and fits the conformal calibrator on a held-out slice. The `step` method runs one streaming cycle. It predicts on a new labeled batch, updates every detector and control chart, and if any detector trips, evolves one more generation and recalibrates the conformal predictor.

A small `synthesize` function rounds out the file by generating synthetic binary classification data with a controllable drift parameter, so the demo in `main` can simulate a stream of batches whose distribution changes over time.

## Requirements

The program runs on any operating system with a working Python installation. The minimum supported Python version is 3.10. The recommended versions are 3.11 and 3.12, both of which have been tested. The four required libraries are NumPy for numerical arrays, pandas for data tables, SciPy for the statistical tests, and scikit-learn for the underlying classifier. All four are available from the Python Package Index and install in one command.

The computational requirements are modest. The full demo runs in under a minute on any laptop manufactured in the last ten years and uses less than five hundred megabytes of memory. No graphics card is required. No internet connection is required after the libraries are installed.

The disk requirement is approximately three hundred megabytes for the libraries themselves plus a few kilobytes for the source file. If you create a virtual environment for the project, the entire installation lives inside that environment and can be removed by deleting one folder.

## Installation

There are two common ways to set up the environment, and the right one depends on whether you already use Anaconda or Miniconda. The instructions below cover both. Pick the one that matches your existing setup. If you have neither, the venv path is simpler and works on every operating system.

### Option A: Standard Python With venv

Open a terminal. On Windows that is either Command Prompt or PowerShell. On macOS the terminal lives in Applications under Utilities. On Linux any terminal that came with your distribution will work.

Verify your Python version by running `python --version` and confirming the output shows 3.10 or higher. On some macOS systems the Python 3 interpreter is called `python3` rather than `python`; if so, substitute `python3` and `pip3` for `python` and `pip` in every command that follows.

Create a project folder, change into it, and place `evolutionary_ml_engine.py` inside. For example:

```
mkdir evolutionary_ml_engine
cd evolutionary_ml_engine
```

Copy or download the source file into the new folder.

Create a virtual environment:

```
python -m venv .venv
```

Activate the environment. On macOS or Linux:

```
source .venv/bin/activate
```

On Windows in PowerShell:

```
.venv\Scripts\Activate.ps1
```

On Windows in Command Prompt:

```
.venv\Scripts\activate.bat
```

After activation your terminal prompt will show `(.venv)` at the beginning of the line. This is how you confirm that subsequent commands install into and run from this isolated environment.

Install the four required libraries:

```
pip install numpy pandas scipy scikit-learn
```

The install takes one to three minutes depending on your internet connection. When it finishes you should see a line that begins with `Successfully installed` followed by the four packages with their version numbers and a small number of transitive dependencies including `joblib`, `threadpoolctl`, `python-dateutil`, `six`, and `tzdata`.

### Option B: Anaconda Or Miniconda With conda

If you already use conda, create a new environment named for the project and activate it:

```
conda create -n evolutionary_ml_engine python=3.12
conda activate evolutionary_ml_engine
```

After activation your prompt will show `(evolutionary_ml_engine)` at the beginning of the line. You can install the libraries with either conda or pip. The pip path is shown here because it matches the venv instructions above and produces identical results:

```
pip install numpy pandas scipy scikit-learn
```

If you prefer conda for the libraries themselves, the equivalent command is:

```
conda install numpy pandas scipy scikit-learn
```

Either path produces a working environment.

### Verifying The Installation

Confirm that the libraries imported correctly by running:

```
python -c "import numpy, pandas, scipy, sklearn; print('ok')"
```

If the output is `ok`, the installation is complete. If any of the libraries fails to import you will see a `ModuleNotFoundError` naming the missing library, and the fix is to re-run the install command after confirming your environment is still active.

## Quick Start

With the environment active and `evolutionary_ml_engine.py` in your current directory, run:

```
python evolutionary_ml_engine.py
```

The program prints progress to the terminal and exits on its own when finished. Total runtime is fifteen to forty-five seconds depending on your machine.

## Expected Output

A healthy run produces three distinct sections of output.

The first section is the seeding phase, marked by a header line that reads `=== seeding population on stationary data ===`. Three lines follow, each one summarizing a generation of population-based training with the format `seed gen 1/3: best worker=w_XXXXXXXX auc=0.XXXX diversity=0.XXX`. The AUC should climb or hold steady across generations, and the diversity should generally decrease as the population concentrates on the winning configurations. After the third generation, a single longer line beginning with `EVENT {"event_type": "seed_completed",` records the chosen hyperparameters, the conformal quantile, and other metadata in a structured JSON form suitable for ingestion by a monitoring system.

The second section is the streaming phase, marked by a header that reads `=== streaming cycles with a programmed drift schedule ===` followed by a table header. The table has ten rows, one per cycle, with columns showing the cycle index, the simulated drift level, the model's current AUC, the empirical conformal coverage, the average prediction set size, whether Page-Hinkley triggered, whether the system is in or out of control, and the action taken. Between each table row is an EVENT JSON record describing the cycle in full detail.

In the reference run the first six rows of the table all show `ok` in the out-of-control column and `stable` in the action column. The drift then crosses a threshold around cycle seven, the conformal coverage drops below the chart's lower limit, the column flips to `OOC`, and the action column shows `evolved; new_best=w_XXXXXXXX auc=0.XXXX`. Cycles eight through ten continue to show `OOC` because the drift is sustained. One of those cycles will typically show a temporary AUC dip in the live evaluation column even though the freshly evolved validation AUC is high, which is the engine demonstrating that adaptation under sustained drift is not always instantaneous.

The third and final visible piece of output is simply the terminal prompt returning, which is how you know the program has finished cleanly. There is nothing to close, nothing to save, and no follow-up command required.

## Configuration

Every interesting design decision in the engine is exposed as a parameter so you can experiment without modifying the core implementation. The most useful dials are described below in the order most readers want to turn them.

The `drift_schedule` list inside `main` controls the simulated drift across the streaming cycles. Flattening it to all zeros makes every cycle stationary and the engine should never evolve. Steepening it should make the first out-of-control event happen earlier. Lengthening it adds more cycles to the run.

The `alpha` parameter passed to `EvolutionaryMLSystem` sets the target conformal coverage. The default of 0.10 corresponds to 90 percent coverage. Lowering alpha to 0.05 raises the target to 95 percent and the average prediction set size will widen as the engine becomes more conservative.

The `population_size` parameter on `EvolutionaryMLSystem` trades runtime for search quality. The default of 8 keeps the demo fast. Raising it to 16 or 24 doubles or triples the runtime per generation but explores more hyperparameter space.

The `generations` argument to `seed` controls how many evolutionary rounds happen during the initial bootstrap. The default of 3 is enough for the synthetic data; harder real problems benefit from 6 to 10.

Inside `PopulationAutopilot`, the `truncation_fraction` controls how aggressively bad workers are replaced. The default of 0.25 replaces the bottom quarter each generation. Lower values are more conservative and preserve more of the population unchanged. The `elitism` parameter sets how many top workers are kept untouched and immune from mutation; the default of 2 is a good compromise between protecting the current best and allowing the population to evolve.

Inside `PageHinkley`, the `delta` parameter is the magnitude of allowed innocent change and the `lam` parameter is the alarm threshold. Smaller delta and smaller lam make the detector more sensitive at the cost of more false alarms. The defaults are tuned for the synthetic data and real applications should be calibrated by replaying historical traffic.

Inside `ControlChart`, the `baseline_n` parameter sets how many observations are used to estimate the in-control mean and standard deviation before monitoring begins. The default of 6 gets the chart going quickly on the demo. Real applications should use a longer baseline, typically 25 to 30 observations, to get a stable estimate of normal variation.

## Extending The Engine

The engine is structured so that the common extensions are straightforward additions rather than rewrites.

To add a new model family, write a factory function that takes a hyperparameter dictionary and returns a fitted-style scikit-learn classifier, define a corresponding genome spec for its hyperparameters, and pass them into `PopulationAutopilot` alongside the existing gradient booster. The autopilot is family-agnostic. A more substantial extension is to make the population heterogeneous by carrying multiple families at once, which requires modifying the autopilot to track family membership per worker and to limit crossover to within-family pairs.

To add a new diagnostic signal, write a class that follows the same interface as `PageHinkley` or `ControlChart`, instantiate it inside `EvolutionaryMLSystem.__init__`, update it inside `step`, and incorporate its output into the `out_of_control` decision. Useful additions include a per-feature drift detector based on the Population Stability Index, a calibration drift detector based on the gap between predicted probability and empirical frequency in deciles, and a cost-and-latency monitor that vetoes deployments when inference is too slow.

To add a governance layer that vetoes deployments based on rules rather than statistics, insert a check between the detection of an out-of-control condition and the call to `evolve_one_generation`, where a candidate model would be evaluated against application-specific constraints like fairness limits, monotonicity requirements, or approved feature schemas. This is exactly the symbolic-guardrail idiom from the companion `autonomous_ml_engine.py`, and the two designs are explicitly compatible.

To send events to a real monitoring system rather than the terminal, replace the `event_sink` default in `EvolutionaryMLSystem.__init__` with a callable that forwards each event dictionary to your destination of choice. A typical implementation posts the dictionary as JSON to an HTTP endpoint, writes it to a log file in a structured format, or publishes it to a message broker.

## File Structure

The project at minimum contains the single source file:

```
evolutionary_ml_engine/
└── evolutionary_ml_engine.py
```

After running the installation instructions above, the folder additionally contains a virtual environment:

```
evolutionary_ml_engine/
├── .venv/                       (created by venv)
└── evolutionary_ml_engine.py
```

Or, if conda was used:

```
evolutionary_ml_engine/
└── evolutionary_ml_engine.py
```

with the conda environment stored centrally in your conda installation rather than inside the project folder.

The program writes no files to disk during operation. All output goes to standard output. The structured EVENT records that print between cycle summaries are intended to be redirected or piped to a monitoring destination in a real deployment.

## Troubleshooting

If the terminal reports `python: command not found` or `'python' is not recognized as an internal or external command`, then Python is either not installed or not on your PATH. On Windows the standard fix is to reinstall from python.org and tick the "Add Python to PATH" box during installation. On macOS, try `python3` and `pip3` in place of `python` and `pip` throughout the commands above.

If the program starts but prints `ModuleNotFoundError: No module named 'numpy'` or a similar message for one of the other libraries, the active environment is not the one where the libraries are installed. Confirm that your prompt shows either `(.venv)` for venv or `(evolutionary_ml_engine)` for conda, and re-run the activation command if it does not. If the prompt is correct, re-run the install command to confirm the libraries are present in the active environment.

If the program prints sklearn warnings about convergence, deprecation, or future API changes during the run, ignore them. They do not change the result and they will be resolved in future scikit-learn releases.

If the cycle table prints all `ok` rows and never shows `OOC`, the run still succeeded; the random seed combined with your particular Python and library versions may produce a slightly different trajectory. Scroll back through the EVENT records to confirm the engine evaluated the cycles correctly even if the table looks different.

If the program seems to hang, give it sixty seconds before investigating. Population-based training fits eight models per generation, which on a slow laptop can take fifteen seconds for a single generation. If nothing has appeared after sixty seconds, press Ctrl and C to interrupt and check for errors.

If pip is itself out of date and refuses to install some package version, upgrade it with `python -m pip install --upgrade pip` and retry.

## Honest Limits

The reference engine is a single-file teaching artifact and not a production platform. A short list of the most important things it does not yet do, so the picture is honest.

It evaluates new candidates on a slice of the same batch that triggered the retraining, which is acceptable for illustration but in production would require a held-out evaluation window so the model is judged on data it has not even indirectly seen.

It assumes labels arrive at the same rate as predictions, which is rarely true in real systems. A production deployment would need a label-delay queue that holds predictions until their ground-truth labels arrive and only then feeds them into the diagnostics.

It uses scikit-learn's gradient booster as the only family in the population. A serious deployment would benefit from a heterogeneous mix that might include linear models, stronger boosters such as LightGBM or XGBoost, and possibly small neural network ensembles.

Its conformal calibration is split conformal, the simplest variant, which requires a held-out calibration slice. Deployments that cannot afford to lose a calibration slice could use Mondrian conformal, cross-conformal, or full conformal at additional compute cost.

Its Page-Hinkley parameters are tuned for the synthetic data in the demo. Real applications need to be calibrated against historical traffic to find a tolerance and threshold that catch known incidents without firing on routine noise.

It has no persistent model storage. Restarting the program starts fresh. A production version would persist the population, the conformal quantile, and the control chart baselines across restarts.

It has no traffic management. A production version would shadow-deploy a new model alongside the existing primary, route a fraction of live traffic to the candidate, and promote only after the candidate's behavior on real traffic has been observed.

These limitations are deliberate. The goal of this reference is to make the architecture concrete enough to argue about, not to ship a turnkey product. The companion design document explains what each missing piece would look like and where in the codebase it would attach.

## References

The techniques in this engine are not invented for the project; they are well-established methods from machine learning, statistics, and industrial process control, assembled into one coherent system.

Population-based training was introduced by Jaderberg and collaborators at DeepMind in 2017 and has been used to train state-of-the-art reinforcement learning agents and large language models since.

Split conformal prediction was developed by Vladimir Vovk, Alexander Gammerman, and Glenn Shafer in the early 2000s. The textbook "Algorithmic Learning in a Random World" by Vovk, Gammerman, and Shafer is the foundational reference, and a more recent and accessible introduction is the survey "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" by Anastasios Angelopoulos and Stephen Bates.

The Page-Hinkley change-point test was introduced by E. S. Page in 1954 and named for him and the statistician Donald Hinkley. It is one of the oldest and most reliable sequential detectors in the literature.

Shewhart control charts and the Western Electric runs rules were developed at Bell Laboratories in the 1920s and 1930s and codified in the Western Electric Statistical Quality Control Handbook in 1956. They remain the foundation of industrial quality control.

The combination of all four into an autonomous machine learning system is the contribution of this reference implementation. None of the parts are novel; the architecture is the contribution.

## Contact

Email: h.alesso@comcast.net
GitHub: https://github.com/alessoh


