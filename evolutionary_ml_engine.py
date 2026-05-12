from __future__ import annotations

import json
import logging
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)-14s | %(message)s",
)
log = logging.getLogger("evo_ml")


# ---------------------------------------------------------------------------
# 1. Types
# ---------------------------------------------------------------------------

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def safe_log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    """log_loss that survives sklearn API churn around the eps argument."""
    p = np.clip(proba, 1e-7, 1.0 - 1e-7)
    return float(log_loss(y_true, p))


@dataclass
class Worker:
    """A single member of the population: a hyperparameter configuration
    paired with the model fit at the latest generation."""
    worker_id: str
    hyperparameters: Dict[str, Any]
    fitness: float = float("-inf")
    age: int = 0
    lineage: List[str] = field(default_factory=list)
    model: Any = None
    metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2. Hyperparameter genome and operators
# ---------------------------------------------------------------------------

class HPGenome:
    """Defines the search space and the operators (sample, mutate, crossover)
    that act on hyperparameter dictionaries.

    A genome spec is a recipe: for each HP, declare its type, range, and an
    optional mutation rate. This lets evolution operate uniformly across
    heterogeneous spaces of floats, integers, and discrete choices.
    """

    def __init__(self, spec: Dict[str, Dict[str, Any]], rng: np.random.Generator):
        self.spec = spec
        self.rng = rng

    def sample(self) -> Dict[str, Any]:
        hp: Dict[str, Any] = {}
        for name, s in self.spec.items():
            t = s["type"]
            if t == "float":
                lo, hi = s["range"]
                if s.get("log"):
                    hp[name] = float(
                        np.exp(self.rng.uniform(np.log(lo), np.log(hi)))
                    )
                else:
                    hp[name] = float(self.rng.uniform(lo, hi))
            elif t == "int":
                lo, hi = s["range"]
                hp[name] = int(self.rng.integers(lo, hi + 1))
            elif t == "choice":
                hp[name] = random.choice(s["choices"])
            else:
                raise ValueError(f"unknown hp type {t}")
        return hp

    def mutate(self, hp: Dict[str, Any], strength: float = 0.25) -> Dict[str, Any]:
        new = dict(hp)
        for name, s in self.spec.items():
            if self.rng.uniform() > s.get("mutation_rate", 0.4):
                continue
            t = s["type"]
            if t == "float":
                lo, hi = s["range"]
                if s.get("log"):
                    span = np.log(hi) - np.log(lo)
                    log_val = np.log(max(new[name], 1e-12))
                    log_val += self.rng.normal(0.0, span * strength)
                    new[name] = float(np.clip(np.exp(log_val), lo, hi))
                else:
                    span = hi - lo
                    new[name] = float(np.clip(
                        new[name] + self.rng.normal(0.0, span * strength), lo, hi,
                    ))
            elif t == "int":
                lo, hi = s["range"]
                span = max(1, hi - lo)
                step = max(1, int(round(span * strength)))
                delta = int(self.rng.integers(-step, step + 1))
                new[name] = int(np.clip(new[name] + delta, lo, hi))
            elif t == "choice":
                new[name] = random.choice(s["choices"])
        return new

    def crossover(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        return {
            name: (a[name] if self.rng.uniform() < 0.5 else b[name])
            for name in self.spec
        }


# ---------------------------------------------------------------------------
# 3. Population-Based Training autopilot
# ---------------------------------------------------------------------------

class PopulationAutopilot:
    """Population-Based Training over a single model family.

    One generation:
        1. Fit every worker on (X_train, y_train).
        2. Score every worker on (X_val, y_val) by primary_metric.
        3. Sort by fitness descending.
        4. Elitism: keep top `elitism` workers untouched.
        5. Truncation: bottom k workers are replaced by children of the
           top k. A child copies a random top parent's hyperparameters,
           then mutates them.
        6. Middle workers carry their hyperparameters forward and refit
           on the next generation's data.
    """

    def __init__(
        self,
        model_factory: Callable[[Dict[str, Any]], Any],
        genome: HPGenome,
        population_size: int = 10,
        truncation_fraction: float = 0.25,
        elitism: int = 2,
        seed: int = 42,
    ):
        self.model_factory = model_factory
        self.genome = genome
        self.population_size = population_size
        self.truncation_fraction = truncation_fraction
        self.elitism = elitism
        self.rng = np.random.default_rng(seed)
        self.population: List[Worker] = []
        self.generation: int = 0

    def seed_population(self) -> None:
        self.population = [
            Worker(worker_id=short_id("w"), hyperparameters=self.genome.sample())
            for _ in range(self.population_size)
        ]

    def _fit_score(
        self, worker: Worker, X_tr: np.ndarray, y_tr: np.ndarray,
        X_va: np.ndarray, y_va: np.ndarray,
    ) -> None:
        model = self.model_factory(worker.hyperparameters)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        worker.model = model
        worker.fitness = float(roc_auc_score(y_va, proba))
        worker.metrics = {
            "roc_auc": worker.fitness,
            "brier": float(brier_score_loss(y_va, proba)),
            "log_loss": safe_log_loss(y_va, proba),
        }

    def evolve_one_generation(
        self, X_tr: np.ndarray, y_tr: np.ndarray,
        X_va: np.ndarray, y_va: np.ndarray,
    ) -> List[Worker]:
        if not self.population:
            self.seed_population()

        for w in self.population:
            self._fit_score(w, X_tr, y_tr, X_va, y_va)
            w.age += 1
        self.population.sort(key=lambda w: w.fitness, reverse=True)

        k = max(1, int(self.population_size * self.truncation_fraction))
        top = self.population[:k]
        survivors = self.population[: self.elitism]
        middle = self.population[self.elitism : self.population_size - k]

        # Exploit-and-explore: bottom k replaced by mutated children of top k
        children: List[Worker] = []
        for _ in range(k):
            parent = top[int(self.rng.integers(0, len(top)))]
            child_hp = self.genome.mutate(dict(parent.hyperparameters), strength=0.25)
            children.append(Worker(
                worker_id=short_id("w"),
                hyperparameters=child_hp,
                lineage=[parent.worker_id] + parent.lineage[:5],
            ))

        self.population = survivors + middle + children
        self.generation += 1
        return self.population

    def best(self) -> Worker:
        return max(self.population, key=lambda w: w.fitness)

    def diversity(self) -> float:
        """Mean pairwise distance over normalized numeric HPs in [0,1]."""
        names = [n for n, s in self.genome.spec.items() if s["type"] in ("int", "float")]
        if not names or len(self.population) < 2:
            return 0.0
        rows: List[List[float]] = []
        for w in self.population:
            row: List[float] = []
            for n in names:
                lo, hi = self.genome.spec[n]["range"]
                if self.genome.spec[n].get("log"):
                    v = (np.log(max(w.hyperparameters[n], 1e-12)) - np.log(lo)) / (
                        np.log(hi) - np.log(lo)
                    )
                else:
                    v = (w.hyperparameters[n] - lo) / max(hi - lo, 1e-12)
                row.append(float(v))
            rows.append(row)
        arr = np.array(rows)
        dists = []
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                dists.append(float(np.linalg.norm(arr[i] - arr[j])))
        return float(np.mean(dists)) if dists else 0.0


# ---------------------------------------------------------------------------
# 4. Split conformal classifier
# ---------------------------------------------------------------------------

class ConformalCalibrator:
    """Split conformal prediction for classifiers.

    Given a held-out calibration set, compute nonconformity scores
        s_i = 1 - p_model(y_i | x_i)
    At prediction time, return the set
        { y : 1 - p_model(y | x) <= q_hat }
    where q_hat is the ceil((n+1)(1-alpha))/n quantile of calibration
    scores. This yields marginal coverage P(y in set) >= 1 - alpha
    without distributional assumptions, provided calibration and test
    data are exchangeable.
    """

    def __init__(self, alpha: float = 0.1):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = alpha
        self.q_hat: Optional[float] = None
        self.n_calib: int = 0

    def fit(self, model: Any, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        proba = model.predict_proba(X_cal)
        n = len(y_cal)
        scores = np.array([1.0 - proba[i, int(y_cal[i])] for i in range(n)])
        level = math.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = float(min(max(level, 0.0), 1.0))
        self.q_hat = float(np.quantile(scores, level))
        self.n_calib = n

    def predict_set(self, model: Any, X: np.ndarray) -> List[List[int]]:
        if self.q_hat is None:
            raise RuntimeError("ConformalCalibrator must be fit first")
        proba = model.predict_proba(X)
        sets: List[List[int]] = []
        for row in proba:
            keep = [int(c) for c, p in enumerate(row) if (1.0 - p) <= self.q_hat]
            sets.append(keep)
        return sets

    @staticmethod
    def coverage(sets: List[List[int]], y_true: np.ndarray) -> float:
        if not sets:
            return float("nan")
        hits = sum(int(int(y_true[i]) in s) for i, s in enumerate(sets))
        return hits / len(sets)


# ---------------------------------------------------------------------------
# 5. Page-Hinkley sequential change-point detector
# ---------------------------------------------------------------------------

class PageHinkley:
    """Sequential change-point detector on a stream of scalars.

    Maintains a running mean and a cumulative sum
        m_t = sum_{i=1}^{t} (x_i - mean_t - delta)
    and tracks M_T = min(m_t for t <= T). When (m_T - M_T) > lam, the
    detector declares a positive shift at index T. delta is the
    "magnitude of allowed change" and lam is the alarm threshold; both
    are tunable.
    """

    def __init__(self, delta: float = 0.005, lam: float = 10.0, warmup: int = 5):
        self.delta = delta
        self.lam = lam
        self.warmup = warmup
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.mean = 0.0
        self.m_t = 0.0
        self.M_T = 0.0
        self.change_detected = False
        self.change_index: Optional[int] = None

    def update(self, x: float) -> bool:
        self.t += 1
        self.mean += (x - self.mean) / self.t
        self.m_t += x - self.mean - self.delta
        self.M_T = min(self.M_T, self.m_t)
        if self.t < self.warmup:
            return False
        if (self.m_t - self.M_T) > self.lam:
            self.change_detected = True
            self.change_index = self.t
            return True
        return False


# ---------------------------------------------------------------------------
# 6. Shewhart x-bar control chart with Western Electric runs rules
# ---------------------------------------------------------------------------

class ControlChart:
    """Shewhart x-bar chart with a subset of the Western Electric rules.

    A baseline phase estimates mean and std from the first `baseline_n`
    observations, then freezes them. After that, every new observation
    is evaluated against:
        R1: single point more than 3 sigma from the mean.
        R2: 8 consecutive points on the same side of the mean.
        R3: 6 consecutive monotone points (trend).
        R4: 2 of 3 consecutive points beyond 2 sigma on the same side.
    """

    def __init__(self, baseline_n: int = 8):
        self.baseline_n = baseline_n
        self.history: List[float] = []
        self.mean: Optional[float] = None
        self.std: Optional[float] = None

    def update(self, x: float) -> Dict[str, Any]:
        self.history.append(x)
        if len(self.history) == self.baseline_n:
            arr = np.array(self.history, dtype=float)
            self.mean = float(arr.mean())
            self.std = float(arr.std(ddof=1) or 1e-9)
        if self.mean is None or self.std is None:
            return {"in_control": True, "rules_fired": [], "phase": "baseline"}

        rules: List[str] = []
        if abs(x - self.mean) > 3.0 * self.std:
            rules.append("R1_3sigma")

        recent_8 = self.history[-8:]
        if len(recent_8) == 8:
            if all(p > self.mean for p in recent_8) or all(p < self.mean for p in recent_8):
                rules.append("R2_run8")

        recent_6 = self.history[-6:]
        if len(recent_6) == 6:
            diffs = np.diff(recent_6)
            if np.all(diffs > 0) or np.all(diffs < 0):
                rules.append("R3_trend6")

        recent_3 = self.history[-3:]
        if len(recent_3) == 3:
            above = sum(1 for p in recent_3 if (p - self.mean) > 2.0 * self.std)
            below = sum(1 for p in recent_3 if (self.mean - p) > 2.0 * self.std)
            if above >= 2 or below >= 2:
                rules.append("R4_2of3_2sigma")

        return {
            "in_control": not rules,
            "rules_fired": rules,
            "phase": "monitoring",
            "value": float(x),
            "mean": self.mean,
            "std": self.std,
            "ucl_3sigma": self.mean + 3.0 * self.std,
            "lcl_3sigma": self.mean - 3.0 * self.std,
        }


# ---------------------------------------------------------------------------
# 7. Top-level orchestrator
# ---------------------------------------------------------------------------

class EvolutionaryMLSystem:
    """Top-level autonomous engine that wires PBT, conformal, and SPC together.

    Lifecycle:
        seed(X, y):     three-way split (train/val/calib), evolve N
                        generations, fit the conformal calibrator on the
                        best worker.
        step(X, y):     one streaming cycle. Predict on the batch, update
                        all sequential detectors and control charts, and
                        evolve the population if any detector trips.
    """

    def __init__(
        self,
        model_factory: Callable[[Dict[str, Any]], Any],
        genome: HPGenome,
        population_size: int = 10,
        alpha: float = 0.1,
        seed: int = 42,
    ):
        self.autopilot = PopulationAutopilot(
            model_factory=model_factory,
            genome=genome,
            population_size=population_size,
            seed=seed,
        )
        self.conformal = ConformalCalibrator(alpha=alpha)
        self.ph_loss = PageHinkley(delta=0.005, lam=2.0, warmup=4)
        self.auc_chart = ControlChart(baseline_n=6)
        self.coverage_chart = ControlChart(baseline_n=6)
        self.alpha = alpha
        self.cycle_idx = 0
        self.events: List[Dict[str, Any]] = []

    def _emit(self, event: Dict[str, Any]) -> None:
        event["emitted_at"] = utcnow()
        self.events.append(event)
        log.info("EVENT %s", json.dumps(event, default=str)[:280])

    def seed(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        generations: int = 4,
        val_size: float = 0.3,
        calib_size: float = 0.2,
        seed: int = 42,
    ) -> None:
        """Train the initial population and calibrate the conformal predictor."""
        X_rem, X_cal, y_rem, y_cal = train_test_split(
            X.values, y, test_size=calib_size, stratify=y, random_state=seed
        )
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_rem, y_rem, test_size=val_size, stratify=y_rem, random_state=seed
        )
        for g in range(generations):
            self.autopilot.evolve_one_generation(X_tr, y_tr, X_va, y_va)
            best = self.autopilot.best()
            log.info(
                f"seed gen {g + 1}/{generations}: best worker={best.worker_id} "
                f"auc={best.fitness:.4f} diversity={self.autopilot.diversity():.3f}"
            )

        best = self.autopilot.best()
        self.conformal.fit(best.model, X_cal, y_cal)
        self._emit({
            "event_type": "seed_completed",
            "best_worker_id": best.worker_id,
            "best_hyperparameters": best.hyperparameters,
            "best_metrics": best.metrics,
            "conformal_q_hat": self.conformal.q_hat,
            "conformal_target_coverage": 1.0 - self.alpha,
            "calibration_n": self.conformal.n_calib,
            "population_diversity": self.autopilot.diversity(),
        })

    def step(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """One streaming cycle on a new labeled batch."""
        self.cycle_idx += 1
        cycle_id = short_id("c")
        best = self.autopilot.best()

        proba = best.model.predict_proba(X.values)[:, 1]
        auc = float(roc_auc_score(y, proba))
        ll = safe_log_loss(y, proba)

        sets = self.conformal.predict_set(best.model, X.values)
        coverage = self.conformal.coverage(sets, y)
        avg_set_size = float(np.mean([len(s) for s in sets]))

        ph_fired = self.ph_loss.update(ll)
        auc_status = self.auc_chart.update(auc)
        cov_status = self.coverage_chart.update(coverage)

        out_of_control = (
            ph_fired
            or not auc_status["in_control"]
            or not cov_status["in_control"]
        )

        action = "stable"
        if out_of_control:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X.values, y, test_size=0.3, stratify=y,
                random_state=1000 + self.cycle_idx,
            )
            self.autopilot.evolve_one_generation(X_tr, y_tr, X_va, y_va)
            new_best = self.autopilot.best()
            self.conformal.fit(new_best.model, X_va, y_va)
            self.ph_loss.reset()
            action = f"evolved; new_best={new_best.worker_id} auc={new_best.fitness:.4f}"

        cycle_event = {
            "event_type": "cycle",
            "cycle_id": cycle_id,
            "cycle_index": self.cycle_idx,
            "primary_worker_id": best.worker_id,
            "auc": auc,
            "log_loss": ll,
            "conformal_coverage": coverage,
            "conformal_target_coverage": 1.0 - self.alpha,
            "avg_set_size": avg_set_size,
            "page_hinkley_triggered": ph_fired,
            "auc_chart": auc_status,
            "coverage_chart": cov_status,
            "out_of_control": out_of_control,
            "population_diversity": self.autopilot.diversity(),
            "generation": self.autopilot.generation,
            "action": action,
        }
        self._emit(cycle_event)
        return cycle_event


# ---------------------------------------------------------------------------
# 8. Demo
# ---------------------------------------------------------------------------

def make_genome(rng: np.random.Generator) -> HPGenome:
    spec = {
        "n_estimators": {"type": "int", "range": (40, 220), "mutation_rate": 0.4},
        "learning_rate": {"type": "float", "range": (1e-3, 0.3), "log": True, "mutation_rate": 0.5},
        "max_depth": {"type": "int", "range": (2, 6), "mutation_rate": 0.4},
        "subsample": {"type": "float", "range": (0.5, 1.0), "mutation_rate": 0.3},
        "min_samples_leaf": {"type": "int", "range": (1, 10), "mutation_rate": 0.3},
    }
    return HPGenome(spec, rng)


def gb_factory(hp: Dict[str, Any]) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=hp["n_estimators"],
        learning_rate=hp["learning_rate"],
        max_depth=hp["max_depth"],
        subsample=hp["subsample"],
        min_samples_leaf=hp["min_samples_leaf"],
        random_state=42,
    )


def synthesize(
    n: int, drift_strength: float = 0.0, seed: int = 0
) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_features = 6
    X = rng.normal(
        loc=drift_strength * 0.5,
        scale=1.0 + drift_strength * 0.2,
        size=(n, n_features),
    )
    weights = np.array([1.2, -0.8, 0.6, 0.4, -1.0, 0.3])
    if drift_strength > 0.5:
        weights = weights + rng.normal(0.0, drift_strength * 0.3, size=weights.shape)
    logits = X @ weights - 0.3
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < probs).astype(int)
    cols = [f"feature_{i:02d}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), y


def main() -> None:
    random.seed(42)
    rng = np.random.default_rng(42)
    genome = make_genome(rng)
    system = EvolutionaryMLSystem(
        model_factory=gb_factory,
        genome=genome,
        population_size=8,
        alpha=0.1,
    )

    log.info("=== seeding population on stationary data ===")
    X0, y0 = synthesize(n=1500, drift_strength=0.0, seed=0)
    system.seed(X0, y0, generations=3)

    log.info("=== streaming cycles with a programmed drift schedule ===")
    drift_schedule = [0.0, 0.05, 0.1, 0.2, 0.4, 0.9, 1.3, 1.4, 1.4, 1.4]
    print()
    print(
        f"{'cycle':>5} {'drift':>6} {'auc':>6} {'cov':>6} "
        f"{'avg|set|':>9} {'PH':>3} {'ooc':>4} action"
    )
    print("-" * 80)
    for i, d in enumerate(drift_schedule):
        Xb, yb = synthesize(n=400, drift_strength=d, seed=100 + i)
        r = system.step(Xb, yb)
        print(
            f"{i + 1:>5} {d:>6.2f} {r['auc']:>6.3f} {r['conformal_coverage']:>6.3f} "
            f"{r['avg_set_size']:>9.2f} {'Y' if r['page_hinkley_triggered'] else '.':>3} "
            f"{'OOC' if r['out_of_control'] else 'ok':>4} {r['action']}"
        )


if __name__ == "__main__":
    main()
