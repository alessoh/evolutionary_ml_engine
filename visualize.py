"""
Visualization utilities for the Evolutionary ML Engine.

Generates publication-quality charts showing population evolution,
drift detection, and performance metrics over time.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def parse_event_logs(log_file: str = "events.log") -> pd.DataFrame:
    """Parse EVENT logs from stdout or a saved log file."""
    events = []
    
    # If log file exists, read from it
    if Path(log_file).exists():
        with open(log_file, 'r') as f:
            for line in f:
                if "EVENT {" in line:
                    json_str = line.split("EVENT ")[1]
                    events.append(json.loads(json_str))
    
    return pd.DataFrame(events) if events else None


def plot_population_evolution(events: List[Dict[str, Any]], save_path: str = "population_evolution.png"):
    """Plot population fitness and diversity over generations."""
    seed_events = [e for e in events if e.get("event_type") == "seed_completed"]
    cycle_events = [e for e in events if e.get("event_type") == "cycle"]
    
    if not cycle_events:
        print("No cycle events found for plotting")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Evolutionary ML Engine Performance", fontsize=16, fontweight='bold')
    
    # Extract metrics
    cycles = [e["cycle_index"] for e in cycle_events]
    auc = [e["auc"] for e in cycle_events]
    coverage = [e["conformal_coverage"] for e in cycle_events]
    target_coverage = cycle_events[0]["conformal_target_coverage"]
    diversity = [e["population_diversity"] for e in cycle_events]
    ooc = [e["out_of_control"] for e in cycle_events]
    
    # Plot 1: Model Performance (AUC)
    ax1.plot(cycles, auc, 'b-', linewidth=2, label='ROC AUC')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
    
    # Highlight out-of-control points
    ooc_cycles = [c for c, o in zip(cycles, ooc) if o]
    ooc_auc = [a for a, o in zip(auc, ooc) if o]
    ax1.scatter(ooc_cycles, ooc_auc, color='red', s=100, zorder=5, 
                label='Retraining Event', marker='X')
    
    ax1.set_ylabel('ROC AUC', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Over Time', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.0])
    
    # Plot 2: Conformal Coverage
    ax2.plot(cycles, coverage, 'g-', linewidth=2, label='Actual Coverage')
    ax2.axhline(y=target_coverage, color='orange', linestyle='--', linewidth=2, 
                label=f'Target Coverage ({target_coverage:.0%})')
    
    # Add tolerance bands
    ax2.axhspan(target_coverage - 0.05, target_coverage + 0.05, 
                alpha=0.2, color='orange', label='Tolerance Band')
    
    # Highlight out-of-control points
    ooc_cov = [c for c, o in zip(coverage, ooc) if o]
    ax2.scatter(ooc_cycles, ooc_cov, color='red', s=100, zorder=5, marker='X')
    
    ax2.set_ylabel('Coverage', fontsize=12, fontweight='bold')
    ax2.set_title('Conformal Prediction Coverage', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.7, 1.0])
    
    # Plot 3: Population Diversity
    ax3.plot(cycles, diversity, 'purple', linewidth=2, label='Population Diversity')
    ax3.scatter(ooc_cycles, [d for d, o in zip(diversity, ooc) if o], 
                color='red', s=100, zorder=5, marker='X', label='Retraining Event')
    
    ax3.set_xlabel('Cycle', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Diversity', fontsize=12, fontweight='bold')
    ax3.set_title('Population Hyperparameter Diversity', fontsize=14)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_drift_detection(events: List[Dict[str, Any]], save_path: str = "drift_detection.png"):
    """Plot drift detection signals and control chart status."""
    cycle_events = [e for e in events if e.get("event_type") == "cycle"]
    
    if not cycle_events:
        print("No cycle events found for plotting")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Drift Detection and Statistical Process Control", 
                 fontsize=16, fontweight='bold')
    
    cycles = [e["cycle_index"] for e in cycle_events]
    log_loss = [e["log_loss"] for e in cycle_events]
    ph_triggered = [e["page_hinkley_triggered"] for e in cycle_events]
    
    # Extract control chart info
    auc_chart_fired = []
    cov_chart_fired = []
    for e in cycle_events:
        auc_chart_fired.append(not e["auc_chart"]["in_control"])
        cov_chart_fired.append(not e["coverage_chart"]["in_control"])
    
    # Plot 1: Log Loss with Page-Hinkley Detection
    ax1.plot(cycles, log_loss, 'b-', linewidth=2, label='Log Loss')
    
    # Highlight Page-Hinkley triggers
    ph_cycles = [c for c, p in zip(cycles, ph_triggered) if p]
    ph_loss = [ll for ll, p in zip(log_loss, ph_triggered) if p]
    ax1.scatter(ph_cycles, ph_loss, color='red', s=150, zorder=5, 
                marker='^', label='Page-Hinkley Alert')
    
    ax1.set_ylabel('Log Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Sequential Change Detection (Page-Hinkley)', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Control Chart Violations
    width = 0.35
    x = np.arange(len(cycles))
    
    ax2.bar(x - width/2, auc_chart_fired, width, label='AUC Chart Alert', 
            color='orange', alpha=0.7)
    ax2.bar(x + width/2, cov_chart_fired, width, label='Coverage Chart Alert', 
            color='purple', alpha=0.7)
    
    ax2.set_xlabel('Cycle', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Alert Status', fontsize=12, fontweight='bold')
    ax2.set_title('Control Chart Violations', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cycles)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['In Control', 'Out of Control'])
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_adaptation_timeline(events: List[Dict[str, Any]], save_path: str = "adaptation_timeline.png"):
    """Create a timeline showing when the system adapted."""
    cycle_events = [e for e in events if e.get("event_type") == "cycle"]
    
    if not cycle_events:
        print("No cycle events found for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("System Adaptation Timeline", fontsize=16, fontweight='bold')
    
    cycles = [e["cycle_index"] for e in cycle_events]
    auc = [e["auc"] for e in cycle_events]
    actions = [e["action"] for e in cycle_events]
    
    # Plot performance
    ax.plot(cycles, auc, 'b-', linewidth=2, marker='o', markersize=6, label='ROC AUC')
    
    # Mark adaptation events
    adapted_cycles = []
    adapted_auc = []
    for c, a, action in zip(cycles, auc, actions):
        if action.startswith("evolved"):
            adapted_cycles.append(c)
            adapted_auc.append(a)
    
    if adapted_cycles:
        ax.scatter(adapted_cycles, adapted_auc, color='red', s=200, zorder=5, 
                  marker='*', label='Auto-Adaptation Event', edgecolors='darkred', linewidths=2)
    
    ax.set_xlabel('Cycle', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROC AUC', fontsize=12, fontweight='bold')
    ax.set_title('Performance with Automatic Adaptations Marked', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    # Add annotation for each adaptation
    for i, (c, a) in enumerate(zip(adapted_cycles, adapted_auc)):
        ax.annotate(f'Adapt #{i+1}', xy=(c, a), xytext=(c, a + 0.05),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=10, ha='center', color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def generate_all_visualizations(events: List[Dict[str, Any]]):
    """Generate all visualization plots from event data."""
    print("Generating visualizations...")
    
    plot_population_evolution(events, "population_evolution.png")
    plot_drift_detection(events, "drift_detection.png")
    plot_adaptation_timeline(events, "adaptation_timeline.png")
    
    print("\nAll visualizations generated successfully!")
    print("Files created:")
    print("  - population_evolution.png")
    print("  - drift_detection.png")
    print("  - adaptation_timeline.png")


def main():
    """Main entry point for visualization script."""
    import sys
    
    # Import the main engine to run it and capture events
    from evolutionary_ml_engine import EvolutionaryMLSystem, make_genome, gb_factory, synthesize
    import random
    import numpy as np
    
    print("Running Evolutionary ML Engine to generate data...")
    random.seed(42)
    rng = np.random.default_rng(42)
    genome = make_genome(rng)
    system = EvolutionaryMLSystem(
        model_factory=gb_factory,
        genome=genome,
        population_size=8,
        alpha=0.1,
    )
    
    # Seed the system
    from evolutionary_ml_engine import synthesize
    X0, y0 = synthesize(n=1500, drift_strength=0.0, seed=0)
    system.seed(X0, y0, generations=3)
    
    # Run streaming cycles
    drift_schedule = [0.0, 0.05, 0.1, 0.2, 0.4, 0.9, 1.3, 1.4, 1.4, 1.4]
    for i, d in enumerate(drift_schedule):
        Xb, yb = synthesize(n=400, drift_strength=d, seed=100 + i)
        system.step(Xb, yb)
    
    print("\n" + "="*80)
    print("Generating visualizations from event data...")
    print("="*80 + "\n")
    
    # Generate visualizations
    generate_all_visualizations(system.events)


if __name__ == "__main__":
    main()
