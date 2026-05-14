# -*- coding: utf-8 -*-
"""
Interactive Streamlit Dashboard for the Evolutionary ML Engine.

Provides real-time visualization of population evolution, drift detection,
and performance metrics with interactive parameter controls.

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from evolutionary_ml_engine import (
    EvolutionaryMLSystem,
    make_genome,
    gb_factory,
    synthesize,
)


# Page configuration
st.set_page_config(
    page_title="Evolutionary ML Engine Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_system(population_size, alpha, seed_generations):
    """Initialize the evolutionary ML system."""
    random.seed(42)
    rng = np.random.default_rng(42)
    genome = make_genome(rng)
    
    system = EvolutionaryMLSystem(
        model_factory=gb_factory,
        genome=genome,
        population_size=population_size,
        alpha=alpha,
    )
    
    # Seed the system
    X0, y0 = synthesize(n=1500, drift_strength=0.0, seed=0)
    with st.spinner("Seeding population..."):
        system.seed(X0, y0, generations=seed_generations)
    
    return system


def run_streaming_cycles(system, drift_schedule):
    """Run streaming cycles with the given drift schedule."""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, d in enumerate(drift_schedule):
        status_text.text(f"Processing cycle {i+1}/{len(drift_schedule)} (drift={d:.2f})")
        
        Xb, yb = synthesize(n=400, drift_strength=d, seed=100 + i)
        result = system.step(Xb, yb)
        results.append(result)
        
        progress_bar.progress((i + 1) / len(drift_schedule))
    
    status_text.text("All cycles completed!")
    return pd.DataFrame(results)


def plot_performance_metrics(df):
    """Create interactive plot of performance metrics."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Model Performance (ROC AUC)", 
                       "Conformal Coverage",
                       "Population Diversity"),
        vertical_spacing=0.12,
        row_heights=[0.33, 0.33, 0.34],
    )
    
    # ROC AUC
    fig.add_trace(
        go.Scatter(x=df["cycle_index"], y=df["auc"],
                  mode='lines+markers',
                  name='ROC AUC',
                  line=dict(color='royalblue', width=2),
                  marker=dict(size=6)),
        row=1, col=1
    )
    
    # Highlight retraining events
    retraining = df[df["out_of_control"] == True]
    if not retraining.empty:
        fig.add_trace(
            go.Scatter(x=retraining["cycle_index"], y=retraining["auc"],
                      mode='markers',
                      name='Retraining Event',
                      marker=dict(size=12, color='red', symbol='star')),
            row=1, col=1
        )
    
    # Random baseline
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                 annotation_text="Random", row=1, col=1)
    
    # Conformal Coverage
    fig.add_trace(
        go.Scatter(x=df["cycle_index"], y=df["conformal_coverage"],
                  mode='lines+markers',
                  name='Actual Coverage',
                  line=dict(color='green', width=2),
                  marker=dict(size=6)),
        row=2, col=1
    )
    
    target = df["conformal_target_coverage"].iloc[0]
    fig.add_hline(y=target, line_dash="dash", line_color="orange",
                 annotation_text=f"Target ({target:.0%})", row=2, col=1)
    
    # Add tolerance band
    fig.add_hrect(y0=target-0.05, y1=target+0.05, 
                 fillcolor="orange", opacity=0.1,
                 line_width=0, row=2, col=1)
    
    if not retraining.empty:
        fig.add_trace(
            go.Scatter(x=retraining["cycle_index"], 
                      y=retraining["conformal_coverage"],
                      mode='markers',
                      name='Retraining Event',
                      marker=dict(size=12, color='red', symbol='star'),
                      showlegend=False),
            row=2, col=1
        )
    
    # Population Diversity
    fig.add_trace(
        go.Scatter(x=df["cycle_index"], y=df["population_diversity"],
                  mode='lines+markers',
                  name='Diversity',
                  line=dict(color='purple', width=2),
                  marker=dict(size=6)),
        row=3, col=1
    )
    
    if not retraining.empty:
        fig.add_trace(
            go.Scatter(x=retraining["cycle_index"], 
                      y=retraining["population_diversity"],
                      mode='markers',
                      name='Retraining Event',
                      marker=dict(size=12, color='red', symbol='star'),
                      showlegend=False),
            row=3, col=1
        )
    
    fig.update_xaxes(title_text="Cycle", row=3, col=1)
    fig.update_yaxes(title_text="ROC AUC", row=1, col=1, range=[0.5, 1.0])
    fig.update_yaxes(title_text="Coverage", row=2, col=1, range=[0.7, 1.0])
    fig.update_yaxes(title_text="Diversity", row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, hovermode='x unified')
    
    return fig


def plot_drift_detection(df):
    """Create interactive plot of drift detection signals."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Log Loss & Page-Hinkley Detection",
                       "Control Chart Violations"),
        vertical_spacing=0.15,
    )
    
    # Log Loss
    fig.add_trace(
        go.Scatter(x=df["cycle_index"], y=df["log_loss"],
                  mode='lines+markers',
                  name='Log Loss',
                  line=dict(color='blue', width=2),
                  marker=dict(size=6)),
        row=1, col=1
    )
    
    # Page-Hinkley triggers
    ph_triggered = df[df["page_hinkley_triggered"] == True]
    if not ph_triggered.empty:
        fig.add_trace(
            go.Scatter(x=ph_triggered["cycle_index"], 
                      y=ph_triggered["log_loss"],
                      mode='markers',
                      name='Page-Hinkley Alert',
                      marker=dict(size=14, color='red', symbol='triangle-up')),
            row=1, col=1
        )
    
    # Control Chart Status
    auc_violations = []
    cov_violations = []
    for _, row in df.iterrows():
        auc_violations.append(0 if row["auc_chart"]["in_control"] else 1)
        cov_violations.append(0 if row["coverage_chart"]["in_control"] else 1)
    
    fig.add_trace(
        go.Bar(x=df["cycle_index"], y=auc_violations,
              name='AUC Chart Alert',
              marker_color='orange',
              opacity=0.7),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=df["cycle_index"], y=cov_violations,
              name='Coverage Chart Alert',
              marker_color='purple',
              opacity=0.7),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Cycle", row=2, col=1)
    fig.update_yaxes(title_text="Log Loss", row=1, col=1)
    fig.update_yaxes(title_text="Status", row=2, col=1,
                    ticktext=['In Control', 'Out of Control'],
                    tickvals=[0, 1])
    
    fig.update_layout(height=600, showlegend=True, barmode='group')
    
    return fig


def main():
    """Main dashboard application."""
    
    # Title and description
    st.title("Evolutionary ML Engine Dashboard")
    st.markdown("""
    **Autonomous machine learning that adapts itself to changing data without human intervention**
    
    This dashboard demonstrates the Evolutionary ML Engine detecting and responding to data drift
    through population-based training, conformal prediction, and statistical process control.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    st.sidebar.subheader("System Parameters")
    population_size = st.sidebar.slider("Population Size", 4, 20, 8, 
                                       help="Number of hyperparameter configurations")
    alpha = st.sidebar.slider("Conformal Alpha", 0.05, 0.20, 0.10, 0.05,
                              help="Target coverage = 1 - alpha")
    seed_generations = st.sidebar.slider("Seed Generations", 2, 10, 3,
                                        help="Initial evolution rounds")
    
    st.sidebar.subheader("Drift Scenario")
    scenario = st.sidebar.selectbox(
        "Select Scenario",
        ["Gradual Drift", "Sudden Shift", "Oscillating", "Extreme Drift", "Custom"]
    )
    
    if scenario == "Gradual Drift":
        drift_schedule = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0]
    elif scenario == "Sudden Shift":
        drift_schedule = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    elif scenario == "Oscillating":
        drift_schedule = [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5]
    elif scenario == "Extreme Drift":
        drift_schedule = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0]
    else:  # Custom
        st.sidebar.markdown("**Custom Drift Values** (comma-separated)")
        custom_input = st.sidebar.text_input("Drift schedule", 
                                             "0.0,0.1,0.2,0.4,0.8,1.2,1.4,1.4,1.4,1.4")
        drift_schedule = [float(x.strip()) for x in custom_input.split(",")]
    
    st.sidebar.markdown(f"**Cycles:** {len(drift_schedule)}")
    st.sidebar.markdown(f"**Max Drift:** {max(drift_schedule):.2f}")
    
    # Run button
    run_button = st.sidebar.button("Run Simulation", type="primary")
    
    # Main content
    if run_button:
        # Initialize system
        st.header("System Initialization")
        system = initialize_system(population_size, alpha, seed_generations)
        
        seed_event = [e for e in system.events if e["event_type"] == "seed_completed"][0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best ROC AUC", f"{seed_event['best_metrics']['roc_auc']:.4f}")
        with col2:
            st.metric("Conformal q-hat", f"{seed_event['conformal_q_hat']:.4f}")
        with col3:
            st.metric("Population Diversity", f"{seed_event['population_diversity']:.3f}")
        
        with st.expander("View Best Hyperparameters"):
            st.json(seed_event['best_hyperparameters'])
        
        # Run streaming cycles
        st.header("Streaming Cycles")
        results_df = run_streaming_cycles(system, drift_schedule)
        
        # Summary metrics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            retraining_count = results_df["out_of_control"].sum()
            st.metric("Retraining Events", retraining_count)
        
        with col2:
            avg_auc = results_df["auc"].mean()
            st.metric("Average ROC AUC", f"{avg_auc:.4f}")
        
        with col3:
            avg_coverage = results_df["conformal_coverage"].mean()
            target = results_df["conformal_target_coverage"].iloc[0]
            delta = avg_coverage - target
            st.metric("Average Coverage", f"{avg_coverage:.4f}", 
                     delta=f"{delta:+.4f}")
        
        with col4:
            final_diversity = results_df["population_diversity"].iloc[-1]
            st.metric("Final Diversity", f"{final_diversity:.3f}")
        
        # Performance plots
        st.header("Performance Metrics")
        st.plotly_chart(plot_performance_metrics(results_df), use_container_width=True)
        
        # Drift detection plots
        st.header("Drift Detection")
        st.plotly_chart(plot_drift_detection(results_df), use_container_width=True)
        
        # Detailed results table
        st.header("Detailed Results")
        
        display_df = results_df[[
            "cycle_index", "auc", "log_loss", "conformal_coverage",
            "avg_set_size", "page_hinkley_triggered", "out_of_control",
            "population_diversity", "generation"
        ]].copy()
        
        display_df.columns = [
            "Cycle", "ROC AUC", "Log Loss", "Coverage", "Avg Set Size",
            "PH Alert", "Out of Control", "Diversity", "Generation"
        ]
        
        st.dataframe(
            display_df.style.format({
                "ROC AUC": "{:.4f}",
                "Log Loss": "{:.4f}",
                "Coverage": "{:.4f}",
                "Avg Set Size": "{:.2f}",
                "Diversity": "{:.3f}",
            }).background_gradient(subset=["ROC AUC"], cmap="RdYlGn"),
            hide_index=True,
            use_container_width=True
        )
        
        # Export data
        st.header("Export Data")
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="evolutionary_ml_results.csv",
            mime="text/csv"
        )
        
    else:
        # Instructions when not running
        st.info("""
        Configure the system parameters and drift scenario in the sidebar, 
        then click **Run Simulation** to see the Evolutionary ML Engine in action!
        
        **What to expect:**
        - Initial population seeding and evolution
        - Real-time streaming cycles with configurable drift
        - Automatic detection and adaptation to distribution shifts
        - Interactive visualizations of all key metrics
        """)
        
        st.markdown("---")
        st.markdown("""
        ### About This Dashboard
        
        The Evolutionary ML Engine combines three powerful techniques:
        
        1. **Population-Based Training**: Maintains diverse hyperparameter configurations 
           that evolve toward better performance
        2. **Conformal Prediction**: Provides distribution-free uncertainty estimates 
           with mathematical coverage guarantees
        3. **Statistical Process Control**: Detects drift using Page-Hinkley tests 
           and Shewhart control charts
        
        When drift is detected, the system automatically evolves new hyperparameters 
        and recalibrates its uncertainty estimates - no human intervention required.
        """)


if __name__ == "__main__":
    main()
