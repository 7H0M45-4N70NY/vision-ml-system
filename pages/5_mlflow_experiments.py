"""MLflow Experiments Dashboard - Experiment tracking and run comparison."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.vision_ml.mlflow_integration import MLflowManager
from src.vision_ml.analytics.analytics_db import AnalyticsDB

st.set_page_config(page_title="MLflow Experiments", page_icon="📈", layout="wide")

st.title("📈 MLflow Experiments")
st.markdown("Track, compare, and manage ML experiments with DagsHub")

# Initialize managers
mlflow_manager = MLflowManager()
db = AnalyticsDB()

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    
    view_type = st.radio(
        "View Type",
        ["Experiments", "Runs", "Best Models", "Model Registry"],
        horizontal=False
    )
    
    if view_type in ["Runs", "Best Models"]:
        time_range = st.selectbox(
            "Time Range",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"]
        )
    
    if view_type == "Runs":
        sort_by = st.selectbox(
            "Sort By",
            ["val_loss (ascending)", "train_loss (ascending)", "accuracy (descending)", "timestamp (newest)"]
        )


# ── Experiments View ──────────────────────────────────────────────

if view_type == "Experiments":
    st.subheader("📂 All Experiments")
    
    try:
        # Get all experiments
        client = mlflow_manager.client
        experiments = client.search_experiments()
        
        if experiments:
            # Create experiments table
            exp_data = []
            for exp in experiments:
                runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1)
                exp_data.append({
                    "Experiment": exp.name,
                    "Experiment ID": exp.experiment_id,
                    "Runs": len(client.search_runs(experiment_ids=[exp.experiment_id])),
                    "Created": datetime.fromtimestamp(exp.creation_time / 1000).strftime("%Y-%m-%d %H:%M"),
                })
            
            df_exp = pd.DataFrame(exp_data)
            st.dataframe(df_exp, use_container_width=True, hide_index=True)
            
            # Detailed view
            st.divider()
            st.subheader("📊 Experiment Details")
            
            selected_exp_name = st.selectbox(
                "Select Experiment",
                options=[e["Experiment"] for e in exp_data]
            )
            
            if selected_exp_name:
                selected_exp = next(e for e in experiments if e.name == selected_exp_name)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Experiment ID", selected_exp.experiment_id)
                with col2:
                    runs = client.search_runs(experiment_ids=[selected_exp.experiment_id])
                    st.metric("Total Runs", len(runs))
                with col3:
                    st.metric("Created", datetime.fromtimestamp(selected_exp.creation_time / 1000).strftime("%Y-%m-%d"))
                
                # Show runs in this experiment
                st.markdown("**Recent Runs**")
                runs = client.search_runs(
                    experiment_ids=[selected_exp.experiment_id],
                    max_results=20,
                    order_by=["start_time DESC"]
                )
                
                if runs:
                    runs_data = []
                    for run in runs:
                        runs_data.append({
                            "Run ID": run.info.run_id[:8],
                            "Status": run.info.status,
                            "Start Time": datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
                            "Duration (s)": (run.info.end_time - run.info.start_time) // 1000 if run.info.end_time else "Running",
                            "Val Loss": f"{run.data.metrics.get('val_loss', 'N/A'):.4f}" if 'val_loss' in run.data.metrics else "N/A",
                            "Accuracy": f"{run.data.metrics.get('accuracy', 'N/A'):.4f}" if 'accuracy' in run.data.metrics else "N/A",
                        })
                    
                    df_runs = pd.DataFrame(runs_data)
                    st.dataframe(df_runs, use_container_width=True, hide_index=True)
        else:
            st.info("No experiments found. Start training to create experiments.")
    
    except Exception as e:
        st.error(f"❌ Error loading experiments: {str(e)}")


# ── Runs View ──────────────────────────────────────────────────────

elif view_type == "Runs":
    st.subheader("🏃 Training Runs")
    
    try:
        client = mlflow_manager.client
        experiments = client.search_experiments()
        
        # Collect all runs from all experiments
        all_runs = []
        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=100)
            for run in runs:
                all_runs.append({
                    "experiment": exp.name,
                    "run": run
                })
        
        if all_runs:
            # Sort by selected metric
            if "val_loss" in sort_by:
                all_runs.sort(key=lambda x: x["run"].data.metrics.get("val_loss", float('inf')))
            elif "train_loss" in sort_by:
                all_runs.sort(key=lambda x: x["run"].data.metrics.get("train_loss", float('inf')))
            elif "accuracy" in sort_by:
                all_runs.sort(key=lambda x: x["run"].data.metrics.get("accuracy", 0), reverse=True)
            else:
                all_runs.sort(key=lambda x: x["run"].info.start_time, reverse=True)
            
            # Display runs table
            runs_data = []
            for item in all_runs[:50]:  # Show top 50
                run = item["run"]
                runs_data.append({
                    "Experiment": item["experiment"],
                    "Run ID": run.info.run_id[:8],
                    "Status": run.info.status,
                    "Start Time": datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
                    "Train Loss": f"{run.data.metrics.get('train_loss', 'N/A'):.4f}" if 'train_loss' in run.data.metrics else "N/A",
                    "Val Loss": f"{run.data.metrics.get('val_loss', 'N/A'):.4f}" if 'val_loss' in run.data.metrics else "N/A",
                    "Accuracy": f"{run.data.metrics.get('accuracy', 'N/A'):.4f}" if 'accuracy' in run.data.metrics else "N/A",
                })
            
            df_runs = pd.DataFrame(runs_data)
            st.dataframe(df_runs, use_container_width=True, hide_index=True)
            
            # Detailed run view
            st.divider()
            st.subheader("📋 Run Details")
            
            selected_run_id = st.selectbox(
                "Select Run",
                options=[r["Run ID"] for r in runs_data],
                format_func=lambda x: f"{x} - {next(r['Experiment'] for r in runs_data if r['Run ID'] == x)}"
            )
            
            if selected_run_id:
                # Find full run ID
                full_run_id = next(
                    item["run"].info.run_id 
                    for item in all_runs 
                    if item["run"].info.run_id.startswith(selected_run_id)
                )
                
                run = client.get_run(full_run_id)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Status", run.info.status)
                with col2:
                    duration = (run.info.end_time - run.info.start_time) // 1000 if run.info.end_time else "Running"
                    st.metric("Duration (s)", duration)
                with col3:
                    st.metric("Val Loss", f"{run.data.metrics.get('val_loss', 'N/A'):.4f}")
                with col4:
                    st.metric("Accuracy", f"{run.data.metrics.get('accuracy', 'N/A'):.4f}")
                
                # Parameters
                st.markdown("**Hyperparameters**")
                params_df = pd.DataFrame([
                    {"Parameter": k, "Value": v}
                    for k, v in run.data.params.items()
                ])
                st.dataframe(params_df, use_container_width=True, hide_index=True)
                
                # Metrics
                st.markdown("**Metrics**")
                metrics_df = pd.DataFrame([
                    {"Metric": k, "Value": f"{v:.4f}"}
                    for k, v in run.data.metrics.items()
                ])
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                # Tags
                if run.data.tags:
                    st.markdown("**Tags**")
                    tags_df = pd.DataFrame([
                        {"Tag": k, "Value": v}
                        for k, v in run.data.tags.items()
                    ])
                    st.dataframe(tags_df, use_container_width=True, hide_index=True)
        else:
            st.info("No runs found. Start training to create runs.")
    
    except Exception as e:
        st.error(f"❌ Error loading runs: {str(e)}")


# ── Best Models View ──────────────────────────────────────────────

elif view_type == "Best Models":
    st.subheader("🏆 Best Models by Experiment")
    
    try:
        client = mlflow_manager.client
        experiments = client.search_experiments()
        
        best_models = []
        
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=100,
                order_by=["metrics.val_loss ASC"]
            )
            
            if runs:
                best_run = runs[0]
                best_models.append({
                    "Experiment": exp.name,
                    "Best Run ID": best_run.info.run_id[:8],
                    "Val Loss": f"{best_run.data.metrics.get('val_loss', 'N/A'):.4f}" if 'val_loss' in best_run.data.metrics else "N/A",
                    "Accuracy": f"{best_run.data.metrics.get('accuracy', 'N/A'):.4f}" if 'accuracy' in best_run.data.metrics else "N/A",
                    "Status": best_run.info.status,
                })
        
        if best_models:
            df_best = pd.DataFrame(best_models)
            st.dataframe(df_best, use_container_width=True, hide_index=True)
            
            # Visualization
            if HAS_PLOTLY and len(best_models) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        df_best,
                        x="Experiment",
                        y="Val Loss",
                        title="Best Val Loss by Experiment",
                        labels={"Val Loss": "Validation Loss"}
                    )
                    fig.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        df_best,
                        x="Experiment",
                        y="Accuracy",
                        title="Best Accuracy by Experiment",
                        labels={"Accuracy": "Accuracy"}
                    )
                    fig.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No models found.")
    
    except Exception as e:
        st.error(f"❌ Error loading best models: {str(e)}")


# ── Model Registry View ──────────────────────────────────────────

elif view_type == "Model Registry":
    st.subheader("🎯 Model Registry")
    
    try:
        client = mlflow_manager.client
        registered_models = client.list_registered_models()
        
        if registered_models:
            for model in registered_models:
                with st.expander(f"📦 {model.name}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Versions", len(model.latest_versions))
                    
                    with col2:
                        # Find production version
                        prod_versions = [v for v in model.latest_versions if v.current_stage == "Production"]
                        st.metric("Production Version", prod_versions[0].version if prod_versions else "N/A")
                    
                    with col3:
                        # Find staging version
                        staging_versions = [v for v in model.latest_versions if v.current_stage == "Staging"]
                        st.metric("Staging Version", staging_versions[0].version if staging_versions else "N/A")
                    
                    st.markdown("**Versions**")
                    
                    versions_data = []
                    for version in model.latest_versions:
                        run = client.get_run(version.run_id)
                        versions_data.append({
                            "Version": version.version,
                            "Stage": version.current_stage,
                            "Val Loss": f"{run.data.metrics.get('val_loss', 'N/A'):.4f}" if 'val_loss' in run.data.metrics else "N/A",
                            "Created": datetime.fromtimestamp(version.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M"),
                        })
                    
                    df_versions = pd.DataFrame(versions_data)
                    st.dataframe(df_versions, use_container_width=True, hide_index=True)
        else:
            st.info("No registered models found.")
    
    except Exception as e:
        st.error(f"❌ Error loading model registry: {str(e)}")


# Footer
st.divider()
st.caption("📈 MLflow Experiments Dashboard | Powered by DagsHub")
