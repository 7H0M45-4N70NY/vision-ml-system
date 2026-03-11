"""Model Registry & Deployment - Manage model versions and promotions."""

import streamlit as st
import pandas as pd
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.vision_ml.mlflow_integration import MLflowManager
from src.vision_ml.analytics.analytics_db import AnalyticsDB


def _fmt(val, fmt=".4f"):
    """Safely format a metric value; returns 'N/A' for missing/non-numeric."""
    if val is None or val == 'N/A':
        return "N/A"
    try:
        return f"{float(val):{fmt}}"
    except (TypeError, ValueError):
        return str(val)


st.title("🎯 Model Registry & Deployment")
st.markdown("Manage model versions, stages, and deployments")

# Initialize managers
mlflow_manager = MLflowManager()
db = AnalyticsDB()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    action_type = st.radio(
        "Action",
        ["View Models", "Promote Model", "Compare Versions", "Deployment History"],
        horizontal=False
    )


# ── View Models ──────────────────────────────────────────────────

if action_type == "View Models":
    st.subheader("📦 Registered Models")
    
    try:
        client = mlflow_manager.client
        registered_models = client.search_registered_models()
        
        if registered_models:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Models", len(registered_models))
            
            with col2:
                total_versions = sum(len(m.latest_versions) for m in registered_models)
                st.metric("Total Versions", total_versions)
            
            with col3:
                prod_count = sum(
                    1 for m in registered_models 
                    for v in m.latest_versions 
                    if v.current_stage == "Production"
                )
                st.metric("In Production", prod_count)
            
            with col4:
                staging_count = sum(
                    1 for m in registered_models 
                    for v in m.latest_versions 
                    if v.current_stage == "Staging"
                )
                st.metric("In Staging", staging_count)
            
            st.divider()
            
            # Models table
            models_data = []
            for model in registered_models:
                prod_versions = [v for v in model.latest_versions if v.current_stage == "Production"]
                staging_versions = [v for v in model.latest_versions if v.current_stage == "Staging"]
                
                models_data.append({
                    "Model Name": model.name,
                    "Total Versions": len(model.latest_versions),
                    "Production": prod_versions[0].version if prod_versions else "—",
                    "Staging": staging_versions[0].version if staging_versions else "—",
                    "Last Updated": datetime.fromtimestamp(model.last_updated_timestamp / 1000).strftime("%Y-%m-%d %H:%M"),
                })
            
            df_models = pd.DataFrame(models_data)
            st.dataframe(df_models, use_container_width=True, hide_index=True)
            
            st.divider()
            st.subheader("📊 Model Details")
            
            # Select model for detailed view
            selected_model_name = st.selectbox(
                "Select Model",
                options=[m.name for m in registered_models]
            )
            
            if selected_model_name:
                selected_model = next(m for m in registered_models if m.name == selected_model_name)
                
                # Model info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Name", selected_model.name)
                with col2:
                    st.metric("Total Versions", len(selected_model.latest_versions))
                with col3:
                    st.metric("Last Updated", datetime.fromtimestamp(selected_model.last_updated_timestamp / 1000).strftime("%Y-%m-%d"))
                
                # Versions by stage
                st.markdown("**Versions by Stage**")
                
                stages = {}
                for version in selected_model.latest_versions:
                    stage = version.current_stage
                    if stage not in stages:
                        stages[stage] = []
                    stages[stage].append(version)
                
                for stage in ["Production", "Staging", "Archived"]:
                    if stage in stages:
                        with st.expander(f"🔹 {stage} ({len(stages[stage])} versions)"):
                            versions_data = []
                            for version in stages[stage]:
                                run = client.get_run(version.run_id)
                                versions_data.append({
                                    "Version": version.version,
                                    "Run ID": version.run_id[:8],
                                    "Val Loss": f"{run.data.metrics.get('val_loss', 'N/A'):.4f}" if 'val_loss' in run.data.metrics else "N/A",
                                    "Accuracy": f"{run.data.metrics.get('accuracy', 'N/A'):.4f}" if 'accuracy' in run.data.metrics else "N/A",
                                    "Created": datetime.fromtimestamp(version.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M"),
                                })
                            
                            df_versions = pd.DataFrame(versions_data)
                            st.dataframe(df_versions, use_container_width=True, hide_index=True)
        else:
            st.info("No registered models found. Train a model and register it to see it here.")
    
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")


# ── Promote Model ────────────────────────────────────────────────

elif action_type == "Promote Model":
    st.subheader("🚀 Promote Model")
    
    try:
        client = mlflow_manager.client
        registered_models = client.search_registered_models()
        
        if registered_models:
            # Select model
            selected_model_name = st.selectbox(
                "Select Model",
                options=[m.name for m in registered_models]
            )
            
            if selected_model_name:
                selected_model = next(m for m in registered_models if m.name == selected_model_name)
                
                # Show current stages
                st.markdown("**Current Stages**")
                
                stages = {}
                for version in selected_model.latest_versions:
                    stage = version.current_stage
                    if stage not in stages:
                        stages[stage] = version
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    prod_version = stages.get("Production")
                    st.metric("Production", f"v{prod_version.version}" if prod_version else "—")
                with col2:
                    staging_version = stages.get("Staging")
                    st.metric("Staging", f"v{staging_version.version}" if staging_version else "—")
                with col3:
                    archived_versions = [v for v in selected_model.latest_versions if v.current_stage == "Archived"]
                    st.metric("Archived", len(archived_versions))
                
                st.divider()
                st.markdown("**Promotion Workflow**")
                
                # Select version to promote
                available_versions = [v for v in selected_model.latest_versions if v.current_stage != "Production"]
                
                if available_versions:
                    selected_version = st.selectbox(
                        "Select Version to Promote",
                        options=[v.version for v in available_versions],
                        format_func=lambda v: f"v{v} ({next(ver.current_stage for ver in available_versions if ver.version == v)})"
                    )
                    
                    if selected_version:
                        selected_ver_obj = next(v for v in available_versions if v.version == selected_version)
                        
                        # Show version details
                        run = client.get_run(selected_ver_obj.run_id)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Version", selected_version)
                        with col2:
                            st.metric("Current Stage", selected_ver_obj.current_stage)
                        with col3:
                            st.metric("Val Loss", _fmt(run.data.metrics.get('val_loss')))
                        
                        # Promotion target
                        st.markdown("**Promotion Target**")
                        
                        if selected_ver_obj.current_stage == "Staging":
                            target_stage = "Production"
                            st.info(f"✅ Ready to promote to {target_stage}")
                        else:
                            target_stage = st.selectbox(
                                "Target Stage",
                                ["Staging", "Production"]
                            )
                        
                        # Comparison with current production
                        if target_stage == "Production" and prod_version:
                            st.markdown("**Comparison with Current Production**")
                            
                            prod_run = client.get_run(prod_version.run_id)
                            
                            comparison_data = {
                                "Metric": ["Val Loss", "Accuracy", "Train Loss"],
                                "Current (v" + str(prod_version.version) + ")": [
                                    _fmt(prod_run.data.metrics.get('val_loss')),
                                    _fmt(prod_run.data.metrics.get('accuracy')),
                                    _fmt(prod_run.data.metrics.get('train_loss')),
                                ],
                                "Candidate (v" + str(selected_version) + ")": [
                                    _fmt(run.data.metrics.get('val_loss')),
                                    _fmt(run.data.metrics.get('accuracy')),
                                    _fmt(run.data.metrics.get('train_loss')),
                                ],
                            }
                            
                            df_comparison = pd.DataFrame(comparison_data)
                            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                            
                            # Improvement indicator
                            val_loss_current = prod_run.data.metrics.get('val_loss', float('inf'))
                            val_loss_candidate = run.data.metrics.get('val_loss', float('inf'))
                            
                            if val_loss_candidate < val_loss_current:
                                improvement = ((val_loss_current - val_loss_candidate) / val_loss_current) * 100
                                st.success(f"✅ Candidate is {improvement:.1f}% better!")
                            else:
                                degradation = ((val_loss_candidate - val_loss_current) / val_loss_current) * 100
                                st.warning(f"⚠️ Candidate is {degradation:.1f}% worse")
                        
                        # Promotion button
                        st.divider()
                        
                        if st.button(f"🚀 Promote v{selected_version} to {target_stage}", key="promote_btn"):
                            try:
                                mlflow_manager.promote_model(
                                    model_name=selected_model_name,
                                    from_stage=selected_ver_obj.current_stage,
                                    to_stage=target_stage
                                )
                                
                                # Save to analytics DB
                                db.save_training_event({
                                    'trigger_type': 'model_promotion',
                                    'dataset_size': 0,
                                    'drift_score': 0.0,
                                    'model_version': f"{selected_model_name}:v{selected_version}",
                                })
                                
                                st.success(f"✅ Model promoted to {target_stage}!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Promotion failed: {str(e)}")
                else:
                    st.info("All versions are already in Production.")
        else:
            st.info("No registered models found.")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


# ── Compare Versions ────────────────────────────────────────────

elif action_type == "Compare Versions":
    st.subheader("📊 Compare Model Versions")
    
    try:
        client = mlflow_manager.client
        registered_models = client.search_registered_models()
        
        if registered_models:
            selected_model_name = st.selectbox(
                "Select Model",
                options=[m.name for m in registered_models]
            )
            
            if selected_model_name:
                selected_model = next(m for m in registered_models if m.name == selected_model_name)
                
                # Select versions to compare
                col1, col2 = st.columns(2)
                
                with col1:
                    version1 = st.selectbox(
                        "Version 1",
                        options=[v.version for v in selected_model.latest_versions],
                        key="v1"
                    )
                
                with col2:
                    version2 = st.selectbox(
                        "Version 2",
                        options=[v.version for v in selected_model.latest_versions if v.version != version1],
                        key="v2"
                    )
                
                if version1 and version2:
                    ver1_obj = next(v for v in selected_model.latest_versions if v.version == version1)
                    ver2_obj = next(v for v in selected_model.latest_versions if v.version == version2)
                    
                    run1 = client.get_run(ver1_obj.run_id)
                    run2 = client.get_run(ver2_obj.run_id)
                    
                    # Comparison table
                    st.markdown("**Metrics Comparison**")
                    
                    comparison_data = {
                        "Metric": ["Val Loss", "Train Loss", "Accuracy", "Precision", "Recall", "F1 Score"],
                        f"v{version1}": [
                            _fmt(run1.data.metrics.get('val_loss')),
                            _fmt(run1.data.metrics.get('train_loss')),
                            _fmt(run1.data.metrics.get('accuracy')),
                            _fmt(run1.data.metrics.get('precision')),
                            _fmt(run1.data.metrics.get('recall')),
                            _fmt(run1.data.metrics.get('f1_score')),
                        ],
                        f"v{version2}": [
                            _fmt(run2.data.metrics.get('val_loss')),
                            _fmt(run2.data.metrics.get('train_loss')),
                            _fmt(run2.data.metrics.get('accuracy')),
                            _fmt(run2.data.metrics.get('precision')),
                            _fmt(run2.data.metrics.get('recall')),
                            _fmt(run2.data.metrics.get('f1_score')),
                        ],
                    }
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                    
                    # Parameters comparison
                    st.markdown("**Hyperparameters Comparison**")
                    
                    all_params = set(run1.data.params.keys()) | set(run2.data.params.keys())
                    
                    params_data = {
                        "Parameter": list(all_params),
                        f"v{version1}": [run1.data.params.get(p, "—") for p in all_params],
                        f"v{version2}": [run2.data.params.get(p, "—") for p in all_params],
                    }
                    
                    df_params = pd.DataFrame(params_data)
                    st.dataframe(df_params, use_container_width=True, hide_index=True)
        else:
            st.info("No registered models found.")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


# ── Deployment History ──────────────────────────────────────────

elif action_type == "Deployment History":
    st.subheader("📜 Deployment History")
    
    try:
        # Get training events (which include model promotions)
        training_events = db.get_training_events(limit=50)
        
        if training_events:
            df_events = pd.DataFrame(training_events)
            df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
            df_events = df_events.sort_values('timestamp', ascending=False)
            
            # Filter for promotions
            promotion_events = df_events[df_events['trigger_type'] == 'model_promotion']
            
            if not promotion_events.empty:
                st.dataframe(
                    promotion_events[[
                        'event_id', 'timestamp', 'model_version', 'status'
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )
                
                # Timeline visualization
                if HAS_PLOTLY:
                    fig = px.timeline(
                        promotion_events,
                        x_start='timestamp',
                        x_end='timestamp',
                        y='model_version',
                        title='Model Promotion Timeline',
                        labels={'timestamp': 'Date', 'model_version': 'Model'},
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No deployment history found.")
        else:
            st.info("No deployment history found.")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


# Footer
st.divider()
st.caption("🎯 Model Registry & Deployment | Powered by MLflow & DagsHub")
