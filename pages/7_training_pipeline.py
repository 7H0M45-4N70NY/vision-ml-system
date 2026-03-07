"""Training Pipeline Orchestration - Monitor and manage automated training workflows."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.vision_ml.analytics.analytics_db import AnalyticsDB
from src.vision_ml.mlflow_integration import MLflowManager

st.set_page_config(page_title="Training Pipeline", page_icon="🔄", layout="wide")

st.title("🔄 Training Pipeline Orchestration")
st.markdown("Monitor and manage automated training workflows with Airflow")

# Initialize managers
db = AnalyticsDB()
mlflow_manager = MLflowManager()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    view_type = st.radio(
        "View",
        ["Pipeline Status", "Job Queue", "Drift Detection", "Scheduled Runs"],
        horizontal=False
    )
    
    if view_type == "Pipeline Status":
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)


# ── Pipeline Status ──────────────────────────────────────────────

if view_type == "Pipeline Status":
    st.subheader("📊 Current Pipeline Status")
    
    # Get training events
    training_events = db.get_training_events(limit=100)
    
    if training_events:
        df_events = pd.DataFrame(training_events)
        df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
        df_events = df_events.sort_values('timestamp', ascending=False)
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_runs = len(df_events)
            st.metric("Total Runs", total_runs)
        
        with col2:
            successful = len(df_events[df_events['status'] == 'completed'])
            st.metric("Successful", successful)
        
        with col3:
            failed = len(df_events[df_events['status'] == 'failed'])
            st.metric("Failed", failed)
        
        with col4:
            running = len(df_events[df_events['status'] == 'running'])
            st.metric("Running", running)
        
        with col5:
            success_rate = (successful / total_runs * 100) if total_runs > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        st.divider()
        
        # DAG Visualization
        st.subheader("🔀 Pipeline DAG")
        
        dag_info = """
        ```
        check_drift
            ↓
        trigger_train (conditional)
            ├→ prepare_data
            │   ├─ Download from Roboflow
            │   ├─ Validate dataset
            │   └─ Split train/val/test
            │
            ├→ train_model
            │   ├─ Initialize model
            │   ├─ Training loop (with MLflow logging)
            │   ├─ Save checkpoints
            │   └─ Log metrics
            │
            ├→ evaluate_model
            │   ├─ Compute metrics
            │   ├─ Generate plots
            │   └─ Compare with baseline
            │
            └→ promote_model (conditional)
                ├─ Register in MLflow
                ├─ Transition to Staging
                └─ Compare with Production
        ```
        """
        st.markdown(dag_info)
        
        st.divider()
        
        # Recent runs
        st.subheader("📋 Recent Training Runs")
        
        recent_runs = df_events.head(20)
        
        runs_display = []
        for _, row in recent_runs.iterrows():
            status_emoji = {
                'completed': '✅',
                'running': '🔄',
                'failed': '❌',
                'pending': '⏳'
            }.get(row['status'], '❓')
            
            runs_display.append({
                "Status": status_emoji,
                "Timestamp": row['timestamp'].strftime("%Y-%m-%d %H:%M"),
                "Trigger": row['trigger_type'],
                "Dataset Size": row['dataset_size'],
                "Drift Score": f"{row['drift_score']:.3f}" if row['drift_score'] else "N/A",
                "Model Version": row['model_version'] if row['model_version'] else "—",
            })
        
        df_display = pd.DataFrame(runs_display)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Pipeline metrics visualization
        st.divider()
        st.subheader("📈 Pipeline Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Runs over time
            if HAS_PLOTLY:
                df_daily = df_events.set_index('timestamp').resample('D').size()
                fig = px.bar(
                    x=df_daily.index,
                    y=df_daily.values,
                    title="Training Runs per Day",
                    labels={'x': 'Date', 'y': 'Number of Runs'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Install plotly for visualizations")
        
        with col2:
            # Success rate over time
            if HAS_PLOTLY:
                df_events_sorted = df_events.sort_values('timestamp')
                df_events_sorted['success'] = (df_events_sorted['status'] == 'completed').astype(int)
                df_rolling = df_events_sorted.set_index('timestamp').rolling('7D')['success'].mean()
                
                fig = px.line(
                    x=df_rolling.index,
                    y=df_rolling.values,
                    title="7-Day Rolling Success Rate",
                    labels={'x': 'Date', 'y': 'Success Rate'}
                )
                fig.update_layout(height=400, yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No training runs yet. Trigger training to start the pipeline.")


# ── Job Queue ────────────────────────────────────────────────────

elif view_type == "Job Queue":
    st.subheader("📦 Training Job Queue")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Queue Status**")
        
        # Simulated queue (in production, this would come from Airflow)
        queue_data = {
            "Job ID": ["job_001", "job_002", "job_003", "job_004"],
            "Status": ["Running", "Queued", "Queued", "Scheduled"],
            "Priority": ["High", "Medium", "Low", "High"],
            "Trigger": ["Manual", "Drift", "Scheduled", "Manual"],
            "Submitted": [
                (datetime.now() - timedelta(minutes=15)).strftime("%H:%M"),
                (datetime.now() - timedelta(minutes=5)).strftime("%H:%M"),
                (datetime.now() - timedelta(minutes=2)).strftime("%H:%M"),
                (datetime.now() + timedelta(hours=1)).strftime("%H:%M"),
            ],
            "ETA": ["5 min", "20 min", "25 min", "Scheduled"],
        }
        
        df_queue = pd.DataFrame(queue_data)
        st.dataframe(df_queue, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Queue Statistics**")
        st.metric("Total Jobs", 4)
        st.metric("Running", 1)
        st.metric("Queued", 2)
        st.metric("Scheduled", 1)
    
    st.divider()
    
    # Job details
    st.subheader("📊 Running Job Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Job ID", "job_001")
    with col2:
        st.metric("Progress", "35%")
    with col3:
        st.metric("Time Elapsed", "5 min 23 sec")
    
    # Progress bar
    progress_value = 0.35
    st.progress(progress_value)
    
    # Current task
    st.markdown("**Current Task: train_model**")
    
    task_info = """
    - Epoch: 35/100
    - Train Loss: 0.2341
    - Val Loss: 0.2567
    - Learning Rate: 0.001
    - Batch Size: 32
    - GPU Memory: 4.2 GB / 8 GB
    - Estimated Time Remaining: 10 minutes
    """
    st.markdown(task_info)
    
    # Task history
    st.divider()
    st.subheader("✅ Completed Tasks")
    
    tasks = [
        {"Task": "check_drift", "Status": "✅ Completed", "Duration": "2 sec", "Result": "Drift detected (score: 0.25)"},
        {"Task": "prepare_data", "Status": "✅ Completed", "Duration": "45 sec", "Result": "1000 images prepared"},
    ]
    
    df_tasks = pd.DataFrame(tasks)
    st.dataframe(df_tasks, use_container_width=True, hide_index=True)


# ── Drift Detection ──────────────────────────────────────────────

elif view_type == "Drift Detection":
    st.subheader("🔍 Drift Detection & Auto-Retraining")
    
    # Get inference runs for drift calculation
    inference_runs = db.get_inference_runs(limit=50)
    
    if len(inference_runs) >= 2:
        df_runs = pd.DataFrame(inference_runs)
        df_runs['timestamp'] = pd.to_datetime(df_runs['timestamp'])
        df_runs = df_runs.sort_values('timestamp')
        
        st.markdown("**Drift Monitoring**")
        
        col1, col2, col3 = st.columns(3)
        
        # Current metrics
        latest_run = df_runs.iloc[-1]
        previous_run = df_runs.iloc[-2]
        
        current_ratio = latest_run['secondary_ratio']
        previous_ratio = previous_run['secondary_ratio']
        drift_score = abs(current_ratio - previous_ratio)
        
        with col1:
            st.metric("Current Secondary Ratio", f"{current_ratio:.1%}")
        
        with col2:
            st.metric("Previous Secondary Ratio", f"{previous_ratio:.1%}")
        
        with col3:
            st.metric("Drift Score", f"{drift_score:.3f}")
        
        st.divider()
        
        # Drift threshold
        drift_threshold = st.slider("Drift Threshold", 0.0, 1.0, 0.2)
        
        if drift_score > drift_threshold:
            st.warning(
                f"⚠️ **DRIFT DETECTED**\n\n"
                f"Drift score ({drift_score:.3f}) exceeds threshold ({drift_threshold:.3f})\n\n"
                f"Automatic retraining will be triggered."
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Trigger Retraining Now", key="trigger_retrain"):
                    event_id = db.save_training_event({
                        'trigger_type': 'drift',
                        'dataset_size': len(df_runs),
                        'drift_score': drift_score,
                        'model_version': 'auto_retrain',
                    })
                    st.success(f"✅ Retraining triggered! Event ID: {event_id}")
            
            with col2:
                if st.button("🔕 Snooze Alert (1 hour)", key="snooze_alert"):
                    st.info("Alert snoozed for 1 hour")
        else:
            st.success(
                f"✅ **NO DRIFT DETECTED**\n\n"
                f"Drift score ({drift_score:.3f}) is below threshold ({drift_threshold:.3f})\n\n"
                f"Model is performing well."
            )
        
        st.divider()
        
        # Drift history
        st.subheader("📊 Drift Score History")
        
        if HAS_PLOTLY:
            secondary_ratios = df_runs['secondary_ratio'].values
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_runs['timestamp'],
                y=secondary_ratios,
                mode='lines+markers',
                name='Secondary Ratio',
                line=dict(color='blue', width=2),
            ))
            
            fig.add_hline(
                y=drift_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold ({drift_threshold:.2f})",
                annotation_position="right"
            )
            
            fig.update_layout(
                title="Secondary Detector Usage Over Time",
                xaxis_title="Timestamp",
                yaxis_title="Secondary Ratio",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Drift statistics
        st.markdown("**Drift Statistics**")
        
        stats_data = {
            "Metric": [
                "Mean Secondary Ratio",
                "Max Secondary Ratio",
                "Min Secondary Ratio",
                "Std Dev",
                "Trend (last 5 runs)",
            ],
            "Value": [
                f"{df_runs['secondary_ratio'].mean():.3f}",
                f"{df_runs['secondary_ratio'].max():.3f}",
                f"{df_runs['secondary_ratio'].min():.3f}",
                f"{df_runs['secondary_ratio'].std():.3f}",
                "↑ Increasing" if df_runs['secondary_ratio'].iloc[-5:].mean() > df_runs['secondary_ratio'].iloc[-10:-5].mean() else "↓ Decreasing",
            ]
        }
        
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
    else:
        st.info("Need at least 2 inference runs to detect drift")


# ── Scheduled Runs ──────────────────────────────────────────────

elif view_type == "Scheduled Runs":
    st.subheader("📅 Scheduled Training Runs")
    
    st.markdown("**Airflow DAG Schedule**")
    
    schedule_config = """
    - **Daily Drift Check**: Every day at 2:00 AM UTC
    - **Weekly Retraining**: Every Monday at 3:00 AM UTC (if drift detected)
    - **Monthly Full Retrain**: First day of month at 4:00 AM UTC
    - **On-Demand**: Manual trigger anytime
    """
    
    st.markdown(schedule_config)
    
    st.divider()
    
    # Upcoming scheduled runs
    st.subheader("📋 Upcoming Scheduled Runs")
    
    now = datetime.now()
    scheduled_runs = [
        {
            "Run ID": "scheduled_001",
            "Type": "Drift Check",
            "Scheduled For": (now + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
            "Frequency": "Daily",
            "Status": "Scheduled",
        },
        {
            "Run ID": "scheduled_002",
            "Type": "Weekly Retrain",
            "Scheduled For": (now + timedelta(days=3)).strftime("%Y-%m-%d %H:%M"),
            "Frequency": "Weekly",
            "Status": "Scheduled",
        },
        {
            "Run ID": "scheduled_003",
            "Type": "Monthly Full Retrain",
            "Scheduled For": (now + timedelta(days=25)).strftime("%Y-%m-%d %H:%M"),
            "Frequency": "Monthly",
            "Status": "Scheduled",
        },
    ]
    
    df_scheduled = pd.DataFrame(scheduled_runs)
    st.dataframe(df_scheduled, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Schedule configuration
    st.subheader("⚙️ Configure Schedule")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Drift Check Schedule**")
        drift_check_time = st.time_input("Time (UTC)", value=datetime.strptime("02:00", "%H:%M").time())
        drift_check_enabled = st.checkbox("Enable Daily Drift Check", value=True)
    
    with col2:
        st.markdown("**Retraining Schedule**")
        retrain_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        retrain_time = st.time_input("Time (UTC)", value=datetime.strptime("03:00", "%H:%M").time(), key="retrain_time")
    
    if st.button("💾 Save Schedule Configuration"):
        st.success("✅ Schedule configuration saved!")
        st.info(f"Daily drift check at {drift_check_time} UTC")
        st.info(f"Weekly retraining on {retrain_day} at {retrain_time} UTC")


# Footer
st.divider()
st.caption("🔄 Training Pipeline Orchestration | Powered by Airflow & MLflow")
