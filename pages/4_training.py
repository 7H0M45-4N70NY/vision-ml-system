"""Training page - Model retraining with drift detection and manual triggers."""

import streamlit as st
import pandas as pd

from src.vision_ml.analytics.analytics_db import AnalyticsDB

st.title("🚀 Training")
st.markdown("Automated model retraining with drift detection and manual triggers")

# Initialize database
db = AnalyticsDB()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    trigger_type = st.radio(
        "Trigger Type",
        ["Manual", "Drift Detection"],
        horizontal=True
    )
    
    if trigger_type == "Drift Detection":
        drift_threshold = st.slider("Drift Threshold", 0.0, 1.0, 0.2)
    
    model_version = st.text_input("Model Version", value="v1")


# Main content
col_main, col_info = st.columns([2, 1])

with col_main:
    st.subheader("Training Status")
    
    # Get training events
    training_events = db.get_training_events(limit=20)
    
    if training_events:
        df_training = pd.DataFrame(training_events)
        df_training['timestamp'] = pd.to_datetime(df_training['timestamp'])
        df_training = df_training.sort_values('timestamp', ascending=False)
        
        # Display recent training events
        st.markdown("**Recent Training Events**")
        st.dataframe(
            df_training[[
                'event_id', 'timestamp', 'trigger_type', 'dataset_size', 'drift_score', 'status'
            ]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No training events yet")
    
    st.divider()
    
    # Training trigger section
    st.subheader("Trigger Training")
    
    if trigger_type == "Manual":
        st.markdown("**Manual Training Trigger**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_size = st.number_input(
                "Dataset Size",
                min_value=1,
                value=100,
                help="Number of images in training dataset"
            )
        
        with col2:
            model_version_input = st.text_input("Model Version", value=model_version)
        
        if st.button("🚀 Start Training", key="start_training"):
            import subprocess, sys
            try:
                event_id = db.save_training_event({
                    'trigger_type': 'manual',
                    'dataset_size': dataset_size,
                    'drift_score': 0.0,
                    'model_version': model_version_input,
                })
                subprocess.Popen(
                    [sys.executable, "scripts/train.py", "--trigger", "manual"],
                    cwd=".",
                )
                st.success(f"✅ Training launched! Event ID: {event_id}")
                st.info("Training running in background — check MLflow Experiments for progress")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    else:  # Drift Detection
        st.markdown("**Confidence-Based Drift Detection**")
        
        # Get inference runs to calculate drift
        inference_runs = db.get_inference_runs(limit=50)
        
        if len(inference_runs) >= 2:
            df_runs = pd.DataFrame(inference_runs)
            df_runs['timestamp'] = pd.to_datetime(df_runs['timestamp'])
            df_runs = df_runs.sort_values('timestamp')
            
            # Ensure columns exist (backward compat)
            if 'avg_confidence' not in df_runs.columns:
                df_runs['avg_confidence'] = 0.0
            if 'drift_score' not in df_runs.columns:
                df_runs['drift_score'] = 0.0
            df_runs['avg_confidence'] = df_runs['avg_confidence'].fillna(0.0)
            df_runs['drift_score'] = df_runs['drift_score'].fillna(0.0)
            
            latest = df_runs.iloc[-1]
            previous = df_runs.iloc[-2]
            current_conf = float(latest.get('avg_confidence', 0))
            previous_conf = float(previous.get('avg_confidence', 0))
            current_drift = float(latest.get('drift_score', 0))
            conf_delta = current_conf - previous_conf
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Confidence", f"{current_conf:.3f}",
                          delta=f"{conf_delta:+.3f}" if previous_conf > 0 else None)
            with col2:
                st.metric("Previous Confidence", f"{previous_conf:.3f}")
            with col3:
                st.metric("Drift Score", f"{current_drift:.3f}")
            
            st.divider()
            
            if current_drift > drift_threshold:
                st.warning(
                    f"⚠️ **Drift Detected!**\n\n"
                    f"Drift score ({current_drift:.3f}) exceeds threshold ({drift_threshold:.3f})\n\n"
                    f"Model confidence has dropped — consider retraining."
                )
                
                if st.button("🚀 Start Training (Drift)", key="start_training_drift"):
                    import subprocess, sys
                    try:
                        event_id = db.save_training_event({
                            'trigger_type': 'drift',
                            'dataset_size': len(df_runs),
                            'drift_score': current_drift,
                            'model_version': model_version,
                        })
                        subprocess.Popen(
                            [sys.executable, "scripts/train.py", "--trigger", "drift"],
                            cwd=".",
                        )
                        st.success(f"✅ Training triggered! Event ID: {event_id}")
                        st.info("Training running in background — check MLflow Experiments for progress")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.success(
                    f"✅ **No Drift Detected**\n\n"
                    f"Drift score ({current_drift:.3f}) is below threshold ({drift_threshold:.3f})\n\n"
                    f"Model is performing well. No retraining needed."
                )
        else:
            remaining = 2 - len(inference_runs)
            st.info(f"Need at least 2 inference runs to detect drift. "
                    f"Run {remaining} more inference{'s' if remaining > 1 else ''}.")
    
    st.divider()
    
    st.subheader("Training Configuration")
    
    try:
        from src.vision_ml.utils.config import load_config as _lc
        _tcfg = _lc('config/training/base.yaml').get('training', {})
        st.markdown(f"""
    **Hyperparameters** *(from config/training/base.yaml)*:
    - Learning Rate: {_tcfg.get('learning_rate', 0.01)}
    - Batch Size: {_tcfg.get('batch_size', 16)}
    - Epochs: {_tcfg.get('epochs', 10)}
    - Optimizer: {_tcfg.get('optimizer', 'auto')}
    - Patience: {_tcfg.get('patience', 5)}
    - Device: {_tcfg.get('device', 'cpu')}
        """)
    except Exception:
        st.markdown("*Could not load training config*")

with col_info:
    st.subheader("ℹ️ Info")
    st.markdown("""
    **Training Triggers:**
    - **Manual**: Explicitly start training
    - **Drift**: Auto-trigger on model degradation
    
    **Drift Detection:**
    - Monitors avg inference confidence
    - Drift score = 1 - avg_confidence
    - Triggers retraining when score > threshold
    
    **Best Practices:**
    - Monitor drift scores regularly
    - Validate before deployment
    - Keep version history
    - Test on holdout set
    
    **Integration:**
    - MLflow for experiment tracking
    - Model versioning
    - Automatic deployment
    """)
