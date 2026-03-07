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
            try:
                # Save training event to database
                event_id = db.save_training_event({
                    'trigger_type': 'manual',
                    'dataset_size': dataset_size,
                    'drift_score': 0.0,
                    'model_version': model_version_input,
                })
                
                st.success(f"✅ Training triggered! Event ID: {event_id}")
                st.info("""
                **Next Steps:**
                1. Training job queued
                2. Monitor progress in MLflow
                3. Validate model performance
                4. Deploy when ready
                """)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    else:  # Drift Detection
        st.markdown("**Drift Detection**")
        
        # Get inference runs to calculate drift
        inference_runs = db.get_inference_runs(limit=50)
        
        if len(inference_runs) >= 2:
            df_runs = pd.DataFrame(inference_runs)
            df_runs['timestamp'] = pd.to_datetime(df_runs['timestamp'])
            df_runs = df_runs.sort_values('timestamp')
            
            # Calculate drift as change in secondary ratio
            secondary_ratios = df_runs['secondary_ratio'].values
            
            if len(secondary_ratios) > 1:
                recent_ratio = secondary_ratios[-1]
                previous_ratio = secondary_ratios[-2]
                drift_score = abs(recent_ratio - previous_ratio)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Secondary Ratio", f"{recent_ratio:.1%}")
                with col2:
                    st.metric("Previous Secondary Ratio", f"{previous_ratio:.1%}")
                with col3:
                    st.metric("Drift Score", f"{drift_score:.3f}")
                
                st.divider()
                
                if drift_score > drift_threshold:
                    st.warning(
                        f"⚠️ **Drift Detected!**\n\n"
                        f"Drift score ({drift_score:.3f}) exceeds threshold ({drift_threshold:.3f})\n\n"
                        f"Model performance may be degrading. Consider retraining."
                    )
                    
                    if st.button("🚀 Start Training (Drift)", key="start_training_drift"):
                        try:
                            event_id = db.save_training_event({
                                'trigger_type': 'drift',
                                'dataset_size': len(df_runs),
                                'drift_score': drift_score,
                                'model_version': model_version,
                            })
                            
                            st.success(f"✅ Training triggered! Event ID: {event_id}")
                            st.info("""
                            **Next Steps:**
                            1. Training job queued
                            2. Monitor progress in MLflow
                            3. Validate model performance
                            4. Deploy when ready
                            """)
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.success(
                        f"✅ **No Drift Detected**\n\n"
                        f"Drift score ({drift_score:.3f}) is below threshold ({drift_threshold:.3f})\n\n"
                        f"Model is performing well. No retraining needed."
                    )
            else:
                st.info("Need at least 2 inference runs to detect drift")
        else:
            st.info("Need at least 2 inference runs to detect drift")
    
    st.divider()
    
    st.subheader("Training Configuration")
    
    st.markdown("""
    **Hyperparameters:**
    - Learning Rate: 0.001
    - Batch Size: 32
    - Epochs: 100
    - Optimizer: Adam
    
    **Data Split:**
    - Training: 80%
    - Validation: 10%
    - Test: 10%
    
    **Augmentation:**
    - Random flip
    - Random rotation
    - Color jitter
    """)

with col_info:
    st.subheader("ℹ️ Info")
    st.markdown("""
    **Training Triggers:**
    - **Manual**: Explicitly start training
    - **Drift**: Auto-trigger on model degradation
    
    **Drift Detection:**
    - Monitors secondary detector usage
    - Triggers when ratio exceeds threshold
    - Prevents model degradation
    
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
