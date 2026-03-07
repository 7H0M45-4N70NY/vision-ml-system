"""Auto-Labeling page - Automatic label generation and export."""

import streamlit as st

from src.vision_ml.labeling.auto_labeler import AutoLabeler
from src.vision_ml.utils.config import load_config
from src.vision_ml.analytics.analytics_db import AnalyticsDB

st.title("🏷️ Auto-Labeling")
st.markdown("Automatic label generation from low-confidence detections")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    labeler_provider = st.radio("Provider", ["Local", "Roboflow"], horizontal=True)
    min_confidence = st.slider("Min Confidence for Labels", 0.0, 1.0, 0.7)


# Main content
col_main, col_info = st.columns([2, 1])

with col_main:
    st.subheader("Step 1: Load Pseudo-Labels")
    
    frame_dir = st.text_input(
        "Frame directory",
        value="data/low_confidence_frames",
        help="Directory where DualDetector saved frames"
    )
    
    if st.button("📂 Load Frames", key="load_frames"):
        try:
            config = load_config('config/training/base.yaml')
            labeler = AutoLabeler(config)
            count = labeler.load_dual_detector_frames(frame_dir)
            
            st.success(f"✅ Loaded {count} pseudo-labels")
            st.session_state.labeler = labeler
            st.session_state.loaded_count = count
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Step 2: Preview labels
    if 'labeler' in st.session_state and st.session_state.loaded_count > 0:
        st.subheader("Step 2: Preview Labels")
        
        if st.checkbox("Show label preview"):
            labeler = st.session_state.labeler
            if labeler.pending_labels:
                # Show first 3 labels
                for i, label in enumerate(labeler.pending_labels[:3]):
                    with st.expander(f"Label {i+1}: {label['image_id']}"):
                        st.json(label)
    
    # Step 3: Export/Upload
    st.subheader("Step 3: Export / Upload")
    
    if 'labeler' in st.session_state and st.session_state.loaded_count > 0:
        col_export, col_upload = st.columns(2)
        
        with col_export:
            if st.button("💾 Export Local", key="export_local"):
                try:
                    labeler = st.session_state.labeler
                    labeler.flush(output_dir='data/auto_labeled')
                    
                    # Save to analytics DB
                    db = AnalyticsDB()
                    db.save_labeling_event({
                        'frames_processed': st.session_state.loaded_count,
                        'labels_created': len(labeler.pending_labels),
                        'provider': 'local',
                    })
                    
                    st.success("✅ Exported to data/auto_labeled/")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col_upload:
            if st.button("☁️ Upload to Roboflow", key="upload_roboflow"):
                try:
                    labeler = st.session_state.labeler
                    labeler.flush()  # Uses provider from config
                    
                    # Save to analytics DB
                    db = AnalyticsDB()
                    db.save_labeling_event({
                        'frames_processed': st.session_state.loaded_count,
                        'labels_created': len(labeler.pending_labels),
                        'provider': 'roboflow',
                    })
                    
                    st.success("✅ Uploaded to Roboflow!")
                    st.info("👉 Review labels in Roboflow UI, then create new dataset version")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        st.info("Load frames first to see export options")

with col_info:
    st.subheader("ℹ️ Info")
    st.markdown("""
    **Auto-Labeling Flow:**
    1. DualDetector saves low-confidence frames
    2. Load pseudo-labels from disk
    3. Export locally or upload to Roboflow
    4. Human reviews in Roboflow UI
    5. Create dataset version → Train
    
    **Providers:**
    - **Local**: Save to JSON file
    - **Roboflow**: Upload for human review
    
    **Best Practices:**
    - Review labels before training
    - Use min confidence threshold
    - Monitor label quality
    """)
