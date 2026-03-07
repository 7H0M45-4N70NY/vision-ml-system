"""Inference page - Real-time video processing with dual-detector."""

import streamlit as st
import cv2
import os

from src.vision_ml.inference.pipeline import InferencePipeline
from src.vision_ml.utils.config import load_config
from src.vision_ml.analytics.analytics_db import AnalyticsDB

st.title("📹 Inference")
st.markdown("Real-time video processing with dual-detector (YOLO + RF-DETR)")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    use_dual = st.checkbox("Use Dual-Detector", value=True)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    source_type = st.radio("Source Type", ["Video Upload", "Webcam"], horizontal=True)


# Main content
col_main, col_info = st.columns([2, 1])

with col_main:
    st.subheader("Input Source")
    
    if source_type == "Video Upload":
        uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
        
        if uploaded_file:
            video_path = f"temp_{uploaded_file.name}"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("▶️ Run Inference", key="run_inference"):
                with st.spinner("Processing video..."):
                    try:
                        # Load config
                        config = load_config('config/inference/base.yaml')
                        config['detection']['use_dual_detector'] = use_dual
                        config['detection']['dual_confidence_threshold'] = confidence_threshold
                        
                        # Run inference
                        pipeline = InferencePipeline(config)
                        summary = pipeline.run_offline(video_path)
                        
                        # Save to analytics DB
                        db = AnalyticsDB()
                        run_id = db.save_inference_run({
                            'source_type': 'video_upload',
                            'duration_seconds': summary.get('duration_seconds', 0),
                            'total_frames': summary.get('total_frames', 0),
                            'unique_visitors': summary.get('unique_visitors', 0),
                            'avg_dwell_time_seconds': summary.get('avg_dwell_time_seconds', 0),
                            'use_dual_detector': use_dual,
                            'secondary_ratio': summary.get('dual_detector', {}).get('secondary_ratio', 0),
                            'frames_saved': summary.get('dual_detector', {}).get('frames_saved', 0),
                        })
                        
                        # Save visitor analytics
                        if 'dwell_times' in summary:
                            db.save_visitor_analytics(run_id, summary['dwell_times'])
                        
                        # Save to session state
                        st.session_state.pipeline = pipeline
                        st.session_state.analytics_summary = summary
                        st.session_state.run_id = run_id
                        
                        st.success("✅ Inference complete!")
                        
                        # Show results
                        st.json(summary)
                        
                        # Show dual-detector stats if enabled
                        if use_dual and 'dual_detector' in summary:
                            st.subheader("📊 Dual-Detector Stats")
                            dual_stats = summary['dual_detector']
                            
                            col_a, col_b, col_c, col_d = st.columns(4)
                            with col_a:
                                st.metric("Total Frames", dual_stats['total_frames'])
                            with col_b:
                                st.metric("Secondary Calls", dual_stats['secondary_calls'])
                            with col_c:
                                st.metric("Secondary Ratio", f"{dual_stats['secondary_ratio']:.1%}")
                            with col_d:
                                st.metric("Frames Saved", dual_stats['frames_saved'])
                            
                            if dual_stats['secondary_ratio'] > 0.20:
                                st.warning(
                                    f"⚠️ Model degradation detected! "
                                    f"Secondary ratio is {dual_stats['secondary_ratio']:.1%}. "
                                    f"Consider retraining."
                                )
                            else:
                                st.success(f"✅ Model is healthy ({dual_stats['secondary_ratio']:.1%} secondary usage)")
                        
                        # Show output video
                        if os.path.exists('runs/inference/output.mp4'):
                            st.video('runs/inference/output.mp4')
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
                    
                    finally:
                        if os.path.exists(video_path):
                            os.remove(video_path)
    
    else:  # Webcam
        st.markdown("### 📷 Real-Time Webcam Inference")
        
        col_controls, col_display = st.columns([1, 2])
        
        with col_controls:
            st.markdown("**Controls**")
            start_webcam = st.button("▶️ Start Webcam", key="start_webcam")
            stop_webcam = st.button("⏹️ Stop", key="stop_webcam")
            
            st.markdown("**Settings**")
            st.write(f"Dual-Detector: {'✅' if use_dual else '❌'}")
            st.write(f"Confidence: {confidence_threshold:.2f}")
        
        with col_display:
            video_placeholder = st.empty()
            stats_placeholder = st.empty()
        
        if start_webcam:
            try:
                config = load_config('config/inference/base.yaml')
                config['detection']['use_dual_detector'] = use_dual
                config['detection']['dual_confidence_threshold'] = confidence_threshold
                
                pipeline = InferencePipeline(config)
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("❌ Could not open webcam. Check camera permissions.")
                else:
                    st.success("✅ Webcam opened")
                    
                    frame_count = 0
                    stats_update_interval = 10
                    
                    while start_webcam and not stop_webcam:
                        success, frame = cap.read()
                        if not success:
                            st.error("Failed to read frame from webcam")
                            break
                        
                        detections, annotated = pipeline.process_frame(frame, frame_count)
                        
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(annotated_rgb)
                        
                        if frame_count % stats_update_interval == 0:
                            current_stats = pipeline.analytics.get_summary()
                            
                            if use_dual and hasattr(pipeline.detector, 'stats'):
                                dual_stats = pipeline.detector.stats
                                
                                with stats_placeholder.container():
                                    st.markdown("**📊 Live Stats**")
                                    col_s1, col_s2, col_s3 = st.columns(3)
                                    with col_s1:
                                        st.metric("Frames", frame_count)
                                    with col_s2:
                                        st.metric("Secondary Ratio", f"{dual_stats['secondary_ratio']:.1%}")
                                    with col_s3:
                                        st.metric("Visitors", current_stats.get('unique_visitors', 0))
                        
                        frame_count += 1
                    
                    cap.release()
                    st.info("✅ Webcam closed")
            
            except Exception as e:
                st.error(f"❌ Webcam error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

with col_info:
    st.subheader("ℹ️ Info")
    st.markdown("""
    **Dual-Detector Mode:**
    - Primary: YOLO11n (fast)
    - Secondary: RF-DETR (accurate)
    
    **Output:**
    - Annotated video
    - Analytics JSON
    - Low-confidence frames saved
    
    **Health Check:**
    - Secondary ratio < 20% ✅
    - Secondary ratio > 20% ⚠️
    """)
