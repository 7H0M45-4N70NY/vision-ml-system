"""Inference page - Real-time video processing with dual-detector."""

import streamlit as st
import cv2
import os

from src.vision_ml.inference.pipeline import InferencePipeline
from src.vision_ml.utils.config import load_config
from src.vision_ml.analytics.analytics_db import AnalyticsDB

st.title("📹 Inference")
st.markdown("Real-time video processing with configurable detector modes")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Detector mode selection
    st.subheader("Detector Mode")
    detector_mode = st.radio(
        "Choose detection strategy:",
        options=[
            ("🚀 Hot Path (fastest)", False),
            ("⚡ Inline (real-time dual)", 'inline'),
            ("📦 Batch (fast + deferred)", 'batch'),
        ],
        format_func=lambda x: x[0],
        horizontal=False,
    )
    detector_mode_value = detector_mode[1]
    
    # Show mode description
    mode_desc = {
        False: "Primary detector only. No secondary overhead. Best for speed.",
        'inline': "Primary + secondary during inference. Best for accuracy.",
        'batch': "Primary only; save frames for offline secondary analysis.",
    }
    st.caption(f"ℹ️ {mode_desc[detector_mode_value]}")
    
    st.divider()
    
    # Dual detector parameters
    confidence_threshold = st.slider("Dual Confidence Threshold", 0.0, 1.0, 0.5,
                                     help="Primary detections below this → check secondary (inline/batch modes)")
    save_low_conf_frames = st.checkbox("Save Low-Confidence Frames", value=True,
                                       help="Save frames for training/batch analysis")
    
    st.divider()
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
                        config['detection']['use_dual_detector'] = detector_mode_value
                        config['detection']['dual_confidence_threshold'] = confidence_threshold
                        config['detection']['save_low_confidence_frames'] = save_low_conf_frames
                        
                        # Run inference
                        pipeline = InferencePipeline(config)
                        summary = pipeline.run_offline(video_path)
                        
                        # Compute drift metrics via DriftDetector
                        pipeline.drift_detector.check()
                        drift_metrics = pipeline.drift_detector.get_metrics()
                        
                        # Save to analytics DB
                        db = AnalyticsDB()
                        run_id = db.save_inference_run({
                            'source_type': 'video_upload',
                            'duration_seconds': summary.get('duration_seconds', 0),
                            'total_frames': summary.get('total_frames', 0),
                            'unique_visitors': summary.get('unique_visitors', 0),
                            'avg_dwell_time_seconds': summary.get('avg_dwell_time_seconds', 0),
                            'use_dual_detector': str(detector_mode_value),
                            'secondary_ratio': summary.get('dual_detector', {}).get('secondary_ratio', 0),
                            'frames_saved': summary.get('dual_detector', {}).get('frames_saved', 0),
                            'avg_confidence': drift_metrics['avg_confidence'],
                            'drift_score': drift_metrics['drift_score'],
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
                        
                        # Show dual-detector stats if enabled (inline or batch mode)
                        if detector_mode_value in ('inline', 'batch') and 'dual_detector' in summary:
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
            st.write(f"Tracking: ✅ (ByteTrack)")
            mode_label = {False: '🚀 Hot', 'inline': '⚡ Inline', 'batch': '📦 Batch'}
            st.write(f"Detector Mode: {mode_label.get(detector_mode_value, 'Unknown')}")
            st.write(f"Confidence: {confidence_threshold:.2f}")
            st.write(f"Active Learning: 🎯 (low-conf frames)")
        
        with col_display:
            video_placeholder = st.empty()
            stats_placeholder = st.empty()
        
        if start_webcam:
            try:
                config = load_config('config/inference/base.yaml')
                config['detection']['use_dual_detector'] = detector_mode_value
                config['detection']['dual_confidence_threshold'] = confidence_threshold
                config['detection']['save_low_confidence_frames'] = save_low_conf_frames
                
                pipeline = InferencePipeline(config)
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("❌ Could not open webcam. Check camera permissions.")
                else:
                    st.success("✅ Webcam opened — tracking + auto-labeling active")
                    
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
                            
                            with stats_placeholder.container():
                                st.markdown("**📊 Live Stats**")
                                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                                with col_s1:
                                    st.metric("Frames", frame_count)
                                with col_s2:
                                    st.metric("Unique Visitors", current_stats.get('unique_visitors', 0))
                                with col_s3:
                                    st.metric("Avg Dwell (s)", f"{current_stats.get('avg_dwell_time_seconds', 0):.1f}")
                                with col_s4:
                                    if hasattr(pipeline.detector, 'frames_saved'):
                                        st.metric("Low-Conf Frames", pipeline.detector.frames_saved)
                                    else:
                                        st.metric("Low-Conf Frames", "N/A")
                                
                                if detector_mode_value == 'inline' and hasattr(pipeline.detector, 'stats'):
                                    dual_stats = pipeline.detector.stats
                                    st.metric("Secondary Ratio", f"{dual_stats['secondary_ratio']:.1%}")
                        
                        frame_count += 1
                    
                    cap.release()
                    
                    # Compute drift metrics via DriftDetector
                    pipeline.drift_detector.check()
                    drift_metrics = pipeline.drift_detector.get_metrics()
                    
                    # Save analytics to DB after webcam session
                    summary = pipeline.analytics.get_summary()
                    db = AnalyticsDB()
                    run_id = db.save_inference_run({
                        'source_type': 'webcam',
                        'duration_seconds': summary.get('total_frames', 0) / max(pipeline.analytics.fps, 1),
                        'total_frames': summary.get('total_frames', 0),
                        'unique_visitors': summary.get('unique_visitors', 0),
                        'avg_dwell_time_seconds': summary.get('avg_dwell_time_seconds', 0),
                        'use_dual_detector': str(detector_mode_value),
                        'avg_confidence': drift_metrics['avg_confidence'],
                        'drift_score': drift_metrics['drift_score'],
                    })
                    if summary.get('dwell_times'):
                        db.save_visitor_analytics(run_id, summary['dwell_times'])
                    
                    # Store pipeline in session state for manual label flush
                    st.session_state.pipeline = pipeline
                    st.session_state.webcam_summary = summary
                    st.session_state.webcam_run_id = run_id
                    
                    low_conf_frames = len([f for f in os.listdir('data/low_confidence_frames') if f.endswith('.jpg')]) if os.path.exists('data/low_confidence_frames') else 0
                    st.info(f"✅ Webcam closed — {summary.get('unique_visitors', 0)} visitors tracked, "
                            f"{low_conf_frames} low-confidence frames saved for active learning")
            
            except Exception as e:
                st.error(f"❌ Webcam error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        # Active learning: load low-confidence frames and flush for training
        if os.path.exists('data/low_confidence_frames'):
            low_conf_files = [f for f in os.listdir('data/low_confidence_frames') if f.endswith('.json')]
            if low_conf_files:
                st.divider()
                st.markdown(f"### 🎯 Active Learning — {len(low_conf_files)} Low-Confidence Frames")
                st.markdown("These are frames where the primary detector struggled. Load them for training dataset.")

                col_flush_local, col_flush_rf = st.columns(2)
                with col_flush_local:
                    if st.button("💾 Load & Export (Local)", key="flush_local"):
                        with st.spinner("Loading low-confidence frames..."):
                            try:
                                labeler = st.session_state.pipeline.auto_labeler if 'pipeline' in st.session_state else __import__('src.vision_ml.labeling.auto_labeler', fromlist=['AutoLabeler']).AutoLabeler(load_config('config/inference/base.yaml'))
                                count = labeler.load_dual_detector_frames('data/low_confidence_frames')
                                labeler.provider = 'local'
                                labeler.flush('data/auto_labeled')
                                db = AnalyticsDB()
                                db.save_labeling_event({
                                    'frames_processed': count,
                                    'labels_created': count,
                                    'provider': 'local',
                                })
                                st.success(f"✅ Exported {count} low-confidence labels to data/auto_labeled/auto_labels.json")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")

                with col_flush_rf:
                    if st.button("☁️ Upload to Roboflow", key="flush_rf"):
                        with st.spinner("Uploading to Roboflow..."):
                            try:
                                labeler = st.session_state.pipeline.auto_labeler if 'pipeline' in st.session_state else __import__('src.vision_ml.labeling.auto_labeler', fromlist=['AutoLabeler']).AutoLabeler(load_config('config/inference/base.yaml'))
                                count = labeler.load_dual_detector_frames('data/low_confidence_frames')
                                labeler.provider = 'roboflow'
                                labeler.flush('data/auto_labeled')
                                db = AnalyticsDB()
                                db.save_labeling_event({
                                    'frames_processed': count,
                                    'labels_created': count,
                                    'provider': 'roboflow',
                                })
                                st.success(f"✅ Uploaded {count} low-confidence labels to Roboflow")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")

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
