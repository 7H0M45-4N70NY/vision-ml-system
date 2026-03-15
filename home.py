"""Vision ML System - Home Page.

Main entrypoint for multi-page Streamlit app.
Sub-pages in pages/ are auto-discovered by Streamlit and shown in the sidebar.
"""

import streamlit as st
from src.vision_ml.analytics.analytics_db import AnalyticsDB

# Page config — only set here (main entrypoint)
st.set_page_config(
    page_title="Vision ML System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern SaaS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: box-shadow 0.2s;
    }
    .card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .card h3 {
        margin-top: 0;
    }
    .badge-success {
        background-color: #d1fae5; color: #065f46;
        padding: 0.2rem 0.6rem; border-radius: 9999px;
        font-size: 0.8rem; font-weight: 600;
    }
    .badge-info {
        background-color: #dbeafe; color: #1e40af;
        padding: 0.2rem 0.6rem; border-radius: 9999px;
        font-size: 0.8rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar — shared across all pages
with st.sidebar:
    st.markdown("## 🎯 Vision ML")
    st.caption("Solid infrastructure beats a good model")
    st.divider()
    st.markdown("### System Status")
    st.markdown('<span class="badge-success">Models Ready</span> &nbsp; YOLO + RF-DETR', unsafe_allow_html=True)
    st.divider()
    st.markdown("### Quick Links")
    st.page_link("home.py", label="🏠 Home")
    st.page_link("pages/1_inference.py", label="📹 Inference")
    st.page_link("pages/2_auto_labeling.py", label="🏷️ Auto-Labeling")
    st.page_link("pages/3_analytics.py", label="📊 Analytics")
    st.page_link("pages/4_training.py", label="🚀 Training")

# ── Home Page Content ──────────────────────────────────────────────

st.markdown('<h1 class="main-header">Vision ML System</h1>', unsafe_allow_html=True)
st.markdown("**Modern infrastructure for computer vision inference, auto-labeling, and continuous learning.**")

st.divider()

# Quick stats from SQLite
try:
    db = AnalyticsDB()
    summary = db.get_analytics_summary()
except Exception:
    summary = {
        'total_runs': 0, 'total_visitors': 0, 'avg_dwell_time_seconds': 0.0,
        'total_frames': 0, 'total_labeling_events': 0, 'total_training_events': 0,
    }

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Runs", summary['total_runs'])
col2.metric("Total Visitors", summary['total_visitors'])
col3.metric("Avg Dwell (s)", round(summary['avg_dwell_time_seconds'], 1))
col4.metric("Total Frames", summary['total_frames'])
col5.metric("Label Events", summary['total_labeling_events'])
col6.metric("Train Events", summary['total_training_events'])

st.divider()

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
    <h3>📹 Inference</h3>
    Real-time video processing with dual-detector
    <ul>
    <li>Video upload &amp; webcam</li>
    <li>YOLO + RF-DETR ensemble</li>
    <li>Live analytics overlay</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <h3>🏷️ Auto-Labeling</h3>
    Automatic label generation from detections
    <ul>
    <li>Pseudo-label collection</li>
    <li>Local export or Roboflow</li>
    <li>Human-in-the-loop review</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
    <h3>📊 Analytics</h3>
    Comprehensive visitor analytics
    <ul>
    <li>Per-person dwell times</li>
    <li>Historical run data</li>
    <li>Interactive charts</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
    <h3>🚀 Training</h3>
    Automated model retraining
    <ul>
    <li>Drift detection triggers</li>
    <li>Manual training trigger</li>
    <li>Model version management</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <h3>💾 Infrastructure</h3>
    Solid foundation for ML systems
    <ul>
    <li>SQLite analytics persistence</li>
    <li>Model caching (~/.cache/yolo/)</li>
    <li>Session state management</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("Vision ML System — Built with Streamlit, SQLite, YOLO, RF-DETR, Supervision, and Plotly.")
