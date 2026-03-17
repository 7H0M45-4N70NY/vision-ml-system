"""Analytics Dashboard page - Comprehensive visitor analytics with SQLite persistence."""

import streamlit as st
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.vision_ml.analytics.analytics_db import AnalyticsDB

st.title("📊 Analytics Dashboard")
st.markdown("Comprehensive visitor analytics and system metrics")

# Initialize database
try:
    db = AnalyticsDB()
except Exception as e:
    st.error(f"Database unavailable: {e}")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    
    time_range = st.selectbox(
        "Time Range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"]
    )
    
    view_type = st.radio(
        "View",
        ["Summary", "Runs", "Visitors", "Events"],
        horizontal=True
    )


# Get data
summary = db.get_analytics_summary()
inference_runs = db.get_inference_runs(limit=50)

# Display summary metrics
st.header("📈 Summary Metrics")

# Add custom CSS for better spacing
st.markdown("""
<style>
.metric-container {
    padding: 1rem;
    margin: 0.5rem;
    background-color: #f8f9fa;
    border-radius: 0.5rem;
    border: 1px solid #e9ecef;
}
.chart-container {
    padding: 1.5rem;
    margin: 1rem 0;
    background-color: white;
    border-radius: 0.5rem;
    border: 1px solid #e9ecef;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5, gap="medium")

with col1:
    st.metric("Total Runs", summary['total_runs'])
with col2:
    st.metric("Total Visitors", summary['total_visitors'])
with col3:
    st.metric("Avg Dwell Time (s)", round(summary['avg_dwell_time_seconds'], 1))
with col4:
    st.metric("Total Frames", summary['total_frames'])
with col5:
    st.metric("Labeling Events", summary['total_labeling_events'])


# Main content based on view type
if view_type == "Summary":
    st.divider()
    st.header("🎯 System Overview")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Inference Runs Over Time")
        
        if inference_runs:
            df_runs = pd.DataFrame(inference_runs)
            df_runs['timestamp'] = pd.to_datetime(df_runs['timestamp'])
            df_runs_daily = df_runs.set_index('timestamp').resample('D').size()
            
            if HAS_PLOTLY:
                fig = px.bar(
                    x=df_runs_daily.index,
                    y=df_runs_daily.values,
                    title="Daily Inference Runs",
                    labels={'x': 'Date', 'y': 'Number of Runs'},
                )
                fig.update_layout(
                    height=400,
                    xaxis_tickformat='%Y-%m-%d',
                    yaxis_tickformat=',.0f'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df_runs_daily)
        else:
            st.info("No inference runs yet")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Model Health")
        
        if inference_runs:
            df_runs = pd.DataFrame(inference_runs)
            
            # Use avg_confidence for model health (works for all runs)
            if 'avg_confidence' in df_runs.columns:
                df_runs['avg_confidence'] = df_runs['avg_confidence'].fillna(0)
                valid_conf = df_runs[df_runs['avg_confidence'] > 0]['avg_confidence']
                avg_conf = float(valid_conf.mean()) if len(valid_conf) > 0 else 0
            else:
                avg_conf = 0
            
            if HAS_PLOTLY:
                fig = go.Figure(data=[go.Indicator(
                    mode="gauge+number+delta",
                    value=avg_conf * 100,
                    title={'text': "Avg Model Confidence (%)"},
                    delta={'reference': 70},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightcoral"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 30
                        }
                    }
                )])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
                if avg_conf < 0.30:
                    st.warning("Model degradation detected — confidence below 30%")
                else:
                    st.success("Model is healthy")
        else:
            st.info("No model data yet")
        st.markdown('</div>', unsafe_allow_html=True)


elif view_type == "Runs":
    st.divider()
    st.header("📹 Inference Runs")
    
    if inference_runs:
        df_runs = pd.DataFrame(inference_runs)
        df_runs['timestamp'] = pd.to_datetime(df_runs['timestamp'])
        df_runs = df_runs.sort_values('timestamp', ascending=False)
        
        # Display table — select only columns that exist
        display_cols = ['run_id', 'timestamp', 'source_type', 'total_frames',
                        'unique_visitors', 'avg_dwell_time_seconds',
                        'avg_confidence', 'drift_score', 'status']
        display_cols = [c for c in display_cols if c in df_runs.columns]
        st.dataframe(
            df_runs[display_cols],
            use_container_width=True,
            hide_index=True,
        )
        
        # Detailed view
        st.subheader("Run Details")
        selected_run = st.selectbox(
            "Select run",
            options=df_runs['run_id'].tolist(),
            format_func=lambda x: f"{x} - {df_runs[df_runs['run_id']==x]['timestamp'].values[0]}"
        )
        
        if selected_run:
            run_data = df_runs[df_runs['run_id'] == selected_run].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Frames", int(run_data['total_frames']))
            with col2:
                st.metric("Unique Visitors", int(run_data['unique_visitors']))
            with col3:
                st.metric("Avg Dwell Time (s)", round(run_data['avg_dwell_time_seconds'], 2))
            
            # Get visitor details for this run
            visitors = db.get_visitor_analytics(selected_run)
            if visitors:
                st.subheader("Visitor Details")
                df_visitors = pd.DataFrame(visitors)
                v_cols = [c for c in ['tracker_id', 'duration_seconds', 'first_frame', 'last_frame']
                          if c in df_visitors.columns]
                st.dataframe(df_visitors[v_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No inference runs yet")


elif view_type == "Visitors":
    st.divider()
    st.header("👥 Visitor Analytics")
    
    if inference_runs:
        # Aggregate visitor data across all runs
        all_visitors = []
        for run in inference_runs:
            visitors = db.get_visitor_analytics(run['run_id'])
            all_visitors.extend(visitors)
        
        if all_visitors:
            df_visitors = pd.DataFrame(all_visitors)
            df_visitors = df_visitors.sort_values('duration_seconds', ascending=False)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Unique Visitors", len(df_visitors))
            with col2:
                st.metric("Max Dwell Time (s)", round(df_visitors['duration_seconds'].max(), 1))
            with col3:
                st.metric("Min Dwell Time (s)", round(df_visitors['duration_seconds'].min(), 1))
            with col4:
                st.metric("Avg Dwell Time (s)", round(df_visitors['duration_seconds'].mean(), 1))
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                top_visitors = df_visitors.nlargest(10, 'duration_seconds')
                if HAS_PLOTLY:
                    fig = px.bar(
                        top_visitors,
                        x='visitor_id',
                        y='duration_seconds',
                        title='Top 10 Visitors by Dwell Time',
                        labels={'duration_seconds': 'Dwell Time (seconds)'},
                    )
                    fig.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("**Top 10 Visitors by Dwell Time**")
                    st.bar_chart(top_visitors.set_index('visitor_id')['duration_seconds'])
            
            with col2:
                if HAS_PLOTLY:
                    fig = px.histogram(
                        df_visitors,
                        x='duration_seconds',
                        nbins=20,
                        title='Distribution of Dwell Times',
                        labels={'duration_seconds': 'Dwell Time (seconds)'},
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("**Distribution of Dwell Times**")
                    st.bar_chart(df_visitors['duration_seconds'].value_counts().sort_index())
            
            # Detailed table
            st.subheader("All Visitors")
            all_v_cols = [c for c in ['visitor_id', 'run_id', 'duration_seconds', 'first_frame', 'last_frame']
                          if c in df_visitors.columns]
            st.dataframe(df_visitors[all_v_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No visitor data yet")
    else:
        st.info("No inference runs yet")


elif view_type == "Events":
    st.divider()
    st.header("📋 System Events")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Labeling Events")
        labeling_events = db.get_labeling_events(limit=20)
        
        if labeling_events:
            df_labeling = pd.DataFrame(labeling_events)
            df_labeling['timestamp'] = pd.to_datetime(df_labeling['timestamp'])
            
            st.dataframe(
                df_labeling[[
                    'event_id', 'timestamp', 'frames_processed', 'labels_created', 'provider', 'status'
                ]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No labeling events yet")
    
    with col2:
        st.subheader("Training Events")
        training_events = db.get_training_events(limit=20)
        
        if training_events:
            df_training = pd.DataFrame(training_events)
            df_training['timestamp'] = pd.to_datetime(df_training['timestamp'])
            
            st.dataframe(
                df_training[[
                    'event_id', 'timestamp', 'trigger_type', 'dataset_size', 'drift_score', 'status'
                ]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No training events yet")


# Footer
st.divider()
st.markdown("""
---
**Analytics Dashboard** | Real-time visitor analytics with SQLite persistence
""")
