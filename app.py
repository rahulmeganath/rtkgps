"""
GPS/IMU Analysis Dashboard
A production-grade tool for validating GPS/IMU data for autonomous junction decisions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Analysis modules
from analysis.csv_parser import parse_csv, get_data_summary
from analysis.coordinates import (
    latlon_array_to_enu, compute_cumulative_distance, compute_speed_from_position
)
from analysis.stop_detector import detect_stops, get_stop_summary_table
from analysis.gps_quality import (
    analyze_gps_quality, get_jump_histogram_data, compare_reported_vs_measured
)
from analysis.zone_simulator import (
    Zone, simulate_zone, create_zone_from_stop, sweep_hysteresis
)
from analysis.imu_analyzer import analyze_imu, get_turn_summary_table, get_imu_interpretation
from analysis.insight_engine import (
    compute_recommendations, compute_verdict, 
    get_recommendation_explanation, generate_executive_summary
)
from utils.export import export_zone_config_json, export_metrics_csv, generate_pdf_summary


# Page configuration
st.set_page_config(
    page_title="GPS/IMU Analysis Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean engineering look
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .verdict-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px 25px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
    }
    .verdict-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px 25px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
    }
    .verdict-red {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 15px 25px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 20px;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🛰️ GPS/IMU Analysis Dashboard")
    st.markdown("*Decision-validation tool for autonomous junction detection*")
    
    # Sidebar
    with st.sidebar:
        st.header("📂 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload GPS/IMU CSV", 
            type=['csv'],
            help="Upload a CSV file containing GPS and optionally IMU data"
        )
        
        if uploaded_file is not None:
            st.success(f"✓ {uploaded_file.name}")
        
        st.divider()
        st.header("⚙️ Parameters")
        
        speed_threshold = st.slider(
            "Stop Speed Threshold (m/s)",
            min_value=0.1, max_value=1.0, value=0.3, step=0.05,
            help="Speed below which vehicle is considered stopped"
        )
        
        min_stop_duration = st.slider(
            "Minimum Stop Duration (s)",
            min_value=0.5, max_value=5.0, value=1.5, step=0.5,
            help="Minimum duration to count as a stop"
        )
        
        jump_threshold = st.slider(
            "Position Jump Threshold (m)",
            min_value=1.0, max_value=10.0, value=5.0, step=0.5,
            help="Max acceptable position jump between samples"
        )
    
    if uploaded_file is None:
        show_welcome_screen()
        return
    
    # Parse CSV
    with st.spinner("Parsing CSV..."):
        try:
            data = parse_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
            return
    
    # Show warnings
    if data.warnings:
        with st.expander("⚠️ Data Warnings", expanded=False):
            for w in data.warnings:
                st.warning(w)
    
    # Convert to ENU
    if data.latitude is None or data.longitude is None:
        st.error("No latitude/longitude data found in CSV")
        return
    
    x, y, lat_ref, lon_ref = latlon_array_to_enu(data.latitude, data.longitude)
    
    # Compute speed if needed
    if data.speed is None:
        data.speed = compute_speed_from_position(x, y, data.time, window=50)
    
    # Run analyses
    with st.spinner("Analyzing data..."):
        # Stop detection
        stop_analysis = detect_stops(
            x, y, data.time, data.speed,
            data.latitude, data.longitude,
            speed_threshold=speed_threshold,
            min_duration=min_stop_duration
        )
        
        # GPS quality
        gps_metrics = analyze_gps_quality(
            x, y, data.time,
            data.pos_std_xy,
            stop_analysis.stops,
            jump_threshold=jump_threshold
        )
        
        # IMU analysis
        imu_analysis = analyze_imu(
            data.gyro_z, data.time, 
            stop_analysis.is_stopped
        )
        
        # Recommendations and verdict
        recommendations = compute_recommendations(
            gps_metrics, stop_analysis, data.sample_rate
        )
        verdict = compute_verdict(gps_metrics, stop_analysis)
        
        # Total distance
        cumulative_dist = compute_cumulative_distance(x, y)
        total_distance = cumulative_dist[-1] if len(cumulative_dist) > 0 else 0
        
        # Executive summary
        summary = generate_executive_summary(
            data.duration, data.sample_rate, total_distance,
            stop_analysis, verdict
        )
    
    # Store in session for exports
    st.session_state['data'] = data
    st.session_state['x'] = x
    st.session_state['y'] = y
    st.session_state['lat_ref'] = lat_ref
    st.session_state['lon_ref'] = lon_ref
    st.session_state['stop_analysis'] = stop_analysis
    st.session_state['gps_metrics'] = gps_metrics
    st.session_state['imu_analysis'] = imu_analysis
    st.session_state['recommendations'] = recommendations
    st.session_state['verdict'] = verdict
    st.session_state['summary'] = summary
    
    # Executive Summary Section
    render_executive_summary(summary, verdict)
    
    st.divider()
    
    # Tabbed sections
    tabs = st.tabs([
        "🗺️ Trajectory", 
        "🛑 Stop-Go Analysis",
        "📡 GPS Quality",
        "🎯 Zone Analysis",
        "🔄 IMU Analysis",
        "📊 Speed & Motion",
        "💾 Export"
    ])
    
    with tabs[0]:
        render_trajectory_tab(x, y, data.time, data.speed, stop_analysis, gps_metrics)
    
    with tabs[1]:
        render_stop_analysis_tab(data.time, stop_analysis)
    
    with tabs[2]:
        render_gps_quality_tab(data.time, gps_metrics, data.pos_std_xy)
    
    with tabs[3]:
        render_zone_analysis_tab(x, y, data.time, stop_analysis, lat_ref, lon_ref)
    
    with tabs[4]:
        render_imu_tab(data.time, imu_analysis)
    
    with tabs[5]:
        render_speed_motion_tab(data.time, data.speed, cumulative_dist, x, y)
    
    with tabs[6]:
        render_export_tab()


def show_welcome_screen():
    """Show instructions when no file is uploaded."""
    st.markdown("""
    ## Welcome to the GPS/IMU Analysis Dashboard
    
    This tool helps you validate GPS/IMU data for autonomous junction decisions by analyzing:
    
    - **Stop-Go Behavior**: Detect when vehicle stops and measure GPS noise at each stop
    - **GPS Stability**: Evaluate position noise, jumps, and uncertainty
    - **Zone Detection Quality**: Simulate zone entry/exit with hysteresis
    - **Turn Decision Readiness**: Verify data suitability for junction decisions
    
    ### Getting Started
    
    1. **Upload a CSV file** using the sidebar
    2. **Review the executive summary** with overall verdict
    3. **Explore each tab** for detailed analysis
    4. **Export results** as PDF, JSON, or CSV
    
    ### Expected CSV Format
    
    The parser auto-detects columns including:
    - `latitude`, `longitude` (required)
    - `timestamp` or `ros_secs` 
    - `speed` or velocity components
    - `pos_std_lat`, `pos_std_lon` (position uncertainty)
    - `status_name` (GNSS fix status)
    - `ang_vel_z` (IMU yaw rate)
    """)


def render_executive_summary(summary: dict, verdict):
    """Render the executive summary section."""
    
    # Verdict banner
    verdict_class = f"verdict-{verdict.verdict_color}"
    verdict_emoji = "🟢" if verdict.verdict_color == "green" else (
        "🟡" if verdict.verdict_color == "orange" else "🔴"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
        <div class="{verdict_class}">
            {verdict_emoji} GPS Quality: {verdict.verdict}
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"*{verdict.summary}*")
    
    with col2:
        if verdict.checks:
            with st.expander("Checks"):
                for check, passed in verdict.checks.items():
                    icon = "✅" if passed else "❌"
                    st.write(f"{icon} {check}")
    
    st.markdown("---")
    
    # Metrics row
    cols = st.columns(6)
    
    with cols[0]:
        st.metric("Duration", f"{summary['duration_min']:.1f} min")
    
    with cols[1]:
        st.metric("Sample Rate", f"{summary['sample_rate']:.0f} Hz")
    
    with cols[2]:
        st.metric("Distance", f"{summary['total_distance_m']:.0f} m")
    
    with cols[3]:
        st.metric("Stops", summary['stop_count'])
    
    with cols[4]:
        st.metric("% Stopped", f"{summary['stop_percent']:.1f}%")
    
    with cols[5]:
        st.metric("% Moving", f"{summary['move_percent']:.1f}%")


def render_trajectory_tab(x, y, time, speed, stop_analysis, gps_metrics):
    """Render trajectory visualization."""
    st.subheader("Trajectory Visualization")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        color_by = st.selectbox(
            "Color by",
            ["Speed", "GPS Noise", "Time"],
            index=0
        )
        
        show_stops = st.checkbox("Show Stop Markers", value=True)
        
        if stop_analysis.stops:
            show_zones = st.checkbox("Show Stop Zones", value=False)
            zone_radius = st.slider("Zone Radius (m)", 2.0, 20.0, 8.0)
        else:
            show_zones = False
            zone_radius = 8.0
    
    with col1:
        # Create figure
        fig = go.Figure()
        
        # Trajectory colored by selected metric
        if color_by == "Speed":
            color_data = speed
            colorbar_title = "Speed (m/s)"
        elif color_by == "GPS Noise" and gps_metrics.jumps is not None:
            # Use jump distances as proxy for noise
            color_data = np.concatenate([[gps_metrics.jumps[0]], gps_metrics.jumps])
            colorbar_title = "Position Jump (m)"
        else:
            color_data = time
            colorbar_title = "Time (s)"
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(width=1, color='rgba(100,100,100,0.3)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(
                size=3,
                color=color_data,
                colorscale='Viridis',
                colorbar=dict(title=colorbar_title),
                showscale=True
            ),
            customdata=np.column_stack([time, speed]),
            hovertemplate="East: %{x:.1f}m<br>North: %{y:.1f}m<br>Time: %{customdata[0]:.1f}s<br>Speed: %{customdata[1]:.2f}m/s<extra></extra>",
            showlegend=False
        ))
        
        # Start/end markers
        fig.add_trace(go.Scatter(
            x=[x[0]], y=[y[0]],
            mode='markers+text',
            marker=dict(size=15, color='green', symbol='circle'),
            text=['START'],
            textposition='top center',
            name='Start'
        ))
        
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[y[-1]],
            mode='markers+text',
            marker=dict(size=15, color='red', symbol='square'),
            text=['END'],
            textposition='top center',
            name='End'
        ))
        
        # Stop markers
        if show_stops and stop_analysis.stops:
            stop_x = [s.mean_x for s in stop_analysis.stops]
            stop_y = [s.mean_y for s in stop_analysis.stops]
            stop_labels = [f"Stop {s.index}" for s in stop_analysis.stops]
            stop_info = [f"Duration: {s.duration:.1f}s<br>Noise: {s.pos_std*100:.1f}cm" 
                        for s in stop_analysis.stops]
            
            fig.add_trace(go.Scatter(
                x=stop_x, y=stop_y,
                mode='markers+text',
                marker=dict(size=20, color='orange', symbol='circle',
                           line=dict(color='black', width=2)),
                text=[str(s.index) for s in stop_analysis.stops],
                textposition='middle center',
                textfont=dict(color='black', size=10),
                customdata=stop_info,
                hovertemplate="%{text}<br>%{customdata}<extra></extra>",
                name='Stops'
            ))
        
        # Zone circles
        if show_zones and stop_analysis.stops:
            for s in stop_analysis.stops:
                theta = np.linspace(0, 2*np.pi, 50)
                circle_x = s.mean_x + zone_radius * np.cos(theta)
                circle_y = s.mean_y + zone_radius * np.sin(theta)
                
                fig.add_trace(go.Scatter(
                    x=circle_x, y=circle_y,
                    mode='lines',
                    line=dict(color='rgba(255,165,0,0.5)', width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            height=600,
            margin=dict(l=50, r=50, t=30, b=50),
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_stop_analysis_tab(time, stop_analysis):
    """Render stop-go analysis section."""
    st.subheader("Stop-Go Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Speed vs time with stops shaded
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time, y=stop_analysis.speed,
            mode='lines',
            line=dict(color='#3498db', width=1),
            name='Speed'
        ))
        
        # Shade stop regions
        for stop in stop_analysis.stops:
            fig.add_vrect(
                x0=stop.start_time, x1=stop.end_time,
                fillcolor='rgba(231,76,60,0.2)',
                line_width=0,
                annotation_text=f"Stop {stop.index}",
                annotation_position="top left"
            )
        
        fig.add_hline(
            y=0.3, line_dash="dash", line_color="red",
            annotation_text="Stop Threshold (0.3 m/s)"
        )
        
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Speed (m/s)",
            height=400,
            margin=dict(l=50, r=20, t=30, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Summary Statistics**")
        
        st.metric("Total Stops", stop_analysis.stop_count)
        st.metric("Total Stop Time", f"{stop_analysis.total_stop_time:.1f} s")
        
        if stop_analysis.stops:
            st.metric("Shortest Stop", f"{stop_analysis.shortest_stop:.1f} s")
            st.metric("Longest Stop", f"{stop_analysis.longest_stop:.1f} s")
            st.metric("Worst GPS Noise", f"{stop_analysis.worst_gps_noise*100:.1f} cm")
            st.metric("Mean GPS Noise", f"{stop_analysis.mean_gps_noise*100:.1f} cm")
    
    # Stops table
    if stop_analysis.stops:
        st.markdown("---")
        st.subheader("Detected Stops")
        
        stops_df = pd.DataFrame(get_stop_summary_table(stop_analysis.stops))
        st.dataframe(stops_df, use_container_width=True, hide_index=True)


def render_gps_quality_tab(time, gps_metrics, pos_std_xy):
    """Render GPS quality analysis."""
    st.subheader("GPS Quality Analysis")
    
    # Verdict
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        verdict_color = gps_metrics.verdict_color
        st.markdown(f"""
        <div style="background-color: {'#27ae60' if verdict_color == 'green' else '#e74c3c' if verdict_color == 'red' else '#f39c12'}; 
                    padding: 15px; border-radius: 8px; text-align: center; color: white;">
            <strong>GPS Quality: {gps_metrics.verdict}</strong>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"*{gps_metrics.verdict_explanation}*")
    
    st.divider()
    
    # Metrics row
    cols = st.columns(4)
    
    with cols[0]:
        if gps_metrics.reported_std_available:
            st.metric("Reported Std (mean)", f"{gps_metrics.reported_std_mean:.2f} m")
            st.metric("Reported Std (max)", f"{gps_metrics.reported_std_max:.2f} m")
        else:
            st.info("No reported uncertainty")
    
    with cols[1]:
        if gps_metrics.measured_available:
            st.metric("Measured Std (mean)", f"{gps_metrics.measured_std_mean:.3f} m")
            st.metric("Measured Radius (max)", f"{gps_metrics.measured_radius_max:.3f} m")
        else:
            st.info("No stops for measurement")
    
    with cols[2]:
        st.metric("Max Position Jump", f"{gps_metrics.jump_max:.2f} m")
        st.metric("Mean Position Jump", f"{gps_metrics.jump_mean:.4f} m")
    
    with cols[3]:
        st.metric("Large Jumps (>{gps_metrics.jump_threshold}m)", gps_metrics.large_jump_count)
    
    st.divider()
    
    # Plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Jump histogram
        if gps_metrics.jumps is not None and len(gps_metrics.jumps) > 0:
            st.markdown("**Position Jump Distribution**")
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=gps_metrics.jumps,
                nbinsx=50,
                marker_color='#3498db',
                name='Jumps'
            ))
            
            fig.add_vline(
                x=gps_metrics.jump_threshold, line_dash="dash", line_color="red",
                annotation_text=f"Threshold ({gps_metrics.jump_threshold}m)"
            )
            
            fig.add_vline(
                x=gps_metrics.jump_mean, line_dash="solid", line_color="green",
                annotation_text=f"Mean ({gps_metrics.jump_mean:.4f}m)"
            )
            
            fig.update_layout(
                xaxis_title="Jump Distance (m)",
                yaxis_title="Count",
                height=350,
                xaxis=dict(range=[0, min(gps_metrics.jump_max * 1.5, gps_metrics.jump_threshold * 2)])
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Position std over time
        if pos_std_xy is not None:
            st.markdown("**Reported Position Uncertainty Over Time**")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time, y=pos_std_xy,
                mode='lines',
                line=dict(color='#9b59b6', width=1),
                name='Position Std'
            ))
            
            fig.update_layout(
                xaxis_title="Time (s)",
                yaxis_title="Position Std (m)",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Position uncertainty data not available in CSV")
    
    # Reported vs Measured comparison
    comparison = compare_reported_vs_measured(gps_metrics)
    with st.expander("📊 Reported vs Measured Noise Interpretation"):
        st.markdown(comparison['interpretation'])


def render_zone_analysis_tab(x, y, time, stop_analysis, lat_ref, lon_ref):
    """Render zone analysis and simulation."""
    st.subheader("Zone Analysis & Simulation")
    
    # Zone creation options
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Create Zone**")
        
        zone_source = st.radio(
            "Zone Source",
            ["From Stop", "Manual Entry"],
            horizontal=True
        )
        
        if zone_source == "From Stop" and stop_analysis.stops:
            stop_options = {f"Stop {s.index}": s for s in stop_analysis.stops}
            selected_stop = st.selectbox("Select Stop", list(stop_options.keys()))
            base_stop = stop_options[selected_stop]
            
            zone_radius = st.slider(
                "Zone Radius (m)", 3.0, 25.0, 
                max(base_stop.pos_max_radius * 2 + 3.0, 5.0)
            )
            zone_hysteresis = st.slider(
                "Hysteresis (m)", 0.0, 10.0,
                max(base_stop.pos_std * 1.5, 2.0)
            )
            
            zone = Zone(
                id=1, name=selected_stop,
                center_x=base_stop.mean_x, center_y=base_stop.mean_y,
                radius=zone_radius, hysteresis=zone_hysteresis,
                lat=base_stop.lat, lon=base_stop.lon
            )
        
        elif zone_source == "Manual Entry":
            zone_lat = st.number_input("Latitude", value=lat_ref, format="%.6f")
            zone_lon = st.number_input("Longitude", value=lon_ref, format="%.6f")
            zone_radius = st.slider("Zone Radius (m)", 3.0, 25.0, 10.0)
            zone_hysteresis = st.slider("Hysteresis (m)", 0.0, 10.0, 2.5)
            
            from analysis.zone_simulator import create_zone_from_latlon
            zone = create_zone_from_latlon(
                zone_lat, zone_lon, lat_ref, lon_ref,
                zone_radius, zone_hysteresis, 1, "Manual Zone"
            )
        
        else:
            st.info("No stops detected. Use manual entry.")
            zone = None
        
        if zone:
            st.session_state['current_zone'] = zone
    
    with col2:
        if zone:
            # Run simulation
            sim_result = simulate_zone(zone, x, y, time)
            
            # Verdict display
            verdict_bg = {'green': '#27ae60', 'orange': '#f39c12', 'red': '#e74c3c', 'gray': '#95a5a6'}
            st.markdown(f"""
            <div style="background-color: {verdict_bg.get(sim_result.verdict_color, '#95a5a6')}; 
                        padding: 10px; border-radius: 8px; color: white; text-align: center;">
                <strong>Zone Verdict: {sim_result.verdict}</strong>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"*{sim_result.verdict_explanation}*")
            
            # Metrics
            mcols = st.columns(4)
            with mcols[0]:
                st.metric("Entry Count", len(sim_result.entries))
            with mcols[1]:
                st.metric("Exit Count", len(sim_result.exits))
            with mcols[2]:
                st.metric("Flicker Count", sim_result.flip_count_hysteresis)
            with mcols[3]:
                st.metric("Time Inside", f"{sim_result.time_inside:.1f} s")
            
            # Distance to zone plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time, y=sim_result.distances,
                mode='lines',
                line=dict(color='#3498db', width=1),
                name='Distance to Zone'
            ))
            
            fig.add_hline(y=zone.radius, line_color='green', line_dash='solid',
                         annotation_text=f"Zone Radius ({zone.radius:.1f}m)")
            fig.add_hline(y=zone.radius + zone.hysteresis, line_color='orange', line_dash='dash',
                         annotation_text=f"Exit Threshold")
            
            # Shade inside region
            inside_indices = np.where(sim_result.inside_hysteresis)[0]
            if len(inside_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=time[sim_result.inside_hysteresis],
                    y=sim_result.distances[sim_result.inside_hysteresis],
                    mode='markers',
                    marker=dict(size=3, color='green', opacity=0.3),
                    name='Inside Zone'
                ))
            
            fig.update_layout(
                xaxis_title="Time (s)",
                yaxis_title="Distance to Zone Center (m)",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Hysteresis sweep
    if zone:
        st.divider()
        st.subheader("Hysteresis Optimization")
        
        sweep_results = sweep_hysteresis(zone, x, y, time)
        sweep_df = pd.DataFrame(sweep_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(sweep_df, use_container_width=True, hide_index=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[r['hysteresis'] for r in sweep_results],
                y=[r['flip_count'] for r in sweep_results],
                marker_color=['green' if r['stable'] else 'orange' for r in sweep_results]
            ))
            fig.update_layout(
                xaxis_title="Hysteresis (m)",
                yaxis_title="Flip Count",
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)


def render_imu_tab(time, imu_analysis):
    """Render IMU analysis section."""
    st.subheader("IMU Analysis")
    
    if not imu_analysis.gyro_z_available:
        st.warning("No IMU gyroscope data available in CSV")
        st.markdown(get_imu_interpretation())
        return
    
    # Explanation
    with st.expander("ℹ️ About IMU Analysis"):
        st.markdown(get_imu_interpretation())
    
    # Metrics row
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Detected Turns", imu_analysis.turn_count)
    
    with cols[1]:
        st.metric("Left / Right", f"{imu_analysis.left_turns} / {imu_analysis.right_turns}")
    
    with cols[2]:
        if imu_analysis.noise_computed_at_stops:
            st.metric("Gyro Noise (std)", f"{np.degrees(imu_analysis.gyro_noise_std):.2f} deg/s")
        else:
            st.metric("Gyro Std", f"{np.degrees(imu_analysis.gyro_z_std):.2f} deg/s")
    
    with cols[3]:
        if imu_analysis.gyro_bias is not None:
            st.metric("Gyro Bias", f"{np.degrees(imu_analysis.gyro_bias):.3f} deg/s")
    
    st.divider()
    
    # Gyro Z plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time, y=np.degrees(imu_analysis.gyro_z),
        mode='lines',
        line=dict(color='#9b59b6', width=1),
        name='Gyro Z (yaw rate)'
    ))
    
    # Mark turns
    for turn in imu_analysis.turns:
        color = '#27ae60' if turn.direction == 'left' else '#e74c3c'
        fig.add_vrect(
            x0=turn.start_time, x1=turn.end_time,
            fillcolor=f'rgba({231 if turn.direction == "right" else 39},{76 if turn.direction == "right" else 174},{60 if turn.direction == "right" else 96},0.2)',
            line_width=0,
            annotation_text=f"{turn.direction.capitalize()}",
            annotation_position="top left"
        )
    
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Yaw Rate (deg/s)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Turns table
    if imu_analysis.turns:
        st.subheader("Detected Turns")
        turns_df = pd.DataFrame(get_turn_summary_table(imu_analysis.turns))
        st.dataframe(turns_df, use_container_width=True, hide_index=True)


def render_speed_motion_tab(time, speed, cumulative_dist, x, y):
    """Render speed and motion view."""
    st.subheader("Speed & Motion Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Speed over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time, y=speed,
            mode='lines',
            line=dict(color='#3498db', width=1),
            name='Speed'
        ))
        
        fig.update_layout(
            title="Speed vs Time",
            xaxis_title="Time (s)",
            yaxis_title="Speed (m/s)",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cumulative distance
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time, y=cumulative_dist,
            mode='lines',
            line=dict(color='#27ae60', width=2),
            name='Distance'
        ))
        
        fig.update_layout(
            title="Cumulative Distance",
            xaxis_title="Time (s)",
            yaxis_title="Distance (m)",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Speed statistics
    st.divider()
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Max Speed", f"{np.max(speed):.2f} m/s")
    with cols[1]:
        st.metric("Mean Speed (moving)", f"{np.mean(speed[speed > 0.3]):.2f} m/s" if np.any(speed > 0.3) else "N/A")
    with cols[2]:
        st.metric("Total Distance", f"{cumulative_dist[-1]:.1f} m")
    with cols[3]:
        st.metric("Straight-line Distance", 
                 f"{np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2):.1f} m")


def render_export_tab():
    """Render export options."""
    st.subheader("Export Results")
    
    if 'summary' not in st.session_state:
        st.warning("Run analysis first to enable exports")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📄 PDF Report")
        st.markdown("Complete analysis summary with visualizations")
        
        if st.button("Generate PDF"):
            pdf_bytes = generate_pdf_summary(
                st.session_state['summary'],
                st.session_state['stop_analysis'].stops,
                st.session_state['gps_metrics'],
                st.session_state['recommendations'],
                st.session_state['verdict']
            )
            
            st.download_button(
                label="📥 Download PDF",
                data=pdf_bytes,
                file_name="gps_analysis_report.pdf",
                mime="application/pdf"
            )
    
    with col2:
        st.markdown("### 📋 JSON Configuration")
        st.markdown("Zone parameters for integration")
        
        zones = []
        if 'current_zone' in st.session_state:
            zones.append(st.session_state['current_zone'])
        
        json_str = export_zone_config_json(
            zones,
            st.session_state['recommendations']
        )
        
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name="zone_config.json",
            mime="application/json"
        )
        
        with st.expander("Preview JSON"):
            st.code(json_str, language='json')
    
    with col3:
        st.markdown("### 📊 CSV Metrics")
        st.markdown("Computed metrics for analysis")
        
        csv_str = export_metrics_csv(
            st.session_state['summary'],
            st.session_state['stop_analysis'].stops,
            st.session_state['gps_metrics']
        )
        
        st.download_button(
            label="📥 Download CSV",
            data=csv_str,
            file_name="analysis_metrics.csv",
            mime="text/csv"
        )
    
    # Recommendations display
    st.divider()
    st.subheader("📐 Recommended Configuration")
    
    rec = st.session_state['recommendations']
    st.markdown(get_recommendation_explanation(rec))


if __name__ == "__main__":
    main()
