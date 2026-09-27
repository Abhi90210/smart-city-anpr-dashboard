"""
app.py: Smart City AI Traffic & Surveillance Dashboard
"""
import streamlit as st
import pandas as pd
import pydeck as pdk
import random
from datetime import datetime, timedelta
from engine import CityTrafficEngine, seed_mock_city_data, ANPRDetection

st.set_page_config(
    page_title="AI Smart City ANPR Matrix",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dashboard Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222B;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #00D2FF;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📡 Smart City AI ANPR & Traffic Control Center")

# Initialize Engine
if "engine" not in st.session_state:
    engine = CityTrafficEngine()
    seed_mock_city_data(engine)
    st.session_state.engine = engine

engine = st.session_state.engine

# Sidebar Simulation Controls
st.sidebar.title("🎛️ Command Console")

st.sidebar.subheader("Live Simulation Feed")
if st.sidebar.button("⚡ Simulate Live Plate Detection"):
    random_cam = random.choice(list(engine.cameras.keys()))
    random_plate = random.choice(list(set([d.plate_number for d in engine.detections])))
    
    new_event = ANPRDetection(
        plate_number=random_plate,
        camera_id=random_cam,
        timestamp=datetime.now(),
        confidence=round(random.uniform(0.94, 0.99), 3)
    )
    engine.log_detection(new_event)
    st.sidebar.success(f"Log: {random_plate} @ {random_cam}")

st.sidebar.markdown("---")
navigation = st.sidebar.radio(
    "Modules",
    ["City Overview & 3D Map", "Vehicle Forensic Tracker", "AI Threat & Violation Center"]
)

# MODULE 1: CITY OVERVIEW & 3D MAP
if navigation == "City Overview & 3D Map":
    st.subheader("Network Grid Status & Traffic Density")

    analytics_df = engine.compute_traffic_analytics()
    anomalies_df = engine.detect_anomalies()

    # Executive KPI Bar
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Active Camera Nodes", len(engine.cameras))
    kpi2.metric("Total ANPR Detections", len(engine.detections))
    
    critical_alerts = len(anomalies_df[anomalies_df["Severity"] == "Critical"]) if not anomalies_df.empty else 0
    kpi3.metric("Plate Cloning Alerts", critical_alerts, delta_color="inverse")
    
    speed_alerts = len(anomalies_df[anomalies_df["Severity"] == "Warning"]) if not anomalies_df.empty else 0
    kpi4.metric("Speed Violations", speed_alerts, delta_color="inverse")

    st.markdown("---")

    # Prepare Data for 3D PyDeck Visuals
    cam_volume = {}
    for d in engine.detections:
        cam_volume[d.camera_id] = cam_volume.get(d.camera_id, 0) + 1

    map_data = []
    for cam_id, node in engine.cameras.items():
        map_data.append({
            "camera_id": cam_id,
            "name": node.name,
            "lat": node.lat,
            "lon": node.lon,
            "detections_count": cam_volume.get(cam_id, 0),
            "height": cam_volume.get(cam_id, 0) * 120
        })
    map_df = pd.DataFrame(map_data)

    col_map, col_table = st.columns([1.5, 1])

    with col_map:
        st.write("### 3D Junction Activity Volume")
        
        column_layer = pdk.Layer(
            "ColumnLayer",
            data=map_df,
            get_position=["lon", "lat"],
            get_elevation="height",
            elevation_scale=1,
            radius=100,
            get_fill_color="[0, 210, 255, 180]",
            pickable=True,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(
            latitude=map_df["lat"].mean(),
            longitude=map_df["lon"].mean(),
            zoom=11.5,
            pitch=45,
            bearing=20
        )

        st.pydeck_chart(pdk.Deck(
            layers=[column_layer],
            initial_view_state=view_state,
            tooltip={"text": "{name}\nID: {camera_id}\nTotal Passes: {detections_count}"}
        ))

    with col_table:
        st.write("### Corridor Congestion Index")
        
        def highlight_status(val):
            if val == "Heavy Traffic":
                return "background-color: #721c24; color: white"
            elif val == "Moderate Traffic":
                return "background-color: #856404; color: white"
            return "background-color: #155724; color: white"

        st.dataframe(
            analytics_df[["Corridor", "Avg Speed (km/h)", "Congestion Index", "Status"]].style.map(
                highlight_status, subset=["Status"]
            ),
            use_container_width=True,
            height=380
        )

# MODULE 2: VEHICLE FORENSIC TRACKER
elif navigation == "Vehicle Forensic Tracker":
    st.subheader("🕵️ Vehicle Target Trajectory Analysis")

    unique_plates = sorted(list(set([d.plate_number for d in engine.detections])))
    target_plate = st.selectbox("Select Target Vehicle License Plate", unique_plates)

    if target_plate:
        traj_df = engine.get_vehicle_trajectory(target_plate)

        if not traj_df.empty:
            has_speeding = traj_df["is_speeding"].any()
            
            if has_speeding:
                st.error(f"⚠️ SPEEDING DETECTED: Target vehicle {target_plate} exceeded corridor limits during movement.")
            else:
                st.success(f"Target {target_plate} profile clean. No localized speed violations recorded.")

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Checkpoints Crossed", len(traj_df))
            m2.metric("Peak Speed Recorded", f"{traj_df['speed_kmh'].max()} km/h")
            m3.metric("First Sight Time", traj_df["timestamp"].min().strftime("%H:%M:%S"))

            # Spatial Path Mapping
            st.write("### Route Trajectory Path")
            
            line_data = []
            for i in range(len(traj_df) - 1):
                line_data.append({
                    "start": [traj_df.loc[i, "lon"], traj_df.loc[i, "lat"]],
                    "end": [traj_df.loc[i + 1, "lon"], traj_df.loc[i + 1, "lat"]],
                    "is_speeding": bool(traj_df.loc[i + 1, "is_speeding"])
                })

            line_layer = pdk.Layer(
                "LineLayer",
                data=line_data,
                get_source_position="start",
                get_target_position="end",
                get_color="is_speeding ? [255, 0, 0, 255] : [0, 255, 150, 255]",
                get_width=5,
            )

            point_layer = pdk.Layer(
                "ScatterplotLayer",
                data=traj_df,
                get_position=["lon", "lat"],
                get_color="[255, 255, 255, 255]",
                get_radius=80,
                pickable=True,
            )

            v_state = pdk.ViewState(
                latitude=traj_df["lat"].mean(),
                longitude=traj_df["lon"].mean(),
                zoom=12.2,
                pitch=30
            )

            st.pydeck_chart(pdk.Deck(
                layers=[line_layer, point_layer],
                initial_view_state=v_state,
                tooltip={"text": "Checkpoint: {camera_name}\nTime: {timestamp}"}
            ))

            st.write("### Passage Event Timeline")
            st.dataframe(
                traj_df[["camera_id", "camera_name", "timestamp", "speed_kmh", "speed_limit", "is_speeding", "confidence"]],
                use_container_width=True
            )

# MODULE 3: AI THREAT & VIOLATION CENTER
elif navigation == "AI Threat & Violation Center":
    st.subheader("🚨 Automated Security & Anomaly Logs")

    anomalies_df = engine.detect_anomalies()

    if not anomalies_df.empty:
        criticals = anomalies_df[anomalies_df["Severity"] == "Critical"]
        warnings = anomalies_df[anomalies_df["Severity"] == "Warning"]

        if not criticals.empty:
            st.write("### 🚨 Critical Alerts (Suspected Plate Cloning / Fraud)")
            for _, row in criticals.iterrows():
                st.error(f"**{row['Plate']}** - {row['Type']} at {row['Time']} | {row['Description']}")

        if not warnings.empty:
            st.write("### ⚠️ Traffic Violations (Speeding)")
            st.dataframe(warnings[["Time", "Plate", "Type", "Description"]], use_container_width=True)
    else:
        st.success("No traffic anomalies or plate cloning detected in current logs.")

        from copilot import generate_fir_and_dispatch

# ... inside your app.py ...

elif navigation == "AI Threat & Violation Center":
    st.subheader("🚨 Automated Security & Anomaly Logs")

    anomalies_df = engine.detect_anomalies()

    if not anomalies_df.empty:
        criticals = anomalies_df[anomalies_df["Severity"] == "Critical"]
        warnings = anomalies_df[anomalies_df["Severity"] == "Warning"]

        # ---------------- CRITICAL ALERTS (CLONING / SPOOFING) ----------------
        if not criticals.empty:
            st.markdown("### 🚨 Critical Security Threats (Plate Cloning / Double Presence)")
            
            for idx, row in criticals.iterrows():
                plate = row["Plate"]
                
                with st.expander(f"🛑 CRITICAL: Plate {plate} | {row['Type']}", expanded=True):
                    col_alert, col_action = st.columns([2, 1])
                    
                    with col_alert:
                        st.error(f"**Alert:** {row['Description']}")
                        st.caption(f"Detected at: {row['Time']}")
                    
                    with col_action:
                        # Button to trigger AI dispatch dossier
                        trigger_ai = st.button("⚖️ Generate AI FIR Dossier", key=f"btn_crit_{idx}")

                    if trigger_ai:
                        # Pull trajectory history from engine for context
                        traj = engine.get_vehicle_trajectory(plate)
                        history = []
                        if not traj.empty:
                            for _, t_row in traj.iterrows():
                                history.append({
                                    "time": str(t_row.get("timestamp")),
                                    "camera": t_row.get("camera_name", t_row.get("camera_id")),
                                    "speed": t_row.get("speed_kmh", 0)
                                })

                        with st.spinner("AI Forensic Agent compiling surveillance logs and drafting legal notice..."):
                            report = generate_fir_and_dispatch(
                                plate_number=plate,
                                anomaly_data=row.to_dict(),
                                trajectory_summary=history
                            )
                            
                            st.markdown("---")
                            st.markdown(report)
                            
                            # Add download button for the generated dossier
                            st.download_button(
                                label="📥 Export Legal Incident Dossier (TXT)",
                                data=report,
                                file_name=f"FIR_Dossier_{plate}_{row['Time']}.txt",
                                mime="text/plain",
                                key=f"dl_crit_{idx}"
                            )

        # ---------------- WARNINGS (SPEEDING VIOLATIONS) ----------------
        if not warnings.empty:
            st.markdown("### ⚠️ Traffic Violations (Speeding)")
            st.dataframe(warnings[["Time", "Plate", "Type", "Description"]], use_container_width=True)

            selected_warn_plate = st.selectbox(
                "Select a speeding violator to draft automated E-Challan:",
                warnings["Plate"].unique()
            )
            
            if st.button("📄 Generate E-Challan Notice via Copilot"):
                warn_row = warnings[warnings["Plate"] == selected_warn_plate].iloc[0].to_dict()
                with st.spinner("Drafting official violation citation..."):
                    challan_text = generate_fir_and_dispatch(
                        plate_number=selected_warn_plate,
                        anomaly_data=warn_row,
                        trajectory_summary=[]
                    )
                    st.info(challan_text)

    else:
        st.success("No traffic anomalies or plate cloning detected in current logs.")