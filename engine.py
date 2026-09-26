"""
engine.py: Core Analytics Engine with Anomaly & Speed Fraud Detection
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class CameraNode:
    camera_id: str
    name: str
    lat: float
    lon: float


@dataclass
class ANPRDetection:
    plate_number: str
    camera_id: str
    timestamp: datetime
    confidence: float


class CityTrafficEngine:

    def __init__(self):
        self.cameras: dict[str, CameraNode] = {}
        self.city_graph = nx.DiGraph()
        self.detections: list[ANPRDetection] = []

    def register_camera(
        self, camera_id: str, name: str, lat: float, lon: float
    ):
        """Register camera location in the spatial network."""
        node = CameraNode(camera_id, name, lat, lon)
        self.cameras[camera_id] = node
        self.city_graph.add_node(
            camera_id, name=name, pos=(lat, lon), lat=lat, lon=lon
        )

    def add_road_segment(
        self, from_cam: str, to_cam: str, distance_km: float, speed_limit: float
    ):
        """Define road link between two monitoring points."""
        self.city_graph.add_edge(
            from_cam, to_cam, distance_km=distance_km, speed_limit=speed_limit
        )

    def log_detection(self, detection: ANPRDetection):
        """Ingest single ANPR event."""
        self.detections.append(detection)

    def get_vehicle_trajectory(self, plate_number: str) -> pd.DataFrame:
        """Reconstruct spatio-temporal route and calculate speeds between checkpoints."""
        records = [
            {
                "camera_id": d.camera_id,
                "camera_name": self.cameras[d.camera_id].name,
                "lat": self.cameras[d.camera_id].lat,
                "lon": self.cameras[d.camera_id].lon,
                "timestamp": d.timestamp,
                "confidence": d.confidence,
            }
            for d in self.detections
            if d.plate_number == plate_number
        ]

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
        df["prev_cam"] = df["camera_id"].shift(1)
        df["prev_time"] = df["timestamp"].shift(1)

        speeds = []
        speed_limits = []
        is_speeding = []

        for _, row in df.iterrows():
            if pd.isna(row["prev_cam"]):
                speeds.append(0.0)
                speed_limits.append(0.0)
                is_speeding.append(False)
                continue

            u, v = row["prev_cam"], row["camera_id"]
            if self.city_graph.has_edge(u, v):
                dist = self.city_graph[u][v]["distance_km"]
                limit = self.city_graph[u][v]["speed_limit"]
                elapsed_hrs = (
                    row["timestamp"] - row["prev_time"]
                ).total_seconds() / 3600.0
                speed = dist / elapsed_hrs if elapsed_hrs > 0 else 0
                speeds.append(round(speed, 2))
                speed_limits.append(limit)
                is_speeding.append(speed > limit)
            else:
                speeds.append(0.0)
                speed_limits.append(0.0)
                is_speeding.append(False)

        df["speed_kmh"] = speeds
        df["speed_limit"] = speed_limits
        df["is_speeding"] = is_speeding
        return df

    def detect_anomalies(self) -> pd.DataFrame:
        """Detect plate cloning (impossible speed) and severe speed violations."""
        df_all = pd.DataFrame(
            [
                {
                    "plate": d.plate_number,
                    "camera_id": d.camera_id,
                    "timestamp": d.timestamp,
                    "confidence": d.confidence,
                }
                for d in self.detections
            ]
        ).sort_values(["plate", "timestamp"])

        anomalies = []

        for plate, group in df_all.groupby("plate"):
            group = group.reset_index(drop=True)
            for i in range(len(group) - 1):
                u, v = group.loc[i, "camera_id"], group.loc[i + 1, "camera_id"]
                t1, t2 = group.loc[i, "timestamp"], group.loc[i + 1, "timestamp"]
                dt_hrs = (t2 - t1).total_seconds() / 3600.0

                if self.city_graph.has_edge(u, v) and dt_hrs > 0:
                    dist = self.city_graph[u][v]["distance_km"]
                    speed_limit = self.city_graph[u][v]["speed_limit"]
                    calc_speed = dist / dt_hrs

                    # Anomaly Rule 1: Impossible Speed / Plate Cloning Detection (>180 km/h)
                    if calc_speed > 180:
                        anomalies.append({
                            "Plate": plate,
                            "Type": "🚨 Plate Cloning Alert",
                            "Description": f"Traveled from {self.cameras[u].name} to {self.cameras[v].name} at impossible speed ({round(calc_speed, 1)} km/h)",
                            "Time": t2.strftime("%H:%M:%S"),
                            "Severity": "Critical"
                        })
                    # Anomaly Rule 2: Standard Speed Violation
                    elif calc_speed > speed_limit:
                        anomalies.append({
                            "Plate": plate,
                            "Type": "⚠️ Speed Limit Violation",
                            "Description": f"Exceeded limit on corridor {self.cameras[u].name} ➔ {self.cameras[v].name} ({round(calc_speed, 1)} km/h vs {speed_limit} limit)",
                            "Time": t2.strftime("%H:%M:%S"),
                            "Severity": "Warning"
                        })

        return pd.DataFrame(anomalies)

    def compute_traffic_analytics(self) -> pd.DataFrame:
        """Compute corridor stats."""
        df_all = pd.DataFrame(
            [
                {
                    "plate": d.plate_number,
                    "camera_id": d.camera_id,
                    "timestamp": d.timestamp,
                }
                for d in self.detections
            ]
        ).sort_values(["plate", "timestamp"])

        analytics = []
        for u, v, data in self.city_graph.edges(data=True):
            dist = data["distance_km"]
            speed_limit = data["speed_limit"]

            travel_times = []
            for plate, group in df_all.groupby("plate"):
                cams = list(group["camera_id"])
                times = list(group["timestamp"])
                for i in range(len(cams) - 1):
                    if cams[i] == u and cams[i + 1] == v:
                        dt = (times[i + 1] - times[i]).total_seconds() / 3600.0
                        if dt > 0:
                            travel_times.append(dt)

            vehicle_count = len(travel_times)
            if vehicle_count > 0:
                avg_time = np.mean(travel_times)
                avg_speed = round(dist / avg_time, 2)
            else:
                avg_speed = speed_limit

            free_flow_time = dist / speed_limit
            avg_travel_time = (
                dist / avg_speed if avg_speed > 0 else free_flow_time
            )
            congestion_index = round(avg_travel_time / free_flow_time, 2)

            status = "Free Flow"
            if congestion_index > 1.8:
                status = "Heavy Traffic"
            elif congestion_index > 1.3:
                status = "Moderate Traffic"

            analytics.append(
                {
                    "Corridor": f"{self.cameras[u].name} ➔ {self.cameras[v].name}",
                    "From": u,
                    "To": v,
                    "Distance (km)": dist,
                    "Vehicles Count": vehicle_count,
                    "Avg Speed (km/h)": avg_speed,
                    "Speed Limit": speed_limit,
                    "Congestion Index": congestion_index,
                    "Status": status,
                }
            )

        return pd.DataFrame(analytics)


def seed_mock_city_data(engine: CityTrafficEngine):
    engine.register_camera("CAM_01", "North Gate Junction", 18.628, 73.805)
    engine.register_camera("CAM_02", "Central Expressway A", 18.618, 73.815)
    engine.register_camera("CAM_03", "Tech Park Crossing", 18.605, 73.825)
    engine.register_camera("CAM_04", "South Ring Interchange", 18.595, 73.812)
    engine.register_camera("CAM_05", "East Industrial Avenue", 18.612, 73.835)

    engine.add_road_segment("CAM_01", "CAM_02", distance_km=2.2, speed_limit=60)
    engine.add_road_segment("CAM_02", "CAM_03", distance_km=1.8, speed_limit=50)
    engine.add_road_segment("CAM_03", "CAM_04", distance_km=2.5, speed_limit=60)
    engine.add_road_segment("CAM_02", "CAM_05", distance_km=3.0, speed_limit=70)
    engine.add_road_segment("CAM_05", "CAM_03", distance_km=1.5, speed_limit=40)

    base_time = datetime.now() - timedelta(minutes=45)
    plates = [f"MH14-AB-{random.randint(1000, 9999)}" for _ in range(15)]
    
    # VIP vehicle and suspected cloned vehicle
    plates.append("MH14-VIP-0007")
    plates.append("MH14-CLONE-99")

    for plate in plates:
        cur_time = base_time + timedelta(minutes=random.randint(0, 10))
        seq = (
            ["CAM_01", "CAM_02", "CAM_03", "CAM_04"]
            if random.random() > 0.4
            else ["CAM_01", "CAM_02", "CAM_05", "CAM_03"]
        )

        for cam in seq:
            engine.log_detection(
                ANPRDetection(
                    plate_number=plate,
                    camera_id=cam,
                    timestamp=cur_time,
                    confidence=round(random.uniform(0.93, 0.99), 3),
                )
            )
            
            # Simulate impossible speed for clone vehicle
            if plate == "MH14-CLONE-99":
                cur_time += timedelta(seconds=15)
            else:
                cur_time += timedelta(minutes=random.randint(1, 5))