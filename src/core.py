"""
Core perception, visual odometry, mapping, and planning system.

Provides a single Core class with a step(frame, ts) API that processes frames
and returns a JSON-serializable state dict containing pose, detections, world
map, and navigation commands.

Example:
    core = Core(task_string="find bottle, avoid person")
    state = core.step(frame, time.time())
"""

import time
import math
import json
from collections import deque
from typing import List, Dict, Any, Optional

import numpy as np
import cv2
from loguru import logger

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(
        "ultralytics YOLO required. Install with `pip install ultralytics`"
    ) from e

CONFIG = {
    "MODEL_PATH": "yolov8n.pt",
    "DETECTION_CONF": 0.45,
    "MAP_MERGE_DIST": 0.5,
    "MAP_PRUNE_SEC": 12.0,
    "RADAR_SCALE": 80,
    "VO_FEATURES": 500,
    "VO_MATCH_TOPK": 60,
    "STEP_MAP_UPDATE_EVERY_N": 4,
}


def _now():
    return time.time()


def _bbox_to_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class Core:
    """
    Main processing pipeline for perception, visual odometry, mapping, and planning.

    Processes video frames to detect objects, estimate robot motion, maintain a
    persistent world map, and generate navigation commands based on task objectives.
    """

    def __init__(
        self,
        task_string: str = "find bottle, avoid person",
        model_path: Optional[str] = None,
    ):
        logger.info("Core init")
        self.cfg = CONFIG.copy()
        if model_path:
            self.cfg["MODEL_PATH"] = model_path

        self.model = YOLO(self.cfg["MODEL_PATH"])
        try:
            self.model.fuse()
        except Exception:
            pass

        self.robot_pos = np.array([0.0, 0.0, 1.5], dtype=float)
        self.robot_yaw = 0.0
        self.robot_pitch = 0.0

        self.world_objects: List[Dict[str, Any]] = []
        self._next_obj_id = 1

        self.prev_frame_gray = None
        self.feature_detector = cv2.ORB_create(nfeatures=self.cfg["VO_FEATURES"])

        self.task_string = (task_string or "").lower()
        self.task_targets = self._parse_targets(self.task_string)
        self.task_obstacles = self._parse_obstacles(self.task_string)

        self.frame_id = 0
        self._last_map_update = 0

        logger.info(
            f"Detector targets={self.task_targets} obstacles={self.task_obstacles}"
        )

    def _parse_targets(self, s: str):
        candidates = [
            "cup",
            "bottle",
            "book",
            "cell phone",
            "laptop",
            "remote",
            "keyboard",
            "mouse",
        ]
        return [c for c in candidates if c in s]

    def _parse_obstacles(self, s: str):
        candidates = ["person", "chair", "couch", "potted plant", "dining table"]
        return [c for c in candidates if c in s]

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run object detection on frame and return detections.

        Returns list of detection dicts with keys: class, confidence, bbox,
        screen (pixel center), distance (estimated depth in meters), and
        height_offset.
        """
        h, w = frame.shape[:2]
        results = self.model(frame, verbose=False, conf=self.cfg["DETECTION_CONF"])[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = results.names[cls]

            cx, cy = _bbox_to_center((x1, y1, x2, y2))
            bw = x2 - x1
            bh = y2 - y1
            area = max(1.0, bw * bh)

            distance = 4.0 / (math.sqrt(area / (w * h)) + 0.08)
            distance = float(np.clip(distance, 0.35, 12.0))

            normalized_y = cy / h
            height_offset = (0.5 - normalized_y) * 1.2

            detections.append(
                {
                    "class": str(name),
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "screen": [float(cx), float(cy)],
                    "distance": distance,
                    "height_offset": height_offset,
                }
            )

        return detections

    def estimate_motion(self, frame: np.ndarray):
        """Estimate robot motion using visual odometry.

        Returns tuple of (forward, lateral, vertical, yaw_change, pitch_change)
        motion estimates in meters/radians.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return 0.0, 0.0, 0.0, 0.0, 0.0

        kp1, des1 = self.feature_detector.detectAndCompute(self.prev_frame_gray, None)
        kp2, des2 = self.feature_detector.detectAndCompute(gray, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            self.prev_frame_gray = gray
            return 0.0, 0.0, 0.0, 0.0, 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[: self.cfg["VO_MATCH_TOPK"]]

        if len(matches) < 6:
            self.prev_frame_gray = gray
            return 0.0, 0.0, 0.0, 0.0, 0.0

        dx_sum = 0.0
        dy_sum = 0.0
        for m in matches:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            dx_sum += pt2[0] - pt1[0]
            dy_sum += pt2[1] - pt1[1]

        dx = dx_sum / len(matches)
        dy = dy_sum / len(matches)

        yaw_change = -dx / frame.shape[1] * 0.35
        pitch_change = dy / frame.shape[0] * 0.2

        forward = -dy / 160.0
        lateral = dx / 160.0
        vertical = dy / 220.0

        self.prev_frame_gray = gray

        return (
            float(forward),
            float(lateral),
            float(vertical),
            float(yaw_change),
            float(pitch_change),
        )

    def update_robot_pose(self, forward, lateral, vertical, yaw_change, pitch_change):
        """Update robot pose by integrating motion estimates."""
        self.robot_yaw += yaw_change
        self.robot_pitch += pitch_change
        self.robot_pitch = float(np.clip(self.robot_pitch, -np.pi / 3, np.pi / 3))

        dx = forward * math.cos(self.robot_yaw) - lateral * math.sin(self.robot_yaw)
        dy = forward * math.sin(self.robot_yaw) + lateral * math.cos(self.robot_yaw)
        dz = vertical

        self.robot_pos[0] += dx
        self.robot_pos[1] += dy
        self.robot_pos[2] += dz
        self.robot_pos[2] = float(np.clip(self.robot_pos[2], 0.35, 3.0))

    def project_to_3d_world(
        self, detection: Dict[str, Any], frame_width: int, frame_height: int
    ):
        """Project 2D detection to 3D world coordinates."""
        sx, sy = detection["screen"]
        h_angle_offset = (sx - frame_width / 2) / frame_width * math.radians(62)
        v_angle_offset = (sy - frame_height / 2) / frame_height * math.radians(44)
        object_yaw = self.robot_yaw + h_angle_offset
        object_pitch = self.robot_pitch + v_angle_offset

        distance = float(detection["distance"])
        horizontal_dist = distance * math.cos(object_pitch)
        vertical_offset = distance * math.sin(object_pitch)

        wx = self.robot_pos[0] + horizontal_dist * math.cos(object_yaw)
        wy = self.robot_pos[1] + horizontal_dist * math.sin(object_yaw)
        wz = self.robot_pos[2] + vertical_offset

        return np.array([float(wx), float(wy), float(wz)], dtype=float)

    def update_map(
        self,
        detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
        ts: Optional[float] = None,
    ):
        """Update persistent world map with new detections and prune stale objects."""
        if ts is None:
            ts = _now()

        for det in detections:
            pos = self.project_to_3d_world(det, frame_width, frame_height)
            cls = det["class"]
            merged = False
            for obj in self.world_objects:
                if obj["class"] == cls:
                    dist = float(np.linalg.norm(obj["pos"] - pos))
                    if dist < self.cfg["MAP_MERGE_DIST"]:
                        obj["pos"] = (obj["pos"] + pos) / 2.0
                        obj["conf"] = max(
                            obj["conf"], det.get("confidence", det.get("conf", 0.0))
                        )
                        obj["last_seen"] = ts
                        obj["trail"].appendleft(obj["pos"].copy())
                        merged = True
                        break
            if not merged:
                self.world_objects.append(
                    {
                        "id": self._next_obj_id,
                        "class": cls,
                        "pos": pos,
                        "conf": det.get("confidence", det.get("conf", 0.0)),
                        "last_seen": ts,
                        "trail": deque([pos.copy()], maxlen=60),
                    }
                )
                self._next_obj_id += 1

        cutoff = ts - self.cfg["MAP_PRUNE_SEC"]
        before = len(self.world_objects)
        self.world_objects = [o for o in self.world_objects if o["last_seen"] >= cutoff]
        after = len(self.world_objects)
        if after < before:
            logger.debug(f"pruned {before - after} world objects")

    def plan_path(self):
        """Generate path to nearest target object, avoiding obstacles."""
        if not self.task_targets:
            return []
        target_obj = None
        min_dist = float("inf")
        for obj in self.world_objects:
            if obj["class"] in self.task_targets:
                d = float(np.linalg.norm(obj["pos"] - self.robot_pos))
                if d < min_dist:
                    min_dist = d
                    target_obj = obj
        if target_obj is None:
            return []
        target_pos = target_obj["pos"].copy()

        avoid_radius = 0.8
        path = []
        num_points = 8

        for i in range(num_points + 1):
            t = i / num_points
            waypoint = (1 - t) * self.robot_pos + t * target_pos

            for obj in self.world_objects:
                if obj["class"] in self.task_obstacles:
                    obj_pos = obj["pos"]
                    dist_to_obstacle = float(np.linalg.norm(waypoint - obj_pos))
                    if dist_to_obstacle < avoid_radius:
                        offset_dir = waypoint - obj_pos
                        if np.linalg.norm(offset_dir) > 0.01:
                            offset_dir = offset_dir / np.linalg.norm(offset_dir)
                            waypoint = obj_pos + offset_dir * avoid_radius

            path.append(waypoint)

        return path

    def get_navigation_command(self, current_path: List[np.ndarray]):
        """Generate navigation command string from current path."""
        if not self.task_targets:
            return "EXPLORING"
        if not current_path or len(current_path) < 2:
            found = any(obj["class"] in self.task_targets for obj in self.world_objects)
            return "TARGET LOCATED - PLANNING" if found else "SEARCHING"

        next_waypoint = np.array(current_path[1])
        to_waypoint = next_waypoint - self.robot_pos
        distance_3d = float(np.linalg.norm(to_waypoint))
        if distance_3d < 0.3:
            return "TARGET REACHED"

        angle_to_waypoint = math.atan2(float(to_waypoint[1]), float(to_waypoint[0]))
        angle_diff = angle_to_waypoint - self.robot_yaw
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        angle_deg = int(math.degrees(angle_diff))

        horizontal_dist = math.sqrt(
            float(to_waypoint[0]) ** 2 + float(to_waypoint[1]) ** 2
        )
        pitch_to_waypoint = math.atan2(float(to_waypoint[2]), horizontal_dist)
        pitch_diff = int(math.degrees(pitch_to_waypoint - self.robot_pitch))

        instr = []
        if abs(angle_deg) > 15:
            instr.append(
                f"TURN {'LEFT' if angle_deg > 0 else 'RIGHT'} {abs(angle_deg)}°"
            )
        if abs(pitch_diff) > 10:
            instr.append(
                f"LOOK {'UP' if pitch_diff > 0 else 'DOWN'} {abs(pitch_diff)}°"
            )
        if not instr:
            instr.append(f"FORWARD {distance_3d:.2f}m")
        return " + ".join(instr)

    def step(self, frame: np.ndarray, ts: Optional[float] = None) -> Dict[str, Any]:
        """Process frame and return current system state.

        Performs detection and visual odometry on every frame, updates robot pose,
        and updates the persistent world map every N frames.
        """
        if ts is None:
            ts = _now()
        self.frame_id += 1

        h, w = frame.shape[:2]
        detections = self.detect(frame)

        fwd, lat, vert, yawc, pitchc = self.estimate_motion(frame)
        self.update_robot_pose(fwd, lat, vert, yawc, pitchc)

        if (self.frame_id - self._last_map_update) >= self.cfg[
            "STEP_MAP_UPDATE_EVERY_N"
        ]:
            self.update_map(detections, w, h, ts)
            self._last_map_update = self.frame_id

        path = self.plan_path()
        command = self.get_navigation_command(path)

        world_ser = []
        for o in self.world_objects:
            world_ser.append(
                {
                    "id": int(o["id"]),
                    "class": str(o["class"]),
                    "pos": [float(x) for x in o["pos"]],
                    "conf": float(o["conf"]),
                    "last_seen": float(o["last_seen"]),
                }
            )

        state = {
            "frame_id": int(self.frame_id),
            "ts": float(ts),
            "pose": {
                "x": float(self.robot_pos[0]),
                "y": float(self.robot_pos[1]),
                "z": float(self.robot_pos[2]),
                "yaw": float(self.robot_yaw),
                "pitch": float(self.robot_pitch),
            },
            "detections": detections,
            "world": world_ser,
            "command": str(command),
        }

        return state

    def save_state_ndjson(self, filename: str, state: Dict[str, Any]):
        """Append state dict to ndjson file (one JSON object per line).

        Note: Does not save frame images. Use a separate recorder for video data.
        """
        with open(filename, "a") as f:
            f.write(json.dumps(state) + "\n")

    def load_states_ndjson(self, filename: str) -> List[Dict[str, Any]]:
        out = []
        with open(filename, "r") as f:
            for line in f:
                out.append(json.loads(line))
        return out


if __name__ == "__main__":
    logger.info(
        "Core module run as script — demo (no camera)\nUsage: import Core from this file in your UI script."
    )
    c = Core(task_string="find bottle, avoid person")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    s = c.step(blank)
    print(json.dumps(s, indent=2))
