"""
PySide6 GUI demo for Pathfinder.
"""

import sys
import time
import math
import threading
import queue
from collections import deque

import cv2
import numpy as np
from loguru import logger

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QStatusBar,
)

try:
    from core import Core
except Exception as e:
    raise RuntimeError(
        "core.py not found or import failed. Put core.py next to this file."
    ) from e

CAM_W, CAM_H = 1280, 720
CAM_PREVIEW_W, CAM_PREVIEW_H = 960, 540
RADAR_SIZE = 420
FWD_W, FWD_H = 480, 220
MODEL_PATH = None

COL_BG = (18, 20, 23)
COL_PANEL = (28, 30, 33)
COL_ACCENT = (255, 190, 0)
COL_TEXT = (230, 230, 230)
COL_TARGET = (36, 200, 100)
COL_OBST = (200, 60, 60)
COL_NEUTRAL = (140, 140, 140)
COL_PATH = (100, 150, 255)


logger.remove()
logger.add(lambda m: print(m, end=""))


def bgr_to_qpixmap(img_bgr):
    """Convert BGR numpy array to QPixmap."""
    h, w = img_bgr.shape[:2]
    # Convert to RGB
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bytes_per_line = 3 * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class Worker(threading.Thread):
    """Background worker for frame capture and processing."""

    def __init__(self, out_q: queue.Queue, task_string: str, model_path: str = None):
        super().__init__(daemon=True)
        self.out_q = out_q
        self.task = task_string
        self.model_path = model_path
        self._stop = threading.Event()
        self._pause = threading.Event()
        self.cap = None
        self.core = Core(task_string=self.task, model_path=self.model_path)

    def run(self):
        logger.info("Worker: opening camera")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        if not self.cap.isOpened():
            logger.error("Worker: cannot open camera")
            return
        logger.info("Worker: camera opened, starting loop")

        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(0.05)
                continue
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            ts = time.time()
            try:
                state = self.core.step(frame, ts)
            except Exception as e:
                logger.error(f"Worker: core.step error: {e}")
                continue

            try:
                if self.out_q.full():
                    try:
                        self.out_q.get_nowait()
                    except Exception:
                        pass
                self.out_q.put_nowait((frame, state))
            except Exception:
                pass

        logger.info("Worker: stopping, releasing camera")
        try:
            self.cap.release()
        except Exception:
            pass

    def stop(self):
        self._stop.set()

    def pause(self, p: bool):
        if p:
            self._pause.set()
        else:
            self._pause.clear()

    def reset_map(self):
        self.core.world_objects = []
        self.core._next_obj_id = 1
        self.core.robot_pos = np.array([0.0, 0.0, 1.5], dtype=float)
        self.core.robot_yaw = 0.0
        self.core.robot_pitch = 0.0
        logger.info("Worker: map reset")


class MainWindow(QMainWindow):
    def __init__(self, task_string: str, model_path: str = None):
        super().__init__()
        self.setWindowTitle("Pathfinder")
        self.resize(1400, 820)
        self.task_string = task_string
        self.model_path = model_path

        self.q = queue.Queue(maxsize=1)
        self.worker = Worker(self.q, self.task_string, self.model_path)
        self.worker.start()

        self.cam_label = QLabel()
        self.cam_label.setFixedSize(CAM_PREVIEW_W, CAM_PREVIEW_H)
        self.cam_label.setStyleSheet(
            "background-color: rgb(32,34,37); border-radius: 6px;"
        )
        self.cam_label.setAlignment(Qt.AlignCenter)

        self.radar_label = QLabel()
        self.radar_label.setFixedSize(RADAR_SIZE, RADAR_SIZE)
        self.radar_label.setStyleSheet(
            "background-color: rgb(35,35,38); border-radius: 6px;"
        )
        self.radar_label.setAlignment(Qt.AlignCenter)

        self.forward_label = QLabel()
        self.forward_label.setFixedSize(FWD_W, FWD_H)
        self.forward_label.setStyleSheet(
            "background-color: rgb(28,28,30); border-radius: 6px;"
        )
        self.forward_label.setAlignment(Qt.AlignCenter)

        self.direction_label = QLabel()
        self.direction_label.setFixedSize(CAM_PREVIEW_W, 120)
        self.direction_label.setStyleSheet(
            "background-color: rgb(28,28,30); border-radius: 6px;"
        )
        self.direction_label.setAlignment(Qt.AlignCenter)

        self.btn_reset = QPushButton("Reset (R)")
        self.btn_pause = QPushButton("Pause (P)")
        self.btn_save = QPushButton("Save (S)")

        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_save.clicked.connect(self.on_save)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.cam_label)
        left_layout.addSpacing(6)
        left_layout.addWidget(self.direction_label)
        left_layout.addSpacing(6)
        left_layout.addWidget(self.btn_reset)
        left_layout.addWidget(self.btn_pause)
        left_layout.addWidget(self.btn_save)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.radar_label)
        right_layout.addSpacing(6)
        right_layout.addWidget(self.forward_label)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget, stretch=3)
        main_layout.addWidget(right_widget, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.latest_frame = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        self.latest_state = None
        self.paused = False
        self.trajectory = deque(maxlen=200)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(33)

        self._update_status("Starting...")
        self._fill_placeholders()

    def _fill_placeholders(self):
        black = np.zeros((CAM_PREVIEW_H, CAM_PREVIEW_W, 3), dtype=np.uint8)
        self.cam_label.setPixmap(bgr_to_qpixmap(black))
        rad = np.zeros((RADAR_SIZE, RADAR_SIZE, 3), dtype=np.uint8)
        rad[:] = (35, 35, 38)
        self.radar_label.setPixmap(bgr_to_qpixmap(rad))
        fwd = np.zeros((FWD_H, FWD_W, 3), dtype=np.uint8)
        fwd[:] = (28, 28, 30)
        self.forward_label.setPixmap(bgr_to_qpixmap(fwd))
        dir_img = np.zeros((120, CAM_PREVIEW_W, 3), dtype=np.uint8)
        dir_img[:] = (28, 28, 30)
        self.direction_label.setPixmap(bgr_to_qpixmap(dir_img))

    def closeEvent(self, event):
        logger.info("Main: shutting down")
        self.worker.stop()
        self.worker.join(timeout=2.0)
        event.accept()

    def _on_tick(self):
        try:
            item = self.q.get_nowait()
        except queue.Empty:
            return
        if not item:
            return
        frame, state = item
        self.latest_frame = frame
        self.latest_state = state

        task_obstacles = []
        if self.worker and self.worker.core:
            task_obstacles = self.worker.core.task_obstacles

        cam_preview = self._compose_camera_preview(
            frame, state.get("detections", []), task_obstacles
        )
        self.cam_label.setPixmap(bgr_to_qpixmap(cam_preview))

        pose = state.get("pose", {})
        if pose:
            self.trajectory.append((pose.get("x", 0), pose.get("y", 0)))

        trajectory_img = self._compose_trajectory(
            state.get("world", []), state.get("pose", {}), task_obstacles
        )
        self.radar_label.setPixmap(bgr_to_qpixmap(trajectory_img))

        fwd_img = self._compose_forward(state.get("detections", []))
        self.forward_label.setPixmap(bgr_to_qpixmap(fwd_img))

        direction_img = self._compose_direction(state)
        self.direction_label.setPixmap(bgr_to_qpixmap(direction_img))

        self._update_status_bar(state)

    def _compose_camera_preview(self, frame, detections, task_obstacles=None):
        preview = cv2.resize(frame, (CAM_PREVIEW_W, CAM_PREVIEW_H))
        overlay = preview.copy()

        if task_obstacles is None:
            task_obstacles = [
                "person",
                "chair",
                "couch",
                "potted plant",
                "dining table",
            ]

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            scale_x = CAM_PREVIEW_W / CAM_W
            scale_y = CAM_PREVIEW_H / CAM_H
            x1_ = int(x1 * scale_x)
            y1_ = int(y1 * scale_y)
            x2_ = int(x2 * scale_x)
            y2_ = int(y2 * scale_y)
            cls = d["class"]
            conf = d.get("confidence", d.get("conf", 0.0))
            is_target = cls in ["cup", "bottle", "book", "laptop", "cell phone"]
            is_obstacle = cls.lower() in [o.lower() for o in task_obstacles]

            if is_target:
                col = COL_TARGET
            elif is_obstacle:
                col = COL_OBST
            else:
                col = COL_NEUTRAL

            thickness = 3 if is_obstacle else 2
            cv2.rectangle(overlay, (x1_, y1_), (x2_, y2_), col, thickness, cv2.LINE_AA)

            if is_obstacle:
                cx = (x1_ + x2_) // 2
                warning_size = 20
                cv2.line(
                    overlay, (cx, y1_), (cx, y1_ + warning_size), col, 2, cv2.LINE_AA
                )
                cv2.line(
                    overlay,
                    (cx - warning_size // 2, y1_ + warning_size // 2),
                    (cx + warning_size // 2, y1_ + warning_size // 2),
                    col,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    "!",
                    (cx - 5, y1_ + warning_size + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    col,
                    2,
                    cv2.LINE_AA,
                )

            label = f"{cls} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            lx = max(4, x1_)
            ly = max(12, y1_ - 6)
            cv2.rectangle(
                overlay, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), (0, 0, 0), -1
            )
            cv2.putText(
                overlay,
                label,
                (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            bw = int((x2_ - x1_) * 0.4)
            bx = x1_ + 6
            by = y2_ + 8
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + 8), (70, 70, 70), -1)
            cv2.rectangle(overlay, (bx, by), (bx + int(bw * conf), by + 8), col, -1)

        card = np.full_like(overlay, COL_PANEL, dtype=np.uint8)
        final = cv2.addWeighted(overlay, 0.96, card, 0.04, 0)
        return final

    def _compose_trajectory(self, world_objs, pose, task_obstacles=None):
        """Create trajectory visualization showing robot path and nearby objects."""
        size = RADAR_SIZE
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:] = (35, 35, 38)

        if task_obstacles is None:
            task_obstacles = [
                "person",
                "chair",
                "couch",
                "potted plant",
                "dining table",
            ]

        if len(self.trajectory) == 0:
            return canvas

        current_x = pose.get("x", 0.0) if pose else 0.0
        current_y = pose.get("y", 0.0) if pose else 0.0

        traj_array = np.array(list(self.trajectory))
        min_x, max_x = traj_array[:, 0].min(), traj_array[:, 0].max()
        min_y, max_y = traj_array[:, 1].min(), traj_array[:, 1].max()

        range_x = max_x - min_x if max_x != min_x else 1.0
        range_y = max_y - min_y if max_y != min_y else 1.0
        max_range = max(range_x, range_y, 2.0)

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        scale = (size * 0.8) / max_range
        cx = size // 2
        cy = size // 2

        def world_to_screen(wx, wy):
            dx = wx - center_x
            dy = wy - center_y
            sx = int(cx + dx * scale)
            sy = int(cy - dy * scale)
            return sx, sy

        for o in world_objs:
            pos = o.get("pos", [0.0, 0.0, 0.0])
            obj_x, obj_y = pos[0], pos[1]
            dist = math.sqrt((obj_x - center_x) ** 2 + (obj_y - center_y) ** 2)

            if dist <= max_range * 0.6:
                sx, sy = world_to_screen(obj_x, obj_y)
                cls = o.get("class", "")
                conf = o.get("conf", 0.0)
                is_obstacle = cls.lower() in [o.lower() for o in task_obstacles]
                col = COL_OBST if is_obstacle else COL_TARGET

                if 10 <= sx < size - 10 and 10 <= sy < size - 10:
                    radius = int(4 + conf * 3)
                    cv2.circle(canvas, (sx, sy), radius, col, -1, cv2.LINE_AA)
                    cv2.circle(
                        canvas, (sx, sy), radius, (255, 255, 255), 1, cv2.LINE_AA
                    )

                    if is_obstacle:
                        warning_size = 8
                        cv2.line(
                            canvas,
                            (sx, sy - radius - 2),
                            (sx, sy - radius - warning_size),
                            col,
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.line(
                            canvas,
                            (sx - warning_size // 2, sy - radius - warning_size // 2),
                            (sx + warning_size // 2, sy - radius - warning_size // 2),
                            col,
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            canvas,
                            "!",
                            (sx - 3, sy - radius - warning_size - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            col,
                            1,
                            cv2.LINE_AA,
                        )

                    label = f"{cls[:6]}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
                    )
                    cv2.rectangle(
                        canvas,
                        (sx - 2, sy - th - 6),
                        (sx + tw + 2, sy + 2),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        canvas,
                        label,
                        (sx, sy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (230, 230, 230),
                        1,
                        cv2.LINE_AA,
                    )

        if len(self.trajectory) > 1:
            traj_points = []
            for tx, ty in self.trajectory:
                sx, sy = world_to_screen(tx, ty)
                if 0 <= sx < size and 0 <= sy < size:
                    traj_points.append((sx, sy))

            if len(traj_points) > 1:
                for i in range(len(traj_points) - 1):
                    alpha = i / max(len(traj_points) - 1, 1)
                    color_intensity = int(100 + alpha * 155)
                    cv2.line(
                        canvas,
                        traj_points[i],
                        traj_points[i + 1],
                        (color_intensity, color_intensity, color_intensity),
                        2,
                        cv2.LINE_AA,
                    )

        if pose:
            curr_sx, curr_sy = world_to_screen(current_x, current_y)
            if 0 <= curr_sx < size and 0 <= curr_sy < size:
                yaw = pose.get("yaw", 0.0)
                cv2.circle(
                    canvas, (curr_sx, curr_sy), 10, (80, 220, 150), -1, cv2.LINE_AA
                )
                cv2.circle(
                    canvas, (curr_sx, curr_sy), 10, (60, 180, 120), 2, cv2.LINE_AA
                )

                heading_len = 25
                hx = int(curr_sx + math.cos(yaw) * heading_len)
                hy = int(curr_sy - math.sin(yaw) * heading_len)
                cv2.arrowedLine(
                    canvas,
                    (curr_sx, curr_sy),
                    (hx, hy),
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                    tipLength=0.3,
                )

        cv2.putText(
            canvas,
            "Trajectory",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def _compose_forward(self, detections):
        canvas = np.zeros((FWD_H, FWD_W, 3), dtype=np.uint8)
        canvas[:] = (28, 28, 30)
        baseline_y = int(FWD_H * 0.72)
        cv2.line(
            canvas,
            (10, baseline_y),
            (FWD_W - 10, baseline_y),
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
        for d in detections:
            cls = d.get("class", "")
            dist = d.get("distance", d.get("dist", 2.0))
            cx = d["screen"][0]
            offset = (cx - CAM_W / 2) / (CAM_W / 2)
            x = int(FWD_W / 2 + offset * FWD_W * 0.35)
            size = int(max(6, min(120, 80 / max(0.1, dist))))
            y = baseline_y - int(min(80, 30 / max(0.1, dist)))
            col = (36, 200, 100) if cls in ["cup", "bottle", "book"] else (200, 60, 60)
            cv2.circle(canvas, (x, y), max(4, size // 6), col, -1, cv2.LINE_AA)
            cv2.circle(
                canvas, (x, y), max(4, size // 6), (255, 255, 255), 1, cv2.LINE_AA
            )
            label = f"{cls[:8]} {dist:.1f}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(
                canvas, (x + 8, y - th - 4), (x + tw + 12, y + 4), (0, 0, 0), -1
            )
            cv2.putText(
                canvas,
                label,
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            bar_w = 120
            bx = max(10, min(FWD_W - bar_w - 10, x - bar_w // 2))
            by = baseline_y + 8
            pct = max(0.0, min(1.0, 1.0 - (dist / 6.0)))
            cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + 8), (60, 60, 60), -1)
            cv2.rectangle(canvas, (bx, by), (bx + int(bar_w * pct), by + 8), col, -1)
        return canvas

    def _compose_direction(self, state):
        """Create directional indicator showing movement commands."""
        width = CAM_PREVIEW_W
        height = 120
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (28, 28, 30)

        cmd = state.get("command", "")

        if not cmd or cmd == "EXPLORING":
            cv2.putText(
                canvas,
                "No active navigation",
                (width // 2 - 100, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (150, 150, 150),
                1,
                cv2.LINE_AA,
            )
            return canvas

        center_x = width // 2
        center_y = height // 2

        turn_angle = 0
        pitch_angle = 0
        forward_dist = 0
        turn_dir = None
        pitch_dir = None

        cmd_parts = cmd.split(" + ")
        for cmd_part in cmd_parts:
            parts = cmd_part.split()
            if "TURN" in cmd_part:
                for i, part in enumerate(parts):
                    if part == "LEFT":
                        turn_dir = "LEFT"
                        if i + 1 < len(parts):
                            try:
                                turn_angle = int(parts[i + 1].replace("°", ""))
                            except (ValueError, IndexError):
                                pass
                    elif part == "RIGHT":
                        turn_dir = "RIGHT"
                        if i + 1 < len(parts):
                            try:
                                turn_angle = int(parts[i + 1].replace("°", ""))
                            except (ValueError, IndexError):
                                pass

            if "LOOK" in cmd_part:
                for i, part in enumerate(parts):
                    if part == "UP":
                        pitch_dir = "UP"
                        if i + 1 < len(parts):
                            try:
                                pitch_angle = int(parts[i + 1].replace("°", ""))
                            except (ValueError, IndexError):
                                pass
                    elif part == "DOWN":
                        pitch_dir = "DOWN"
                        if i + 1 < len(parts):
                            try:
                                pitch_angle = int(parts[i + 1].replace("°", ""))
                            except (ValueError, IndexError):
                                pass

            if "FORWARD" in cmd_part:
                for i, part in enumerate(parts):
                    if part == "FORWARD" and i + 1 < len(parts):
                        try:
                            forward_dist = float(parts[i + 1].replace("m", ""))
                        except (ValueError, IndexError):
                            pass

        arrow_size = 30
        arrow_thickness = 3

        if turn_dir:
            turn_rad = math.radians(turn_angle if turn_dir == "LEFT" else -turn_angle)
            arrow_x = int(center_x + math.cos(turn_rad + math.pi / 2) * arrow_size)
            arrow_y = int(center_y - math.sin(turn_rad + math.pi / 2) * arrow_size)

            if turn_dir == "LEFT":
                cv2.arrowedLine(
                    canvas,
                    (center_x, center_y),
                    (arrow_x, arrow_y),
                    COL_ACCENT,
                    arrow_thickness,
                    cv2.LINE_AA,
                    tipLength=0.3,
                )
                cv2.putText(
                    canvas,
                    f"LEFT {turn_angle}°",
                    (center_x - 80, center_y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COL_ACCENT,
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.arrowedLine(
                    canvas,
                    (center_x, center_y),
                    (arrow_x, arrow_y),
                    COL_ACCENT,
                    arrow_thickness,
                    cv2.LINE_AA,
                    tipLength=0.3,
                )
                cv2.putText(
                    canvas,
                    f"RIGHT {turn_angle}°",
                    (center_x + 20, center_y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COL_ACCENT,
                    1,
                    cv2.LINE_AA,
                )

        if pitch_dir:
            pitch_rad = math.radians(pitch_angle if pitch_dir == "UP" else -pitch_angle)
            arrow_x = int(center_x + math.cos(pitch_rad) * arrow_size)
            arrow_y = int(center_y - math.sin(pitch_rad) * arrow_size)

            if pitch_dir == "UP":
                cv2.arrowedLine(
                    canvas,
                    (center_x, center_y),
                    (arrow_x, arrow_y),
                    (100, 200, 255),
                    arrow_thickness,
                    cv2.LINE_AA,
                    tipLength=0.3,
                )
                cv2.putText(
                    canvas,
                    f"UP {pitch_angle}°",
                    (center_x - 50, center_y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (100, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.arrowedLine(
                    canvas,
                    (center_x, center_y),
                    (arrow_x, arrow_y),
                    (100, 200, 255),
                    arrow_thickness,
                    cv2.LINE_AA,
                    tipLength=0.3,
                )
                cv2.putText(
                    canvas,
                    f"DOWN {pitch_angle}°",
                    (center_x - 60, center_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (100, 200, 255),
                    1,
                    cv2.LINE_AA,
                )

        if forward_dist > 0:
            if turn_dir:
                cv2.arrowedLine(
                    canvas,
                    (center_x, center_y + 20),
                    (center_x, center_y + 20 - arrow_size * 0.7),
                    COL_TARGET,
                    arrow_thickness - 1,
                    cv2.LINE_AA,
                    tipLength=0.3,
                )
                cv2.putText(
                    canvas,
                    f"{forward_dist:.1f}m",
                    (center_x - 30, center_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    COL_TARGET,
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.arrowedLine(
                    canvas,
                    (center_x, center_y),
                    (center_x, center_y - arrow_size),
                    COL_TARGET,
                    arrow_thickness,
                    cv2.LINE_AA,
                    tipLength=0.3,
                )
                cv2.putText(
                    canvas,
                    f"FORWARD {forward_dist:.2f}m",
                    (center_x - 70, center_y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COL_TARGET,
                    1,
                    cv2.LINE_AA,
                )

        if "TARGET REACHED" in cmd:
            cv2.putText(
                canvas,
                "TARGET REACHED",
                (center_x - 80, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COL_TARGET,
                2,
                cv2.LINE_AA,
            )

        if "SEARCHING" in cmd:
            cv2.putText(
                canvas,
                "SEARCHING...",
                (center_x - 60, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 100),
                1,
                cv2.LINE_AA,
            )

        return canvas

    def _update_status_bar(self, state):
        """Update status bar with clean, concise information."""
        cmd = state.get("command", "")
        mapped = len(state.get("world", []))
        pose = state.get("pose", {})

        if cmd:
            if "TURN" in cmd or "LOOK" in cmd or "FORWARD" in cmd:
                cmd = "NAVIGATING"
            elif "TARGET REACHED" in cmd:
                cmd = "TARGET REACHED"
            elif "SEARCHING" in cmd:
                cmd = "SEARCHING"
            elif "EXPLORING" in cmd:
                cmd = "EXPLORING"

        status_text = f"Status: {cmd}  |  Objects: {mapped}  |  Position: ({pose.get('x', 0):.2f}, {pose.get('y', 0):.2f}, {pose.get('z', 0):.2f})"
        self._update_status(status_text)

    def on_reset(self):
        self.worker.reset_map()
        self.trajectory.clear()
        self._update_status("Map reset")

    def on_pause(self):
        self.paused = not self.paused
        self.worker.pause(self.paused)
        self._update_status("Paused" if self.paused else "Running")

    def on_save(self):
        ts = int(time.time())
        try:
            cam = self.latest_frame.copy()
            task_obstacles = []
            if self.worker and self.worker.core:
                task_obstacles = self.worker.core.task_obstacles

            rad = (
                self._compose_trajectory(
                    self.latest_state.get("world", []),
                    self.latest_state.get("pose", {}),
                    task_obstacles,
                )
                if self.latest_state
                else np.zeros((RADAR_SIZE, RADAR_SIZE, 3), np.uint8)
            )
            fwd = (
                self._compose_forward(self.latest_state.get("detections", []))
                if self.latest_state
                else np.zeros((FWD_H, FWD_W, 3), np.uint8)
            )
            h_total = CAM_PREVIEW_H + FWD_H + 30
            w_total = CAM_PREVIEW_W + RADAR_SIZE + 40
            canvas = np.zeros((h_total, w_total, 3), dtype=np.uint8)
            canvas[:] = (18, 20, 23)
            cam_r = cv2.resize(cam, (CAM_PREVIEW_W, CAM_PREVIEW_H))
            canvas[20 : 20 + CAM_PREVIEW_H, 20 : 20 + CAM_PREVIEW_W] = cam_r
            canvas[
                20 : 20 + RADAR_SIZE,
                40 + CAM_PREVIEW_W : 40 + CAM_PREVIEW_W + RADAR_SIZE,
            ] = rad
            fwd_r = cv2.resize(fwd, (FWD_W, FWD_H))
            canvas[40 + CAM_PREVIEW_H : 40 + CAM_PREVIEW_H + FWD_H, 20 : 20 + FWD_W] = (
                fwd_r
            )
            fn = f"pathfinder_shot_{ts}.png"
            cv2.imwrite(fn, canvas)
            self._update_status(f"Saved {fn}")
        except Exception as e:
            logger.error(f"Save failed: {e}")
            self._update_status("Save failed")

    def _update_status(self, text):
        self.status.showMessage(text)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Q:
            self.close()
        elif key == Qt.Key_R:
            self.on_reset()
        elif key == Qt.Key_P:
            self.on_pause()
        elif key == Qt.Key_S:
            self.on_save()
        else:
            super().keyPressEvent(event)


def main(argv):
    task = " ".join(argv[1:]) if len(argv) > 1 else "find bottle, avoid person"
    app = QApplication(sys.argv)
    window = MainWindow(task, model_path=MODEL_PATH)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main(sys.argv)
