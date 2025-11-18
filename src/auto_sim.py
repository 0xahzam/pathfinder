from pipeline import PathfinderPipeline
from planner import Action
from loguru import logger
import cv2
import numpy as np
from pathlib import Path


class AutoSimulator:
    def __init__(self, target_class: str = "cup"):
        self.pipeline = PathfinderPipeline(target_class=target_class)
        self.robot_angle = 0
        self.robot_pos = np.array([400.0, 300.0])
        self.world_size = (800, 600)
        self.objects = []
        self.setup_world(target_class)
        logger.success("auto simulator initialized")

    def setup_world(self, target_class):
        self.objects = [
            {"class": "chair", "pos": (200, 200), "size": 80},
            {"class": "table", "pos": (600, 400), "size": 100},
            {"class": target_class, "pos": (400, 100), "size": 40},
            {"class": "person", "pos": (150, 450), "size": 60},
        ]
        logger.info(f"world setup with {len(self.objects)} objects")

    def render_world(self):
        world = (
            np.ones((self.world_size[1], self.world_size[0], 3), dtype=np.uint8) * 240
        )

        for obj in self.objects:
            color = self.get_object_color(obj["class"])
            pos = obj["pos"]
            size = obj["size"]
            cv2.rectangle(
                world,
                (pos[0] - size // 2, pos[1] - size // 2),
                (pos[0] + size // 2, pos[1] + size // 2),
                color,
                -1,
            )
            cv2.putText(
                world,
                obj["class"],
                (pos[0] - size // 2, pos[1] - size // 2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

        rx, ry = int(self.robot_pos[0]), int(self.robot_pos[1])
        cv2.circle(world, (rx, ry), 15, (0, 255, 0), -1)

        angle_rad = np.radians(self.robot_angle)
        end_x = int(rx + 30 * np.cos(angle_rad))
        end_y = int(ry - 30 * np.sin(angle_rad))
        cv2.arrowedLine(world, (rx, ry), (end_x, end_y), (0, 150, 0), 3)

        return world

    def get_object_color(self, obj_class):
        colors = {
            "chair": (139, 69, 19),
            "table": (160, 82, 45),
            "cup": (255, 200, 0),
            "bottle": (0, 200, 255),
            "person": (255, 100, 100),
        }
        return colors.get(obj_class, (128, 128, 128))

    def render_robot_view(self):
        view_width = 640
        view_height = 480
        fov = 60

        camera = np.ones((view_height, view_width, 3), dtype=np.uint8) * 200

        detections = []

        for obj in self.objects:
            obj_pos = np.array(obj["pos"])
            rel_pos = obj_pos - self.robot_pos

            distance = np.linalg.norm(rel_pos)

            if distance > 400:
                continue

            angle_to_obj = np.degrees(np.arctan2(-rel_pos[1], rel_pos[0]))
            angle_diff = (angle_to_obj - self.robot_angle + 180) % 360 - 180

            if abs(angle_diff) < fov / 2:
                screen_x = int(view_width / 2 + (angle_diff / fov) * view_width)

                apparent_size = int(obj["size"] * 300 / max(distance, 50))
                screen_y = int(view_height / 2 + apparent_size / 4)

                x1 = max(0, screen_x - apparent_size // 2)
                y1 = max(0, screen_y - apparent_size)
                x2 = min(view_width, screen_x + apparent_size // 2)
                y2 = min(view_height, screen_y)

                if x2 > x1 and y2 > y1:
                    color = self.get_object_color(obj["class"])
                    cv2.rectangle(camera, (x1, y1), (x2, y2), color, -1)

                    confidence = max(0.5, 1.0 - distance / 400)
                    detections.append(
                        {
                            "class": obj["class"],
                            "confidence": confidence,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        }
                    )

        return camera, detections

    def update_robot_state(self, action):
        if action == Action.TURN_LEFT:
            self.robot_angle = (self.robot_angle + 10) % 360
        elif action == Action.TURN_RIGHT:
            self.robot_angle = (self.robot_angle - 10) % 360
        elif action == Action.APPROACH_TARGET or action == Action.MOVE_FORWARD:
            angle_rad = np.radians(self.robot_angle)
            new_pos = self.robot_pos + np.array(
                [8 * np.cos(angle_rad), -8 * np.sin(angle_rad)]
            )

            new_pos[0] = np.clip(new_pos[0], 20, self.world_size[0] - 20)
            new_pos[1] = np.clip(new_pos[1], 20, self.world_size[1] - 20)

            self.robot_pos = new_pos
        elif action == Action.SEARCH:
            self.robot_angle = (self.robot_angle + 5) % 360

    def create_display(self, world, camera, action, detections):
        display = np.zeros((600, 1440, 3), dtype=np.uint8)

        display[:600, :800] = cv2.resize(world, (800, 600))

        camera_display = cv2.resize(camera, (640, 480))
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.rectangle(camera_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                camera_display,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        display[60:540, 800:1440] = camera_display

        cv2.putText(
            display,
            "TOP-DOWN VIEW",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display,
            "ROBOT CAMERA",
            (810, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        action_color = {
            Action.TURN_LEFT: (0, 255, 255),
            Action.TURN_RIGHT: (255, 0, 255),
            Action.APPROACH_TARGET: (0, 255, 0),
            Action.SEARCH: (255, 255, 0),
            Action.STOP: (0, 0, 255),
        }.get(action, (128, 128, 128))

        cv2.putText(
            display,
            f"ACTION: {action.value.upper()}",
            (810, 560),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            action_color,
            2,
        )

        return display

    def run(self, steps: int = 500):
        logger.info(f"running simulation for {steps} steps")
        logger.info("press 'q' to quit early")

        for step in range(steps):
            world = self.render_world()
            camera, detections = self.render_robot_view()

            result = self.pipeline.planner.decide(
                detections, frame_width=640, frame_height=480
            )
            action, context = result

            display = self.create_display(world, camera, action, detections)

            cv2.imshow("Pathfinder Auto Simulation", display)

            if step % 50 == 0:
                logger.info(
                    f"step {step}: action={action.value}, detections={len(detections)}"
                )

            self.update_robot_state(action)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                logger.info("simulation stopped by user")
                break

        cv2.destroyAllWindows()
        logger.success("simulation complete")


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "cup"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    sim = AutoSimulator(target_class=target)
    sim.run(steps=steps)
