from pipeline import PathfinderPipeline
from planner import Action
from loguru import logger
import cv2
import numpy as np


class SimpleSimulator:
    def __init__(self, target_class: str = "cup"):
        self.pipeline = PathfinderPipeline(target_class=target_class)
        self.robot_angle = 0
        self.robot_pos = [320, 400]
        logger.success("simulator initialized")

    def draw_robot_view(self, frame, action):
        h, w = frame.shape[:2]
        view = np.zeros((h, w + 300, 3), dtype=np.uint8)
        view[:h, :w] = frame

        info_x = w + 10
        cv2.putText(
            view,
            "ROBOT STATUS",
            (info_x, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            view,
            f"Action: {action.value.upper()}",
            (info_x, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        cv2.putText(
            view,
            f"Angle: {self.robot_angle}deg",
            (info_x, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            view,
            f"Pos: ({self.robot_pos[0]}, {self.robot_pos[1]})",
            (info_x, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        self.draw_top_down_view(view, info_x, 160)

        if action == Action.TURN_LEFT:
            color = (0, 255, 255)
        elif action == Action.TURN_RIGHT:
            color = (255, 0, 255)
        elif action == Action.APPROACH_TARGET:
            color = (0, 255, 0)
        elif action == Action.SEARCH:
            color = (255, 255, 0)
        else:
            color = (128, 128, 128)

        cv2.rectangle(view, (0, 0), (w, h), color, 5)

        return view

    def draw_top_down_view(self, view, x, y):
        map_size = 200
        cv2.rectangle(view, (x, y), (x + map_size, y + map_size), (50, 50, 50), -1)
        cv2.rectangle(view, (x, y), (x + map_size, y + map_size), (255, 255, 255), 2)

        robot_x = int(x + map_size / 2)
        robot_y = int(y + map_size / 2)

        cv2.circle(view, (robot_x, robot_y), 8, (0, 255, 0), -1)

        angle_rad = np.radians(self.robot_angle)
        end_x = int(robot_x + 20 * np.cos(angle_rad))
        end_y = int(robot_y - 20 * np.sin(angle_rad))
        cv2.arrowedLine(view, (robot_x, robot_y), (end_x, end_y), (0, 255, 0), 2)

        cv2.putText(
            view,
            "Top View",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    def update_robot_state(self, action):
        if action == Action.TURN_LEFT:
            self.robot_angle = (self.robot_angle + 15) % 360
        elif action == Action.TURN_RIGHT:
            self.robot_angle = (self.robot_angle - 15) % 360
        elif action == Action.APPROACH_TARGET or action == Action.MOVE_FORWARD:
            angle_rad = np.radians(self.robot_angle)
            self.robot_pos[0] += int(5 * np.cos(angle_rad))
            self.robot_pos[1] -= int(5 * np.sin(angle_rad))
        elif action == Action.SEARCH:
            self.robot_angle = (self.robot_angle + 5) % 360

    def run_on_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"failed to open video: {video_path}")
            return

        logger.info("press 'q' to quit")

        while True:
            ret, frame = cap.read()

            if not ret:
                logger.info("video ended, looping")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (640, 480))

            result = self.pipeline.process_frame(frame)
            action = Action(result["action"])

            annotated = self.pipeline.vision.draw_detections(
                frame.copy(), result["detections"]
            )
            view = self.draw_robot_view(annotated, action)

            self.update_robot_state(action)

            cv2.imshow("Pathfinder Simulation", view)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_on_webcam(self, camera_id: int = 0):
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            logger.error("failed to open webcam")
            return

        logger.info("press 'q' to quit")

        while True:
            ret, frame = cap.read()

            if not ret:
                logger.error("failed to read frame")
                break

            frame = cv2.resize(frame, (640, 480))

            result = self.pipeline.process_frame(frame)
            action = Action(result["action"])

            annotated = self.pipeline.vision.draw_detections(
                frame.copy(), result["detections"]
            )
            view = self.draw_robot_view(annotated, action)

            self.update_robot_state(action)

            cv2.imshow("Pathfinder Simulation", view)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "cup"

    sim = SimpleSimulator(target_class=target)

    logger.info("starting webcam demo")
    sim.run_on_webcam()
