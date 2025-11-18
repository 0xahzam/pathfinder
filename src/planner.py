from loguru import logger
from enum import Enum


class Action(Enum):
    STOP = "stop"
    MOVE_FORWARD = "move_forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    APPROACH_TARGET = "approach_target"
    SEARCH = "search"


class BehaviorPlanner:
    def __init__(self, target_class: str = "cup"):
        self.target_class = target_class
        self.obstacle_classes = {
            "chair",
            "couch",
            "dining table",
            "bed",
            "toilet",
            "tv",
            "laptop",
        }
        self.dynamic_classes = {"person", "dog", "cat"}
        logger.info(f"planner initialized, target: {self.target_class}")

    def classify_detection(self, detection):
        cls = detection["class"]
        conf = detection["confidence"]

        if cls == self.target_class and conf > 0.5:
            return "target"
        elif cls in self.obstacle_classes and conf > 0.5:
            return "obstacle"
        elif cls in self.dynamic_classes and conf > 0.5:
            return "dynamic"
        else:
            return "ignore"

    def calculate_bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return cx, cy

    def decide(self, detections, frame_width=640, frame_height=480):
        targets = []
        obstacles = []
        dynamics = []

        for det in detections:
            category = self.classify_detection(det)

            if category == "target":
                targets.append(det)
            elif category == "obstacle":
                obstacles.append(det)
            elif category == "dynamic":
                dynamics.append(det)

        logger.debug(
            f"targets: {len(targets)}, obstacles: {len(obstacles)}, dynamics: {len(dynamics)}"
        )

        center_x = frame_width / 2
        center_y = frame_height / 2

        if not targets:
            logger.info("no target visible, searching")
            return Action.SEARCH, None

        target = targets[0]
        tx, ty = self.calculate_bbox_center(target["bbox"])

        if obstacles:
            for obs in obstacles:
                ox, oy = self.calculate_bbox_center(obs["bbox"])
                if abs(ox - center_x) < frame_width * 0.3 and oy > center_y:
                    logger.warning(f"obstacle {obs['class']} blocking path")

                    if tx < center_x:
                        logger.info("turning left to avoid")
                        return Action.TURN_LEFT, {"reason": "avoid_obstacle"}
                    else:
                        logger.info("turning right to avoid")
                        return Action.TURN_RIGHT, {"reason": "avoid_obstacle"}

        if dynamics:
            logger.info(f"dynamic entity detected, slowing approach")

        offset = tx - center_x
        threshold = frame_width * 0.1

        if abs(offset) < threshold:
            logger.info(f"target centered, approaching {target['class']}")
            return Action.APPROACH_TARGET, {"target": target}
        elif offset < -threshold:
            logger.info("target left, turning left")
            return Action.TURN_LEFT, {"target": target}
        else:
            logger.info("target right, turning right")
            return Action.TURN_RIGHT, {"target": target}
