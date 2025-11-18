import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from planner import BehaviorPlanner, Action
from loguru import logger


def test_no_target():
    planner = BehaviorPlanner(target_class="cup")

    detections = [
        {"class": "chair", "confidence": 0.9, "bbox": [100, 200, 300, 400]},
        {"class": "person", "confidence": 0.85, "bbox": [400, 150, 500, 450]},
    ]

    action, context = planner.decide(detections)
    logger.info(f"action: {action.value}")
    assert action == Action.SEARCH


def test_target_centered():
    planner = BehaviorPlanner(target_class="cup")

    detections = [{"class": "cup", "confidence": 0.8, "bbox": [300, 200, 340, 250]}]

    action, context = planner.decide(detections, frame_width=640)
    logger.info(f"action: {action.value}")
    assert action == Action.APPROACH_TARGET


def test_target_left():
    planner = BehaviorPlanner(target_class="cup")

    detections = [{"class": "cup", "confidence": 0.8, "bbox": [50, 200, 100, 250]}]

    action, context = planner.decide(detections, frame_width=640)
    logger.info(f"action: {action.value}")
    assert action == Action.TURN_LEFT


def test_obstacle_blocking():
    planner = BehaviorPlanner(target_class="bottle")

    detections = [
        {"class": "bottle", "confidence": 0.75, "bbox": [100, 150, 150, 300]},
        {"class": "chair", "confidence": 0.9, "bbox": [280, 250, 360, 400]},
    ]

    action, context = planner.decide(detections, frame_width=640)
    logger.info(f"action: {action.value}, reason: {context}")
    assert action in [Action.TURN_LEFT, Action.TURN_RIGHT]


if __name__ == "__main__":
    logger.info("test: no target detected")
    test_no_target()

    logger.info("\ntest: target centered")
    test_target_centered()

    logger.info("\ntest: target on left")
    test_target_left()

    logger.info("\ntest: obstacle blocking")
    test_obstacle_blocking()

    logger.success("\nall tests passed")
