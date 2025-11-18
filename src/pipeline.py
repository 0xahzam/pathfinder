from vision import VisionPipeline
from planner import BehaviorPlanner
from loguru import logger
import cv2
from pathlib import Path
import json
from datetime import datetime


class PathfinderPipeline:
    def __init__(self, target_class: str = "cup"):
        self.vision = VisionPipeline()
        self.planner = BehaviorPlanner(target_class=target_class)
        logger.success("pipeline initialized")

    def process_frame(self, frame):
        h, w = frame.shape[:2]

        detections = self.vision.detect(frame)
        action, context = self.planner.decide(detections, frame_width=w, frame_height=h)

        return {
            "timestamp": datetime.now().isoformat(),
            "detections": detections,
            "action": action.value,
            "context": context,
        }

    def process_image(self, image_path: str, output_dir: str = "data/pipeline_outputs"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"processing: {image_path}")
        frame = cv2.imread(image_path)

        if frame is None:
            logger.error(f"failed to load: {image_path}")
            return None

        result = self.process_frame(frame)

        logger.info(f"action: {result['action']}")
        logger.info(f"detections: {len(result['detections'])}")

        annotated = self.vision.draw_detections(frame.copy(), result["detections"])

        action_text = f"ACTION: {result['action'].upper()}"
        cv2.putText(
            annotated,
            action_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        img_name = Path(image_path).stem
        output_img = output_path / f"{img_name}_result.jpg"
        output_json = output_path / f"{img_name}_result.json"

        cv2.imwrite(str(output_img), annotated)

        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)

        logger.success(f"saved: {output_img}")
        logger.success(f"saved: {output_json}")

        return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logger.error("usage: python pipeline.py <image_path> [target_class]")
        sys.exit(1)

    image_path = sys.argv[1]
    target_class = sys.argv[2] if len(sys.argv) > 2 else "cup"

    pipeline = PathfinderPipeline(target_class=target_class)
    pipeline.process_image(image_path)
