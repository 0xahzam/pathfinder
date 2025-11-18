import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vision import VisionPipeline
from loguru import logger
import cv2


def test_on_sample():
    vision = VisionPipeline()

    test_dir = Path(__file__).parent.parent / "data" / "test_images"
    output_dir = Path(__file__).parent.parent / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images = list(test_dir.glob("*.jpg"))

    if not test_images:
        logger.error(f"no images found in {test_dir}")
        return

    for img_path in test_images:
        logger.info(f"processing: {img_path.name}")

        frame = cv2.imread(str(img_path))
        detections = vision.detect(frame)

        logger.info(f"  found {len(detections)} objects")
        for det in detections:
            logger.info(f"    {det['class']}: {det['confidence']:.3f}")

        annotated = vision.draw_detections(frame, detections)
        output_path = output_dir / f"detected_{img_path.name}"
        cv2.imwrite(str(output_path), annotated)
        logger.success(f"  saved: {output_path}")


if __name__ == "__main__":
    test_on_sample()
