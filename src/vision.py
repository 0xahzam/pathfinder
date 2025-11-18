from ultralytics import YOLO
import cv2
from loguru import logger


class VisionPipeline:
    def __init__(self, model_name: str = "yolov8n.pt"):
        logger.info(f"loading model: {model_name}")
        self.model = YOLO(model_name)
        logger.success("model loaded")

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = results.names[cls]

            detections.append(
                {
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

        return detections

    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = f"{det['class']} {det['confidence']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return frame


def process_image(image_path: str, output_path: str = None):
    vision = VisionPipeline()

    logger.info(f"processing: {image_path}")
    frame = cv2.imread(image_path)

    if frame is None:
        logger.error(f"failed to load image: {image_path}")
        return None

    detections = vision.detect(frame)
    logger.info(f"detected {len(detections)} objects")

    for det in detections:
        logger.debug(f"  {det['class']}: {det['confidence']:.3f}")

    annotated = vision.draw_detections(frame.copy(), detections)

    if output_path:
        cv2.imwrite(output_path, annotated)
        logger.success(f"saved to: {output_path}")

    return detections, annotated


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logger.error("usage: python vision.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.jpg"

    process_image(image_path, output_path)
