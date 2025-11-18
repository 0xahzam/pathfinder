import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import PathfinderPipeline
from loguru import logger


def test_pipeline():
    test_dir = Path(__file__).parent.parent / "data" / "test_images"
    test_images = list(test_dir.glob("*.jpg"))

    if not test_images:
        logger.error("no test images found")
        return

    pipeline = PathfinderPipeline(target_class="dog")

    for img_path in test_images:
        logger.info(f"\n{'=' * 50}")
        result = pipeline.process_image(str(img_path))

        if result:
            logger.info(f"final action: {result['action']}")


if __name__ == "__main__":
    test_pipeline()
