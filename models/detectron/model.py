from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.base import BaseModel  # noqa: E402

logger = logging.getLogger(__name__)

THING_CLASSES = ["jaguar", "collared_peccary", "cattle"]
GBIF_IDS = {
    "jaguar": "https://www.gbif.org/species/5219426",
    "collared_peccary": "https://www.gbif.org/species/2440995",
    "cattle": "https://www.gbif.org/species/2441022",
}

BASE_DIR = os.path.dirname(__file__)


class Model(BaseModel):

    def load(self) -> None:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine.defaults import DefaultPredictor

        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml")
        )
        cfg.merge_from_file(os.path.join(BASE_DIR, "custom_model_config.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(BASE_DIR, "model_final.pth")
        cfg.freeze()
        self._predictor = DefaultPredictor(cfg)
        logger.info("Detectron v1 model loaded")

    def predict(self, image_bytes: bytes) -> object:
        from PIL import Image
        from detectron2.data.detection_utils import convert_PIL_to_numpy

        img_array = convert_PIL_to_numpy(Image.open(io.BytesIO(image_bytes)), "BGR")
        return self._predictor(img_array)

    def map_result(self, raw: object) -> list[dict]:
        instances = raw["instances"]
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes
        scores = instances.scores.numpy()

        results = []
        for i in range(len(boxes)):
            class_name = THING_CLASSES[classes[i]]
            results.append({
                "bbox": boxes[i].tolist(),
                "acceptedNameUsageID": GBIF_IDS.get(class_name, ""),
                "score": float(scores[i]),
            })
            logger.debug(f"Detected: {class_name} ({scores[i]:.3f})")

        logger.info(f"Detectron v1: {len(results)} detections")
        return results


instance = Model()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test this model locally")
    parser.add_argument("image", nargs="?", help="Path to a test image")
    parser.add_argument("--step", choices=["structure", "load", "predict", "all"], default="all")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    instance.test(image_path=args.image, step=args.step, threshold=args.threshold)
