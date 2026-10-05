from __future__ import annotations

import importlib.util
import io
import logging
import os
from pathlib import Path

from detectron2.data.detection_utils import convert_PIL_to_numpy
from PIL import Image

_spec = importlib.util.spec_from_file_location("_base", Path(__file__).parent.parent / "base.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
BaseModel = _base.BaseModel

logger = logging.getLogger(__name__)

# The class labels this model can output. Their scientific names are declared
# in classes.json; the platform resolves each label to its name and the
# middleware resolves that to a COL id — no ids or vernacular names here.
THING_CLASSES = ["jaguar", "collared_peccary", "cattle"]

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

        img_array = convert_PIL_to_numpy(Image.open(io.BytesIO(image_bytes)), "BGR")
        return self._predictor(img_array)

    def map_result(self, raw: object) -> list[dict]:
        instances = raw["instances"]
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes
        scores = instances.scores.numpy()

        results = []
        for i in range(len(boxes)):
            class_idx = int(classes[i])
            if class_idx >= len(THING_CLASSES):
                # Model predicted a class index outside our known set — likely a
                # config/weights mismatch. Skip rather than 500 the request.
                logger.warning(f"Class index {class_idx} out of range (have {len(THING_CLASSES)} classes) — skipping")
                continue
            class_name = THING_CLASSES[class_idx]
            results.append({
                "class": class_name,
                "bbox": boxes[i].tolist(),
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
