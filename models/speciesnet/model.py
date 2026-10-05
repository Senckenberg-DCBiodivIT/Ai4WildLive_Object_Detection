from __future__ import annotations

import importlib.util
import io
import logging
import tempfile
from pathlib import Path

from PIL import Image

_spec = importlib.util.spec_from_file_location("_base", Path(__file__).parent.parent / "base.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
BaseModel = _base.BaseModel

logger = logging.getLogger(__name__)

SCORE_THRESHOLD: float = 0.5


class SpeciesNetModel(BaseModel):

    def load(self) -> None:
        from speciesnet import DEFAULT_MODEL, SpeciesNet
        self._model = SpeciesNet(DEFAULT_MODEL)

    def predict(self, image_bytes: bytes) -> object:

        with Image.open(io.BytesIO(image_bytes)) as img:
            self._img_width, self._img_height = img.size

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            result = self._model.predict(filepaths=[tmp_path], run_mode="multi_thread")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return result

    def map_result(self, raw: object) -> list[dict]:
        if not raw or not raw.get("predictions"):
            return []

        pred = raw["predictions"][0]
        prediction_class = pred.get("prediction", "")
        prediction_score = float(pred.get("prediction_score", 0.0))

        if prediction_score < SCORE_THRESHOLD:
            return []

        # Class format: "uuid;class;order;family;genus;species;common_name"
        # the middleware resolves the vernacular name from the Catalogue of Life.
        parts = prediction_class.split(";")
        genus = parts[4].strip() if len(parts) > 4 else ""
        species = parts[5].strip() if len(parts) > 5 else ""

        if not genus or not species:
            return []

        scientific_name = f"{genus.capitalize()} {species}"

        detections = pred.get("detections", [])
        if not detections:
            return []

        results = []
        for det in detections:
            det_score = float(det.get("conf", det.get("score", 0.0)))
            if det_score < SCORE_THRESHOLD:
                continue

            bbox = det.get("bbox", [])
            if isinstance(bbox, dict):
                bbox = [bbox.get("x_min", 0.0), bbox.get("y_min", 0.0),
                        bbox.get("x_max", 0.0), bbox.get("y_max", 0.0)]
            if len(bbox) != 4:
                # SpeciesNet sometimes emits classification-only predictions
                # with no detection box. Skip — we have nothing to localise.
                continue

            x, y, w, h = bbox
            bbox = [
                x * self._img_width,
                y * self._img_height,
                (x + w) * self._img_width,
                (y + h) * self._img_height,
            ]

            results.append({
                "scientificName": scientific_name,
                "bbox": bbox,
                "score": prediction_score,
            })

        return results


instance = SpeciesNetModel()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Path to a test image")
    parser.add_argument("--step", choices=["structure", "load", "predict", "all"], default="all")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    instance.test(image_path=args.image, step=args.step, threshold=args.threshold)
