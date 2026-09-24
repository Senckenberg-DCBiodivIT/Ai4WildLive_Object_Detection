"""
Copy this file to models/<your_model_id>/model.py and implement the three methods.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_base", Path(__file__).parent.parent / "base.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
BaseModel = _base.BaseModel


# classes.json maps every label your model can output to its scientific name
# (or null for non-taxon classes like "vehicle"). The platform resolves that
# name against the Catalogue of Life for the taxon ID, vernacular name, and
# higher taxonomy — declare the scientific name only, nothing else.
CLASSES: dict[str, str | None] = json.loads((Path(__file__).parent / "classes.json").read_text())

SCORE_THRESHOLD: float = 0.5


class Model(BaseModel):

    def load(self) -> None:
        """Load your model weights here. Called once before first prediction."""
        raise NotImplementedError

    def predict(self, image_bytes: bytes) -> object:  # noqa: ARG002
        """Run inference. Return the raw output from your framework."""
        raise NotImplementedError

    def map_result(self, raw: object) -> list[dict]:  # noqa: ARG002
        """
        Convert raw output to the standard format.

        Each entry must have:
            "bbox":  [x1, y1, x2, y2]
            "class": "jaguar"          # must be a key in classes.json
            "score": 0.95
        """
        raise NotImplementedError


instance = Model()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Path to a test image")
    parser.add_argument("--step", choices=["structure", "load", "predict", "all"],
                        default="all")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    instance.test(image_path=args.image, step=args.step, threshold=args.threshold)
