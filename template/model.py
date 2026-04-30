"""
Copy this file to models/<your_model_id>/model.py and implement the three methods.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Makes 'from models.base import BaseModel' work when running this file directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.base import BaseModel  # noqa: E402


# Map your model's class labels to GBIF species URLs
GBIF_IDS: dict[str, str] = {
    # "jaguar": "https://www.gbif.org/species/5219426",
}

CLASS_NAMES: list[str] = [
    # "jaguar",
]

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
            "bbox": [x1, y1, x2, y2]
            "acceptedNameUsageID": "https://www.gbif.org/species/<id>"
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
