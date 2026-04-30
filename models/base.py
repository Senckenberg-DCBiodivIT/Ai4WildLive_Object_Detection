from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path


class BaseModel(ABC):
    """
    Base class for all local object detection model providers.

    Subclass this in your models/<name>/model.py and implement
    load(), predict(), and map_result().

    The model_adapter calls instance.run(image_bytes) which
    orchestrates load-once → predict → map_result automatically.

    During development, call instance.test() directly from your model file:
        python models/my_model/model.py --step structure
        python models/my_model/model.py image.jpg
    """

    _loaded: bool = False

    # Abstract interface — must be implemented by every model
    @abstractmethod
    def load(self) -> None:
        """
        Load model weights and initialise the predictor.
        Called once on first use; result is kept in memory.
        """

    @abstractmethod
    def predict(self, image_bytes: bytes) -> object:
        """
        Run inference on raw image bytes.
        Returns whatever the underlying framework produces
        (e.g. Detectron2 Instances, YOLO Results, …).
        map_result() will convert this to the standard format.
        """

    @abstractmethod
    def map_result(self, raw: object) -> list[dict]:
        """
        Convert raw framework output to the standard prediction format.

        Each entry in the returned list must be:
            {
                "bbox": [x1, y1, x2, y2],          # pixel coords
                "acceptedNameUsageID": "<gbif_url>", # GBIF species URL
                "score": 0.95,                       # confidence 0–1
            }
        """

    # Orchestration — called by model_adapter, no need to override
    def run(self, image_bytes: bytes) -> list[dict]:
        """Load once, predict, map — the single entry point used by model_adapter."""
        if not self._loaded:
            self.load()
            self._loaded = True
        raw = self.predict(image_bytes)
        return self.map_result(raw)

    # Local testing — call instance.test() from your model file

    def test(self, image_path: str | None = None, step: str = "all",
             threshold: float = 0.5) -> None:
        """
        Verify your implementation step by step.

        Steps (run in order):
            structure  — check class structure, which methods are implemented
            load       — call load(), verify weights load without error
            predict    — load() + predict(), print raw output
            all        — full pipeline including map_result() (default)

        Called automatically when running your model file directly:
            python models/my_model/model.py --step structure
            python models/my_model/model.py image.jpg
        """
        _print_header(type(self).__name__, step)

        _check_structure(self)
        if step == "structure":
            return

        _check_load(self)
        if step == "load":
            return

        if not image_path:
            _die("An image path is required for --step predict / all")

        image_bytes = _load_image(image_path)
        raw = _check_predict(self, image_bytes)
        if step == "predict":
            return

        _check_map_result(self, raw, threshold)


# ── internal test helpers (not part of the public API) ───────────────────────

def _die(msg: str) -> None:
    print(f"\nERROR: {msg}")
    sys.exit(1)


def _print_header(class_name: str, step: str) -> None:
    print("=" * 58)
    print(f"  WildLive Model Tester")
    print(f"  Model: {class_name}   Step: {step}")
    print("=" * 58)


def _load_image(path: str) -> bytes:
    p = Path(path)
    if not p.exists():
        _die(f"Image file not found: {path}\n  Please check the path and try again.")
    data = p.read_bytes()
    print(f"  Image loaded ({len(data):,} bytes)")
    return data


def _check_structure(inst: BaseModel) -> None:
    print("\nStep 1 of 4 - Checking your code structure")
    print("  Looking for the three required methods ...\n")
    not_done = []
    for method in ("load", "predict", "map_result"):
        try:
            if method == "load":
                inst.load()
            elif method == "predict":
                inst.predict(b"")
            else:
                inst.map_result(None)
        except NotImplementedError:
            not_done.append(method)
            print(f"  {method:<12}  not implemented yet")
        except Exception:
            print(f"  {method:<12}  implemented")
        else:
            print(f"  {method:<12}  implemented")

    if not_done:
        print(f"\n  Next: implement {', '.join(not_done)} in model.py, then run this again.")
        if "load" in not_done:
            sys.exit(0)
    else:
        print("\n  All three methods are in place.")
        print("  Next step: run  python model.py --step load")


def _check_load(inst: BaseModel) -> None:
    print("\nStep 2 of 4 - Loading your model weights")
    print("  Calling load() — this may take a moment ...\n")
    t0 = time.perf_counter()
    try:
        inst.load()
        inst._loaded = True
    except NotImplementedError:
        _die("load() is not implemented yet.\n  Open model.py and fill in the load() method.")
    except Exception as e:
        _die(f"load() failed:\n\n  {e}\n\n  Check that your weights file exists and the path is correct.")
    print(f"  Model loaded in {(time.perf_counter() - t0) * 1000:.0f} ms")
    print("\n  Next step: run  python model.py /path/to/image.jpg --step predict")


def _check_predict(inst: BaseModel, image_bytes: bytes) -> object:
    print("\nStep 3 of 4 - Running inference on your image")
    print("  Calling predict() ...\n")
    t0 = time.perf_counter()
    try:
        raw = inst.predict(image_bytes)
    except NotImplementedError:
        _die("predict() is not implemented yet.\n  Open model.py and fill in the predict() method.")
    except Exception as e:
        _die(f"predict() failed:\n\n  {e}")
    print(f"  Inference completed in {(time.perf_counter() - t0) * 1000:.0f} ms")
    print(f"\n  Output type   : {type(raw).__name__}")
    print(f"  Output preview: {repr(raw)[:400]}")
    print("\n  Use the output above to implement map_result() in model.py.")
    print("  Next step: run  python model.py /path/to/image.jpg")
    return raw


def _check_map_result(inst: BaseModel, raw: object, threshold: float) -> None:
    print("\nStep 4 of 4 - Checking your output format")
    print("  Calling map_result() and validating the output ...\n")
    try:
        predictions = inst.map_result(raw)
    except NotImplementedError:
        _die("map_result() is not implemented yet.\n  Open model.py and fill in the map_result() method.")
    except Exception as e:
        _die(f"map_result() failed:\n\n  {e}")

    if not isinstance(predictions, list):
        _die(
            f"map_result() must return a list of detections, but got: {type(predictions).__name__}\n"
            f"  Each item must be a dict with 'bbox', 'score', and 'acceptedNameUsageID'."
        )

    errors = []
    for i, pred in enumerate(predictions):
        bbox = pred.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"  Detection {i + 1}: 'bbox' must be a list of 4 numbers [x1, y1, x2, y2]")
        if "score" not in pred:
            errors.append(f"  Detection {i + 1}: missing 'score' (confidence between 0.0 and 1.0)")
        if "acceptedNameUsageID" not in pred:
            errors.append(f"  Detection {i + 1}: missing 'acceptedNameUsageID' (GBIF species URL)")

    if errors:
        print("  Problems found:\n")
        for e in errors:
            print(e)
        _die("Fix the output format in map_result() and run again.")

    above = [p for p in predictions if p.get("score", 0.0) >= threshold]
    print(f"  Found {len(predictions)} detection(s)  ({len(above)} above confidence threshold {threshold})")
    print("  Output format is correct\n")

    if not predictions:
        print("  No objects detected in this image. Try a different image to verify your model works.")
    else:
        print("  Detections:\n")
        for i, pred in enumerate(predictions):
            score = pred.get("score", 0.0)
            bbox  = pred.get("bbox", [])
            sp    = pred.get("acceptedNameUsageID", "")
            flag  = "  (below threshold)" if score < threshold else ""
            bbox_str = (f"[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"
                        if len(bbox) == 4 else str(bbox))
            print(f"  [{i + 1}] Confidence : {score:.1%}{flag}")
            print(f"       Bounding box: {bbox_str}")
            print(f"       Species     : {sp}")
            print()

    print("  Your model is ready to be submitted!")
    print("  Open a pull request with your models/<your_model_id>/ folder.")
    print("=" * 58)
