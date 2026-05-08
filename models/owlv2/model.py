"""
OWLv2 + LoRA adapter for the object-detection-service model template.

The folder is meant to be copied to:
    object-detection-service/models/template_owlv2/

It loads the HF OWLv2 base model, injects the same LoRA structure used during
fine-tuning, loads the local checkpoint, and returns detections in the standard
template format.
"""
from __future__ import annotations

import os
# Keep HF fast tokenizer safe with multiprocessing workers.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import io
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision.ops import nms
from transformers import Owlv2ForObjectDetection, Owlv2Processor  # type: ignore


THIS_DIR = Path(__file__).resolve().parent
SERVICE_ROOT = THIS_DIR.parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

try:
    from models.base import BaseModel  # type: ignore  # noqa: E402
except Exception:  # Allows local smoke tests outside object-detection-service.
    class BaseModel:  # type: ignore
        def test(self, image_path: str | None = None, step: str = "all", threshold: float | None = None) -> None:
            if threshold is not None:
                setattr(self, "score_threshold", float(threshold))
            if step in ("structure", "all"):
                print(json.dumps({"classes": CLASS_NAMES, "threshold": getattr(self, "score_threshold", None)}, indent=2))
            if step in ("load", "predict", "all"):
                self.load()
                print("Model loaded")
            if step in ("predict", "all"):
                if not image_path:
                    raise ValueError("Pass an image path for predict/all tests.")
                raw = self.predict(Path(image_path).read_bytes())
                print(json.dumps(self.map_result(raw), indent=2))

try:
    from .lora_utils import LoRALinear, resolve_parent_and_attr, set_child_module
except Exception:
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    from lora_utils import LoRALinear, resolve_parent_and_attr, set_child_module  # type: ignore  # noqa: E402


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else float(default)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_default_checkpoint_path() -> Path:
    candidates = sorted(
        THIS_DIR.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return THIS_DIR / "owlv2_cct_epoch400.pt"


MODEL_NAME = os.environ.get("OWLV2_MODEL_NAME", "google/owlv2-base-patch16-ensemble").strip()
DEFAULT_CHECKPOINT_PATH = _find_default_checkpoint_path()
SCORE_THRESHOLD = _env_float("OWLV2_SCORE_THRESHOLD", 0.50)
MAX_DETECTIONS = _env_int("OWLV2_MAX_DETECTIONS", 50)
NMS_IOU_THRESHOLD = _env_float("OWLV2_NMS_IOU_THRESHOLD", 0.50)
PROCESSOR_IMAGE_SIZE = _env_int("OWLV2_PROCESSOR_IMAGE_SIZE", 0)
REQUIRE_GBIF_IDS = _env_flag("OWLV2_REQUIRE_GBIF_IDS", False)


CLASS_NAMES: list[str] = [
    "human",
    "horse",
    "bare_faced_curassow",
    "nine_banded_armadillo",
    "jaguar",
    "vehicle",
    "equipment",
    "bird_sp",
    "amazonian_motmot",
    "cattle",
    "central_american_agouti",
    "red_legged_seriema",
    "gray_brocket_deer",
    "argentine_black_and_white_tegu",
    "lowland_paca",
    "something_unidentifiable",
    "collared_peccary",
    "cocoi_heron",
    "boat_billed_heron",
    "amphibian_sp",
    "sunbittern",
    "insect_sp",
    "grey_necked_wood_rail",
    "mammal_sp",
    "ocelot",
    "rusty_margined_guan",
    "reptile_sp",
    "bolivian_squirrel",
    "undulated_tinamou",
    "Tapeti",
    "common_opossum",
    "south_american_coati",
    "south_american_tapir",
    "white_lipped_peccary",
    "crab_eating_raccoon",
    "king_vulture",
    "jaguarundi",
    "yellow_armadillo",
    "white_tipped_dove",
    "crab_eating_fox",
    "peccari_sp",
    "artiodactyla",
    "reptilia",
    "others",
    "bat_sp",
    "scaled_pigeon",
    "armadillo_sp",
    "puma",
    "capped_heron",
    "lesser_yellow_headed_vulture",
    "primates_sp",
    "plush_crested_jay",
    "tufted_capuchin",
    "capybara",
    "nandu",
    "dog",
    "cattle_or_human",
    "tayra",
    "some_rat_like_rodent",
    "margay",
    "tamandua",
    "opossum_sp",
    "giant_wood_rail",
    "aves",
    "xenarthra",
    "tinamou_sp",
    "limpkin",
    "carnivora",
    "green_cheeked_parakeet",
    "rufescent_tiger_heron",
    "black_vulture",
    "red_necked_woodpecker",
    "rodentia",
    "neotropical_otter",
    "white_eared_opossum",
    "giant_anteater",
    "small_billed_tinamou",
    "guan_sp",
    "red_footed_tortoise",
    "brazilian_porcupine",
    "tataupa_tinamou",
]

# Doubtful GBIF IDs: vehicle, equipment, something_unidentifiable, amphibian_sp,
# reptile_sp, reptilia, others, cattle_or_human, some_rat_like_rodent, xenarthra

GBIF_IDS: dict[str, str] = {
    "human": "https://www.gbif.org/species/2436436",
    "horse": "https://www.gbif.org/species/2440886",
    "bare_faced_curassow": "https://www.gbif.org/species/2482174",
    "nine_banded_armadillo": "https://www.gbif.org/species/2440779",
    "jaguar": "https://www.gbif.org/species/5219426",
    "bird_sp": "https://www.gbif.org/species/212",
    "amazonian_motmot": "https://www.gbif.org/species/2475290",
    "cattle": "https://www.gbif.org/species/2441022",
    "central_american_agouti": "https://www.gbif.org/species/2437576",
    "red_legged_seriema": "https://www.gbif.org/species/5228133",
    "gray_brocket_deer": "https://www.gbif.org/species/7261375",
    "argentine_black_and_white_tegu": "https://www.gbif.org/species/5227370",
    "lowland_paca": "https://www.gbif.org/species/4263039",
    "collared_peccary": "https://www.gbif.org/species/2440995",
    "cocoi_heron": "https://www.gbif.org/species/2480951",
    "boat_billed_heron": "https://www.gbif.org/species/2480833",
    "sunbittern": "https://www.gbif.org/species/2474332",
    "insect_sp": "https://www.gbif.org/species/216",
    "grey_necked_wood_rail": "https://www.gbif.org/species/2474659",
    "mammal_sp": "https://www.gbif.org/species/359",
    "ocelot": "https://www.gbif.org/species/2434982",
    "rusty_margined_guan": "https://www.gbif.org/species/2482204",
    "bolivian_squirrel": "https://www.gbif.org/species/5219662",
    "undulated_tinamou": "https://www.gbif.org/species/2477560",
    "Tapeti": "https://www.gbif.org/species/2436860",
    "common_opossum": "https://www.gbif.org/species/2439920",
    "south_american_coati": "https://www.gbif.org/species/2433536",
    "south_american_tapir": "https://www.gbif.org/species/2440898",
    "white_lipped_peccary": "https://www.gbif.org/species/2440993",
    "crab_eating_raccoon": "https://www.gbif.org/species/9518324",
    "king_vulture": "https://www.gbif.org/species/2481923",
    "jaguarundi": "https://www.gbif.org/species/2435146",
    "yellow_armadillo": "https://www.gbif.org/species/2440799",
    "white_tipped_dove": "https://www.gbif.org/species/2496060",
    "crab_eating_fox": "https://www.gbif.org/species/2434584",
    "peccari_sp": "https://www.gbif.org/species/5303",
    "artiodactyla": "https://www.gbif.org/species/731",
    "bat_sp": "https://www.gbif.org/species/734",
    "scaled_pigeon": "https://www.gbif.org/species/2495339",
    "armadillo_sp": "https://www.gbif.org/species/735",
    "puma": "https://www.gbif.org/species/2435099",
    "capped_heron": "https://www.gbif.org/species/2480816",
    "lesser_yellow_headed_vulture": "https://www.gbif.org/species/2481928",
    "primates_sp": "https://www.gbif.org/species/798",
    "plush_crested_jay": "https://www.gbif.org/species/2482575",
    "tufted_capuchin": "https://www.gbif.org/species/7477334",
    "capybara": "https://www.gbif.org/species/5786666",
    "nandu": "https://www.gbif.org/species/2495165",
    "dog": "https://www.gbif.org/species/6164210",
    "tayra": "https://www.gbif.org/species/2433716",
    "margay": "https://www.gbif.org/species/2434950",
    "tamandua": "https://www.gbif.org/species/2436340",
    "opossum_sp": "https://www.gbif.org/species/5455",
    "giant_wood_rail": "https://www.gbif.org/species/2474657",
    "aves": "https://www.gbif.org/species/212",
    "tinamou_sp": "https://www.gbif.org/species/9353",
    "limpkin": "https://www.gbif.org/species/2474337",
    "carnivora": "https://www.gbif.org/species/732",
    "green_cheeked_parakeet": "https://www.gbif.org/species/2479804",
    "rufescent_tiger_heron": "https://www.gbif.org/species/2480869",
    "black_vulture": "https://www.gbif.org/species/2481942",
    "red_necked_woodpecker": "https://www.gbif.org/species/2478573",
    "rodentia": "https://www.gbif.org/species/1459",
    "neotropical_otter": "https://www.gbif.org/species/2433738",
    "white_eared_opossum": "https://www.gbif.org/species/2439930",
    "giant_anteater": "https://www.gbif.org/species/2436346",
    "small_billed_tinamou": "https://www.gbif.org/species/2477558",
    "guan_sp": "https://www.gbif.org/species/5237",
    "red_footed_tortoise": "https://www.gbif.org/species/9533689",
    "brazilian_porcupine": "https://www.gbif.org/species/2437600",
    "tataupa_tinamou": "https://www.gbif.org/species/2477522",
}


_CLASS_LOOKUP = {name.lower(): name for name in CLASS_NAMES}


def _class_from_prompt(label: str) -> str:
    value = str(label).strip()
    lowered = value.lower()
    for prefix in ("a photo of a ", "a photo of an ", "photo of a ", "photo of an ", "a ", "an "):
        if lowered.startswith(prefix):
            value = value[len(prefix):].strip()
            lowered = value.lower()
            break

    if lowered in _CLASS_LOOKUP:
        return _CLASS_LOOKUP[lowered]

    underscored = value.replace(" ", "_")
    return _CLASS_LOOKUP.get(underscored.lower(), underscored)


def _checkpoint_state(ckpt: Any) -> dict[str, Any]:
    if isinstance(ckpt, dict):
        lora_state = ckpt.get("lora_state")
        model_state = ckpt.get("model_state")
        if isinstance(lora_state, dict) and lora_state:
            return lora_state
        if isinstance(model_state, dict) and model_state:
            return model_state
        if all(isinstance(k, str) for k in ckpt.keys()):
            tensor_values = [v for v in ckpt.values() if torch.is_tensor(v)]
            if tensor_values:
                return ckpt
    raise RuntimeError("Unsupported OWLv2 checkpoint format.")


def _checkpoint_config(ckpt: Any) -> dict[str, Any]:
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    return cfg if isinstance(cfg, dict) else {}


def _checkpoint_model_name(ckpt: Any) -> str:
    if os.environ.get("OWLV2_MODEL_NAME", "").strip():
        return MODEL_NAME
    value = str(_checkpoint_config(ckpt).get("model_name") or "").strip()
    return value or MODEL_NAME


def _checkpoint_processor_image_size(ckpt: Any, fallback: int) -> int:
    if fallback > 0:
        return int(fallback)
    value = _checkpoint_config(ckpt).get("processor_image_size")
    try:
        value_i = int(value)
    except Exception:
        value_i = 0
    return value_i if value_i > 0 else 1280


def _checkpoint_lora_dropout(ckpt: Any) -> float:
    raw = _checkpoint_config(ckpt).get("lora", {})
    lora = raw if isinstance(raw, dict) else {}
    value = lora.get("LORA_dropout", lora.get("lora_dropout", 0.0))
    try:
        return float(value)
    except Exception:
        return 0.0


def _lora_pairs_from_state(state: dict[str, Any]) -> dict[str, dict[str, tuple[int, ...]]]:
    pairs: dict[str, dict[str, tuple[int, ...]]] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        if key.endswith(".lora_A"):
            pairs.setdefault(key[:-7], {})["lora_A"] = tuple(int(v) for v in value.shape)
        elif key.endswith(".lora_B"):
            pairs.setdefault(key[:-7], {})["lora_B"] = tuple(int(v) for v in value.shape)
    return pairs


def _group_for_lora_module(name: str) -> str:
    if "owlv2.vision_model" in name:
        return "vision"
    if "owlv2.text_model" in name:
        return "text"
    if "owlv2.visual_projection" in name:
        return "visual_projection"
    if "owlv2.text_projection" in name:
        return "text_projection"
    if (
        name.startswith(("class_head.", "box_head.", "objectness_head."))
        or ".class_head." in name
        or ".box_head." in name
        or ".objectness_head." in name
    ):
        return "heads"
    return "other"


def _layer_index(name: str, token: str) -> int | None:
    marker = f"{token}.encoder.layers."
    if marker not in name:
        return None
    tail = name.split(marker, 1)[1]
    first = tail.split(".", 1)[0]
    try:
        return int(first)
    except Exception:
        return None


def _summarize_lora_pairs(pairs: dict[str, dict[str, tuple[int, ...]]]) -> dict[str, Any]:
    ranks: dict[str, set[int]] = {}
    counts: dict[str, int] = {}
    layers: dict[str, set[int]] = {"vision": set(), "text": set()}

    for module_name, shapes in pairs.items():
        shape_a = shapes.get("lora_A")
        shape_b = shapes.get("lora_B")
        if shape_a:
            rank = int(shape_a[0])
        elif shape_b:
            rank = int(shape_b[1])
        else:
            continue

        group = _group_for_lora_module(module_name)
        ranks.setdefault(group, set()).add(rank)
        counts[group] = counts.get(group, 0) + 1

        vision_idx = _layer_index(module_name, "owlv2.vision_model")
        text_idx = _layer_index(module_name, "owlv2.text_model")
        if vision_idx is not None:
            layers["vision"].add(vision_idx)
        if text_idx is not None:
            layers["text"].add(text_idx)

    return {
        "modules": len(pairs),
        "counts": counts,
        "ranks": {k: sorted(v) for k, v in ranks.items()},
        "layers": {k: sorted(v) for k, v in layers.items() if v},
    }


def _lora_alpha_for_rank(rank: int) -> float:
    alpha = os.environ.get("OWLV2_LORA_ALPHA", "").strip()
    if alpha:
        return float(alpha)
    multiplier = _env_float("OWLV2_LORA_ALPHA_MULTIPLIER", 4.0)
    return float(multiplier) * float(rank)


def _inject_lora_from_state(
    model: nn.Module,
    state: dict[str, Any],
    *,
    dropout_train: float = 0.0,
) -> dict[str, Any]:
    pairs = _lora_pairs_from_state(state)
    summary = _summarize_lora_pairs(pairs)
    if not pairs:
        summary["injected_modules"] = 0
        return summary

    module_map = dict(model.named_modules())
    errors: list[str] = []
    injected = 0

    for module_name, shapes in sorted(pairs.items()):
        shape_a = shapes.get("lora_A")
        shape_b = shapes.get("lora_B")
        if shape_a is None or shape_b is None:
            errors.append(f"{module_name}: incomplete LoRA pair")
            continue
        if len(shape_a) != 2 or len(shape_b) != 2 or shape_a[0] != shape_b[1]:
            errors.append(f"{module_name}: incompatible LoRA shapes A={shape_a} B={shape_b}")
            continue

        module = module_map.get(module_name)
        if module is None:
            errors.append(f"{module_name}: target module not found in base model")
            continue
        if not isinstance(module, nn.Linear):
            errors.append(f"{module_name}: target is {type(module).__name__}, expected Linear")
            continue

        rank = int(shape_a[0])
        in_features = int(shape_a[1])
        out_features = int(shape_b[0])
        if module.in_features != in_features or module.out_features != out_features:
            errors.append(
                f"{module_name}: shape mismatch checkpoint=({in_features}->{out_features}) "
                f"base=({module.in_features}->{module.out_features})"
            )
            continue

        lora = LoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            r=rank,
            alpha=_lora_alpha_for_rank(rank),
            dropout_train=dropout_train,
        )
        with torch.no_grad():
            lora.weight.copy_(module.weight.data)
            if module.bias is not None and lora.bias is not None:
                lora.bias.copy_(module.bias.data)

        parent, attr_name = resolve_parent_and_attr(model, module_name)
        set_child_module(parent, attr_name, lora)
        injected += 1

    if errors:
        preview = "; ".join(errors[:8])
        suffix = f" (+{len(errors) - 8} more)" if len(errors) > 8 else ""
        raise RuntimeError(f"Could not inject LoRA from checkpoint: {preview}{suffix}")

    summary["injected_modules"] = injected
    return summary


def _resolve_device() -> torch.device:
    requested = os.environ.get("OWLV2_DEVICE", "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


class OWLv2ModelHelper:
    """Owns the OWLv2 model lifecycle, inference, and result mapping."""

    def __init__(self) -> None:
        self.model: Owlv2ForObjectDetection | None = None
        self.processor: Owlv2Processor | None = None
        self.device = _resolve_device()
        self.score_threshold = SCORE_THRESHOLD
        self.max_detections = MAX_DETECTIONS
        self.nms_iou_threshold = NMS_IOU_THRESHOLD
        self.processor_image_size = PROCESSOR_IMAGE_SIZE
        self.class_names = CLASS_NAMES
        self.prompts = [f"a photo of a {name}" for name in self.class_names]

        ckpt_env = os.environ.get("OWLV2_CHECKPOINT", "").strip()
        self.checkpoint_path = Path(ckpt_env).expanduser() if ckpt_env else DEFAULT_CHECKPOINT_PATH
        self.load_info: dict[str, Any] = {}

    def load(self) -> None:
        """Load OWLv2 base, inject LoRA, and load the fine-tuned checkpoint."""
        if self.model is not None and self.processor is not None:
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"OWLv2 checkpoint not found: {self.checkpoint_path}. "
                "Set OWLV2_CHECKPOINT or place a .pt checkpoint next to model.py."
            )

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        ckpt = _torch_load(self.checkpoint_path)
        state = _checkpoint_state(ckpt)
        model_name = _checkpoint_model_name(ckpt)
        self.processor_image_size = _checkpoint_processor_image_size(ckpt, self.processor_image_size)

        processor_backend = "torchvision" if self.device.type == "cuda" else "pil"
        try:
            processor = Owlv2Processor.from_pretrained(model_name, backend=processor_backend)
        except TypeError:
            # Older Transformers releases do not accept the backend argument.
            processor = Owlv2Processor.from_pretrained(model_name)

        if hasattr(processor, "image_processor"):
            processor.image_processor.size = {
                "height": self.processor_image_size,
                "width": self.processor_image_size,
            }

        model = Owlv2ForObjectDetection.from_pretrained(model_name)
        if _env_flag("OWLV2_USE_LORA", True):
            lora_summary = _inject_lora_from_state(
                model,
                state,
                dropout_train=_checkpoint_lora_dropout(ckpt),
            )
        else:
            lora_summary = {"modules": 0, "injected_modules": 0, "disabled": True}

        incompatible = model.load_state_dict(state, strict=False)
        model.to(self.device)
        model.eval()

        self.model = model
        self.processor = processor
        self.load_info = {
            "checkpoint": str(self.checkpoint_path),
            "missing_keys": len(getattr(incompatible, "missing_keys", [])),
            "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
            "device": str(self.device),
            "model_name": model_name,
            "processor_backend": processor_backend,
            "processor_image_size": self.processor_image_size,
            "lora": lora_summary,
        }

    @torch.inference_mode()
    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """Run OWLv2 open-vocabulary inference on a single image."""
        if self.model is None or self.processor is None:
            self.load()
        assert self.model is not None
        assert self.processor is not None

        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size

        text_cfg = getattr(getattr(self.model, "config", None), "text_config", None)
        max_txt_len = getattr(text_cfg, "max_position_embeddings", None) if text_cfg is not None else None

        processor_kwargs: dict[str, Any] = {
            "images": image,
            "text": self.prompts,
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
        }
        if max_txt_len is not None:
            processor_kwargs["max_length"] = max_txt_len

        inputs = self.processor(**processor_kwargs).to(self.device)
        use_amp = _env_flag("OWLV2_USE_AMP", self.device.type == "cuda")
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
            if self.device.type == "cuda"
            else nullcontext()
        )

        with amp_ctx:
            outputs = self.model(**inputs, interpolate_pos_encoding=True)

        target_sizes = torch.tensor([(height, width)], device=self.device)
        processed = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=min(0.1, float(self.score_threshold)),
            text_labels=[self.prompts],
        )[0]

        boxes = processed.get("boxes", torch.empty(0, 4, device=self.device)).float()
        scores = processed.get("scores", torch.empty(0, device=self.device)).float()
        labels = processed.get("text_labels", [])
        if not labels and "labels" in processed:
            labels = [self.prompts[int(i)] for i in processed["labels"].detach().cpu().tolist()]

        if boxes.numel() == 0 or scores.numel() == 0:
            return {"boxes": [], "scores": [], "labels": [], "image_size": [height, width]}

        boxes[:, 0::2].clamp_(0, width)
        boxes[:, 1::2].clamp_(0, height)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        keep = (scores >= float(self.score_threshold)) & valid
        if keep.sum().item() == 0:
            return {"boxes": [], "scores": [], "labels": [], "image_size": [height, width]}

        boxes = boxes[keep]
        scores = scores[keep]
        labels = [label for label, keep_one in zip(labels, keep.detach().cpu().tolist()) if keep_one]

        keep_idx = nms(boxes, scores, float(self.nms_iou_threshold))
        keep_idx = keep_idx[scores[keep_idx].argsort(descending=True)]
        keep_idx = keep_idx[: int(self.max_detections)]

        return {
            "boxes": boxes[keep_idx].detach().cpu().tolist(),
            "scores": scores[keep_idx].detach().cpu().tolist(),
            "labels": [labels[i] for i in keep_idx.detach().cpu().tolist()],
            "image_size": [height, width],
        }

    def map_result(self, raw: object) -> list[dict[str, Any]]:
        """Convert raw OWLv2 output to the service response format."""
        if not isinstance(raw, dict):
            return []

        boxes = raw.get("boxes", [])
        scores = raw.get("scores", [])
        labels = raw.get("labels", [])
        mapped: list[dict[str, Any]] = []

        for box, score, label in zip(boxes, scores, labels):
            class_name = _class_from_prompt(str(label))
            gbif_url = GBIF_IDS.get(class_name, "")
            if REQUIRE_GBIF_IDS and not gbif_url:
                continue

            mapped.append(
                {
                    "bbox": [round(float(v), 3) for v in box],
                    "acceptedNameUsageID": gbif_url,
                    "score": float(score),
                    "className": class_name,
                }
            )
        return mapped


class Model(BaseModel):
    """A service-oriented adapter that preserves the intended model methods."""

    _DELEGATED_ATTRIBUTES = {
        "model",
        "processor",
        "device",
        "score_threshold",
        "max_detections",
        "nms_iou_threshold",
        "processor_image_size",
        "class_names",
        "prompts",
        "checkpoint_path",
        "load_info",
    }

    def __init__(self) -> None:
        self._helper = OWLv2ModelHelper()

    def __getattr__(self, name: str) -> Any:
        if name in self._DELEGATED_ATTRIBUTES:
            return getattr(self._helper, name)
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._DELEGATED_ATTRIBUTES and "_helper" in self.__dict__:
            setattr(self._helper, name, value)
            return
        super().__setattr__(name, value)

    def load(self) -> None:
        self._helper.load()

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        return self._helper.predict(image_bytes)

    def map_result(self, raw: object) -> list[dict[str, Any]]:
        return self._helper.map_result(raw)


instance = Model()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Path to a test image")
    parser.add_argument("--step", choices=["structure", "load", "predict", "all"], default="all")
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    args = parser.parse_args()

    instance.test(image_path=args.image, step=args.step, threshold=args.threshold)
