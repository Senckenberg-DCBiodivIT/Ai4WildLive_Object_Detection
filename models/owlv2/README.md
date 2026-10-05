# OWLv2 Bolivia

This folder adapts the OWLv2/LoRA training pipeline to the same interface used
by the base `template` folder.

## Contents

- `model.py`: implements `load`, `predict`, and `map_result`.
- `registry.py`: describes the local model using the shared `ModelConfig`.
- `classes.json`: maps all 81 output labels to scientific names for Catalogue of Life resolution.
- `lora_utils.py`: LoRA modules compatible with the training code.
- `*.pt`: LoRA checkpoints. If `OWLV2_CHECKPOINT` is not set, the newest `.pt`
  file in this folder is used automatically.
- `requirements.txt`: minimal inference dependencies.
- `.python-version`: Python 3.12; validated locally with Python 3.12.7.

## Service Usage

1. Copy `owlv2` into the service `models/` folder.
2. Install the dependencies:

```powershell
pip install -r models/owlv2/requirements.txt
```

3. Review `classes.json`. Names were migrated from the existing GBIF taxon IDs
   using the GBIF species API; WildLive resolves them through Catalogue of Life.
   Broad labels retain their taxonomic rank (for example `Aves` or `Mammalia`),
   rather than claiming a species identification. Review these with the service
   maintainers because the platform prioritizes species-level identification.
   `vehicle`, `equipment`, `something_unidentifiable`, `others`,
   `cattle_or_human`, and `some_rat_like_rodent` map to `null`: the service
   discards their detections. The adapter still returns the original labels.

4. To force a specific checkpoint:

```powershell
$env:OWLV2_CHECKPOINT = "C:\path\to\checkpoint.pt"
```

## Useful Environment Variables

- `OWLV2_CHECKPOINT`: checkpoint path. Default: newest `.pt` file in this folder.
- `OWLV2_MODEL_NAME`: base Hugging Face model. Default: value saved in the
  checkpoint, otherwise `google/owlv2-base-patch16-ensemble`.
- `OWLV2_DEVICE`: `auto`, `cpu`, `cuda`, or `cuda:0`. Default: `auto`.
- `OWLV2_SCORE_THRESHOLD`: score threshold. Default: `0.50`.
- `OWLV2_MAX_DETECTIONS`: maximum detections per image. Default: `50`.
- `OWLV2_NMS_IOU_THRESHOLD`: global NMS IoU threshold. Default: `0.50`.
- `OWLV2_PROCESSOR_IMAGE_SIZE`: square processor resize. Default: value saved
  in the checkpoint, otherwise `1280`.
- `HF_HUB_OFFLINE`: default `0`; set `1` to use only cached Hugging Face files.
- `OWLV2_USE_LORA`: default `true`; controls checkpoint LoRA injection.
- `OWLV2_USE_AMP`: defaults to `true` on CUDA; controls mixed precision.
- `OWLV2_LORA_ALPHA_MULTIPLIER`: alpha/rank multiplier used to reconstruct the
  LoRA scale when the checkpoint only contains `lora_A` and `lora_B`. Default:
  `4.0`, matching the current training setup.

## LoRA Reconstruction

The loader no longer depends on hardcoded ranks, layers, or target groups for
LoRA injection. It reads every `*.lora_A` and `*.lora_B` key from the checkpoint,
finds the corresponding OWLv2 linear module, creates a `LoRALinear` with the
rank inferred from tensor shapes, and then loads the state dict.

This supports checkpoints with different ranks or layer ranges, as long as the
module names match the OWLv2 base model.

## Quick Test

From inside the service:

```powershell
python models/owlv2/model.py path\to\image.jpg --step all --threshold 0.5
```

The `map_result()` output format is:

```json
[
  {
    "bbox": [x1, y1, x2, y2],
    "class": "jaguar",
    "score": 0.95
  }
]
```

`Model(BaseModel)` delegates to `OWLv2ModelHelper`; the shared service base
provides `run()` and the command-line validation. `run()` replaces `class` with
`scientificName` using `classes.json`; null entries become empty names for the
service to discard. Catalogue of Life resolution happens in the middleware.
Bounding boxes remain in
original-image pixel coordinates. Checkpoint weights and inference prompts are
unchanged by the service interface migration.
