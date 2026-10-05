# WildLive Object Detection Models

Models for the WildLive camera trap detection service, and the template for adding
your own.

Each model is a directory under `models/`. The directory name is the model ID. The
service gives every model its own virtual environment, so your dependencies never
clash with another model's.

## Add a model

### 1. Copy the template

```bash
cp -r template/ models/<your_model_id>/
```

Use lowercase letters, numbers and underscores for the ID.

### 2. Declare your classes in `classes.json`

Map every label your model outputs to its scientific name.

```json
{
  "jaguar": "Panthera onca",
  "cattle": "Bos taurus",
  "vehicle": null
}
```

**Please use species names.** WildLive focuses on species-level identification, so give
the full binomial where your model can — `Panthera onca` rather than `Panthera`. If your
model works mainly at genus or family level, get in touch before you start and we'll
talk it through.

Give the name only. WildLive looks it up in the
[Catalogue of Life](https://www.catalogueoflife.org/) to get the taxon ID, common name
and higher taxonomy, so don't add IDs, GBIF URLs or common names yourself.

Use `null` for things that aren't organisms. **A `null` label drops the detection** —
no annotation is saved and the box is discarded. That's usually what you want for a
vehicle or an empty frame.

### 3. Fill in `registry.py`

```python
from models.config import ModelConfig


def get_model() -> ModelConfig:
    return ModelConfig(
        name="My Jaguar Detector",
        version="1.0",
        provider="Your Name / Organisation",
        deployment_type="local",
        description="Detects jaguars in camera trap images",
        supported_taxa="mammals",
        default=False,
    )
```

`name`, `version`, `provider` and `deployment_type` are required; the rest are optional.
Keep `deployment_type="local"` unless your model runs on an external HTTP service, which
needs `endpoint` instead.

### 4. Implement `model.py`

Three methods:

- **`load()`** — load your weights. Called once, before the first prediction.
- **`predict()`** — run inference, return your framework's output unchanged.
- **`map_result()`** — convert that output to the WildLive format.

```python
def map_result(self, raw: object) -> list[dict]:
    return [
        {
            "bbox":  [x1, y1, x2, y2],   # pixels, top-left and bottom-right
            "class": label,              # a key in classes.json
            "score": float(score),       # 0.0 to 1.0
        }
        for box, label, score in raw
    ]
```

Boxes are pixel coordinates. If your framework returns normalised ones, or
`[x, y, width, height]`, convert them here.

### 5. List dependencies in `requirements.txt`

Skip common packages like `pillow` and `requests` — they're already there. Pin versions
as strictly as you like; your model has its own environment.

### 6. Test it

```bash
cd models/<your_model_id>/
pip install -r requirements.txt
```

| Command | Checks |
| --- | --- |
| `python model.py --step structure` | Which methods are implemented |
| `python model.py --step load` | Your weights load |
| `python model.py image.jpg --step predict` | Your raw output, so you can write `map_result()` |
| `python model.py image.jpg` | Everything, including the output format |

**The test does not check your labels against `classes.json`.** An undeclared label
passes every check here and is then dropped in production, with no error anywhere. Read
the `Species` lines in the output and confirm you declared each one.

## Optional files

Put these next to `model.py`:

- **`.python-version`** — pins the interpreter, e.g. `3.10` for an older torch build.
- **`preload.py`** — runs after your dependencies install. Use it to download weights at
  build time instead of on the first prediction.

## If your labels are already scientific names

Some models output scientific names directly, so `classes.json` would just repeat them.
Skip the file and return `scientificName` instead of `class`:

```python
{"bbox": [x1, y1, x2, y2], "scientificName": "Panthera onca", "score": 0.95}
```

You can still add a `classes.json` to fix individual spellings; names not in it pass
through unchanged. This is the exception — use `class` unless your model really does
emit scientific names.

## Examples

- **`models/jaguar_peccary_cattle_detector/`** — follow this one. A Detectron2 model
  using `class` labels and a `classes.json`.
- **`models/speciesnet/`** — read, don't copy. Google's
  [SpeciesNet](https://github.com/google/cameratrapai), showing the `scientificName`
  path. See [its README](models/speciesnet/README.md).

## Submit

Open a pull request with your `models/<your_model_id>/` directory, containing
`model.py`, `registry.py`, `requirements.txt`, `classes.json` and your weights.

Check that `python model.py image.jpg` passes and that every label it printed is in
`classes.json`.

Weights are large — ask us before committing them.
