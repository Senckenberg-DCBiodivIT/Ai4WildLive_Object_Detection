# Adding Your Model to WildLive

Follow these 5 steps to integrate your object detection model into the WildLive platform.

---

## Step 1 — Copy the template

```bash
cp -r template/ models/<your_model_id>/
```

**Model ID rules:** lowercase letters, numbers, and underscores only.
Examples: `yolo_wildlife`, `resnet_mammals`, `my_detector_v2`

---

## Step 2 — Describe your model in `registry.py`

Open `models/<your_model_id>/registry.py` and fill in your model's details.
This is how WildLive registers and displays your model in the platform.

```python
def get_model() -> ModelConfig:
    return ModelConfig(
        name="My Jaguar Detector",           # Display name shown in the UI
        version="1.0",                        # Your model version
        provider="Your Name / Organisation",  # Who built it
        deployment_type="local",              # Keep "local" for bundled models
        endpoint="models.my_model_id.model",  # Must match your folder name exactly
        description="Detects jaguars in camera trap images",
        supported_taxa="mammals",
        default=False,                        # Set True to make this the default model
    )
```

---

## Step 3 — Implement your model in `model.py`

Open `models/<your_model_id>/model.py` and fill in three methods.
You only need to implement these three — the rest is handled automatically.

**`load()`** — Load your model weights. Called once when the service starts.
```python
def load(self) -> None:
    weights_path = os.path.join(os.path.dirname(__file__), "weights.pth")
    self._predictor = MyFramework.load(weights_path)
```

**`predict()`** — Run your model on an image. Return whatever your framework gives you.
```python
def predict(self, image_bytes: bytes) -> object:
    image = Image.open(io.BytesIO(image_bytes))
    return self._predictor(image)
```

**`map_result()`** — Convert your framework's output into the standard WildLive format.
Every detection must have a bounding box, a species ID, and a confidence score.
```python
def map_result(self, raw: object) -> list[dict]:
    results = []
    for box, label, score in raw:
        results.append({
            "bbox": [x1, y1, x2, y2],            # Pixel coordinates: top-left and bottom-right
            "acceptedNameUsageID": GBIF_IDS[label], # GBIF species URL (see note below)
            "score": float(score),                  # Confidence between 0.0 and 1.0
        })
    return results
```

> **Finding GBIF species IDs:**
> Every species your model detects needs a GBIF URL.
> Search for your species at [gbif.org](https://www.gbif.org/), open the species page,
> and copy the URL (e.g. `https://www.gbif.org/species/5219426` for jaguar).
> Add all your species to the `GBIF_IDS` dictionary at the top of `model.py`.

---

## Step 4 — Add your dependencies to `requirements.txt`

List the Python packages your model needs (e.g. PyTorch, ONNX Runtime, TensorFlow).

```
# Example for a PyTorch model:
torch>=2.0.0
torchvision
```

---

## Step 5 — Test your model locally

Run all commands from inside your model folder.
Work through the steps in order — each step builds on the previous one.

```bash
cd models/<your_model_id>/

# Install your dependencies
pip install -r requirements.txt
```

**Check your code structure** (no image needed, run this first):
```bash
python model.py --step structure
```
This tells you which methods are implemented and which still need work.

**Check your model loads correctly:**
```bash
python model.py --step load
```
This calls `load()` and checks your weights file loads without errors.

**Check what your model outputs** (run after load works):
```bash
python model.py /path/to/image.jpg --step predict
```
This prints the raw output from your framework so you can see what to convert in `map_result()`.

**Run the full pipeline** (run when all three methods are done):
```bash
python model.py /path/to/image.jpg
```
This checks everything end-to-end and validates that your output format is correct.

When everything passes you will see:
```
   Your model is ready to be submitted.
```

---

## Step 6 — Submit

Open a pull request with your `models/<your_model_id>/` folder.
Make sure it contains: `model.py`, `registry.py`, `requirements.txt`, and your weights file.
