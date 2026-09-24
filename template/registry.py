from __future__ import annotations

from models.config import ModelConfig


def get_model() -> ModelConfig:
    return ModelConfig(
        # Human-readable name shown in the UI and registry
        name="My Model Name",

        # Semantic version string
        version="1.0",

        # Organisation or person who trained/provides the model
        provider="Your Organisation",

        # "local"  → model.py lives in this directory, loaded via importlib
        # "remote" → an external HTTP endpoint is called at inference time
        deployment_type="local",

        # For local:  Python module path from the object-detection-service root
        #             e.g. "models.my_model.model"
        # For remote: Full HTTPS URL of the prediction endpoint
        #             e.g. "https://api.example.com/predict"
        endpoint="models.template.model",

        # Optional: what the model detects, shown in the registry UI
        description="Describe what species or objects this model detects",

        # Optional: taxonomic group (e.g. "mammals", "birds", "insects")
        supported_taxa="mammals",

        # True  → used automatically when no model_id is specified in the request
        # Only one model should be default=True at a time
        default=False,
    )
