from __future__ import annotations

from models.config import ModelConfig


def get_model() -> ModelConfig:
    return ModelConfig(
        name="Jaguar Peccary Cattle Detector",
        version="1.0",
        provider="Senckenberg",
        deployment_type="local",
        description="Faster RCNN model that detects jaguar, collared peccary, and cattle",
        supported_taxa="mammals",
        default=False,
    )
