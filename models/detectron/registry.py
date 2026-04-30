from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    version: str
    provider: str
    deployment_type: str
    endpoint: str
    description: str | None = None
    supported_taxa: str | None = None
    default: bool = False


def get_model() -> ModelConfig:
    return ModelConfig(
        name="Faster R-CNN Jaguar Detector",
        version="1.0",
        provider="Senckenberg",
        deployment_type="local",
        endpoint="models.detectron.model",
        description="Detects large mammals",
        supported_taxa="mammals",
        default=True,
    )
