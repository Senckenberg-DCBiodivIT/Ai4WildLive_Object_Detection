from __future__ import annotations

from models.config import ModelConfig


def get_model() -> ModelConfig:
    return ModelConfig(
        name="SpeciesNet",
        version="5.0.3",
        provider="Google",
        deployment_type="local",
        description="Two-stage pipeline (detection + classification) for global wildlife species identification in camera trap images",
        supported_taxa="mammals, birds, reptiles",
        default=True,
    )
