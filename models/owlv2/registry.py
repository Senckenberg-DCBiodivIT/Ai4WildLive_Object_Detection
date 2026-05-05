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
        name="AI4WildLive OWLv2 Bolivia",
        version="1.0",
        provider="AI4WildLive / Bolivia camera-trap pipeline",
        deployment_type="local",
        endpoint="models.template_owlv2.model",
        description=(
            "Fine-tuned OWLv2 open-vocabulary detector using the Bolivia "
            "COCO split and LoRA checkpoint from checkpoint."
        ),
        supported_taxa="mammals, birds, reptiles, amphibians, broad camera-trap classes",
        default=False,
    )
