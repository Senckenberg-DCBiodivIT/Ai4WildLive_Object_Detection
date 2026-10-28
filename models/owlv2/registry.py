from __future__ import annotations

from models.config import ModelConfig


def get_model() -> ModelConfig:
    return ModelConfig(
        name="Open-Vocabulary Object Detection (OWLv2)",
        version="1.0",
        provider="THWS AP3 - AI4WildLIVE, contact: simonebianchi@thws.de",
        deployment_type="local",
        description=(
            "Fine-tuned OWLv2 open-vocabulary object detector for Bolivia camera-trap imagery. "
            "The model detects and localizes wildlife, humans, vehicles, and field equipment in camera-trap images. "
            "It supports 81 target categories, including mammals, birds, reptiles, amphibians, insects, unidentified animals, "
            "and broader taxonomic groups such as aves, mammal_sp, carnivora, rodentia, reptilia, and xenarthra."
        ),
        supported_taxa=(
            "Bolivia camera-trap classes, including human, horse, jaguar, puma, ocelot, margay, jaguarundi, "
            "tapir, capybara, peccary species, deer, armadillo species, coati, fox, raccoon, otter, anteater, "
            "opossum species, birds, reptiles, amphibians, insects, vehicles, equipment, and unidentified animals."
        ),
        default=False,
    )
