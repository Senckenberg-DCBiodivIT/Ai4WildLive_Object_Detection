from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ResponseMapping:
    """
    Describes how to extract predictions from a remote model's JSON response.

    predictions_key  — top-level key that holds the list of detections (default: "predictions")
    bbox_key         — key for bounding box inside each detection (default: "bbox")
    score_key        — key for confidence score (default: "score")
    species_key      — key for species identifier (default: "acceptedNameUsageID")
    """
    predictions_key: str = "predictions"
    bbox_key: str = "bbox"
    score_key: str = "score"
    species_key: str = "acceptedNameUsageID"


@dataclass
class ModelConfig:
    """
    Declares a model to the WildLive object detection service.

    Required for every model
    ------------------------
    name            — human-readable display name
    version         — model version string
    provider        — organisation or team that owns the model
    deployment_type — "local"  : weights bundled in the image, runs in an isolated venv
                      "remote" : inference done by an external HTTP service

    Local models only
    -----------------
    (no extra fields — the model ID is derived from the directory name)

    Remote models only
    ------------------
    endpoint         — full URL of the remote inference endpoint
    response_mapping — describes how to parse the remote service's JSON response;
                       omit to use the defaults defined in ResponseMapping

    Optional for all models
    -----------------------
    description     — one-sentence summary shown in the UI
    supported_taxa  — comma-separated taxa the model was trained on
    default         — set True to pre-select this model in the UI (at most one per deployment)
    """

    name: str
    version: str
    provider: str
    deployment_type: Literal["local", "remote"]

    endpoint: str | None = None
    response_mapping: ResponseMapping | None = None

    description: str | None = None
    supported_taxa: str | None = None
    default: bool = False
