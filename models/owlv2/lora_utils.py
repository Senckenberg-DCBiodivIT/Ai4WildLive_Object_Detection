# =========================
# Imports
# =========================
import math
import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# LoRA modules
# =========================
class LoRALayer:
    """
    LoRA mixin.

    - dropout_train is applied only while the module is in train() mode.
    - in eval() dropout is always disabled, even when dropout_train > 0.
    """

    def __init__(self, r: int, alpha: float, dropout_train: float = 0.0):
        self.r = int(r)
        self.alpha = float(alpha)
        self.dropout_train = float(dropout_train)
        self.scaling = (alpha / r) if r > 0 else 1.0

        # nn.Dropout already follows the parent module train()/eval() state, but
        # the forward path below makes the eval bypass explicit.
        self._lora_dropout = nn.Dropout(self.dropout_train) if self.dropout_train > 0.0 else nn.Identity()

    def _apply_lora_dropout(self, x: torch.Tensor) -> torch.Tensor:
        # Guarantee: dropout is never applied in eval mode.
        if not getattr(self, "training", False):
            return x
        return self._lora_dropout(x)


class LoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: float = 1.0,
        dropout_train: float = 0.0,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r, alpha, dropout_train)

        if r > 0:
            self.lora_A = nn.Parameter(torch.empty((r, in_features)))
            self.lora_B = nn.Parameter(torch.empty((out_features, r)))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            # r == 0 disables the LoRA branch while preserving the interface.
            self.lora_A = None
            self.lora_B = None

        # Always freeze the original base layer weight/bias.
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            # Dropout only during training; full bypass during eval.
            x_d = self._apply_lora_dropout(x)
            lora_inter = F.linear(x_d, self.lora_A)
            lora_out = F.linear(lora_inter, self.lora_B) * self.scaling
            return base + lora_out
        return base


# =========================
# Naming helpers / patterns
# =========================
_HEAD_RE = re.compile(r"(^|\.)(class_head|box_head|objectness_head)(\.|$)")
_VISION_LAYER_RE = re.compile(r"^owlv2\.vision_model\.encoder\.layers\.(\d+)\.")
_TEXT_LAYER_RE = re.compile(r"^owlv2\.text_model\.encoder\.layers\.(\d+)\.")
_TEXT_BACKBONE_TOKEN = "owlv2.text_model"
_VISION_BACKBONE_TOKEN = "owlv2.vision_model"
_VISUAL_PROJ_TOKEN = "owlv2.visual_projection"
_TEXT_PROJ_TOKEN = "owlv2.text_projection"


def _is_head(name: str) -> bool:
    return _HEAD_RE.search(name) is not None


def _is_vision_backbone(name: str) -> bool:
    return _VISION_BACKBONE_TOKEN in name


def _is_text_backbone(name: str) -> bool:
    return _TEXT_BACKBONE_TOKEN in name


def _is_projection(name: str) -> bool:
    return (_VISUAL_PROJ_TOKEN in name) or (_TEXT_PROJ_TOKEN in name)


def _is_attn_proj(name: str, want=("q", "k", "v", "out")) -> bool:
    n = name.lower()
    flags = []
    if "q" in want:
        flags += ["q_proj"]
    if "k" in want:
        flags += ["k_proj"]
    if "v" in want:
        flags += ["v_proj"]
    if "out" in want:
        flags += ["out_proj", "o_proj"]  # Covers both naming variants.
    return any(tok in n for tok in flags)


def _is_mlp_fc(name: str, want=("fc1", "fc2")) -> bool:
    n = name.lower()
    return any(re.search(rf"\.mlp\.{w}(\.|$)", n) for w in want)


def _vision_layer_indices(model: nn.Module) -> List[int]:
    idx = set()
    for name, _ in model.named_parameters():
        m = _VISION_LAYER_RE.match(name)
        if m:
            idx.add(int(m.group(1)))
    return sorted(idx)


def _text_layer_indices(model: nn.Module) -> List[int]:
    idx = set()
    for name, _ in model.named_parameters():
        m = _TEXT_LAYER_RE.match(name)
        if m:
            idx.add(int(m.group(1)))
    return sorted(idx)


def _vision_layer_index_from_name(name: str) -> Optional[int]:
    m = _VISION_LAYER_RE.match(name)
    return int(m.group(1)) if m else None


def _text_layer_index_from_name(name: str) -> Optional[int]:
    m = _TEXT_LAYER_RE.match(name)
    return int(m.group(1)) if m else None


# =========================
# Robust module resolution / replacement
# =========================
def resolve_parent_and_attr(model: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    parts = full_name.split(".")
    parent = model
    for p in parts[:-1]:
        if hasattr(parent, p):
            parent = getattr(parent, p)
        elif isinstance(parent, (nn.Sequential, nn.ModuleList)):
            if not p.isdigit():
                raise AttributeError(f"Non-numeric index '{p}' in indexed container for '{full_name}'.")
            parent = parent[int(p)]
        elif isinstance(parent, nn.ModuleDict):
            parent = parent[p]
        else:
            raise AttributeError(f"Could not resolve '{p}' in '{full_name}'. Parent: {type(parent)}")
    return parent, parts[-1]


def set_child_module(parent: nn.Module, name: str, new_module: nn.Module):
    if hasattr(parent, name):
        setattr(parent, name, new_module)
    elif isinstance(parent, (nn.Sequential, nn.ModuleList)):
        if not name.isdigit():
            raise AttributeError(f"Non-numeric index '{name}' in indexed container.")
        parent[int(name)] = new_module
    elif isinstance(parent, nn.ModuleDict):
        parent[name] = new_module
    else:
        raise TypeError(f"Unsupported parent type for set_child_module: {type(parent)}")


# =========================
# Granular freeze / unfreeze helpers
# =========================
def freeze_by_prefix(model: nn.Module, prefixes: Tuple[str, ...], freeze: bool = True):
    for name, p in model.named_parameters():
        if name.startswith(prefixes):
            p.requires_grad = not freeze


def freeze_text_encoder(model: nn.Module, freeze: bool = True):
    freeze_by_prefix(model, (_TEXT_BACKBONE_TOKEN,), freeze)


def freeze_vision_encoder(model: nn.Module, freeze: bool = True):
    freeze_by_prefix(model, (_VISION_BACKBONE_TOKEN,), freeze)


def freeze_heads(model: nn.Module, freeze: bool = True):
    for name, p in model.named_parameters():
        if _is_head(name):
            p.requires_grad = not freeze


def freeze_projections(model: nn.Module, freeze_visual: Optional[bool] = None, freeze_text: Optional[bool] = None):
    for name, p in model.named_parameters():
        if (_VISUAL_PROJ_TOKEN in name) and (freeze_visual is not None):
            p.requires_grad = not freeze_visual
        if (_TEXT_PROJ_TOKEN in name) and (freeze_text is not None):
            p.requires_grad = not freeze_text


def unfreeze_vision_last_n_layers(model: nn.Module, n: int):
    if n <= 0:
        return
    freeze_vision_encoder(model, freeze=True)

    idx_set = set()
    for name, _ in model.named_parameters():
        m = _VISION_LAYER_RE.match(name)
        if m:
            idx_set.add(int(m.group(1)))
    if not idx_set:
        return

    mx = max(idx_set)
    target = {i for i in range(max(0, mx - n + 1), mx + 1)}
    for name, p in model.named_parameters():
        m = _VISION_LAYER_RE.match(name)
        if m and (int(m.group(1)) in target):
            p.requires_grad = True


def unfreeze_text_last_n_layers(model: nn.Module, n: int):
    if n <= 0:
        return
    freeze_text_encoder(model, freeze=True)

    idx_set = set()
    for name, _ in model.named_parameters():
        m = _TEXT_LAYER_RE.match(name)
        if m:
            idx_set.add(int(m.group(1)))
    if not idx_set:
        return

    mx = max(idx_set)
    target = {i for i in range(max(0, mx - n + 1), mx + 1)}
    for name, p in model.named_parameters():
        m = _TEXT_LAYER_RE.match(name)
        if m and (int(m.group(1)) in target):
            p.requires_grad = True


# =========================
# Targeted LoRA injection
# =========================
def add_lora_qkvo_and_heads(
    model: nn.Module,
    *,
    # Rank per group.
    r_backbone_vision: int = 4,
    r_backbone_text: int = 4,
    r_heads: int = 4,
    r_proj_visual: int = 4,
    r_proj_text: int = 4,
    # Alpha / dropout.
    alpha: Optional[float] = None,
    dropout_train: float = 0.0,  # Dropout only in train().
    # Enabled target groups.
    enable_vision_backbone: bool = True,
    enable_text_backbone: bool = False,
    enable_heads: bool = True,
    enable_projections: bool = True,
    # Attention projections to adapt.
    attn_targets=("q", "k", "v", "out"),
    # MLP targets.
    mlp_targets=("fc1", "fc2"),
    enable_vision_mlp: bool = True,
    enable_text_mlp: bool = True,
    r_mlp_vision: Optional[int] = None,
    r_mlp_text: Optional[int] = None,
    # Scope to the last N layers.
    vision_lora_last_n: int = 0,
    text_lora_last_n: int = 0,
    # Base bias handling.
    freeze_lora_base_bias: bool = True,
):
    def _make_lora_from(module: nn.Linear, r_val: int, alpha_val: Optional[float]) -> LoRALinear:
        if alpha_val is None:
            alpha_val = 4 * r_val if r_val > 0 else 1.0
        lora = LoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=(module.bias is not None),
            r=r_val,
            alpha=alpha_val,
            dropout_train=dropout_train,
        )
        with torch.no_grad():
            lora.weight.copy_(module.weight.data)
            if (module.bias is not None) and (lora.bias is not None):
                lora.bias.copy_(module.bias.data)
                if freeze_lora_base_bias:
                    lora.bias.requires_grad = False
        return lora

    # Allowed vision layers when scoping LoRA to the last N layers.
    allowed_vision_layers = None
    if vision_lora_last_n and vision_lora_last_n > 0:
        vis_idx = _vision_layer_indices(model)
        if vis_idx:
            mx = max(vis_idx)
            allowed_vision_layers = {i for i in vis_idx if i >= mx - vision_lora_last_n + 1}

    # Allowed text layers when scoping LoRA to the last N layers.
    allowed_text_layers = None
    if text_lora_last_n and text_lora_last_n > 0:
        txt_idx = _text_layer_indices(model)
        if txt_idx:
            mx = max(txt_idx)
            allowed_text_layers = {i for i in txt_idx if i >= mx - text_lora_last_n + 1}

    # Collect Linear module names once.
    linear_names = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    module_map = dict(model.named_modules())

    for full_name in linear_names:
        module = module_map[full_name]

        # Avoid double wrapping.
        if isinstance(module, LoRALinear) or hasattr(module, "lora_A"):
            continue

        parent, attr_name = resolve_parent_and_attr(model, full_name)

        # Heads.
        if enable_heads and _is_head(full_name):
            set_child_module(parent, attr_name, _make_lora_from(module, r_heads, alpha))
            continue

        # Projections.
        if enable_projections and _is_projection(full_name):
            if _VISUAL_PROJ_TOKEN in full_name:
                set_child_module(parent, attr_name, _make_lora_from(module, r_proj_visual, alpha))
                continue
            if _TEXT_PROJ_TOKEN in full_name:
                set_child_module(parent, attr_name, _make_lora_from(module, r_proj_text, alpha))
                continue

        # Vision backbone attention + MLP.
        if _is_vision_backbone(full_name):
            if allowed_vision_layers is not None:
                li = _vision_layer_index_from_name(full_name)
                if (li is None) or (li not in allowed_vision_layers):
                    continue

            if enable_vision_backbone and _is_attn_proj(full_name, attn_targets):
                set_child_module(parent, attr_name, _make_lora_from(module, r_backbone_vision, alpha))
                continue

            if enable_vision_mlp and _is_mlp_fc(full_name, mlp_targets):
                r_use = r_mlp_vision if r_mlp_vision is not None else r_backbone_vision
                set_child_module(parent, attr_name, _make_lora_from(module, r_use, alpha))
                continue

        # Text backbone attention + optional MLP.
        if _is_text_backbone(full_name):
            if allowed_text_layers is not None:
                li_t = _text_layer_index_from_name(full_name)
                if (li_t is None) or (li_t not in allowed_text_layers):
                    continue

            if enable_text_backbone and _is_attn_proj(full_name, attn_targets):
                set_child_module(parent, attr_name, _make_lora_from(module, r_backbone_text, alpha))
                continue

            if enable_text_mlp and _is_mlp_fc(full_name, mlp_targets):
                r_use = r_mlp_text if r_mlp_text is not None else r_backbone_text
                set_child_module(parent, attr_name, _make_lora_from(module, r_use, alpha))
                continue

    return model


# =========================
# Enforce trainable parameters
# =========================
def enforce_trainable_strict(
    model: nn.Module,
    *,
    keep_vision_last_n: int = 0,
    keep_vision_last_n_attn_full: bool = False,
    keep_vision_last_n_mlp_full: bool = False,
    keep_vision_last_n_ln_full: bool = False,
    keep_text_last_n: int = 0,
    keep_text_last_n_attn_full: bool = False,
    keep_text_last_n_mlp_full: bool = False,
    keep_text_last_n_ln_full: bool = False,
    keep_heads_full: bool = False,
    keep_projections_full: bool = False,
    keep_logit_scale: bool = False,
    keep_top_layernorm: bool = False,
):
    # Identify the last N vision blocks.
    vis_idx = _vision_layer_indices(model)
    allowed_last = set()
    if keep_vision_last_n and vis_idx:
        mx = max(vis_idx)
        allowed_last = {i for i in vis_idx if i >= mx - keep_vision_last_n + 1}

    # Identify the last N text blocks.
    txt_idx = _text_layer_indices(model)
    allowed_text_last = set()
    if keep_text_last_n and txt_idx:
        mx_t = max(txt_idx)
        allowed_text_last = {i for i in txt_idx if i >= mx_t - keep_text_last_n + 1}

    # LoRA module prefixes.
    lora_prefixes = set()
    for n, _ in model.named_parameters():
        if n.endswith(".lora_A") or n.endswith(".lora_B"):
            lora_prefixes.add(n.rsplit(".", 1)[0])

    def _is_lora_base_weight_or_bias(param_name: str) -> bool:
        pref = param_name.rsplit(".", 1)[0]
        return pref in lora_prefixes and (param_name.endswith(".weight") or param_name.endswith(".bias"))

    # Disable everything by default.
    for _, p in model.named_parameters():
        p.requires_grad = False

    # Re-enable parameters selectively.
    for name, p in model.named_parameters():
        ln = name.lower()

        # LoRA A/B are always trainable.
        if ("lora_a" in ln) or ("lora_b" in ln):
            p.requires_grad = True
            continue

        # Never re-enable base weights/biases inside LoRALinear modules.
        if _is_lora_base_weight_or_bias(name):
            continue

        # Last N vision blocks for granular full fine-tuning.
        li = _vision_layer_index_from_name(name)
        if (li is not None) and (li in allowed_last):
            if (".self_attn." in ln) and keep_vision_last_n_attn_full:
                p.requires_grad = True
                continue
            if (".mlp." in ln) and keep_vision_last_n_mlp_full:
                p.requires_grad = True
                continue
            if (".layer_norm1." in ln or ".layer_norm2." in ln) and keep_vision_last_n_ln_full:
                p.requires_grad = True
                continue

        # Last N text blocks for granular full fine-tuning.
        li_t = _text_layer_index_from_name(name)
        if (li_t is not None) and (li_t in allowed_text_last):
            if (".self_attn." in ln) and keep_text_last_n_attn_full:
                p.requires_grad = True
                continue
            if (".mlp." in ln) and keep_text_last_n_mlp_full:
                p.requires_grad = True
                continue
            if (".layer_norm1." in ln or ".layer_norm2." in ln) and keep_text_last_n_ln_full:
                p.requires_grad = True
                continue

        # Full heads, when requested.
        if keep_heads_full and _is_head(name):
            p.requires_grad = True
            continue

        # Full projections, when requested.
        if keep_projections_full and _is_projection(name):
            p.requires_grad = True
            continue

        # logit_scale.
        if keep_logit_scale and ("owlv2.logit_scale" in name):
            p.requires_grad = True
            continue

        # Top-level layer_norm, if the model exposes it this way.
        if keep_top_layernorm and name.startswith("layer_norm."):
            p.requires_grad = True
            continue


# =========================
# Unified configurator
# =========================
def configure_model(
    model: nn.Module,
    *,
    # Freeze / unfreeze.
    freeze_text: bool = False,
    freeze_vision: bool = False,
    unfreeze_vision_last: int = 0,
    unfreeze_text_last: int = 0,
    freeze_heads_flag: bool = False,
    freeze_visual_projection: Optional[bool] = None,  # None = leave unchanged
    freeze_text_projection: Optional[bool] = None,  # None = leave unchanged
    # LoRA.
    use_lora: bool = True,
    lora_dropout: float = 0.0,  # Dropout only in train().
    attn_targets=("q", "k", "v", "out"),
    # Separate ranks.
    r_backbone_vision: int = 4,
    r_backbone_text: int = 0,  # Default 0 = do not apply LoRA to the text backbone.
    r_heads: int = 4,
    r_proj_visual: int = 4,
    r_proj_text: int = 4,
    # LoRA target groups.
    lora_on_vision_backbone: bool = True,
    lora_on_text_backbone: bool = False,
    lora_on_heads: bool = True,
    lora_on_projections: bool = True,
):
    # 1) Freeze base modules.
    if freeze_text:
        freeze_text_encoder(model, True)
    if freeze_vision:
        freeze_vision_encoder(model, True)
    if freeze_heads_flag:
        freeze_heads(model, True)
    if (freeze_visual_projection is not None) or (freeze_text_projection is not None):
        freeze_projections(model, freeze_visual=freeze_visual_projection, freeze_text=freeze_text_projection)

    # 2) Unfreeze the last N vision/text layers; this takes priority over freeze flags.
    if (unfreeze_vision_last is not None) and (unfreeze_vision_last > 0):
        unfreeze_vision_last_n_layers(model, unfreeze_vision_last)
    if (unfreeze_text_last is not None) and (unfreeze_text_last > 0):
        unfreeze_text_last_n_layers(model, unfreeze_text_last)

    # 3) Inject LoRA or use full fine-tuning.
    if use_lora:
        add_lora_qkvo_and_heads(
            model,
            r_backbone_vision=r_backbone_vision,
            r_backbone_text=r_backbone_text,
            r_heads=r_heads,
            r_proj_visual=r_proj_visual,
            r_proj_text=r_proj_text,
            alpha=None,
            dropout_train=lora_dropout,
            enable_vision_backbone=lora_on_vision_backbone,
            enable_text_backbone=lora_on_text_backbone,
            enable_heads=lora_on_heads,
            enable_projections=lora_on_projections,
            attn_targets=attn_targets,
            # Vision/text MLP.
            mlp_targets=("fc1", "fc2"),
            enable_vision_mlp=True,
            enable_text_mlp=True,
            r_mlp_vision=r_backbone_vision,
            r_mlp_text=r_backbone_text,
            # Scope: only the last N layers, matching unfreeze_*_last.
            vision_lora_last_n=unfreeze_vision_last if unfreeze_vision_last else 0,
            text_lora_last_n=unfreeze_text_last if unfreeze_text_last else 0,
            freeze_lora_base_bias=True,
        )

        # 4) Enforce LoRA-only training plus optional extras.
        enforce_trainable_strict(
            model,
            keep_vision_last_n=unfreeze_vision_last if unfreeze_vision_last else 0,
            keep_vision_last_n_attn_full=False,
            keep_vision_last_n_mlp_full=False,
            keep_vision_last_n_ln_full=False,
            keep_text_last_n=unfreeze_text_last if unfreeze_text_last else 0,
            keep_text_last_n_attn_full=False,
            keep_text_last_n_mlp_full=False,
            keep_text_last_n_ln_full=False,
            keep_heads_full=False,
            keep_projections_full=False,
            keep_logit_scale=False,
            keep_top_layernorm=False,
        )
    else:
        # Full fine-tuning: enable all base model parameters.
        for _, p in model.named_parameters():
            p.requires_grad = True

    return model


# =========================
# Debug utility
# =========================
def summarize_trainables_all(model: nn.Module):
    trainables = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    total = sum(s for _, s in trainables)
    print(f"Trainable parameters: {len(trainables)} tensors, {total:,} elements")
    for n, s in trainables:
        print("  ", n, s)
