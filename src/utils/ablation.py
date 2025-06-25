from enum import StrEnum
from typing import Callable

import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


class AblationMode(StrEnum):
    ALL = "all"
    SEPARATE = "separate"


class AblationType(StrEnum):
    ZERO = "zero"
    MEAN = "mean"
    INTERCHANGE = "interchange"


def make_zero_ablation_hook(head: int):
    def hook(
        value: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        value[:, -1, head, :] = 0.0
        return value

    return hook


def make_mean_ablation_hook(
    head: int,
    layer: int,
    ablation_cache: dict[str, torch.Tensor],
):
    def hook(
        value: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        dataset_activations = ablation_cache[f"blocks.{layer}.attn.hook_result"]
        mean_val = dataset_activations[:, -1, head, :].mean(dim=0, keepdim=True)
        value[:, -1, head, :] = mean_val
        return value

    return hook


def make_interchange_hook(
    head: int,
    layer: int,
    ablation_cache: dict[str, torch.Tensor],
):
    def hook(
        value: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        male_activations = ablation_cache[f"blocks.{layer}.attn.hook_result"]
        indices = torch.randperm(male_activations.shape[0])[: value.shape[0]]
        value[:, -1, head, :] = male_activations[indices, -1, head, :]
        return value

    return hook


def run_ablation(
    heads: list[tuple[int, int]],
    model: HookedTransformer,
    tokens: torch.Tensor,
    he_token: int,
    she_token: int,
    ablation_type: AblationType,
    mode: AblationMode,
    ablation_tokens: torch.Tensor | None = None,
) -> torch.Tensor | float:
    ablation_cache = None
    if ablation_type in (AblationType.INTERCHANGE, AblationType.MEAN):
        msg = '"ablation_tokens" cannot be None when type is "interchange" or "mean"'
        assert ablation_tokens is not None, msg

        # layers = range(model.cfg.n_layers)
        # hook_names = [f"blocks.{layer}.attn.hook_result" for layer in layers]

        hook_names = {f"blocks.{layer}.attn.hook_result" for layer, _ in heads}
        _, ablation_cache = model.run_with_cache(
            ablation_tokens, names_filter=lambda name: name in hook_names
        )
        ablation_cache = {k: v.cpu() for k, v in ablation_cache.items()}

    def get_hook(layer: int, head: int) -> Callable:
        if ablation_type == AblationType.ZERO:
            return make_zero_ablation_hook(head)
        elif ablation_type == AblationType.MEAN:
            return make_mean_ablation_hook(head, layer, ablation_cache)
        elif ablation_type == AblationType.INTERCHANGE:
            return make_interchange_hook(head, layer, ablation_cache)
        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")

    with torch.no_grad():
        original_logits = model(tokens, return_type="logits")
        original_probs = torch.softmax(original_logits[:, -1], dim=-1).mean(dim=0)
        original_prob_diff = original_probs[she_token] - original_probs[he_token]

    if mode == AblationMode.ALL:
        hooks = [
            (
                f"blocks.{layer}.attn.hook_result",
                get_hook(layer, head),
            )
            for layer, head in heads
        ]

        with torch.no_grad():
            ablated_logits = model.run_with_hooks(
                tokens, return_type="logits", fwd_hooks=hooks
            )
            ablated_probs = torch.softmax(ablated_logits[:, -1], dim=-1).mean(dim=0)
            ablated_prob_diff = ablated_probs[she_token] - ablated_probs[he_token]
            return original_prob_diff - ablated_prob_diff

    elif mode == AblationMode.SEPARATE:
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        prob_diffs = torch.zeros((n_layers, n_heads), device=tokens.device)

        for layer, head in heads:
            hook = get_hook(layer, head)
            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    tokens,
                    return_type="logits",
                    fwd_hooks=[(f"blocks.{layer}.attn.hook_result", hook)],
                )
                ablated_probs = torch.softmax(ablated_logits[:, -1], dim=-1).mean(dim=0)
                ablated_prob_diff = ablated_probs[she_token] - ablated_probs[he_token]
                prob_diffs[layer, head] = original_prob_diff - ablated_prob_diff

        return prob_diffs
