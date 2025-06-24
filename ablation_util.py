from typing import List, Tuple
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float
from typing import Dict
from enum import Enum

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"{device=}")

class AblationMode(Enum):
    # all heads are ablated together - a single probability diff is returned
    ALL = "all"
    # each head is ablated separately - a probability diff is returned for each head
    SEPARATE = "separate"

# ---------------------------
# ------ Zero ablation ------
# ---------------------------
def make_zero_ablation_hook(head_index_to_ablate: int):
    def zero_ablation_hook(
        value: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        value[:, -1, head_index_to_ablate, :] = 0.
        return value
    return zero_ablation_hook

def perform_zero_ablation(heads: List[Tuple[int, int]], model: HookedTransformer, toks: torch.Tensor, she_token: int, he_token: int, mode: AblationMode = AblationMode.SEPARATE):
    original_probs = torch.softmax(model(toks, return_type="logits")[:, -1], dim=-1).mean(0)
    original_prob_diff = original_probs[she_token] - original_probs[he_token]
    if mode == AblationMode.ALL:
        hooks = [(f'blocks.{layer_to_ablate}.attn.hook_result', make_zero_ablation_hook(head_index_to_ablate)) for layer_to_ablate, head_index_to_ablate in heads]
        ablated_probs = torch.softmax(model.run_with_hooks(
            toks,
            return_type="logits",
            fwd_hooks=hooks
            )[:, -1], dim=-1).mean(0)
        print(f"Original ' she' prob: {original_probs[:, she_token].mean().item():.3f}, Original ' he' prob: {original_probs[:, he_token].mean().item():.3f}")
        print(f"Ablated ' she' prob: {ablated_probs[:, she_token].mean().item():.3f}, Ablated ' he' prob: {ablated_probs[:, he_token].mean().item():.3f}")  
        return original_prob_diff - (ablated_probs[she_token] - ablated_probs[he_token])
    elif mode == AblationMode.SEPARATE:
        prob_diffs = torch.zeros((12, 12)).to(device)
        prob_diffs[:, :] = original_prob_diff
        for layer_to_ablate, head_index_to_ablate in heads:
            ablated_probs = torch.softmax(model.run_with_hooks(
                toks,
                return_type="logits",
                fwd_hooks=[(f'blocks.{layer_to_ablate}.attn.hook_result', make_zero_ablation_hook(head_index_to_ablate))]
                )[:, -1], dim=-1).mean(0)
            prob_diffs[layer_to_ablate, head_index_to_ablate] = ablated_probs[she_token] - ablated_probs[he_token]
            # print(f"Layer {layer_to_ablate}, Head {head_index_to_ablate}: {prob_diffs[layer_to_ablate, head_index_to_ablate]}")
            # print(f"Original prob diff: {original_prob_diff}")
        return original_prob_diff - prob_diffs

# ---------------------------
# ------ Mean ablation ------
# ---------------------------

def make_mean_ablation_hook(head_index_to_ablate: int):
    def mean_ablation_hook(
        value: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        value[:, -1, head_index_to_ablate, :] = value[:, -1, head_index_to_ablate, :].mean(dim=0, keepdim=True)
        return value
    return mean_ablation_hook

def perform_mean_ablation(heads: List[Tuple[int, int]], model: HookedTransformer, toks: torch.Tensor, she_token: int, he_token: int, mode: AblationMode = AblationMode.SEPARATE):
    original_probs = torch.softmax(model(toks, return_type="logits")[:, -1], dim=-1).mean(0)
    original_prob_diff = original_probs[she_token] - original_probs[he_token]

    if mode == AblationMode.ALL:
        hooks = [(f'blocks.{layer_to_ablate}.attn.hook_result', make_mean_ablation_hook(head_index_to_ablate)) for layer_to_ablate, head_index_to_ablate in heads]  
        ablated_probs = torch.softmax(model.run_with_hooks(
            toks,
            return_type="logits",
            fwd_hooks=hooks
            )[:, -1], dim=-1).mean(0)
        return original_prob_diff - (ablated_probs[she_token] - ablated_probs[he_token])
    elif mode == AblationMode.SEPARATE:
        prob_diffs = torch.zeros((12, 12)).to(device)
        prob_diffs[:, :] = original_prob_diff
        for layer_to_ablate, head_index_to_ablate in heads:
            ablated_probs = torch.softmax(model.run_with_hooks(
                toks,
                return_type="logits",
                fwd_hooks=[(f'blocks.{layer_to_ablate}.attn.hook_result', make_mean_ablation_hook(head_index_to_ablate))]
                )[:, -1], dim=-1).mean(0)
            prob_diffs[layer_to_ablate, head_index_to_ablate] = ablated_probs[she_token] - ablated_probs[he_token]
        return original_prob_diff - prob_diffs

# ----------------------------------
# ------ Interchange ablation ------
# ----------------------------------

def make_interchange_intervention_hook(head_index_to_ablate: int, layer_to_ablate: int, ablation_cache: Dict[str, torch.Tensor]):
        def interchange_intervention_hook(
            value: Float[torch.Tensor, "batch pos head_index d_head"],
            hook: HookPoint
        ) -> Float[torch.Tensor, "batch pos head_index d_head"]:

            male_activations = ablation_cache[f'blocks.{layer_to_ablate}.attn.hook_result']
            male_indices = torch.randperm(male_activations.shape[0])[:value.shape[0]]
            value[:, -1, head_index_to_ablate, :] = male_activations[male_indices, -1, head_index_to_ablate, :]
            return value
        return interchange_intervention_hook

def perform_interchange_ablation(heads: List[Tuple[int, int]], model: HookedTransformer, toks: torch.Tensor, ablation_toks: torch.Tensor, she_token: int, he_token: int, mode: AblationMode = AblationMode.SEPARATE):
    hook_names = [f'blocks.{layer}.attn.hook_result' for layer in range(12)]
    _, ablation_cache = model.run_with_cache(ablation_toks, names_filter=lambda name: name in hook_names)

    original_probs = torch.softmax(model(toks, return_type="logits")[:, -1], dim=-1).mean(0)
    original_prob_diff = original_probs[she_token] - original_probs[he_token]

    prob_diffs = torch.zeros((12, 12)).to(device)
    prob_diffs[:, :] = original_prob_diff
    for layer_to_ablate, head_index_to_ablate in heads:
        ablated_probs = torch.softmax(model.run_with_hooks(
            toks,
            return_type="logits",
            fwd_hooks=[(
                f'blocks.{layer_to_ablate}.attn.hook_result',
                make_interchange_intervention_hook(head_index_to_ablate, layer_to_ablate, ablation_cache)
                )]
            )[:, -1], dim=-1).mean(0)
        prob_diffs[layer_to_ablate, head_index_to_ablate] = ablated_probs[she_token] - ablated_probs[he_token]

    return original_prob_diff - prob_diffs
