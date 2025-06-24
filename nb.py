# # Janky code to do different setup when run in a Colab notebook vs VSCode
# DEVELOPMENT_MODE = False
# try:
#     import google.colab
#     IN_COLAB = True
#     print("Running as a Colab notebook")
#     %pip install git+https://github.com/neelnanda-io/TransformerLens.git
#     %pip install circuitsvis
#     %pip install kaleido

#     # PySvelte is an unmaintained visualization library, use it as a backup if circuitsvis isn't working
#     # # Install another version of node that makes PySvelte work way faster
#     # !curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -; sudo apt-get install -y nodejs
#     # %pip install git+https://github.com/neelnanda-io/PySvelte.git
#     !wget https://raw.githubusercontent.com/sebastianGehrmann/CausalMediationAnalysis/master/experiment_data/professions_male_stereo.json
#     !wget https://raw.githubusercontent.com/sebastianGehrmann/CausalMediationAnalysis/master/experiment_data/professions_female_stereo.json
# except:
#     IN_COLAB = False
#     print("Running as a Jupyter notebook - intended for development only!")
#     from IPython import get_ipython

#     ipython = get_ipython()
#     # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
#     # ipython.magic("load_ext autoreload")
#     # ipython.magic("autoreload 2")

import circuitsvis as cv
# Testing that the library works
cv.examples.hello("world")

# Import stuff
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint

from jaxtyping import Float
from typing import List, Union, Tuple
import json

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if not torch.cuda.is_available() and not torch.backends.mps.is_available():
    print("WARNING: Running on CPU. Did you remember to set your Colab accelerator to GPU?")
torch.set_grad_enabled(False)
print(f"Using device: {device}")

model_name = 'gpt2-small'
model = HookedTransformer.from_pretrained(model_name, device=device)

model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. See my explainer for documentation of all supported models, and this table for hyper-parameters and the name used to load them. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""
loss = model(model_description_text, return_type="loss")
print("Model loss:", loss)

def topk(logits: torch.Tensor, k: int) -> List[Tuple[str, float]]:
    all_probs = torch.nn.functional.softmax(logits[-1, :], dim=-1)
    topk_probs, topk_indices = torch.topk(all_probs, k=k)
    topk_tokens = model.tokenizer.convert_ids_to_tokens(topk_indices)
    return list(zip(topk_tokens, topk_probs))

s1 = "The nurse said that"
logits1 = model(s1, return_type="logits").squeeze(0)
print(s1)
for token, prob in topk(logits1, 5):
    print(f'"{token}"', '\t', f'{prob:.2f}')


s2 = "The doctor said that"
logits2 = model(s2, return_type="logits").squeeze(0)
print(s2)
for token, prob in topk(logits2, 5):
    print(f'"{token}"', '\t', f'{prob:.2f}')

# We're going to load up some examples.
with open("professions_female_stereo.json", 'r') as f:
    female_stereo_professions = [x[0] for x in json.load(f)]

with open("professions_male_stereo.json", 'r') as f:
    male_stereo_professions = [x[0] for x in json.load(f)]

# ensure that they're single tokens
female_stereo_professions_lens = [len(model.tokenizer.encode(f' {x}')) for x in female_stereo_professions]
female_stereo_professions = [x for x, l in zip(female_stereo_professions, female_stereo_professions_lens) if l == 1]

male_stereo_professions_lens = [len(model.tokenizer.encode(f' {x}')) for x in male_stereo_professions]
male_stereo_professions = [x for x, l in zip(male_stereo_professions, male_stereo_professions_lens) if l == 1]

# slot them into our sentences
female_stereo_sentences = [f"The {profession} said that" for profession in female_stereo_professions]
male_stereo_sentences = [f"The {profession} said that" for profession in male_stereo_professions]

# convert our sentences into tokens
female_stereo_toks = model.to_tokens(female_stereo_sentences)
male_stereo_toks = model.to_tokens(male_stereo_sentences)
model.cfg.use_attn_result = True
gpt2_logits, gpt2_cache = model.run_with_cache(female_stereo_toks)

def sort_heads(gpt2_cache: ActivationCache, query_index: int, key_index: int) -> List[Tuple[Tuple[int,int], float]]:
    out = []
    for layer in range(12):
        mean_attn_pattern = gpt2_cache[f'blocks.{layer}.attn.hook_pattern'].mean(dim=0)
        for head in range(12):
            head_index = (layer, head)
            head_attn = float(mean_attn_pattern[head, query_index, key_index])
            out.append((head_index, head_attn))
    return sorted(out, key=lambda tuple: tuple[1], reverse=True)

# Show which attention heads attended the most to index 2 ("actress") at the end position (-1)
sorted_heads = sort_heads(gpt2_cache, -1, 2)

print(sorted_heads[:5])

def logit_lens(representation: Float[torch.Tensor, "batch dimension"], unembed_matrix: Float[torch.Tensor, "dimension vocab_size"]) -> Float[torch.Tensor, "batch vocab_size"]:
    return representation @ unembed_matrix

# actually getting the representations in TransformerLens
unembed_matrix = model.unembed.W_U
representations = torch.cat([gpt2_cache[f"blocks.{layer}.attn.hook_result"].mean(0)[-1] for layer in range(12)], dim=0)

# Performing logit lens, and extracting she - he upweighting
logit_lenses = logit_lens(representations, unembed_matrix)

she_token = model.tokenizer.encode(' she')[0]
he_token = model.tokenizer.encode(' he')[0]

she_token_logit_lenses = logit_lenses[:, she_token]
he_token_logit_lenses = logit_lenses[:, he_token]
logit_lens_diff = she_token_logit_lenses - he_token_logit_lenses

heads = [(layer, head) for layer in range(12) for head in range(12)]
values, indices = logit_lens_diff.sort(descending=True)
heads_increasing_she = list(zip([heads[i] for i in indices], values.tolist()))

head_attn_dict = {head: attn for head, attn in sorted_heads}
for head, logit_diff in heads_increasing_she[:5]:
    print(head, logit_diff, head_attn_dict[head])

head_logit_dict = {head: logit_diff for head, logit_diff in heads_increasing_she}
for head, attn in sorted_heads[:5]:
    print(head, head_logit_dict[head], attn)

shape1 = 12  # FILL THIS IN
shape2 = 12  # FILL THIS IN
xlabel = 'head'  # FILL THIS IN
ylabel = 'layer'  # FILL THIS IN

heatmap_input = logit_lens_diff.cpu().view(shape1, shape2)

px.imshow(heatmap_input, color_continuous_scale='RdBu', zmin=-7, zmax=7, labels={'x': xlabel, 'y': ylabel, 'color':'logit diff'}, y=list(range(shape1)), x=list(range(shape2)), width=800, height=600, title=f'Logit Diff by {xlabel} / {ylabel}')

plt.hist(logit_lens_diff.cpu().flatten(), bins=np.arange(np.floor(logit_lens_diff.cpu().min()), np.ceil(logit_lens_diff.cpu().max()) + 0.5, 0.5), edgecolor='black', linewidth=0.5)
plt.xlabel('logit diff')
plt.ylabel('count')
plt.show()

layer_to_ablate = 10
head_index_to_ablate = 9

# We define a head ablation hook
# The type annotations are NOT necessary, they're just a useful guide to the reader

def zero_ablation_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    print(f"Shape of the value tensor: {value.shape}")
    value[:, -1, head_index_to_ablate, :] = 0.
    return value

original_probs = torch.softmax(model(female_stereo_toks, return_type="logits")[:, -1], dim=-1)
ablated_probs = torch.softmax(model.run_with_hooks(
    female_stereo_toks,
    return_type="logits",
    fwd_hooks=[(
        f'blocks.{layer_to_ablate}.attn.hook_result',
        zero_ablation_hook
        )]
    )[:, -1], dim=-1)
print(f"Original ' she' prob: {original_probs[:, she_token].mean().item():.3f}, Original ' he' prob: {original_probs[:, he_token].mean().item():.3f}")
print(f"Ablated ' she' prob: {ablated_probs[:, she_token].mean().item():.3f}, Ablated ' he' prob: {ablated_probs[:, he_token].mean().item():.3f}")

layer_to_ablate = 10
head_index_to_ablate = 9

def perform_zero_ablation(heads: List[Tuple[int, int]], model: HookedTransformer, toks: torch.Tensor):
    hooks = [(f'blocks.{layer_to_ablate}.attn.hook_result', zero_ablation_hook) for layer_to_ablate, _ in heads]

    original_probs = torch.softmax(model(toks, return_type="logits")[:, -1], dim=-1)
    ablated_probs = torch.softmax(model.run_with_hooks(
        toks,
        return_type="logits",
        fwd_hooks=hooks
        )[:, -1], dim=-1)
    print(f"Original ' she' prob: {original_probs[:, she_token].mean().item():.3f}, Original ' he' prob: {original_probs[:, he_token].mean().item():.3f}")
    print(f"Ablated ' she' prob: {ablated_probs[:, she_token].mean().item():.3f}, Ablated ' he' prob: {ablated_probs[:, he_token].mean().item():.3f}")  
    return original_probs, ablated_probs

def mean_ablation_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint
) -> Float[torch.Tensor, "batch pos head_index d_head"]:

    # TODO: Write this part: get the relevant activation from gpt2_cache, and take the mean over examples
    value[:, -1, head_index_to_ablate, :] = value[:, -1, head_index_to_ablate, :].mean(dim=0, keepdim=True)
    return value

original_probs = torch.softmax(model(female_stereo_toks, return_type="logits")[:, -1], dim=-1)
ablated_probs = torch.softmax(model.run_with_hooks(
    female_stereo_toks,
    return_type="logits",
    fwd_hooks=[(
        f'blocks.{layer_to_ablate}.attn.hook_result',
        mean_ablation_hook
        )]
    )[:, -1], dim=-1)
print(f"Original ' she' prob: {original_probs[:, she_token].mean().item():.3f}, Original ' he' prob: {original_probs[:, he_token].mean().item():.3f}")
print(f"Ablated ' she' prob: {ablated_probs[:, she_token].mean().item():.3f}, Ablated ' he' prob: {ablated_probs[:, he_token].mean().item():.3f}")

def perform_mean_ablation(heads: List[Tuple[int, int]], model: HookedTransformer, toks: torch.Tensor):
    hooks = [(f'blocks.{layer_to_ablate}.attn.hook_result', mean_ablation_hook) for layer_to_ablate, _ in heads]

    original_probs = torch.softmax(model(toks, return_type="logits")[:, -1], dim=-1)
    ablated_probs = torch.softmax(model.run_with_hooks(
        toks,
        return_type="logits",
        fwd_hooks=hooks
        )[:, -1], dim=-1)
    print(f"Original ' she' prob: {original_probs[:, she_token].mean().item():.3f}, Original ' he' prob: {original_probs[:, he_token].mean().item():.3f}")
    print(f"Ablated ' she' prob: {ablated_probs[:, she_token].mean().item():.3f}, Ablated ' he' prob: {ablated_probs[:, he_token].mean().item():.3f}")

del gpt2_cache
hook_names = [f'blocks.{layer}.attn.hook_result' for layer in range(12)]
_, male_stereo_cache = model.run_with_cache(male_stereo_toks, names_filter=lambda name: name in hook_names)

layer_to_ablate = 10
head_index_to_ablate = 9

def interchange_intervention_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint
) -> Float[torch.Tensor, "batch pos head_index d_head"]:

    male_activations = male_stereo_cache[f'blocks.{layer_to_ablate}.attn.hook_result']
    male_indices = torch.randperm(male_activations.shape[0])[:value.shape[0]]
    value[:, -1, head_index_to_ablate, :] = male_activations[male_indices, -1, head_index_to_ablate, :]
    return value

original_probs = torch.softmax(model(female_stereo_toks, return_type="logits")[:, -1], dim=-1).mean(0)
ablated_probs = torch.softmax(model.run_with_hooks(
    female_stereo_toks,
    return_type="logits",
    fwd_hooks=[(
        f'blocks.{layer_to_ablate}.attn.hook_result',
        interchange_intervention_hook
        )]
    )[:, -1], dim=-1).mean(0)
print(f"Original ' she' prob: {original_probs[she_token].item():.3f}, Original ' he' prob: {original_probs[he_token].item():.3f}")
print(f"Ablated ' she' prob: {ablated_probs[she_token].item():.3f}, Ablated ' he' prob: {ablated_probs[he_token].item():.3f}")

original_probs = torch.softmax(model(female_stereo_toks, return_type="logits")[:, -1], dim=-1).mean(0)
orig_prob_diff = original_probs[she_token] - original_probs[he_token]

def make_interchange_intervention_hook(head_index_to_ablate: int, layer_to_ablate: int):
    def interchange_intervention_hook(
        value: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:

        # TODO: Fill this in; it's the same as the last one
        male_activations = male_stereo_cache[f'blocks.{layer_to_ablate}.attn.hook_result']
        male_indices = torch.randperm(male_activations.shape[0])[:value.shape[0]]
        value[:, -1, head_index_to_ablate, :] = male_activations[male_indices, -1, head_index_to_ablate, :]
        return value
    return interchange_intervention_hook

prob_diffs = []
for layer_to_ablate in range(12):
    for head_index_to_ablate in range(12):
        ablated_probs = torch.softmax(model.run_with_hooks(
            female_stereo_toks,
            return_type="logits",
            fwd_hooks=[(
                f'blocks.{layer_to_ablate}.attn.hook_result',
                make_interchange_intervention_hook(head_index_to_ablate, layer_to_ablate)
                )]
            )[:, -1], dim=-1).mean(0)
        prob_diff = ablated_probs[she_token] - ablated_probs[he_token]
        prob_diffs.append(prob_diff)

prob_diffs = torch.tensor(prob_diffs).to(device)
prob_change = orig_prob_diff.unsqueeze(0) - prob_diffs

px.imshow(prob_change.cpu().view(12,12), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob diff'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Diff by {xlabel} / {ylabel}')

# Interchange ablation

def perform_interchange_ablation(heads: List[Tuple[int, int]], model: HookedTransformer, toks: torch.Tensor):
    original_probs = torch.softmax(model(toks, return_type="logits")[:, -1], dim=-1).mean(0)
    orig_prob_diff = original_probs[she_token] - original_probs[he_token]

    def make_interchange_intervention_hook(head_index_to_ablate: int, layer_to_ablate: int):
        def interchange_intervention_hook(
            value: Float[torch.Tensor, "batch pos head_index d_head"],
            hook: HookPoint
        ) -> Float[torch.Tensor, "batch pos head_index d_head"]:

            # TODO: Fill this in; it's the same as the last one
            male_activations = male_stereo_cache[f'blocks.{layer_to_ablate}.attn.hook_result']
            male_indices = torch.randperm(male_activations.shape[0])[:value.shape[0]]
            value[:, -1, head_index_to_ablate, :] = male_activations[male_indices, -1, head_index_to_ablate, :]
            return value
        return interchange_intervention_hook

    prob_diffs = []
    for layer_to_ablate in range(12):
        for head_index_to_ablate in range(12):
            ablated_probs = torch.softmax(model.run_with_hooks(
                female_stereo_toks,
                return_type="logits",
                fwd_hooks=[(
                    f'blocks.{layer_to_ablate}.attn.hook_result',
                    make_interchange_intervention_hook(head_index_to_ablate, layer_to_ablate)
                    )]
                )[:, -1], dim=-1).mean(0)
            prob_diff = ablated_probs[she_token] - ablated_probs[he_token]
            prob_diffs.append(prob_diff)

    prob_diffs = torch.tensor(prob_diffs).to(device)
    prob_change = orig_prob_diff.unsqueeze(0) - prob_diffs

    px.imshow(prob_change.cpu().view(12,12), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob diff'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Diff by {xlabel} / {ylabel}')
from typing import Literal

sampling_indices = torch.randint(0, male_stereo_toks.size(0), (female_stereo_toks.size(0),))
_, original_cache = model.run_with_cache(female_stereo_toks)
_, ablated_cache = model.run_with_cache(male_stereo_toks[sampling_indices])

print("Starting direct ablation")

def make_final_residual_stream_hook(A_original: torch.Tensor, A_ablated: torch.Tensor):
    def final_residual_stream_hook(final_residual_stream: Float[torch.Tensor, "batch pos d_residual"], hook: HookPoint):
        # TODO: one-liner; just return the updated value of the final residual stream as described in the last paragraph of the prior section
        # Only update the end / final position
        # print("SHAPES")
        # print(final_residual_stream.shape, A_original.shape, A_ablated.shape)
        final_residual_stream[:, -1, :] = final_residual_stream[:, -1, :] - A_original[:, -1, :] + A_ablated[:, -1, :]
        return final_residual_stream
    return final_residual_stream_hook


def direct_ablation(layer: int, head_or_mlp: Union[Literal['mlp'], int]) -> torch.Tensor:
    # TODO: retrieve the correct values A_original and A_ablation. Then, run the model with the hook you've implemented above to get new logits
    if head_or_mlp == 'mlp':
        hook_str = f'blocks.{layer}.hook_mlp_out'
        A_original = original_cache[hook_str]
        A_ablated = ablated_cache[hook_str]
    else:
        hook_str = f'blocks.{layer}.attn.hook_result'
        A_original = original_cache[hook_str][:, :, head_or_mlp, :].squeeze()
        A_ablated = ablated_cache[hook_str][:, :, head_or_mlp, :].squeeze()
    direct_logits = model.run_with_hooks(
        female_stereo_toks,
        return_type='logits',
        fwd_hooks=[(
            f'blocks.11.hook_resid_post',
            make_final_residual_stream_hook(A_original, A_ablated)
        )]
    )
    return direct_logits


prob_diffs = []
for layer in range(12):
    for head_or_mlp in [*range(12), 'mlp']:
        direct_logits = direct_ablation(layer, head_or_mlp)
        direct_probs = torch.softmax(direct_logits, dim=-1).mean(0)[-1]
        prob_diff = direct_probs[she_token] - direct_probs[he_token]
        prob_diffs.append(prob_diff)
prob_diffs = torch.tensor(prob_diffs).to(device)
prob_change = orig_prob_diff.unsqueeze(0) - prob_diffs
prob_change = prob_change.view(12, 13)
px.imshow(prob_change.cpu(), color_continuous_scale='RdBu', zmin=-0.06, zmax=0.06, labels={'x':'head or mlp', 'y':'layer', 'color':'prob diff change'}, y=list(range(12)), x=[*(str(i) for i in range(12)), 'mlp'], width=800, height=600, title='Prob Diff Change by Layer / Head')

# Unneeded code starts here

sampling_indices = torch.randint(0, male_stereo_toks.size(0), (female_stereo_toks.size(0),))
orig_logits, original_cache = model.run_with_cache(female_stereo_toks)
ablated_logits, ablated_cache = model.run_with_cache(male_stereo_toks[sampling_indices])

def get_output_from_cache(cache: ActivationCache, layer:int, head_or_mlp: Union[int, Literal['mlp']]) -> torch.Tensor:
    if head_or_mlp == 'logits':
        raise ValueError("Just use the actual logits")
    elif head_or_mlp == 'mlp':
        return cache[f'blocks.{layer}.hook_mlp_out']
    else:
        return cache[f'blocks.{layer}.attn.hook_result'][:, :, head_or_mlp]

circuit = [(10, 9), (9, 7), (11, 8), (10, 'mlp'), (9, 'mlp'), (9, 2), (8, 11), (8, 'mlp'), (4,3), (6,0)]

original_values = torch.sum(torch.stack([get_output_from_cache(ablated_cache, *component) for component in circuit]), dim=0)
new_values = torch.sum(torch.stack([get_output_from_cache(original_cache, *component) for component in circuit]), dim=0)

circuit_logits = model.run_with_hooks(
        male_stereo_toks[sampling_indices],
        return_type="logits",
        fwd_hooks=[(
            "blocks.11.hook_resid_post",
            make_final_residual_stream_hook(original_values, new_values)
            )]
        )

for name, logits in zip(['original prob diff', 'ablated prob diff', 'circuit prob diff'], [orig_logits, ablated_logits, circuit_logits]):
    probs = torch.softmax(logits, dim=-1).mean(0)[-1]
    prob_diff = probs[she_token] - probs[he_token]
    print(f'{name}: {prob_diff:0.3f}')

HeadOrMLP: type = Union[Literal['logits'], Literal['mlp'], Tuple[int, Union[Literal['q'],Literal['k'],Literal['v'],None]]]
Node: type = Tuple[int, HeadOrMLP, int]
# A node is a triple of (layer, 'logits' / 'mlp' / (Head #, q/k/v), position)

def assert_valid_path(path: List[Node]) -> None:
    # asserts that a path is valid (by this I mean that for every pair of nodes in the path, the first node has an effect on the second)
    for source, target in zip(path, path[1:]):
        source_layer, source_head_or_mlp, source_position = source
        target_layer, target_head_or_mlp, target_position = target
        if target_head_or_mlp == 'logits':
            if source_position != target_position:
                raise ValueError(f"Path from {source} to {target} has mismatched position (invalid for logit target)")
        elif target_head_or_mlp == 'mlp':
            if source_position != target_position:
                raise ValueError(f"Path from {source} to {target} has mismatched position (invalid for mlp target)")
            if source_layer > target_layer:
                raise ValueError(f"Path from {source} to {target} has source layer > target layer (invalid for mlp target)")
        else:
            if source_layer > target_layer:
                raise ValueError(f"Path from {source} to {target} has source layer >= target layer (invalid for attn target)")

def is_upstream_of(u: Node, v: Node) -> bool:
    # checks if node u is upstream of node v
    u_layer, u_head_or_mlp, _ = u
    v_layer, v_head_or_mlp, _ = v
    if u_head_or_mlp == 'logits':
        return False
    elif v_head_or_mlp == 'logits':
        return True
    elif v_head_or_mlp == 'mlp':
        return (u_layer < v_layer) or ((u_layer == v_layer) and (u_head_or_mlp != 'mlp'))
    else:
        return u_layer < v_layer

def get_output_from_cache(cache: ActivationCache, layer:int, head_or_mlp: HeadOrMLP, position:int) -> torch.Tensor:
    # given specification of a node, get the activations thereof from the cache
    if head_or_mlp == 'logits':
        raise ValueError("Just use the actual logits")
    elif head_or_mlp == 'mlp':
        return cache[f'blocks.{layer}.hook_mlp_out']
    else:
        head, stream = head_or_mlp
        return cache[f'blocks.{layer}.attn.hook_result'][:, :, head]

def intervene_on_input(layer:int,
                       head_or_mlp: HeadOrMLP,
                       position: int,
                       original: Float[torch.Tensor, "batch pos d_residual position"],
                       ablated: Float[torch.Tensor, "batch pos d_residual position"]):

    """
    Given a target layer and head, and a position in its input, modifies the input, subtracting out original and adding in ablated.
    """

    def residual_stream_hook(value: Float[torch.Tensor, "batch pos d_residual"], hook: HookPoint):
        value[:, position] = value[:, position] - original[:, position] + ablated[:, position]
        return value
    if head_or_mlp == 'logits':
        name = "blocks.11.hook_resid_post"
    elif head_or_mlp == 'mlp':
        name = f"blocks.{layer}.ln2.hook_normalized"
    else:
        head, stream = head_or_mlp
        if stream is None:
            name = f"blocks.{layer}.ln1.hook_normalized"
        else:
            name = f"blocks.{layer}.attn.hook_{stream}"
    return name, residual_stream_hook

# Each tuple is (layer, head, position)
def path_ablation(path: List[Node], original_toks, ablation_toks) -> torch.Tensor:
    """
    Given a path (list of nodes), original toks, and ablation toks, performs an interchange ablation.
    The effects of this ablation will propagate only through the specified path.
    """
    assert_valid_path(path)
    _, original_cache = model.run_with_cache(original_toks)
    ablated_logits, ablated_cache = model.run_with_cache(ablation_toks)

    for source_info, target_info in zip(path, path[1:]):
        # get original
        original = get_output_from_cache(original_cache, *source_info)
        # get ablated
        ablated = get_output_from_cache(ablated_cache, *source_info)

        # create and add hook
        _, _, source_position = source_info
        target_layer, target_head_or_mlp, _ = target_info
        model.add_hook(*intervene_on_input(target_layer, target_head_or_mlp, source_position, original, ablated))

        # run_with_hooks, propagating the ablation
        ablated_logits, ablated_cache = model.run_with_cache(original_toks)
        model.reset_hooks()
    return ablated_logits

try:
    profession_position = 2
    end_position = -1
    path = [(10, (9, None), end_position), (11, 'logits', end_position)]
    prob_diffs = []
    for layer in range(12):
        for head_or_mlp in [*zip(range(12), (None,) * 12), 'mlp']:
            if is_upstream_of((layer, head_or_mlp, profession_position), path[0]):
                path_logits = path_ablation([(layer, head_or_mlp, profession_position)] + path, female_stereo_toks, male_stereo_toks[sampling_indices])
                path_probs = torch.softmax(path_logits, dim=-1).mean(0)[-1]
                prob_diff = path_probs[she_token] - path_probs[he_token]
            else:
                prob_diff = orig_prob_diff
            prob_diffs.append(prob_diff)
    prob_diffs = torch.tensor(prob_diffs).to(device)
    prob_change = prob_diffs - orig_prob_diff.unsqueeze(0)
    prob_change = prob_change.view(12, 13)
    fig = px.imshow(prob_change.cpu(), color_continuous_scale='RdBu', zmin=-0.06, zmax=0.06, labels={'x':'head or mlp', 'y':'layer', 'color':'prob diff change'}, y=list(range(12)), x=[*(str(i) for i in range(12)), 'mlp'], width=800, height=600, title='Prob Diff Change by Layer / Head')
    fig.show()
except Exception as e:
    print(layer, head_or_mlp)
    model.reset_hooks()
    raise e