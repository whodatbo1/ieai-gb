from dataset_util import load_professions_dataset, load_names_dataset
from ablation_util import perform_zero_ablation, perform_mean_ablation, perform_interchange_ablation, AblationMode
from transformer_lens import HookedTransformer
import plotly.express as px
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
torch.set_grad_enabled(False)

xlabel = 'head'
ylabel = 'layer'

all_heads = [(layer, head) for layer in range(12) for head in range(12)]

model_name = 'gpt2-small'
model = HookedTransformer.from_pretrained(model_name, device=device)
she_token = model.tokenizer.encode(' she')[0]
he_token = model.tokenizer.encode(' he')[0]

female_stereo_toks, male_stereo_toks, gpt2_logits, gpt2_cache = load_professions_dataset(model)
female_names_toks, male_names_toks, gpt2_names_logits, gpt2_names_cache = load_names_dataset(model)

# Names dataset

print("Performing zero ablation on names dataset")
prob_change = perform_zero_ablation(all_heads, model, female_names_toks, she_token, he_token, mode=AblationMode.SEPARATE)
fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
fig.write_image("results/zero_ablation_names.png")

print("Performing mean ablation on names dataset")
prob_change = perform_mean_ablation(all_heads, model, female_names_toks, she_token, he_token, mode=AblationMode.SEPARATE)
fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
fig.write_image("results/mean_ablation_names.png")

print("Performing interchange ablation on names dataset")
prob_change = perform_interchange_ablation(all_heads, model, female_names_toks, male_names_toks, she_token, he_token, mode=AblationMode.SEPARATE)
fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
fig.write_image("results/interchange_ablation_names.png")

# Professions dataset

print("Performing zero ablation on professions dataset")
prob_change = perform_zero_ablation(all_heads, model, female_stereo_toks, she_token, he_token, mode=AblationMode.SEPARATE)
fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
fig.write_image("results/zero_ablation_stereo.png")

print("Performing mean ablation on professions dataset")
prob_change = perform_mean_ablation(all_heads, model, female_stereo_toks, she_token, he_token, mode=AblationMode.SEPARATE)
fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
fig.write_image("results/mean_ablation_stereo.png")

print("Performing interchange ablation on professions dataset")
prob_change = perform_interchange_ablation(all_heads, model, female_stereo_toks, male_stereo_toks, she_token, he_token, mode=AblationMode.SEPARATE)
fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
fig.write_image("results/interchange_ablation_stereo.png")