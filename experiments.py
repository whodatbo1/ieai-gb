from dataset_util import load_professions_dataset, load_names_dataset
from ablation_util import perform_zero_ablation, perform_mean_ablation, perform_interchange_ablation, AblationMode
from transformer_lens import HookedTransformer
import plotly.express as px
import matplotlib.pyplot as plt
import torch
import argparse

seed = 1234
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, choices=["names", "professions", "phrasing_names", "phrasing_professions", "all"], default="all", help="Which experiment to run")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
torch.set_grad_enabled(False)

xlabel = 'head'
ylabel = 'layer'

all_heads = [(layer, head) for layer in range(12) for head in range(12)]

model_name = 'gpt2-small'
model = HookedTransformer.from_pretrained(model_name, device=device, default_padding_side='left')
she_token = model.tokenizer.encode(' she')[0]
he_token = model.tokenizer.encode(' he')[0]

# Names dataset
if args.experiment in ["names", "all"]:
    female_names_toks, male_names_toks = load_names_dataset(model)
    print("Performing zero ablation on names dataset")
    prob_change = perform_zero_ablation(all_heads, model, female_names_toks, she_token, he_token, mode=AblationMode.SEPARATE)
    fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
    fig.write_image("results/zero_ablation_names.png")

    print("Performing mean ablation on names dataset")
    # mean ablate by taking equally weighted dataset of male and female
    ablation_toks = torch.cat([female_names_toks, male_names_toks], dim=0)
    prob_change = perform_mean_ablation(all_heads, model, female_names_toks, ablation_toks, she_token, he_token, mode=AblationMode.SEPARATE)
    fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
    fig.write_image("results/mean_ablation_names.png")

    print("Performing interchange ablation on names dataset")
    prob_change = perform_interchange_ablation(all_heads, model, female_names_toks, male_names_toks, she_token, he_token, mode=AblationMode.SEPARATE)
    fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
    fig.write_image("results/interchange_ablation_names.png")

# Professions dataset
if args.experiment in ["professions", "all"]:
    female_stereo_toks, male_stereo_toks = load_professions_dataset(model)
    print("Performing zero ablation on professions dataset")
    prob_change = perform_zero_ablation(all_heads, model, female_stereo_toks, she_token, he_token, mode=AblationMode.SEPARATE)
    fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
    fig.write_image("results/zero_ablation_stereo.png")

    print("Performing mean ablation on professions dataset")
    # male dataset is larger, so ensure equal representability
    male_stereo_toks_weighted = male_stereo_toks[torch.randperm(female_stereo_toks.shape[0])]
    ablation_toks = torch.cat([female_stereo_toks, male_stereo_toks_weighted], dim=0)
    prob_change = perform_mean_ablation(all_heads, model, female_stereo_toks, ablation_toks, she_token, he_token, mode=AblationMode.SEPARATE)
    fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
    fig.write_image("results/mean_ablation_stereo.png")

    print("Performing interchange ablation on professions dataset")
    prob_change = perform_interchange_ablation(all_heads, model, female_stereo_toks, male_stereo_toks, she_token, he_token, mode=AblationMode.SEPARATE)
    fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
    fig.write_image("results/interchange_ablation_stereo.png")

# Phrasing datasets
if args.experiment in ["phrasing_names", "all"]:
    rephrases = [lambda name: f"{name} is here, isn't",
                 lambda name: f"{name} told us that",
                 lambda name: f"I met {name} yesterday."]
    for i, rephrase_func in enumerate(rephrases):
        print(f"Loading names dataset with rephrase {i}")
        female_names_toks, male_names_toks = load_names_dataset(model, sentence_structure=rephrase_func)

        print("Performing zero ablation on names dataset on rephrase", i)
        prob_change = perform_zero_ablation(all_heads, model, female_names_toks, she_token, he_token, mode=AblationMode.SEPARATE)
        fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
        fig.write_image(f"results/name_rephrase_{i}_zero_ablation_names.png")

        print("Performing mean ablation on names dataset on rephrase", i)
        # mean ablate by taking equally weighted dataset of male and female
        ablation_toks = torch.cat([female_names_toks, male_names_toks], dim=0)
        prob_change = perform_mean_ablation(all_heads, model, female_names_toks, ablation_toks, she_token, he_token, mode=AblationMode.SEPARATE)
        fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
        fig.write_image(f"results/name_rephrase_{i}_mean_ablation_names.png")

        print("Performing interchange ablation on names dataset on rephrase", i)
        prob_change = perform_interchange_ablation(all_heads, model, female_names_toks, male_names_toks, she_token, he_token, mode=AblationMode.SEPARATE)
        fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
        fig.write_image(f"results/name_rephrase_{i}_interchange_ablation_names.png")

if args.experiment in ["phrasing_professions", "all"]:
    rephrases = [lambda name, pronoun, _: f"The {name} is here, isn't",
                 lambda name, pronoun, _: f"The {name} told us that",
                 lambda name, pronoun, _: f"I met the {name} yesterday and",
                 lambda name, pronoun, _: f"The {name} looked worried as {pronoun} approached us. The {name} said that",
                 lambda name, _, adj: f"The {adj} {name} told us that"]
    for i, rephrase_func in enumerate(rephrases):
        print(f"Loading professions dataset with rephrase {i}") 
        female_stereo_toks, male_stereo_toks = load_professions_dataset(model, sentence_structure=rephrase_func)
        
        print("Performing zero ablation on professions dataset with rephrase", i)
        prob_change = perform_zero_ablation(all_heads, model, female_stereo_toks, she_token, he_token, mode=AblationMode.SEPARATE)
        fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
        fig.write_image(f"results/prof_rephrase_{i}_zero_ablation_stereo.png")

        print("Performing mean ablation on professions dataset with rephrase", i)
        # male dataset is larger, so ensure equal representability
        male_stereo_toks_weighted = male_stereo_toks[torch.randperm(female_stereo_toks.shape[0])]
        ablation_toks = torch.cat([female_stereo_toks, male_stereo_toks_weighted], dim=0)
        prob_change = perform_mean_ablation(all_heads, model, female_stereo_toks, ablation_toks, she_token, he_token, mode=AblationMode.SEPARATE)
        fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
        fig.write_image(f"results/prof_rephrase_{i}_mean_ablation_stereo.png")

        print("Performing interchange ablation on professions dataset with rephrase", i)
        prob_change = perform_interchange_ablation(all_heads, model, female_stereo_toks, male_stereo_toks, she_token, he_token, mode=AblationMode.SEPARATE)
        fig = px.imshow(prob_change.cpu().detach().numpy(), color_continuous_scale='RdBu', zmin=-0.05, zmax=0.05, labels={'x': 'head', 'y': 'layer', 'color':'prob change'}, y=list(range(12)), x=list(range(12)), width=800, height=600, title=f'Prob Change by {xlabel} / {ylabel}')
        fig.write_image(f"results/prof_rephrase_{i}_interchange_ablation_stereo.png")



