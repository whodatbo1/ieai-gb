import json
from transformer_lens import HookedTransformer
from typing import Tuple, List, Union
from transformer_lens import ActivationCache
import torch
import pandas as pd

# tokenizing all sentences together ensures they are padded the same way
def tokenize_together(model: HookedTransformer, female_sentences: List[str], male_sentences: List[str], unisex_sentences: Union[List[str], None] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_sentences = female_sentences + male_sentences
    if unisex_sentences is not None:
        all_sentences += unisex_sentences
    all_tokens = model.to_tokens(all_sentences)

    num_female = len(female_sentences)
    num_male_female = num_female + len(male_sentences)
    female_toks = all_tokens[:num_female]
    male_toks = all_tokens[num_female:num_male_female]
    unisex_toks = all_tokens[num_male_female:]

    return female_toks, male_toks, unisex_toks

def load_professions_dataset(model: HookedTransformer, 
                             sentence_structure=lambda prof, pronoun, adj: f"The {prof} said that") -> Tuple[torch.Tensor, torch.Tensor]:
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
    female_stereo_sentences = [sentence_structure(profession, "he", "broad-shouldered") for profession in female_stereo_professions]
    male_stereo_sentences = [sentence_structure(profession, "he", "broad-shouldered") for profession in male_stereo_professions]

    print(female_stereo_sentences[:3], male_stereo_sentences[:3])
    # convert our sentences into tokens
    female_stereo_toks, male_stereo_toks, _ = tokenize_together(model, female_stereo_sentences, male_stereo_sentences)
    model.cfg.use_attn_result = True

    return female_stereo_toks, male_stereo_toks

def load_names_dataset(model: HookedTransformer,
                       sentence_structure=lambda name, pronoun, adj: f"{name} said that", cap=100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with open("girl_names_2015.json", 'r') as f:
        content = json.load(f)
        female_stereo_names = [x for x in content['names']]

    with open("boy_names_2015.json", 'r') as f:
        content = json.load(f)
        male_stereo_names = [x for x in content['names']]

    df = pd.read_csv('unisex_names_table.csv')
    filtered_df = df[df['gap'] < 0.05]
    unisex_names = filtered_df['name'].tolist()

    # ensure that they're single tokens
    female_stereo_names_lens = [len(model.tokenizer.encode(f' {x}')) for x in female_stereo_names]
    female_stereo_names = [x for x, l in zip(female_stereo_names, female_stereo_names_lens) if l == 1][:cap]

    male_stereo_names_lens = [len(model.tokenizer.encode(f' {x}')) for x in male_stereo_names]
    male_stereo_names = [x for x, l in zip(male_stereo_names, male_stereo_names_lens) if l == 1][:cap]

    unisex_names_lens = [len(model.tokenizer.encode(f' {x}')) for x in unisex_names]
    unisex_names = [x for x, l in zip(unisex_names, unisex_names_lens) if l == 1][:cap]
    print("UNI NAMES: ", len(unisex_names))

    # slot them into our sentences
    female_stereo_sentences = [sentence_structure(name, "he", "broad-shouldered") for name in female_stereo_names]
    male_stereo_sentences = [sentence_structure(name, "he", "broad-shouldered") for name in male_stereo_names]
    unisex_sentences = [sentence_structure(name, "he", "broad-shouldered") for name in unisex_names]

    print(female_stereo_sentences[:10])
    print(male_stereo_sentences[:10])
    print(unisex_sentences[:10])

    # convert our sentences into tokens
    female_stereo_toks, male_stereo_toks, unisex_toks = tokenize_together(model, female_stereo_sentences, male_stereo_sentences, unisex_sentences)

    model.cfg.use_attn_result = True

    return female_stereo_toks, male_stereo_toks, unisex_toks
