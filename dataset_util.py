import json
from transformer_lens import HookedTransformer
from typing import Tuple, List
from transformer_lens import ActivationCache
import torch

# tokenizing all sentences together ensures they are padded the same way
def tokenize_together(model: HookedTransformer, female_sentences: List[str], male_sentences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    all_sentences = female_sentences + male_sentences
    all_tokens = model.to_tokens(all_sentences)

    num_female = len(female_sentences)
    female_toks = all_tokens[:num_female]
    male_toks = all_tokens[num_female:]

    return female_toks, male_toks

def load_professions_dataset(model: HookedTransformer, 
                             sentence_structure=lambda prof, pronoun, adj: f"The {prof} said that") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, ActivationCache]:
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
    male_stereo_sentences = [sentence_structure(profession, "she", "nice-looking") for profession in male_stereo_professions]

    print(female_stereo_sentences[:3], male_stereo_sentences[:3])
    # convert our sentences into tokens
    female_stereo_toks, male_stereo_toks = tokenize_together(model, female_stereo_sentences, male_stereo_sentences)
    model.cfg.use_attn_result = True
    gpt2_logits, gpt2_cache = model.run_with_cache(female_stereo_toks)

    return female_stereo_toks, male_stereo_toks, gpt2_logits, gpt2_cache

def load_names_dataset(model: HookedTransformer,
                       sentence_structure=lambda name: f"{name} said that", cap=100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, ActivationCache]:
    with open("girl_names_2015.json", 'r') as f:
        content = json.load(f)
        female_stereo_names = [x for x in content['names']]

    with open("boy_names_2015.json", 'r') as f:
        content = json.load(f)
        male_stereo_names = [x for x in content['names']]

    # ensure that they're single tokens
    female_stereo_names_lens = [len(model.tokenizer.encode(f' {x}')) for x in female_stereo_names]
    female_stereo_names = [x for x, l in zip(female_stereo_names, female_stereo_names_lens) if l == 1][:cap]

    male_stereo_names_lens = [len(model.tokenizer.encode(f' {x}')) for x in male_stereo_names]
    male_stereo_names = [x for x, l in zip(male_stereo_names, male_stereo_names_lens) if l == 1][:cap]

    # slot them into our sentences
    female_stereo_sentences = [sentence_structure(name) for name in female_stereo_names]
    male_stereo_sentences = [sentence_structure(name) for name in male_stereo_names]

    print(female_stereo_sentences[:10])
    print(male_stereo_sentences[:10])

    # convert our sentences into tokens
    female_stereo_toks, male_stereo_toks = tokenize_together(model, female_stereo_sentences, male_stereo_sentences)

    model.cfg.use_attn_result = True
    gpt2_logits, gpt2_cache = model.run_with_cache(female_stereo_toks)

    return female_stereo_toks, male_stereo_toks, gpt2_logits, gpt2_cache
