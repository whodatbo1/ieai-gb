import argparse

import plotly.express as px
import torch
from transformer_lens import HookedTransformer

from utils import (
    DATASETS_PATH,
    LOGGER,
    RESULTS_PATH,
    AblationMode,
    AblationType,
    get_device,
    load_names_dataset,
    load_professions_dataset,
    run_ablation,
    set_logging_level,
    set_seed,
)


def save_ablation_results(
    results: torch.Tensor,
    dataset_name: str,
    ablation_type: AblationType,
    postfix: str = "",
):
    fig = px.imshow(
        results.cpu().numpy(),
        color_continuous_scale="RdBu",
        zmin=-0.05,
        zmax=0.05,
        labels={"x": "head", "y": "layer", "color": "prob change"},
        y=list(range(results.shape[0])),
        x=list(range(results.shape[1])),
        width=800,
        height=600,
        title=f"Prob Change by Head / Layer ({ablation_type}, {dataset_name.capitalize()})",
    )
    filepath = RESULTS_PATH / f"{dataset_name}_{ablation_type}_ablation.png"

    if postfix:
        filepath = filepath.with_name(f"{filepath.stem}_{postfix}{filepath.suffix}")

    fig.write_image(filepath)


def run_and_save_ablation(
    heads: list[tuple[int, int]],
    model: HookedTransformer,
    tokens: torch.Tensor,
    he_token: int,
    she_token: int,
    ablation_type: AblationType,
    mode: AblationMode,
    dataset_name: str,
    ablation_tokens: torch.Tensor | None = None,
    save_results: bool = True,
    result_filename_postfix: str = "",
):
    results = run_ablation(
        heads=heads,
        model=model,
        tokens=tokens,
        ablation_tokens=ablation_tokens,
        he_token=he_token,
        she_token=she_token,
        ablation_type=ablation_type,
        mode=mode,
    )

    if save_results:
        msg = 'Cannot save results when ablation mode is "All"'
        assert isinstance(results, torch.Tensor), msg
        save_ablation_results(
            results=results,
            dataset_name=dataset_name,
            ablation_type=ablation_type,
            postfix=result_filename_postfix,
        )

    return results


def get_dataset(
    dataset: str,
    model: HookedTransformer,
    num_of_entries: int | None = None,
) -> tuple[list[str], list[str]]:
    LOGGER.debug(f'Loading dataset "{dataset}"...')

    if dataset == "professions":
        male_filepath = DATASETS_PATH / "professions_stereo_male.json"
        female_filepath = DATASETS_PATH / "professions_stereo_female.json"
        male_entries, female_entries = load_professions_dataset(
            male_professions_filepath=male_filepath,
            female_professions_filepath=female_filepath,
            model=model,
            num_of_entries=num_of_entries,
        )
    elif dataset == "names":
        male_filepath = DATASETS_PATH / "names_male.json"
        female_filepath = DATASETS_PATH / "names_female.json"
        male_entries, female_entries = load_names_dataset(
            male_names_filepath=male_filepath,
            female_names_filepath=female_filepath,
            model=model,
            num_of_entries=num_of_entries,
        )
    else:
        raise ValueError(f'Unkown dataset "{dataset}"')

    LOGGER.debug(f'Dataset "{dataset}" loaded!')
    LOGGER.debug(f"Number of male entries: {len(male_entries)}")
    LOGGER.debug(f"Number of female entries: {len(female_entries)}")

    return male_entries, female_entries


def tokenize_sentences_together(
    model: HookedTransformer,
    female_sentences: list[str],
    male_sentences: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    all_sentences = female_sentences + male_sentences
    all_tokens = model.to_tokens(all_sentences)

    num_female = len(female_sentences)
    female_tokens = all_tokens[:num_female]
    male_tokens = all_tokens[num_female:]

    return male_tokens, female_tokens


def run_ablations(
    model: HookedTransformer,
    dataset: str,
    sentence_templates: list[str],
    num_of_entries: int | None,
):
    assert model.tokenizer is not None, 'Model\'s "tokenizer" cannot be None'
    she_token = model.tokenizer.encode(" she")[0]
    he_token = model.tokenizer.encode(" he")[0]

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    heads = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]

    LOGGER.info(f'Evaluating using "{dataset.capitalize()}" dataset:')
    male_entries, female_entries = get_dataset(dataset, model, num_of_entries)

    for index, template in enumerate(sentence_templates):
        LOGGER.info(f'Sentence template: "{template}" (Variation {index})')
        male_sentences = [template.replace("<entry>", p) for p in male_entries]
        female_sentences = [template.replace("<entry>", p) for p in female_entries]

        male_tokens, female_tokens = tokenize_sentences_together(
            model=model,
            female_sentences=female_sentences,
            male_sentences=male_sentences,
        )

        run_and_save_ablation(
            heads=heads,
            model=model,
            tokens=female_tokens,
            he_token=he_token,
            she_token=she_token,
            ablation_type=AblationType.ZERO,
            mode=AblationMode.SEPARATE,
            dataset_name=dataset,
            result_filename_postfix=str(index),
        )

        run_and_save_ablation(
            heads=heads,
            model=model,
            tokens=female_tokens,
            ablation_tokens=torch.cat([female_tokens, male_tokens], dim=0),
            he_token=he_token,
            she_token=she_token,
            ablation_type=AblationType.MEAN,
            mode=AblationMode.SEPARATE,
            dataset_name=dataset,
            result_filename_postfix=str(index),
        )

        run_and_save_ablation(
            heads=heads,
            model=model,
            tokens=female_tokens,
            ablation_tokens=male_tokens,
            he_token=he_token,
            she_token=she_token,
            ablation_type=AblationType.INTERCHANGE,
            mode=AblationMode.SEPARATE,
            dataset_name=dataset,
            result_filename_postfix=str(index),
        )


def main(args: argparse.Namespace):
    set_logging_level(args.log_level)

    if args.seed is not None:
        set_seed(args.seed)

    device = get_device()
    model_name = "gpt2-small"
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.cfg.use_attn_result = True

    sentence_templates = ["The <entry> said that"]

    if any(s in args.experiments for s in ["rephrases", "all"]):
        sentence_templates += [
            "Yesterday, the <entry> said that",
            "After the meeting, the <entry> informed them that",
            "They spoke to the <entry> and said that",
        ]

    if any(s in args.experiments for s in ["adjectives", "all"]):
        sentence_templates += [
            "The nurturing <entry> said that",
            "The competent <entry> informed them that",
            "They spoke to the handsome <entry> and said that",
        ]

    LOGGER.debug(f"Sentence templates: {sentence_templates}")

    for dataset in args.datasets:
        run_ablations(model, dataset, sentence_templates, args.num_of_entries)
