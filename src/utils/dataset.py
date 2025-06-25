import json
from pathlib import Path

from transformer_lens import HookedTransformer

from .logger import LOGGER


def load_profession_words(filepath: Path, num_of_entries: int | None) -> list[str]:
    with filepath.open("r") as f:
        data = json.load(f)
        entries = [x[0] for x in data]
        return entries[:num_of_entries] if num_of_entries is not None else entries


def load_name_words(filepath: Path, num_of_entries: int | None) -> list[str]:
    with filepath.open("r") as f:
        data = json.load(f)
        entries = data["names"]
        return entries[:num_of_entries] if num_of_entries is not None else entries


def filter_single_token_words(words: list[str], model: HookedTransformer) -> list[str]:
    assert model.tokenizer is not None, 'Model\'s tokenizer cannot be "None"'
    return [word for word in words if len(model.tokenizer.encode(f" {word}")) == 1]


def load_professions_dataset(
    male_professions_filepath: Path,
    female_professions_filepath: Path,
    model: HookedTransformer,
    num_of_entries: int | None = None,
) -> tuple[list[str], list[str]]:
    LOGGER.info('Loading "Professions" dataset...')
    male = load_profession_words(male_professions_filepath, num_of_entries)
    female = load_profession_words(female_professions_filepath, num_of_entries)

    return (
        filter_single_token_words(male, model),
        filter_single_token_words(female, model),
    )


def load_names_dataset(
    male_names_filepath: Path,
    female_names_filepath: Path,
    model: HookedTransformer,
    num_of_entries: int | None = None,
) -> tuple[list[str], list[str]]:
    LOGGER.info('Loadding "Names" dataset...')
    male = load_name_words(male_names_filepath, num_of_entries)
    female = load_name_words(female_names_filepath, num_of_entries)

    return (
        filter_single_token_words(male, model),
        filter_single_token_words(female, model),
    )
