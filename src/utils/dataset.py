import json
from pathlib import Path

from transformer_lens import HookedTransformer


def load_profession_words(filepath: Path) -> list[str]:
    with filepath.open("r") as f:
        data = json.load(f)
        return [x[0] for x in data]


def load_name_words(filepath: Path) -> list[str]:
    with filepath.open("r") as f:
        data = json.load(f)
        return data["names"]


def filter_single_token_words(words: list[str], model: HookedTransformer) -> list[str]:
    assert model.tokenizer is not None, 'Model\'s tokenizer cannot be "None"'
    return [word for word in words if len(model.tokenizer.encode(f" {word}")) == 1]


def load_professions_dataset(
    male_professions_filepath: Path,
    female_professions_filepath: Path,
    model: HookedTransformer,
    num_of_entries: int | None = None,
) -> tuple[list[str], list[str]]:
    male = load_profession_words(male_professions_filepath)
    female = load_profession_words(female_professions_filepath)

    filtered_male_entries = filter_single_token_words(male, model)
    filtered_female_entries = filter_single_token_words(female, model)

    if num_of_entries is not None:
        filtered_male_entries = filtered_male_entries[:num_of_entries]
        filtered_female_entries = filtered_female_entries[:num_of_entries]

    return filtered_male_entries, filtered_female_entries


def load_names_dataset(
    male_names_filepath: Path,
    female_names_filepath: Path,
    model: HookedTransformer,
    num_of_entries: int | None = None,
) -> tuple[list[str], list[str]]:
    male = load_name_words(male_names_filepath)
    female = load_name_words(female_names_filepath)

    filtered_male_entries = filter_single_token_words(male, model)
    filtered_female_entries = filter_single_token_words(female, model)

    if num_of_entries is not None:
        filtered_male_entries = filtered_male_entries[:num_of_entries]
        filtered_female_entries = filtered_female_entries[:num_of_entries]

    return filtered_male_entries, filtered_female_entries
