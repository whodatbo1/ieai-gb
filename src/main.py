import argparse
from experiments.sentence_restructure import main


parser = argparse.ArgumentParser(description="Experiments script")

log_levels = ["debug", "info", "warning", "error"]
parser.add_argument(
    "--log-level",
    type=str,
    choices=log_levels,
    default="info",
    help=f"Select the log level. Choices: {log_levels}",
)

experiments = ["simple", "rephrases", "adjectives", "all"]
parser.add_argument(
    "--experiments",
    nargs="+",
    type=str,
    choices=experiments,
    default="all",
    help=f"Select experiments to run. Choices: {experiments}",
)

datasets = ["names", "professions"]
parser.add_argument(
    "--datasets",
    nargs="+",
    type=str,
    choices=datasets,
    default="all",
    help=f"Select datasets to use. Choices: {datasets}",
)

parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility (optional). If not set, seed is None.",
)

parser.add_argument(
    "--num-of-entries",
    type=int,
    default=None,
    help="Number of entries to use per dataset category (optional).",
)
args = parser.parse_args()

main(args)
