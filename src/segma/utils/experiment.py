import random
import time
from datetime import datetime
from pathlib import Path


def _get_random_word(word_list_p: Path | str) -> str:
    words = Path(word_list_p).read_text().replace(" ", "-").split("\n")
    return words[random.randint(0, len(words) - 1)]


def new_experiment_id(wait: bool = False) -> str:
    if wait:
        time.sleep(0.2)
    exp_id = datetime.now().strftime("%y%m%d-%H%M%S-")
    return (
        exp_id
        + _get_random_word("scripts/extra/adj.txt")
        + "-"
        + _get_random_word("scripts/extra/names.txt")
    )
