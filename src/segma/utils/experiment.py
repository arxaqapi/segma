import random
from datetime import datetime
from pathlib import Path

WORD_LIST = ("scripts/extra/names.txt", 3198)


def _get_random_word(word_list_p: Path | str, n_words: int) -> str:
    with Path(word_list_p).open("r") as f:
        return f.readlines()[random.randint(0, n_words)].strip()


def new_experiment_id():
    e_id = datetime.now().strftime("%y%m%d%H%M%S-")
    return e_id + _get_random_word(*WORD_LIST)
