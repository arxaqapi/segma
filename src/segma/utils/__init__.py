import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set the seeds accross the python library random genarator, numpy and pytorch for reproductibility.

    Args:
        seed (int): seed value to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
