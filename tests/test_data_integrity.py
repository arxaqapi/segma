from itertools import combinations
from pathlib import Path


def test_baby_train_no_overlap():
    """Ensures that the given dataset has no data leakage between train, test and validation sets."""
    base: Path = Path("data/baby_train")
    if not base.exists():
        return

    # NOTE - open train, test, val.txt and load uris
    all_uris: dict[str, set[str]] = {}
    for sub in ("train", "test", "val"):
        subset_uris: set[str] = set()
        with (base / f"{sub}.txt").open("r") as f:
            for uri in f.readlines():
                subset_uris.add(uri.strip())
        all_uris[sub] = subset_uris

    # NOTE - test that there is no overlap between sets
    for a, b in combinations(all_uris.keys(), 2):
        inter = all_uris[a].intersection(all_uris[b])
        assert len(inter) == 0
