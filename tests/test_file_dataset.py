from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from scripts.generate_data import gen_classification
from segma.data import SegmaFileDataset, URISubsetLeakageError


@pytest.fixture
def _prepare_dummy_ds():
    gen_classification(Path("tests/sample/debug_5"), per_split=5)
    gen_classification(Path("tests/sample/debug_10"), per_split=10)
    yield
    import shutil

    shutil.rmtree("tests/sample/debug_5", ignore_errors=True)
    shutil.rmtree("tests/sample/debug_10", ignore_errors=True)

    SegmaFileDataset.clean_cache("tests/sample/debug_5")
    SegmaFileDataset.clean_cache("tests/sample/debug_10")


def test_SegmaFileDataset_init(_prepare_dummy_ds):
    sfd = SegmaFileDataset(
        "tests/sample/debug_10",
        classes=["male", "female", "key_child", "other_child"],
        chunk_duration_s=1.0,
    )

    for uris in sfd.subset_to_uris.values():
        assert len(uris) > 0


def test_SegmaFileDataset_load(_prepare_dummy_ds):
    sfd = SegmaFileDataset(
        "tests/sample/debug_10",
        classes=["male", "female", "key_child", "other_child"],
        chunk_duration_s=1.0,
    )

    sfd.load()
    assert sfd.subds_to_durations is not None
    assert sfd.subds_to_interlaps is not None
    assert (
        len(sfd.subds_to_durations["train"])
        == len(sfd.subds_to_interlaps["train"])
        == len(sfd.subset_to_uris["train"])
        == 10
    )


def test_SegmaFileDataset_init_w_exclude(_prepare_dummy_ds):
    with open("tests/sample/debug_5/exclude.txt", "w") as f:
        f.writelines([s + "\n" for s in ("0000", "0005", "0010")])

    sfd = SegmaFileDataset(
        "tests/sample/debug_5",
        classes=["male", "female", "key_child", "other_child"],
        chunk_duration_s=1.0,
    )

    for uris in sfd.subset_to_uris.values():
        assert len(uris) == 4

    assert len(sfd.removed_uris["exclude.txt"]) == 3

    sfd.load()
    assert sfd.subds_to_durations is not None
    assert sfd.subds_to_interlaps is not None
    assert (
        len(sfd.subds_to_durations["train"])
        == len(sfd.subds_to_interlaps["train"])
        == len(sfd.subset_to_uris["train"])
        == 4
    )


def test_data_leakage_detection(_prepare_dummy_ds):
    # NOTE - add contamination to dataset
    with open("tests/sample/debug_10/test.txt", "a") as f:
        f.writelines([s + "\n" for s in ("0000", "0005", "0010")])

    with pytest.raises(URISubsetLeakageError):
        SegmaFileDataset(
            "tests/sample/debug_10",
            classes=["male", "female", "key_child", "other_child"],
            chunk_duration_s=1.0,
        )


def test_SegmaFileDataset_init_w_exclude_invalid(_prepare_dummy_ds):
    # FIXME error whith exlude.txt ???
    ds_to_use = "debug_10"
    for subds, uri in zip(("train", "val", "test"), ("1000", "1001", "1002")):
        with open(f"tests/sample/{ds_to_use}/{subds}.txt", "a") as f:
            f.writelines([s + "\n" for s in (uri,)])
            Path(f"tests/sample/{ds_to_use}/aa/{uri}.aa").touch()

    # NOTE - gen audio >= 2 seconds
    sr = 16_000
    audio = np.zeros((1, 3 * 60 * sr), dtype=np.float32)
    wavfile.write(f"tests/sample/{ds_to_use}/wav/1000.wav", sr, audio.T)
    wavfile.write(f"tests/sample/{ds_to_use}/wav/1001.wav", sr, audio.T)
    wavfile.write(f"tests/sample/{ds_to_use}/wav/1002.wav", sr, audio.T)

    sfd = SegmaFileDataset(
        f"tests/sample/{ds_to_use}",
        classes=["male", "female", "key_child", "other_child"],
        chunk_duration_s=120,
    )

    sfd.load(False)

    assert len(sfd.removed_uris["invalid"]) == 30
    assert len(sfd.removed_uris["invalid"]) == 30

    for uris in sfd.subset_to_uris.values():
        assert len(uris) == 1

    assert sfd.subds_to_durations is not None
    assert sfd.subds_to_interlaps is not None
    assert (
        len(sfd.subds_to_durations["train"])
        == len(sfd.subds_to_interlaps["train"])
        == len(sfd.subset_to_uris["train"])
        == 1
    )


def test_non_existant_ds():
    ds_path = Path("tests/sample/missing_dataset")
    with pytest.raises(
        FileNotFoundError, match=f"Given path to the dataset is non existent. Got `{ds_path}`."
    ):
        SegmaFileDataset(
            ds_path,
            classes=[],
            chunk_duration_s=1,
        )


def test_SegmaFileDataset_save_cache(_prepare_dummy_ds):
    sfd = SegmaFileDataset(
        "tests/sample/debug_10",
        classes=["male", "female", "key_child", "other_child"],
        chunk_duration_s=1.0,
    )

    sfd.load()
    
    assert (Path(".cache/segma") / sfd.base_p / "subds_to_durations").exists()
    assert (Path(".cache/segma") / sfd.base_p / "subds_to_interlaps").exists()


def test_SegmaFileDataset_load_cache(_prepare_dummy_ds):
    SegmaFileDataset(
        "tests/sample/debug_10",
        classes=["male", "female", "key_child", "other_child"],
        chunk_duration_s=1.0,
    ).load()

    sfd = SegmaFileDataset(
        "tests/sample/debug_10",
        classes=["male", "female", "key_child", "other_child"],
        chunk_duration_s=1.0,
    )
    sfd.load_cache()
    assert sfd.subds_to_durations is not None
    assert sfd.subds_to_interlaps is not None
