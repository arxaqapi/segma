"""translation layer.
Loads a database.yml file in pyannote.database format and creates according symbolic links
to all relevant files for segma's data format.

data/
    annotations/
        uri.aa
    rttm/
        uri.rttm
    wav/
        uri.wav
    [test|train|val].txt
"""

import warnings
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import yaml

from segma.annotation import AudioAnnotation


class SplitDataset:
    def __init__(self, uri_path: Path | str, annotations_base: str) -> None:
        # load uris, get annotations path from uris
        self.uri_path: Path = uri_path if isinstance(uri_path, Path) else Path(uri_path)
        self.uris = self._get_uris()
        self.annotations: list[Path] = self._get_annotations(annotations_base)

    def _get_uris(self):
        with self.uri_path.open("r") as f:
            uris = [line.strip() for line in f.readlines()]
        return uris

    def _get_annotations(self, base: str):
        annots = [Path(base.replace("{uri}", uri)) for uri in self.uris]
        for rttm_file in annots:
            assert rttm_file.exists()
        return annots


@dataclass
class Dataset:
    name: str
    train: SplitDataset
    development: SplitDataset
    test: SplitDataset


def load_db(db_path: Path = Path("data/database.yml")):
    assert db_path.exists()

    with db_path.open("r") as f:
        content = yaml.safe_load(f)

    return content


def parse_db(complete_database: dict):
    # NOTE - uris
    all_datasets: list[Dataset] = []
    for protocol in complete_database["Protocols"].keys() - "X":
        database = complete_database["Protocols"][protocol]["SpeakerDiarization"]
        # dataset_name = list(db.keys())
        for dataset_name, ds in database.items():
            dataset = Dataset(
                name=dataset_name,
                train=SplitDataset(
                    uri_path=ds["train"]["uri"],
                    annotations_base=ds["train"]["annotation"],
                ),
                development=SplitDataset(
                    uri_path=ds["development"]["uri"],
                    annotations_base=ds["development"]["annotation"],
                ),
                test=SplitDataset(
                    uri_path=ds["test"]["uri"],
                    annotations_base=ds["test"]["annotation"],
                ),
            )
            all_datasets.append(dataset)
    # TODO - get map from uri to ds name
    from_uri_to_ds: dict[str, str] = {}
    for ds in all_datasets:
        for uri in ds.train.uris + ds.development.uris + ds.test.uris:
            if uri in from_uri_to_ds:
                # FIXME - for the moment we overwrite
                warnings.warn(
                    f"duplicate uri in db: '{uri}' of ds: '{ds.name}' already exists in {from_uri_to_ds[uri]}"
                )
            from_uri_to_ds[uri] = ds.name

    # TODO - add optionnal field in uris for the dataset name. (cross-ref with AudioAnnotation class)
    ds_to_wav_path: dict[str, list[Path]] = defaultdict(list)
    for protocol_name, wav_path in complete_database["Databases"].items():
        wav_per_db = glob(wav_path.replace("{uri}", "*"))
        for wav_f in wav_per_db:
            uri = wav_f.split("/")[-1].removesuffix(".wav")
            # NOTE - only keep WAV with URIs in uri lists.
            if uri in from_uri_to_ds:
                ds = from_uri_to_ds[uri]
                # print(ds, wav_f)
                if not Path(wav_f).exists():
                    raise ValueError(
                        f"wav file '{wav_f}' of dataset '{ds}' does not exists"
                    )
                ds_to_wav_path[ds].append(Path(wav_f))

    return all_datasets, ds_to_wav_path


def symlink_all(
    all_datasets: list[Dataset],
    ds_to_wav: dict[str, list[Path]],
    output_base: Path = Path("data/debug/"),
):
    # create the following folder structure:
    # data/
    #     annotations/uri.aa
    #     rttm/uri.rttm
    #     wav/uri.wav
    #     [train|test|val].txt
    # TODO - specify uris with ds name (ds_name.uri) -- can be handled at the creation of the Dataset isntances by injecting ds_name to the uris
    # NOTE - uri lists
    output_base.mkdir(parents=True, exist_ok=True)

    all_train_uris = [uri for ds in all_datasets for uri in ds.train.uris]
    all_test_uris = [uri for ds in all_datasets for uri in ds.test.uris]
    all_dev_uris = [uri for ds in all_datasets for uri in ds.development.uris]
    for output, split_uris in zip(
        ("train", "test", "val"), (all_train_uris, all_test_uris, all_dev_uris)
    ):
        with (output_base / output).with_suffix(".txt").open("w") as f:
            for uri in split_uris:
                f.write(uri + "\n")
    # print(all_train_uris)

    # NOTE - wav symlinks
    # TODO - fix uri not unique by adding ds name component
    wav_out = output_base / "wav"
    wav_out.mkdir(parents=True, exist_ok=True)
    for _ds, wavs in ds_to_wav.items():
        for wav_p in wavs:
            target_symlink = wav_out / wav_p.name
            if not target_symlink.exists():
                target_symlink.symlink_to(wav_p)

    # NOTE - rttm copy and aa creation
    rttm_out = output_base / "rttm"
    aa_out = output_base / "aa"
    rttm_out.mkdir(parents=True, exist_ok=True)
    aa_out.mkdir(parents=True, exist_ok=True)
    all_train_rttms = [rttm_f for ds in all_datasets for rttm_f in ds.train.annotations]
    all_test_rttms = [rttm_f for ds in all_datasets for rttm_f in ds.test.annotations]
    all_dev_rttms = [
        rttm_f for ds in all_datasets for rttm_f in ds.development.annotations
    ]
    for rttm in all_train_rttms + all_test_rttms + all_dev_rttms:
        # RTTM copy
        dest_rttm = rttm_out / rttm.name
        content = rttm.read_text()
        dest_rttm.write_text(content)
        # AA creation
        dest_aa = (aa_out / rttm.stem).with_suffix(".aa")
        annotations = [AudioAnnotation.from_rttm(line) for line in content.splitlines()]
        with dest_aa.open("w") as aaf:
            for annot in annotations:
                aaf.write(annot.write() + "\n")


if __name__ == "__main__":
    symlink_all(
        *parse_db(load_db(Path("database.yml"))),
        output_base=Path("data/baby_train"),
    )
    print("[log] - all finished :)")
