import logging
from pathlib import Path

import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
import tarfile
import mmap
from io import BytesIO

_LG = logging.getLogger(__name__)

def read_from_archive(archive: Path | str, offset: int, file_size: int, opened_archive = None) -> BytesIO:
    if opened_archive:
        with mmap.mmap(opened_archive.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_o:
            return BytesIO(mmap_o[offset : offset + file_size])
    with Path(archive).open("rb") as path, mmap.mmap(path.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_o:
        return BytesIO(mmap_o[offset : offset + file_size])

class CompressedArchiveError(Exception):
    """Archive must be uncompressed to be read."""

def _manifest_from_bytes_info(archive: Path | str, bytes_info: dict[str, tuple[int | None, int, int]]) -> pd.DataFrame:
    manifest = pd.DataFrame.from_dict(bytes_info, orient="index", columns=["num_frames", "tensor_len", "byte_offset", "byte_size"])
    manifest["archive"] = archive
    manifest.index.name = "path"
    manifest["fileid"] = manifest.index.map(lambda x: Path(x).stem)
    return manifest.reset_index()[["fileid", "path", "num_frames", "tensor_len", "archive", "byte_offset", "byte_size"]]

def build_manifest_tar(path: Path | str, file_extension: str = ".wav", *, read_frames: bool = True) -> pd.DataFrame:
    if not tarfile.is_tarfile(path):
        raise ValueError(path)
    print("opening archive")
    with tarfile.open(path, mode="r") as tar_file:
        if tar_file.mode != "r":
            raise CompressedArchiveError
        infolist = tar_file.getmembers()
    print("starting iteration members archive")
    bytes_info = {}
    with Path(path).open("rb") as archive_desc:
        for info in tqdm(infolist):
            if info.isdir() or not info.name.endswith(file_extension):
                continue
            data = read_from_archive(path, info.offset_data, info.size, archive_desc)
            if file_extension == ".pt":
                splits = info.name.split(".")[0].split("_")
                num_frames = int(splits[-1]) - int(splits[-2]) if read_frames else None
                tensor_len = torch.load(data).shape[0] if read_frames else None
            else:
                num_frames = torchaudio.info(data).num_frames if read_frames else None
            bytes_info[info.name] = (num_frames, tensor_len, info.offset_data, info.size)
    if not bytes_info:
        raise Exception("No bytes info")
        #raise NoAudioFileError(path, file_extension)
    return _manifest_from_bytes_info(path, bytes_info)