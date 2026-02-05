import polars as pl

ANNOTATION_INDEX_SCHEMA = pl.Schema(
    {
        "split": pl.String,
        "uri": pl.String,
        "start_time_s": pl.Float64,
        "duration_s": pl.Float64,
        "label": pl.String,
    }
)
"""The annotation index contains all annotations in a `SegmeFileDataset`.

The name of the file should be `annotation_index.csv`

- `split`: is one of (train, val, test)
- `uri`: is the unique identifier of one audio that links audio files to its annotations. No spaces are allowed in the name.
- `start_time_s`: start time in seconds of the annotated section.
- `duration_s`: diration of the annotated section.
- `label`: Associated string label to the annotated section.
"""

URI_INDEX_SCHEMA = pl.Schema(
    {
        "split": pl.String,
        "uri": pl.String,
        "audio_duration_s": pl.Float64,
        # FIXME - should be sub_dataset
        "dataset": pl.String,
        # FIXME - replace with 'version' instead
        "babytrain_version": pl.String,
    }
)
"""The uri index contains all unique uris in a `SegmeFileDataset`.

The name of the file should be `uri_index.csv`.

- `split`: is one of (train, val, test)
- `uri`: is the unique identifier of one audio that links audio files to its annotations. No spaces are allowed in the name.
- `audio_duration_s`: Duration of the audio file in seconds.
- `dataset`: Dataset which the audio is part of.
- `babytrain_version`: Version identifier for tracking dataset updates.
"""
