from pathlib import Path
from typing import Literal

import polars as pl

from segma.data.stats.schema import ANNOTATION_INDEX_SCHEMA, URI_INDEX_SCHEMA
from segma.utils.conversions import s_to_hms


def load_annotation_index(csv_path: Path) -> pl.DataFrame:
    return pl.read_csv(csv_path, schema_overrides=ANNOTATION_INDEX_SCHEMA)


def load_uri_index(csv_path: Path) -> pl.DataFrame:
    return pl.read_csv(csv_path, schema_overrides=URI_INDEX_SCHEMA)


def duration_per_col_per_split(
    uri_index: pl.DataFrame,
    annotation_index: pl.DataFrame,
    target_col: Literal["split", "dataset"] = "dataset",
    labels: list[str] | None = None,
):
    index = annotation_index.join(
        uri_index, on=("uri", "split"), how="left", validate="m:1"
    )

    total = (
        index.group_by((target_col, "label"))
        .agg(pl.col("duration_s").sum())
        .sort("label")
    )

    if labels is None:
        labels = total["label"].unique().to_list()
    # total col (sum of durations over labels)
    df_dur_s = (
        total.pivot(on="label", index=target_col, values="duration_s")
        .select([target_col] + labels)
        .with_columns(pl.sum_horizontal(labels).alias("TOTAL"))
    )
    # total row (sum of durations over datasets)
    tot_row = df_dur_s.select(
        pl.lit("TOTAL").alias(target_col),
        *[pl.col(c).sum() for c in labels + ["TOTAL"]],
    )

    final = (
        pl.concat([df_dur_s.sort(target_col), tot_row])
        .with_columns(
            pl.exclude(target_col).map_elements(
                lambda e: s_to_hms(e, rjust=4), return_dtype=pl.String
            )
        )
        .select([target_col] + labels + ["TOTAL"])
    )
    return final


def total_duration_per_label(annotation_index: pl.DataFrame) -> pl.DataFrame:
    """
    ```
    # ┌───────┬─────────────┬────────────────┬────────────┐
    # │ label ┆ duration_s  ┆ duration_hms   ┆ duration_% │
    # │ ---   ┆ ---         ┆ ---            ┆ ---        │
    # │ str   ┆ f64         ┆ f64            ┆ f64        │
    # ╞═══════╪═════════════╪════════════════╪════════════╡
    # │ ...   ┆ 943238.814  ┆ 262h 0m 38s    ┆ 59.0       │
    # └───────┴─────────────┴────────────────┴────────────┘
    ```
    """
    return (
        annotation_index.group_by("label")
        .agg(pl.col("duration_s").sum())
        .with_columns(
            pl.col("duration_s")
            .map_elements(lambda e: s_to_hms(e, rjust=4), return_dtype=pl.String)
            .alias("duration_hms"),
            (pl.col("duration_s") / pl.col("duration_s").sum() * 100)
            .round(2)
            .alias("duration_%"),
        )
        .sort("label")
    )
