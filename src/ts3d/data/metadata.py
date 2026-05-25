from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

SplitName = Literal["train", "val", "test"]


class MetadataRow(BaseModel):
    """A single row of the dataset metadata, one candidate cluster."""

    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    split: SplitName
    label: int = Field(..., ge=0, le=1, description="0 = FP, 1 = TP.")

    bbox3d_center_xyz: tuple[float, float, float]
    bbox3d_size_xyz: tuple[float, float, float]

    @field_validator("bbox3d_center_xyz", "bbox3d_size_xyz", mode="before")
    @classmethod
    def _to_tuple3(cls, v: object) -> tuple[float, float, float]:
        try:
            seq = list(v)  # type: ignore[arg-type]
        except TypeError as e:
            raise ValueError("Expected 3-element sequence (x, y, z).") from e
        if len(seq) != 3:
            raise ValueError("Expected 3-element sequence (x, y, z).")
        return (float(seq[0]), float(seq[1]), float(seq[2]))


class Metadata:
    """In-memory view of the dataset metadata.

    The metadata is the single source of truth for which candidates belong to
    which split and what their labels are. The ``Dataset`` reads it and resolves
    LAZ file paths under ``<root>/<clusters_subdir>/<split>/<candidate_id>.laz``.
    """

    REQUIRED_COLUMNS = (
        "candidate_id",
        "split",
        "label",
        "bbox3d_center_xyz",
        "bbox3d_size_xyz",
    )

    def __init__(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Metadata is missing required columns: {missing}")
        self._df = df.reset_index(drop=True)

    @classmethod
    def from_file(cls, path: str | Path) -> Metadata:
        """Load metadata from a records-oriented JSON array."""
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError(
                f"Unsupported metadata format: {path.suffix} (only .json is supported)"
            )
        df = pd.read_json(path, orient="records")
        return cls(df)

    @classmethod
    def from_rows(cls, rows: list[MetadataRow]) -> Metadata:
        records = [r.model_dump() for r in rows]
        return cls(pd.DataFrame.from_records(records))

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._df.to_json(path, orient="records", indent=2)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    def filter_split(self, split: SplitName) -> Metadata:
        sub = self._df[self._df["split"] == split].reset_index(drop=True)
        return Metadata(sub)

    def iter_rows(self) -> Iterator[MetadataRow]:
        for record in self._df.to_dict(orient="records"):
            yield MetadataRow.model_validate(record)

    def row(self, idx: int) -> MetadataRow:
        return MetadataRow.model_validate(self._df.iloc[idx].to_dict())

    def class_counts(self) -> dict[int, int]:
        vc = self._df["label"].astype(int).value_counts().to_dict()
        return {int(k): int(v) for k, v in vc.items()}
