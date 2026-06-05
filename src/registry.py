"""
TransformationRegistry - stateful object that tracks transformation
assignments for all dataset columns.

Replaces the global-state pattern (global transformations + set_transformation
+ sync_transformations) with a single object that owns its state and exposes
a clean interface.
"""

from __future__ import annotations
import pandas as pd

class TransformationRegistry:
    """
    Tracks transformation assignments for all dataset columns.

    Schema of the underlying DataFrame:
        column | dtype | is_target | n_unique | pct_null | transformation | note
    """

    def __init__(self, df: pd.DataFrame, targets: list[str] | None = None) -> None:
        self._targets = targets or []
        self._df      = self._build(df)

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "column":         df.columns,
            "dtype":          df.dtypes.values,
            "is_target":      df.columns.isin(self._targets),
            "n_unique":       [df[c].nunique() for c in df.columns],
            "pct_null":       [round(100 * df[c].isnull().mean(), 2) for c in df.columns],
            "transformation": None,
            "note":           None,
        })

    def set(self, column: str, transformation: str, note: str) -> None:
        """Assign a transformation and note to a single column."""
        mask = self._df["column"] == column
        self._df.loc[mask, "transformation"] = transformation
        self._df.loc[mask, "note"]           = note

    def set_many(self, columns: list[str], transformation: str, note: str) -> None:
        """Assign the same transformation to multiple columns at once."""
        for col in columns:
            self.set(col, transformation, note)

    def sync(self, df: pd.DataFrame) -> None:
        """
        Reconcile registry with the current DataFrame columns.
        Removes dropped columns, registers new ones with no transformation.
        """
        self._df = self._df[self._df["column"].isin(df.columns)].copy()

        new_cols = [c for c in df.columns if c not in self._df["column"].values]
        if new_cols:
            new_rows = pd.DataFrame({
                "column":         new_cols,
                "dtype":          [df[c].dtype for c in new_cols],
                "is_target":      [c in self._targets for c in new_cols],
                "n_unique":       [df[c].nunique() for c in new_cols],
                "pct_null":       [round(100 * df[c].isnull().mean(), 2) for c in new_cols],
                "transformation": None,
                "note":           None,
            })
            self._df = pd.concat([self._df, new_rows], ignore_index=True)

    def get_columns(self, transformation: str) -> list[str]:
        """Return all columns assigned to a given transformation."""
        return self._df.loc[
            self._df["transformation"] == transformation, "column"
        ].tolist()

    def unassigned(self) -> pd.DataFrame:
        """Return columns with no transformation assigned, excluding targets."""
        return self._df[
            self._df["transformation"].isna() & ~self._df["is_target"]
        ]

    def features_only(self) -> "TransformationRegistry":
        """
        Return a new registry containing only feature columns.
        Excludes targets and columns marked for drop.
        Used to build the ColumnTransformer.
        """
        clone = TransformationRegistry.__new__(TransformationRegistry)
        clone._targets = self._targets
        clone._df = self._df[
            ~self._df["transformation"].isin(["target", "drop", None])
        ].copy()
        return clone

    @property
    def data(self) -> pd.DataFrame:
        """Read-only access to the underlying DataFrame."""
        return self._df.copy()
