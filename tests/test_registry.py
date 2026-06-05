import pandas as pd
import pytest

from src.registry import TransformationRegistry


# Fixtures

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "col_a":  [1, 2, 3],
        "col_b":  ["x", "y", "z"],
        "col_c":  [1.0, 2.0, None],
        "target": [100, 200, 300],
    })


@pytest.fixture
def registry(sample_df) -> TransformationRegistry:
    return TransformationRegistry(sample_df, targets=["target"])


# set and set_many

class TestSet:

    def test_set_assigns_transformation(self, registry):
        registry.set("col_a", "standard_scaler", "note")
        row = registry.data[registry.data["column"] == "col_a"]
        assert row["transformation"].iloc[0] == "standard_scaler"

    def test_set_overwrites_existing(self, registry):
        registry.set("col_a", "standard_scaler", "first")
        registry.set("col_a", "robust_scaler",   "second")
        row = registry.data[registry.data["column"] == "col_a"]
        assert row["transformation"].iloc[0] == "robust_scaler"

    def test_set_many_assigns_all(self, registry):
        registry.set_many(["col_a", "col_b"], "drop", "dropping")
        for col in ["col_a", "col_b"]:
            row = registry.data[registry.data["column"] == col]
            assert row["transformation"].iloc[0] == "drop"


# get_columns

class TestGetColumns:

    def test_returns_assigned_columns(self, registry):
        registry.set("col_a", "standard_scaler", "note")
        registry.set("col_c", "standard_scaler", "note")
        assert set(registry.get_columns("standard_scaler")) == {"col_a", "col_c"}

    def test_returns_empty_for_unassigned_transformation(self, registry):
        assert registry.get_columns("robust_scaler") == []

    def test_does_not_return_other_transformations(self, registry):
        registry.set("col_a", "standard_scaler", "note")
        registry.set("col_b", "drop",            "note")
        assert "col_b" not in registry.get_columns("standard_scaler")


# unassigned

class TestUnassigned:

    def test_excludes_targets(self, registry):
        assert "target" not in registry.unassigned()["column"].values

    def test_excludes_assigned_columns(self, registry):
        registry.set("col_a", "standard_scaler", "note")
        assert "col_a" not in registry.unassigned()["column"].values


# sync

class TestSync:

    def test_adds_new_columns_with_no_transformation(self, registry):
        new_df = pd.DataFrame({
            "col_a": [1], "col_b": ["x"], "col_c": [1.0],
            "target": [100], "col_new": [99],
        })
        registry.sync(new_df)
        assert "col_new" in registry.data["column"].values
        new_row = registry.data[registry.data["column"] == "col_new"]
        assert pd.isna(new_row["transformation"].iloc[0])

    def test_preserves_existing_assignments(self, registry):
        registry.set("col_a", "standard_scaler", "note")
        new_df = pd.DataFrame({
            "col_a": [1], "col_b": ["x"], "col_c": [1.0], "target": [100],
        })
        registry.sync(new_df)
        row = registry.data[registry.data["column"] == "col_a"]
        assert row["transformation"].iloc[0] == "standard_scaler"

    def test_removes_dropped_columns(self, registry):
        new_df = pd.DataFrame({"col_a": [1], "target": [100]})
        registry.sync(new_df)
        assert "col_b" not in registry.data["column"].values


# features_only

class TestFeaturesOnly:

    def test_excludes_targets(self, registry):
        registry.set("col_a", "standard_scaler", "note")
        assert "target" not in registry.features_only().data["column"].values

    def test_excludes_drop_columns(self, registry):
        registry.set("col_a", "standard_scaler", "note")
        registry.set("col_b", "drop",            "dropping")
        assert "col_b" not in registry.features_only().data["column"].values

    def test_excludes_unassigned_columns(self, registry):
        registry.set("col_a", "standard_scaler", "note")
        assert "col_c" not in registry.features_only().data["column"].values

    def test_does_not_mutate_original(self, registry):
        registry.set("col_a", "standard_scaler", "note")
        original_len = len(registry.data)
        registry.features_only()
        assert len(registry.data) == original_len
