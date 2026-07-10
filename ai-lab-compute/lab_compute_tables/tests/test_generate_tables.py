"""Smoke tests for the frontier-lab compute table generator.

Verifies the public getters run end-to-end, return pandas DataFrames of the
expected shape, and produce estimates with ordered percentiles in a plausible
range for every tracked lab — and that the intermediates table tells a
consistent story (one final step per lab, matching the year-end table).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_tables import (get_all_tables, LAB_ORDER, LAB_YEAR_KEYS, COLUMNS,
                             INTERMEDIATE_COLUMNS)

TRACKED_LABS = set(LAB_ORDER)
VALID_KINDS = {"input", "constant", "derived", "final"}


@pytest.fixture(scope="module")
def tables():
    """Run the full pipeline once and reuse across tests."""
    tables = get_all_tables()
    assert set(tables.keys()) == {"year_end_by_lab", "intermediates_by_lab"}
    return tables


@pytest.fixture(scope="module")
def table(tables):
    return tables["year_end_by_lab"]


@pytest.fixture(scope="module")
def intermediates(tables):
    return tables["intermediates_by_lab"]


def test_is_nonempty_dataframe_with_expected_columns(table):
    assert isinstance(table, pd.DataFrame)
    assert len(table) > 0
    assert list(table.columns) == COLUMNS


def test_every_lab_has_an_end_2025_row(table):
    labs_2025 = set(table.loc[table["Year"] == 2025, "Lab"])
    assert labs_2025 == TRACKED_LABS


def test_openai_covers_every_disclosed_year_end(table):
    openai_years = set(table.loc[table["Lab"] == "OpenAI", "Year"])
    assert openai_years == {2023, 2024, 2025}


def test_end_2024_backcasts_cover_deepmind_and_meta_but_not_anthropic(table):
    labs_2024 = set(table.loc[table["Year"] == 2024, "Lab"])
    assert labs_2024 == {"OpenAI", "Google DeepMind", "Meta Superintelligence Labs"}


def test_one_row_per_lab_year(table):
    assert not table.duplicated(subset=["Lab", "Year"]).any()


def test_dates_are_year_ends(table):
    assert (table["Date"] == table["Year"].astype(str) + "-12-31").all()


def test_percentiles_are_ordered_and_positive(table):
    assert (table["h100e_p5"] > 0).all()
    assert (table["h100e_p5"] <= table["h100e_med"]).all()
    assert (table["h100e_med"] <= table["h100e_p95"]).all()


def test_medians_are_plausible(table):
    # Order-of-magnitude guards against unit mistakes, loose enough to
    # survive routine prior updates.
    end_2025 = table[table["Year"] == 2025]
    assert end_2025["h100e_med"].between(300_000, 5_000_000).all()

    openai_2023 = table.loc[
        (table["Lab"] == "OpenAI") & (table["Year"] == 2023), "h100e_med"
    ].iloc[0]
    assert 30_000 < openai_2023 < 300_000

    backcasts_2024 = table[(table["Year"] == 2024) & (table["Lab"] != "OpenAI")]
    assert backcasts_2024["h100e_med"].between(100_000, 1_000_000).all()


def test_intermediates_shape_and_columns(intermediates):
    assert isinstance(intermediates, pd.DataFrame)
    assert list(intermediates.columns) == INTERMEDIATE_COLUMNS
    snapshots = set(zip(intermediates["Lab"], intermediates["Year"]))
    assert snapshots == set(LAB_YEAR_KEYS)
    assert set(intermediates["Kind"]) <= VALID_KINDS


def test_intermediates_steps_are_ordered_per_snapshot(intermediates):
    for (lab, year), group in intermediates.groupby(["Lab", "Year"]):
        assert list(group["Step"]) == list(range(1, len(group) + 1))
        assert not group["Variable"].duplicated().any()


def test_intermediates_percentiles_are_ordered(intermediates):
    assert (intermediates["value_p5"] <= intermediates["value_med"]).all()
    assert (intermediates["value_med"] <= intermediates["value_p95"]).all()


def test_intermediates_end_with_one_final_matching_year_end(tables):
    year_end = tables["year_end_by_lab"]
    intermediates = tables["intermediates_by_lab"]
    for (lab, year), group in intermediates.groupby(["Lab", "Year"]):
        finals = group[group["Kind"] == "final"]
        assert len(finals) == 1, f"{lab} {year} should have exactly one final step"
        assert finals["Step"].iloc[0] == len(group), f"{lab} {year} final step must come last"
        headline = year_end.loc[
            (year_end["Lab"] == lab) & (year_end["Year"] == year), "h100e_med"
        ].iloc[0]
        assert finals["value_med"].iloc[0] == pytest.approx(headline)
