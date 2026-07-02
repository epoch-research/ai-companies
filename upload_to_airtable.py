#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv

load_dotenv()

from epochutils.data import airtable

# Estimate notebooks emit csv_export/ (+ most owners_csv_export/); the nvidia
# owner notebooks emit the nvidia owners breakdowns. Run estimates before owners.
ESTIMATE_NOTEBOOKS = [
    "amd_estimates.ipynb",
    "nvidia_estimates.ipynb",
    "tpu_estimates.ipynb",
    "trainium_estimates.ipynb",
]
OWNER_NOTEBOOKS = ["nvidia_owners.ipynb", "nvidia_owners_other.ipynb"]
GENERATION_NOTEBOOKS = ESTIMATE_NOTEBOOKS + OWNER_NOTEBOOKS

DATE_COLS = ("Start date", "End date")


def generate_data(data_dir):
    """Run the estimate + owner notebooks to (re)generate csv_export/ and owners_csv_export/.

    Notebooks pull from public Google Sheets and run a Monte Carlo, so this needs
    network + a Jupyter kernel. Their metadata pins a missing kernel, so force python3.
    """
    import tempfile
    import papermill as pm
    os.environ.setdefault("MPLBACKEND", "Agg")
    tmp = Path(tempfile.gettempdir())
    for nb in GENERATION_NOTEBOOKS:
        path = data_dir / nb
        if not path.exists():
            print(f"  SKIP {nb} (not found)")
            continue
        print(f"  running {nb} …")
        pm.execute_notebook(str(path), str(tmp / nb.replace(".ipynb", "_out.ipynb")),
                            kernel_name="python3", cwd=str(data_dir), progress_bar=False)


def _iso(df):
    for c in DATE_COLS:
        if c in df.columns:
            p = pd.to_datetime(df[c], errors="raise")
            df[c] = p.dt.strftime("%Y-%m-%d").where(p.notna(), None)
    return df


def _linkify(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: [v] if isinstance(v, str) and v else None)
    return df


def _checkbox(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: True if isinstance(v, str) and v.strip() else None)
    return df


def _chip_label(v):
    """Prod's display label for a chip: no "Instinct " prefix, and H100 -> H100/H200."""
    if not isinstance(v, str):
        return v
    v = v.replace("Instinct ", "")
    return "H100/H200" if v == "H100" else v


def _concat(paths):
    """Read every CSV in `paths` and concatenate them into one DataFrame (empty if none)."""
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def normalize(df, rename, fields):
    # Rename column variants, coalesce duplicate-named columns (first non-null per
    # row, dtype-preserving), then keep only `fields`.
    df = df.rename(columns=rename)
    cols = {}
    for name in dict.fromkeys(df.columns):
        sub = df.loc[:, df.columns == name]
        cols[name] = sub.bfill(axis=1).iloc[:, 0] if sub.shape[1] > 1 else sub.iloc[:, 0]
    df = pd.DataFrame(cols)
    return df[[c for c in fields if c in df.columns]].copy()


def sync_ids(base):
    """Resolve the sync tables' ids by name (they must exist in the base)."""
    by_name = {t.name: t.id for t in base.schema().tables}
    missing = [n for n in ("Organization", "Chip types") if n not in by_name]
    if missing:
        sys.exit(f"Sync tables missing from base: {missing}. Create them first (see docs).")
    return by_name["Organization"], by_name["Chip types"]


def main():
    ap = argparse.ArgumentParser(description="Build the AI Compute Stocks tables in Airtable from the ai-chip-counts CSV exports.")
    ap.add_argument("--data-dir", default=".", help="ai-chip-counts repo root (holds csv_export/ and owners_csv_export/)")
    ap.add_argument("--base-id"); ap.add_argument("--api-key")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("AIRTABLE_API_KEY")
    base_id = args.base_id or os.environ.get("AIRTABLE_BASE_ID")
    if not api_key or not base_id:
        sys.exit("AIRTABLE_API_KEY and AIRTABLE_BASE_ID must be set (env, .env, or flags)")

    dd = Path(args.data_dir)
    print("Regenerating csv_export/ and owners_csv_export/ from notebooks…")
    generate_data(dd)

    base = airtable.connect(api_key, base_id)
    org_id, ct_id = sync_ids(base)
    LINK_ORG = {"type": "multipleRecordLinks", "options": {"linkedTableId": org_id}}
    LINK_CT = {"type": "multipleRecordLinks", "options": {"linkedTableId": ct_id}}
    CHECK = {"type": "checkbox", "options": {"icon": "check", "color": "greenBright"}}

    # Match prod's row-name convention: drop the "Instinct " prefix from AMD chip
    # labels everywhere, and render the Nvidia H100 chip as "H100/H200" in the
    # Nvidia + Consolidated names (but not in "Cumulative … by chip type", where
    # prod keeps "H100"). The "Chip type" link is unaffected — it points at the
    # real dimension records ("H100", "Instinct MI…") in both bases.
    NAME_H100_TABLES = {"Timelines by chip (Nvidia)", "Consolidated timelines"}

    def push(name, df, primary, column_types, link_cols=(), select=None):
        if "Name" in df.columns:
            df["Name"] = df["Name"].astype(str).str.replace("Instinct ", "", regex=False)
            if name in NAME_H100_TABLES:
                df["Name"] = df["Name"].str.replace(r"H100(?!/)", "H100/H200", regex=True)
            if name == "Timelines by chip (AMD)":  # prod uses "FY24Q1", the notebook emits "FY24 Q1"
                df["Name"] = df["Name"].str.replace(r"(FY\d{2}) (Q\d)", r"\1\2", regex=True)
        _iso(df)
        _linkify(df, link_cols)
        print(f"[{name}] {len(df)} rows, {len(df.columns)} cols")
        airtable.sync_dataframe(base, name, df, primary, column_types=column_types)

    # ---- per-designer timelines (raw fields only; lookups/formulas are manual) ----
    TL_COLS = [
        "Name",
        "Chip manufacturer",
        "Start date",
        "End date",
        "Compute estimate in H100e (median)",
        "H100e (5th percentile)",
        "H100e (95th percentile)",
        "Number of Units",
        "Number of Units (5th percentile)",
        "Number of Units (95th percentile)",
        "Source / Link",
        "Notes",
        "Chip type",
    ]
    TL_TYPES = {
        "Chip manufacturer": LINK_ORG,
        "Chip type": LINK_CT,
        "Start date": "date",
        "End date": "date",
        "Notes": "multilineText",
        "Source / Link": "multilineText",
    }
    TIMELINE_SOURCES = [
        ("Timelines by chip (Nvidia)", "nvidia_calendar_quarter_chip_timelines"),
        ("Timelines by chip (TPU)", "tpu_calendar_quarter_chip_timelines"),
        ("Timelines by chip (AMD)", "amd_chip_timelines"),
        ("Timelines by chip (Trainium)", "trainium_chip_timelines"),
    ]
    for tbl, stem in TIMELINE_SOURCES:
        df = pd.read_csv(dd / "csv_export" / f"{stem}.csv")
        df = df[[c for c in TL_COLS if c in df.columns]]
        if "Chip type" in df.columns:  # prod's plain-text display label (H100/H200, MI300A)
            df["Chip type name"] = df["Chip type"].map(_chip_label)
        push(tbl, df, "Name", TL_TYPES, link_cols=("Chip manufacturer", "Chip type"))

    # Source files are inconsistent across designers (trainium uses "Number of
    # Units"/"H100e (Nth percentile)" + extra "Power in MW"). Normalize the known
    # variants, then SELECT only the real table's fields so no stray field is created.
    RENAME_UNITS_LOWER = {
        "Number of Units": "Number of units (median)",
        "Number of Units (5th percentile)": "Number of units (5th percentile)",
        "Number of Units (95th percentile)": "Number of units (95th percentile)",
    }
    RENAME_H100E_POWER = {
        "Compute estimate in H100e (median)": "H100e compute power (median)",
        "Compute estimate in H100e (5th percentile)": "H100e compute power (5th percentile)",
        "Compute estimate in H100e (95th percentile)": "H100e compute power (95th percentile)",
        "H100e (5th percentile)": "H100e compute power (5th percentile)",
        "H100e (95th percentile)": "H100e compute power (95th percentile)",
    }
    RENAME_H100E_ESTIMATE = {
        "H100e (5th percentile)": "Compute estimate in H100e (5th percentile)",
        "H100e (95th percentile)": "Compute estimate in H100e (95th percentile)",
    }

    # ---- Cumulative timelines by chip type ----
    fields = [
        "Name",
        "Start date",
        "End date",
        "Chip manufacturer",
        "Chip type",
        "Number of units (median)",
        "Number of units (5th percentile)",
        "Number of units (95th percentile)",
        "H100e compute power (median)",
        "H100e compute power (5th percentile)",
        "H100e compute power (95th percentile)",
        "Notes",
        "Incomplete",
    ]
    df = normalize(
        _concat(sorted((dd / "csv_export").glob("*_cumulative_by_chip.csv"))),
        {**RENAME_UNITS_LOWER, **RENAME_H100E_POWER},
        fields,
    )
    _checkbox(df, ["Incomplete"])
    push(
        "Cumulative timelines by chip type",
        df,
        "Name",
        {
            "Chip manufacturer": LINK_ORG,
            "Chip type": LINK_CT,
            "Start date": "date",
            "End date": "date",
            "Notes": "singleLineText",
            "Incomplete": CHECK,
        },
        link_cols=("Chip manufacturer", "Chip type"),
    )

    # ---- Cumulative timelines by designer ----
    fields = [
        "Name",
        "Chip manufacturer",
        "Start date",
        "End date",
        "Compute estimate in H100e (median)",
        "Compute estimate in H100e (5th percentile)",
        "Compute estimate in H100e (95th percentile)",
        "Number of units (median)",
        "Number of units (5th percentile)",
        "Number of units (95th percentile)",
        "Power in MW (median)",
        "Power in MW (5th percentile)",
        "Power in MW (95th percentile)",
        "Notes",
        "Incomplete",
    ]
    df = normalize(
        _concat(sorted((dd / "csv_export").glob("*_cumulative_totals.csv"))),
        {**RENAME_UNITS_LOWER, **RENAME_H100E_ESTIMATE},
        fields,
    )
    _checkbox(df, ["Incomplete"])
    push(
        "Cumulative timelines by designer",
        df,
        "Name",
        {
            "Chip manufacturer": LINK_ORG,
            "Start date": "date",
            "End date": "date",
            "Notes": "multilineText",
            "Incomplete": CHECK,
        },
        link_cols=("Chip manufacturer",),
    )

    # ---- Consolidated timelines (all designers' calendar timelines) ----
    stems = [
        "amd_chip_timelines",
        "nvidia_calendar_quarter_chip_timelines",
        "tpu_calendar_quarter_chip_timelines",
        "trainium_chip_timelines",
    ]
    df = _concat([dd / "csv_export" / f"{s}.csv" for s in stems])
    df = df[[c for c in TL_COLS if c in df.columns]]
    df["Chip type (linked)"] = df["Chip type"]
    choices = [{"name": x} for x in sorted(df["Chip type"].dropna().unique())]
    push(
        "Consolidated timelines",
        df,
        "Name",
        {
            "Chip manufacturer": LINK_ORG,
            "Chip type (linked)": LINK_CT,
            "Chip type": {"type": "singleSelect", "options": {"choices": choices}},
            "Start date": "date",
            "End date": "date",
            "Source / Link": "multilineText",
            "Notes": "multilineText",
        },
        link_cols=("Chip manufacturer", "Chip type (linked)"),
    )

    # ---- [Owners] tables (flat — no links in the real base) ----
    OWN_TYPES = {
        "Start date": "date",
        "End date": "date",
        "Notes": "multilineText",
        "Source / Link": "multilineText",
        "Incomplete": CHECK,
    }
    RENAME_OWNER_H100E = {
        "Compute estimate in H100e (5th percentile)": "H100e (5th percentile)",
        "Compute estimate in H100e (95th percentile)": "H100e (95th percentile)",
    }
    RENAME_UNITS_MEDIAN = {
        "Number of Units": "Number of Units (median)",
        "Number of units": "Number of Units (median)",
        "Number of units (median)": "Number of Units (median)",
        "Number of units (5th percentile)": "Number of Units (5th percentile)",
        "Number of units (95th percentile)": "Number of Units (95th percentile)",
    }
    RENAME_UNITS_SINGULAR = {
        "Number of Units (median)": "Number of Units",
        "Number of units (median)": "Number of Units",
        "Number of units": "Number of Units",
        "Number of units (5th percentile)": "Number of Units (5th percentile)",
        "Number of units (95th percentile)": "Number of Units (95th percentile)",
    }
    OWN_HEAD = ["Name", "Chip manufacturer", "Owner"]
    H100E = ["Compute estimate in H100e (median)", "H100e (5th percentile)", "H100e (95th percentile)"]
    UNITS_M = ["Number of Units (median)", "Number of Units (5th percentile)", "Number of Units (95th percentile)"]
    UNITS_S = ["Number of Units", "Number of Units (5th percentile)", "Number of Units (95th percentile)"]
    TDP = ["Total TDP (W)", "Total TDP (W) (5th percentile)", "Total TDP (W) (95th percentile)"]
    POWER = ["Power in MW (median)", "Power in MW (5th percentile)", "Power in MW (95th percentile)"]
    TAIL = ["Source / Link", "Notes", "Incomplete"]

    owner_tables = [
        (
            "[Owners] Cumulative timelines by designer",
            "*_owners_cumulative_totals",
            {**RENAME_UNITS_MEDIAN, **RENAME_OWNER_H100E},
            OWN_HEAD + ["Start date", "End date"] + H100E + UNITS_M + POWER + TAIL,
        ),
        (
            "[Owners] Cumulative timelines by chip type",
            "*_owners_cumulative_by_chip",
            {**RENAME_UNITS_MEDIAN, **RENAME_OWNER_H100E},
            OWN_HEAD + ["Chip type", "Start date", "End date"] + H100E + UNITS_M + TDP + TAIL,
        ),
        (
            "[Owners] Quarters by chip type",
            "*_owners_quarters_by_chip",
            {**RENAME_UNITS_SINGULAR, **RENAME_OWNER_H100E},
            OWN_HEAD + ["Chip type", "Start date", "End date"] + H100E + UNITS_S + TDP + TAIL,
        ),
    ]
    for tbl, glob_stem, rename, fields in owner_tables:
        d = normalize(_concat(sorted((dd / "owners_csv_export").glob(f"{glob_stem}.csv"))), rename, fields)
        _checkbox(d, ["Incomplete"])
        push(tbl, d, "Name", OWN_TYPES)

    # NOTE: Timelines by chip (Huawei) and (Cambricon) are not produced by the
    # ai-chip-counts CSV exports — no source data — so they're not built here.
    print("Done. (Huawei/Cambricon timelines skipped — no source data.)")


if __name__ == "__main__":
    main()
