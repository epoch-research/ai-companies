"""Helpers for modeling AI lab compute fleet buildout."""

from pathlib import Path

import numpy as np
import pandas as pd


def convert_it_power_to_chips(
    power_added_mw_by_period,
    chip_specs,
    chip_composition_by_period,
    tolerance=1e-6,
):
    """Run an incremental fleet buildout model.

    In each period, some new IT power (MW) is deployed and allocated across chip
    types by a composition of new units (shares summing to 1). The function
    converts those unit shares into power shares using each chip's IT power,
    then converts deployed power into chip counts and H100-equivalents.
    Cumulative totals carry across periods; chips deployed in an earlier period
    are never retroactively reallocated.

    Args:
        power_added_mw_by_period: mapping of period key → new IT power (MW) in
            that period. Period keys can be any sortable/hashable type (dates,
            quarter labels, ints).
        chip_specs: mapping of chip name → {"IT_power": float, "H100e": float}.
        composition_by_period: mapping of period key → {chip name: share of
            new units added in that period}. Shares must sum to 1 (or to 0 if
            no additions that period). Chips omitted from a period's
            composition are treated as 0.
        tolerance: numerical tolerance for the composition-sum check.

    Returns:
        DataFrame with one row per (period, chip), ordered by period then by
        the order of chips in `chip_specs`.
    """
    periods = sorted(power_added_mw_by_period.keys())
    chips = list(chip_specs.keys())

    for chip, spec in chip_specs.items():
        missing = {'IT_power', 'H100e'} - set(spec.keys())
        if missing:
            raise ValueError(f"chip_specs[{chip!r}] missing keys: {missing}")
        if spec['IT_power'] <= 0:
            raise ValueError(f"chip_specs[{chip!r}].IT_power must be > 0")

    if set(chip_composition_by_period.keys()) != set(periods):
        raise ValueError(
            "composition_by_period keys must exactly match "
            "power_added_mw_by_period keys"
        )

    for period, comp in chip_composition_by_period.items():
        unknown = set(comp.keys()) - set(chips)
        if unknown:
            raise ValueError(
                f"period {period!r}: composition references unknown chips {unknown}"
            )
        total = sum(comp.values())
        sums_to_one = np.isclose(total, 1.0, atol=tolerance)
        sums_to_zero = np.isclose(total, 0.0, atol=tolerance)
        if not (sums_to_one or sums_to_zero):
            raise ValueError(
                f"period {period!r}: composition shares sum to {total:.6f}, "
                "expected 1.0 (or 0.0 if no additions)"
            )

    cumulative_chips = {chip: 0.0 for chip in chips}
    cumulative_h100e = {chip: 0.0 for chip in chips}

    rows = []
    for period in periods:
        new_power_mw = power_added_mw_by_period[period]
        comp = chip_composition_by_period[period]
        weighted_average_it_power = sum(
            comp.get(chip, 0.0) * chip_specs[chip]['IT_power']
            for chip in chips
        )

        for chip in chips:
            share_of_new_units = comp.get(chip, 0.0)
            it_power = chip_specs[chip]['IT_power']
            h100e_per_gpu = chip_specs[chip]['H100e']

            if np.isclose(weighted_average_it_power, 0.0, atol=tolerance):
                share_of_power_added = 0.0
            else:
                # More power-hungry chips get a larger share of the same unit mix.
                share_of_power_added = (
                    share_of_new_units * it_power / weighted_average_it_power
                )

            chip_power_mw = new_power_mw * share_of_power_added
            new_chips = chip_power_mw * 1e6 / it_power
            new_h100e = new_chips * h100e_per_gpu

            cumulative_chips[chip] += new_chips
            cumulative_h100e[chip] += new_h100e

            rows.append({
                'Period': period,
                'Chip Type': chip,
                'Share of Units Added': share_of_new_units,
                'Share of Power Added': share_of_power_added,
                'Power Added (MW)': chip_power_mw,
                'Chips Added': new_chips,
                'H100e Added': new_h100e,
                'Cumulative Chips': cumulative_chips[chip],
                'Cumulative Power (MW)': cumulative_chips[chip] * it_power / 1e6,
                'Cumulative H100e': cumulative_h100e[chip],
            })

    return pd.DataFrame(rows)


PARAMS_CSV = Path(__file__).with_name('lab_model_params.csv')


def load_lab_params(csv_path=None):
    """Load the canonical model priors from lab_model_params.csv.

    The sheet is the single source of truth for the judgment priors shared by
    the lab notebooks and frontier_lab_compute_model.py. Returns
    {lab: {param: squigglepy distribution}}.

    Column meanings: `dist` names the squigglepy constructor (to / norm /
    beta / uniform); `low`/`high` are its two positional arguments — the 90%
    credible interval for `to` and `norm`, the range for `uniform`, and the
    alpha/beta shape parameters for `beta`. Optional `lclip`/`rclip` clip the
    samples. `dist` = "const" returns `low` as a plain float (for scalar
    judgment parameters like mixture weights).

    Besides the per-lab judgment priors, the sheet carries a `chip_specs`
    group of const rows: shared hardware constants (TPU TDPs, the TPU
    IT-power overhead, and the Trainium2 power equivalency) used by the
    Anthropic notebook and the frontier script.

    Every call constructs fresh distribution objects, so a sensitivity cell
    can call again for clean copies (useful before sq.correlate, which ties
    together the objects it is given).
    """
    import squigglepy as sq

    constructors = {'to': sq.to, 'norm': sq.norm, 'beta': sq.beta, 'uniform': sq.uniform}

    params = {}
    for _, row in pd.read_csv(csv_path or PARAMS_CSV).iterrows():
        if row['dist'] == 'const':
            value = float(row['low'])
        elif row['dist'] in constructors:
            kwargs = {}
            for clip in ('lclip', 'rclip'):
                if pd.notna(row[clip]):
                    kwargs[clip] = float(row[clip])
            value = constructors[row['dist']](float(row['low']), float(row['high']), **kwargs)
        else:
            raise ValueError(
                f"{row['lab']}.{row['param']}: unknown dist {row['dist']!r} "
                f"(expected const or one of {sorted(constructors)})"
            )
        params.setdefault(row['lab'], {})[row['param']] = value
    return params


def lab_params_table(lab, csv_path=None):
    """One lab's rows of lab_model_params.csv, for display in a notebook."""
    df = pd.read_csv(csv_path or PARAMS_CSV)
    if lab not in set(df['lab']):
        raise KeyError(f"no rows for lab {lab!r}; labs in sheet: {sorted(set(df['lab']))}")
    return df[df['lab'] == lab].drop(columns=['lab']).reset_index(drop=True)
