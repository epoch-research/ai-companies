"""Shared utilities for chip volume estimation (NVIDIA and TPU)."""

import numpy as np
import pandas as pd


# ===============================
# Core quarterly simulation logic
# ===============================
def estimate_chip_sales(quarters, versions, sample_revenue, sample_shares, sample_price, n_samples=5000):
    """
    Run Monte Carlo simulation to estimate chip volumes.

    Args:
        quarters: list of quarter identifiers (e.g., ['Q1_FY23', 'Q2_FY23', ...])
        versions: list of chip types (e.g., ['v3', 'v4', 'v5e', ...])
        sample_revenue: fn(quarter) -> float, samples or looks up total chip revenue in dollars for a quarter
        sample_shares: fn(quarter) -> dict, samples {version: share} for a quarter (should sum to 1)
        sample_price: fn(quarter, version) -> float, samples or looks up price for a chip type in a quarter
        n_samples: number of Monte Carlo samples

    Returns:
        Dictionary of {quarter: {version: [array of samples of chip unit counts]}}
        To find median, confidence intervals, etc you will need to take the percentiles of the result

    Note on cross-quarter correlations:
        The sampling functions are called independently for each quarter within each iteration.
        This means any parameters you want correlated across quarters (e.g., a single margin
        value affecting all quarters) will NOT be correlated by default. To preserve cross-quarter
        correlations, pre-sample those parameters outside this function and have your sampling
        functions reference them.

        
    """
    results = {quarter: {version: [] for version in versions} for quarter in quarters}

    for _ in range(n_samples):
        for quarter in quarters:
            revenue = sample_revenue(quarter)
            shares = sample_shares(quarter)

            for version in versions:
                share = shares.get(version, 0)
                if share > 0:
                    price = sample_price(quarter, version)
                    chips = (revenue * share) / price
                else:
                    chips = 0
                results[quarter][version].append(chips)

    return results


def normalize_shares(raw_shares):
    """Normalize share values to sum to 1."""
    total = sum(raw_shares.values())
    return {k: v / total for k, v in raw_shares.items()}


def get_percentiles(samples, percentiles=[5, 50, 95]):
    """Get percentile values from samples array."""
    return {p: np.percentile(samples, p) for p in percentiles}


def compute_h100_equivalents(chip_counts, chip_specs, h100_tops=1979):
    """
    Convert chip counts to H100 equivalents based on 8-bit TOPS.

    Args:
        chip_counts: dict of {version: count} or {version: array of samples}
        chip_specs: dict with 'tops' key for each version
        h100_tops: H100 8-bit TOPS (default 1979)

    Returns:
        dict of {version: h100_equivalent_count}
    """
    return {
        version: counts * (chip_specs[version]['tops'] / h100_tops)
        for version, counts in chip_counts.items()
    }


def samples_to_percentile_dict(samples, percentiles=[5, 50, 95]):
    """Convert samples array to dict with percentile keys."""
    return {f'p{p}': int(np.percentile(samples, p)) for p in percentiles}


def export_quarterly_by_version(results, chip_specs, output_path, n_samples, h100_tops=1979):
    """
    Export quarterly chip volumes by version to CSV.

    Args:
        results: dict of {quarter: {version: list of samples}}
        chip_specs: dict with specs for each version
        output_path: path for CSV output
        n_samples: number of samples per distribution
        h100_tops: H100 8-bit TOPS for equivalence calculation

    Returns:
        DataFrame with exported data
    """
    rows = []
    for quarter in results:
        for version in chip_specs:
            arr = np.array(results[quarter][version])
            if arr.sum() > 0:
                h100e_arr = arr * (chip_specs[version]['tops'] / h100_tops)
                rows.append({
                    'quarter': quarter,
                    'version': version,
                    'chips_p5': int(np.percentile(arr, 5)),
                    'chips_p50': int(np.percentile(arr, 50)),
                    'chips_p95': int(np.percentile(arr, 95)),
                    'h100e_p5': int(np.percentile(h100e_arr, 5)),
                    'h100e_p50': int(np.percentile(h100e_arr, 50)),
                    'h100e_p95': int(np.percentile(h100e_arr, 95)),
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def print_cumulative_summary(cumulative, chip_specs, title="Cumulative Production"):
    """Print formatted summary of cumulative chip counts with percentiles."""
    print(f"\n{title}")
    print(f"{'Version':<6} {'p5':>12} {'p50':>12} {'p95':>12}")
    print("-" * 45)

    grand_total = None
    for version in chip_specs:
        arr = cumulative[version]
        if arr.sum() > 0:
            if grand_total is None:
                grand_total = np.zeros_like(arr)
            grand_total += arr
            print(f"{version:<6} {int(np.percentile(arr, 5)):>12,} {int(np.percentile(arr, 50)):>12,} {int(np.percentile(arr, 95)):>12,}")

    if grand_total is not None:
        print("-" * 45)
        print(f"{'TOTAL':<6} {int(np.percentile(grand_total, 5)):>12,} {int(np.percentile(grand_total, 50)):>12,} {int(np.percentile(grand_total, 95)):>12,}")


def format_thousands(n):
    """Format number as Xk (rounded to nearest thousand)."""
    return f"{round(n / 1000)}k"


def summarize_quarterly_by_chip(results, format_fn=format_thousands):
    """
    Create summary DataFrame with each chip type as separate columns with 90% CI.

    Args:
        results: dict of {quarter: {chip_type: list of samples}}
                 Output from estimate_chip_sales()
        format_fn: function to format numbers (default: format_thousands)

    Returns:
        DataFrame with Quarter, one column per chip type, and Total column.
        Each cell shows "median (p5 - p95)".
    """
    # Infer quarters and chip types from results
    quarters = list(results.keys())
    chip_types = list(results[quarters[0]].keys())
    n_samples = len(results[quarters[0]][chip_types[0]])

    rows = []
    for quarter in quarters:
        row = {'Quarter': quarter}
        total = np.zeros(n_samples)

        for chip_type in chip_types:
            arr = np.array(results[quarter][chip_type])
            total += arr
            if arr.sum() > 0:
                p5 = format_fn(np.percentile(arr, 5))
                p50 = format_fn(np.percentile(arr, 50))
                p95 = format_fn(np.percentile(arr, 95))
                row[chip_type] = f"{p50} ({p5}-{p95})"
            else:
                row[chip_type] = "-"

        # Add total column
        p5 = format_fn(np.percentile(total, 5))
        p50 = format_fn(np.percentile(total, 50))
        p95 = format_fn(np.percentile(total, 95))
        row['Total'] = f"{p50} ({p5}-{p95})"

        rows.append(row)

    # Order columns: Quarter, chip types (in order), Total
    cols = ['Quarter'] + chip_types + ['Total']
    return pd.DataFrame(rows)[cols]
