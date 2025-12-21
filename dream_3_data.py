import os
import pandas as pd


def split_tsv_into_runs_no_time_csv(
    input_tsv,
    output_dir="runs",
    time_col="Time",
    sep="\t"
):
    """
    Splits a TSV containing multiple independent time series (runs)
    into separate CSV files, one per run.

    - Detects new runs when time resets or decreases
    - Drops the Time column
    - Saves each run as a CSV
    """

    # Load TSV
    df = pd.read_csv(input_tsv, sep=sep)

    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in TSV")

    time = df[time_col].values

    # Detect run boundaries
    run_starts = [0]
    for i in range(1, len(time)):
        if time[i] <= time[i - 1]:
            run_starts.append(i)

    run_starts.append(len(df))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Split and save
    for i in range(len(run_starts) - 1):
        start, end = run_starts[i], run_starts[i + 1]

        # Drop Time column
        run_df = df.iloc[start:end].drop(columns=[time_col]).reset_index(drop=True)

        out_file = os.path.join(output_dir, f"run_{i+1}.csv")
        run_df.to_csv(out_file, index=False)

        print(f"Saved run {i+1}: rows {start}–{end-1} → {out_file}")

    print(f"\n✅ Total runs detected: {len(run_starts) - 1}")

split_tsv_into_runs_no_time_csv(
    input_tsv="/Users/shreeyansarora/Downloads/causal_discovery_on_time_series/dream3/ecoli2.tsv",
    output_dir="/Users/shreeyansarora/Downloads/causal_discovery_on_time_series/dream3/ecoli2",
    time_col="Time"
)
