import numpy as np
import pandas as pd
from scipy.io import savemat
from pathlib import Path


def load_static_net_from_file(net_file):
    """
    Load GT net assuming ONLY lag=1 exists.
    Format: src,target,lag (0-indexed)
    Returns: net (N, N)
    """
    max_node = 0
    edges = []

    with open(net_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            src, tgt, lag = map(int, line.strip().split(","))
            if lag != 1:
                continue
            edges.append((src, tgt))
            max_node = max(max_node, src, tgt)

    N = max_node + 1
    net = np.zeros((N, N), dtype=int)

    for src, tgt in edges:
        net[src, tgt] = 1

    return net


def create_multirun_mat_S_T_N(
    runs_dir,
    net_file,
    output_mat_path
):
    """
    Create a .mat file with:
        ts: (S, T, N)
        net: (N, N)
    """

    runs_dir = Path(runs_dir)
    csv_files = sorted(runs_dir.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {runs_dir}")

    # Load runs
    runs = [pd.read_csv(f).values for f in csv_files]

    # Sanity checks
    T, N = runs[0].shape
    S = len(runs)

    for i, r in enumerate(runs):
        if r.shape != (T, N):
            raise ValueError(
                f"Run {i+1} has shape {r.shape}, expected {(T, N)}"
            )

    # Stack as (S, T, N)
    ts = np.stack(runs, axis=0)

    # Load static net
    net = load_static_net_from_file(net_file)

    if net.shape != (N, N):
        raise ValueError("Net shape does not match data shape")

    # Save MAT
    savemat(
        output_mat_path,
        {
            "ts": ts,
            "net": net,
            "Nsubjects": np.array([[S]]),
            "Ntimepoints": np.array([[T]]),
            "Nnodes": np.array([[N]])
        }
    )

    print("✅ MAT file created successfully")
    print(f"   ts shape: {ts.shape}  (S, T, N)")
    print(f"   net shape: {net.shape}")
    print(f"   Saved to: {output_mat_path}")

create_multirun_mat_S_T_N(
    runs_dir="/Users/shreeyansarora/Downloads/causal_discovery_on_time_series/dream3/ecoli2",
    net_file="/Users/shreeyansarora/Downloads/causal_discovery_on_time_series/dream3/ecoli2_gt.csv",
    output_mat_path="/Users/shreeyansarora/Downloads/causal_discovery_on_time_series/dream3/ecol21.mat"
)
