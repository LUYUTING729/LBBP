"""
Preprocess distributed instances by overwriting x/y/time windows/demand/service.
This is a one-time, offline step to make instances compatible with test_lbbp.py scales.

Usage:
  python prep_distributed_instances.py
  python prep_distributed_instances.py --path "instances/Distributed Instances/VRPTWD-distributed-instance-1.xlsx"
  python prep_distributed_instances.py --dir "instances/Distributed Instances"
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


REQUIRED_COLS = {"x", "y", "demand", "open", "close", "servicetime"}


def stable_seed_from_name(base_seed: int, name: str) -> int:
    h = 0
    for i, ch in enumerate(name):
        h = (h + (i + 1) * ord(ch)) % 1000003
    return int(base_seed + h)


def gen_customer_attrs(n: int, seed: int) -> List[Tuple[Tuple[float, float], float, float]]:
    random.seed(seed)
    out: List[Tuple[Tuple[float, float], float, float]] = []
    for _ in range(n):
        start = random.choice([0, 60, 120, 180])
        width = 60
        tw = (float(start), float(start + width))
        demand = round(random.uniform(0.5, 2.0), 2)
        service = float(random.uniform(0.5, 2.0))
        out.append((tw, demand, service))
    return out


def minmax_scale(values: List[float], target_min: float, target_max: float) -> List[float]:
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-9:
        return [0.5 * (target_min + target_max) for _ in values]
    scale = (target_max - target_min) / (vmax - vmin)
    return [target_min + (v - vmin) * scale for v in values]


def preprocess_dataframe(df: pd.DataFrame, inst_name: str) -> pd.DataFrame:
    cols = [str(c).strip() for c in df.columns]
    col_map: Dict[str, str] = {c.lower(): c for c in cols}
    missing = REQUIRED_COLS - set(col_map.keys())
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.shape[0] == 0:
        raise ValueError("Empty instance sheet.")

    # Depot (row 0)
    df.loc[0, col_map["x"]] = 0.0
    df.loc[0, col_map["y"]] = 0.0
    df.loc[0, col_map["demand"]] = 0.0
    df.loc[0, col_map["open"]] = 0.0
    df.loc[0, col_map["close"]] = 180.0
    df.loc[0, col_map["servicetime"]] = 0.0

    # Customers (rows 1..)
    cust_df = df.iloc[1:].copy()
    n = cust_df.shape[0]
    if n <= 0:
        return df

    xs = cust_df[col_map["x"]].astype(float).tolist()
    ys = cust_df[col_map["y"]].astype(float).tolist()
    xs_scaled = minmax_scale(xs, 0.0, 50.0)
    ys_scaled = minmax_scale(ys, 0.0, 50.0)

    seed = stable_seed_from_name(42, inst_name)
    attrs = gen_customer_attrs(n=n, seed=seed)

    for idx, row_idx in enumerate(cust_df.index):
        tw, demand, service = attrs[idx]
        df.loc[row_idx, col_map["x"]] = xs_scaled[idx]
        df.loc[row_idx, col_map["y"]] = ys_scaled[idx]
        df.loc[row_idx, col_map["demand"]] = demand
        df.loc[row_idx, col_map["open"]] = tw[0]
        df.loc[row_idx, col_map["close"]] = tw[1]
        df.loc[row_idx, col_map["servicetime"]] = service

    return df


def preprocess_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    xls = pd.ExcelFile(path)
    sheet_name = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet_name)
    df_out = preprocess_dataframe(df, inst_name=path.stem)

    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        df_out.to_excel(writer, sheet_name=sheet_name, index=False)


def list_instance_files(inst_dir: Path) -> List[Path]:
    return sorted([p for p in inst_dir.glob("*.xlsx") if p.is_file()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess distributed instances.")
    parser.add_argument("--path", type=str, default="", help="Single instance file path.")
    parser.add_argument("--dir", type=str, default="Distributed Instances",
                        help="Directory with instance files.")
    args = parser.parse_args()

    if args.path:
        preprocess_file(Path(args.path))
        print(f"Preprocessed: {args.path}")
        return

    inst_dir = Path(args.dir)
    paths = list_instance_files(inst_dir)
    if not paths:
        print(f"No instances found under {inst_dir}")
        return

    for p in paths:
        preprocess_file(p)
        print(f"Preprocessed: {p}")


if __name__ == "__main__":
    main()
