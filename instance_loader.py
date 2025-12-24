"""
Utilities for reading VRPTWD distributed instance Excel files into Problem objects.
"""

import os
from typing import List, Optional, Set

import pandas as pd

from data_model import DroneSpec, Node, Problem, TruckParams

# Columns expected in the distributed-instance spreadsheets
REQUIRED_COLS: Set[str] = {"x", "y", "demand", "open", "close", "servicetime"}


def distributed_instance_path(base_dir: str, instance_id: int) -> str:
    """
    Build the absolute path to an instance file by id (1-based).
    """
    filename = f"VRPTWD-distributed-instance-{instance_id}.xlsx"
    return os.path.join(base_dir, filename)


def load_distributed_instance(
    path: str,
    *,
    drone_spec: Optional[DroneSpec] = None,
    truck_spec: Optional[TruckParams] = None,
    sheet_index: int = 0,
) -> Problem:
    """
    Read a distributed VRPTWD instance Excel file and return a Problem.

    Args:
        path: Excel file path.
        drone_spec: Optional DroneSpec to attach; if omitted a default is used.
        sheet_index: Sheet index to read (default first sheet).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Instance not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_index)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Instance columns missing: {sorted(missing)} in {path}")

    rows = df.reset_index(drop=True)
    nodes: List[Node] = []
    for idx, row in rows.iterrows():
        nodes.append(
            Node(
                id=int(idx),
                demand=float(row["demand"]),
                tw=(float(row["open"]), float(row["close"])),
                service=float(row["servicetime"]),
                x=float(row["x"]),
                y=float(row["y"]),
            )
        )

    # First row is the depot; all others are customers.
    customers = [node.id for node in nodes if node.id != 0]
    drone = drone_spec or DroneSpec(capacity=8.0, endurance=60.0, speed=5.0)
    truck = truck_spec or TruckParams(
        truck_speed=4.0,
        truck_cost_per_time=1.0,
        bigM_time=1e5,
        time_limit=300.0,
    )

    return Problem(nodes=nodes, customers=customers, drone=drone, truck=truck)
