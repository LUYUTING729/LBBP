"""
Batch LBBP/LBBD experiment runner (clean version).
- Reads all instances under instances/Distributed Instances (*.xlsx).
- Uses instance_loader.load_distributed_instance() directly (no data rewriting).
- Writes per-instance outputs into exp/<instance_name>/.
- Writes summary to exp/summary.csv and exp/summary.json.
"""

from __future__ import annotations

import csv
import json
import os
import time
import logging
from pathlib import Path
from typing import List

from instance_loader import load_distributed_instance
from bp_drones import BPSolver, BPParams, ConflictCutParams
from cut_generator import CutGenParams, SRIConfig, CliqueConfig, BucketParams
from label_setting import LabelSettingParams
from rmp import RMPParams
from init_column_generator import generate_init_columns
from logging_utils import init_logging
from data_model import DroneSpec, TruckParams, Node, Problem


EXP_ROOT = "exp"
INST_DIR = os.path.join("Distributed Instances")


def list_instance_files(inst_dir: str) -> List[Path]:
    return sorted([p for p in Path(inst_dir).glob("*.xlsx") if p.is_file()])


def make_params(base_outdir: str) -> BPParams:
    rmp_outdir = os.path.join(base_outdir, "rmp")
    label_outdir = os.path.join(base_outdir, "label_setting")
    cut_gen = CutGenParams(
        enabled=False,
        sri=SRIConfig(
            source="file",
            file_path="cuts.json",
            bucket=BucketParams(enabled=False),
        ),
        clique=CliqueConfig(
            source="file",
            file_path="cuts.json",
        ),
    )
    return BPParams(
        label_setting_params=LabelSettingParams(
            max_len=20,
            depot_idx=0,
            K_per_sig=10,
            require_return=True,
            lambda_route=20.0,
            outdir=label_outdir,
            order_strategy="random",
        ),
        rmp_params=RMPParams(
            relax=True,
            lambda_uncovered=1e4,
            lambda_route=20.0,
            max_routes=None,
            bigM_truck=1e5,
            time_limit=300,
            mip_gap=None,
            solver_log=True,
            export_lp=True,
            outdir=rmp_outdir,
            log_level=logging.INFO,
        ),
        outdir=base_outdir,
        log_level=logging.INFO,
        max_nodes=50,
        max_iterations=50,
        time_limit=30000,
        rc_tolerance=-1e-6,
        pricing_batch=2,
        stabilize_duals=False,
        stabilize_alpha=0.5,
        cut_violation_tol=1e-6,
        cut_time_limit_per_round=0.1,
        cut_max_add_per_round=5,
        enable_branch=True,
        branch_strategy="auto",
        sri_static=[],
        clique_static=[],
        sri_candidates=[],
        clique_candidates=[],
        cut_gen=cut_gen,
        conflict_cut=ConflictCutParams(mode="all"),
    )


def build_problem_with_end_depot(problem: Problem) -> Problem:
    """
    Align with test_lbbp.py: explicit start/end depot nodes and customer list.
    """
    cust_nodes = [n for n in problem.nodes if n.id != 0]
    cust_ids = [n.id for n in cust_nodes]
    end_id = (max(cust_ids) + 1) if cust_ids else 1

    start_depot = Node(0, demand=0.0, tw=(0.0, 180.0), service=0.0, x=0.0, y=0.0)
    end_depot = Node(end_id, demand=0.0, tw=(0.0, 1800.0), service=0.0, x=0.0, y=0.0)

    problem.nodes = [start_depot] + cust_nodes + [end_depot]
    problem.customers = [n.id for n in cust_nodes]
    return problem


def run_one(path: Path, summary: List[dict]) -> None:
    inst_name = path.stem
    outdir = os.path.join(EXP_ROOT, inst_name)
    os.makedirs(outdir, exist_ok=True)

    logger = init_logging(outdir, name=f"exp_{inst_name}")
    logger.info("=== Start instance %s ===", inst_name)
    logger.info("Instance path: %s", path)

    drone_spec = DroneSpec(capacity=8.0, endurance=60.0, speed=8.0)
    truck_spec = TruckParams(truck_speed=4.0, truck_cost_per_time=1.0, bigM_time=1e5, time_limit=300.0)
    problem = load_distributed_instance(str(path), drone_spec=drone_spec, truck_spec=truck_spec)
    problem = build_problem_with_end_depot(problem)

    params = make_params(outdir)
    init_cols = generate_init_columns(
        problem,
        depot_idx=0,
        feasible_only=True,
        logger=logger,
    )

    solver = BPSolver(problem=problem, params=params, logger=logger)

    t0 = time.time()
    row = {
        "instance": inst_name,
        "status": "",
        "objective": "",
        "num_picked": 0,
        "time_sec": "",
        "gap": "",
        "iterations": "",
        "truck_feasible": "",
        "snapshot_json": "",
        "snapshot_csv": "",
        "error": "",
    }
    try:
        result = solver.solve(init_columns=init_cols)
        elapsed = time.time() - t0
        row.update(
            status=result.status,
            objective=f"{result.obj_value:.6f}",
            num_picked=len(result.selected_columns),
            time_sec=f"{elapsed:.2f}",
            snapshot_json=result.snapshot_paths.get("json", ""),
            snapshot_csv=result.snapshot_paths.get("csv", ""),
        )
        if result.stats:
            last = result.stats[-1]
            row.update(
                gap=f"{last.gap:.6g}" if last.gap != float("inf") else "inf",
                iterations=last.iteration,
                truck_feasible=last.truck_feasible if last.truck_feasible is not None else "",
            )

        picked_path = os.path.join(outdir, "picked.json")
        with open(picked_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "id": c.id,
                        "cost": c.cost,
                        "served": sorted(map(int, c.served_set)),
                        "path": c.path,
                        "meta": c.meta,
                    }
                    for c in result.selected_columns
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(
            "Instance %s done. Obj=%.6f, picked=%d, time=%.2fs",
            inst_name,
            result.obj_value,
            len(result.selected_columns),
            elapsed,
        )
    except Exception as e:
        elapsed = time.time() - t0
        row.update(
            status="error",
            time_sec=f"{elapsed:.2f}",
            error=str(e),
        )
        logger.exception("Instance %s failed: %s", inst_name, e)

    summary.append(row)


def write_summary(summary: List[dict]) -> None:
    if not summary:
        return
    os.makedirs(EXP_ROOT, exist_ok=True)
    csv_path = os.path.join(EXP_ROOT, "summary.csv")
    json_path = os.path.join(EXP_ROOT, "summary.json")

    headers = list(summary[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summary)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Summary written to {csv_path} ({len(summary)} rows)")


def main() -> None:
    inst_files = list_instance_files(INST_DIR)
    if not inst_files:
        print(f"No instances found under {INST_DIR}")
        return

    os.makedirs(EXP_ROOT, exist_ok=True)
    summary: List[dict] = []

    for p in inst_files:
        run_one(p, summary)

    write_summary(summary)


if __name__ == "__main__":
    main()
