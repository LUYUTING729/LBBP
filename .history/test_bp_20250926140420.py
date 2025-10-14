# test_bp.py
import os
import json
import random
import logging
from typing import List

from data_model import Node, DroneSpec, Problem
from bp_drones import BPSolver, BPParams
from label_setting import LabelSettingParams
from rmp import RMPParams
from init_column_generator import generate_init_columns

# ----------------------------
# 基础日志（控制台 + 文件）
# ----------------------------
def setup_logger(outdir: str, name: str = "TEST_BP") -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # 文件
    fh = logging.FileHandler(os.path.join(outdir, "test_bp.log"), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger

# ----------------------------
# 随机生成客户
# ----------------------------
def gen_customers(n=20, seed=42) -> List[Node]:
    random.seed(seed)
    customers = []
    for cid in range(1, n + 1):
        x = random.uniform(-12, 12)
        y = random.uniform(-12, 12)
        # 时间窗：适度宽松但有差异
        start = random.uniform(0, 60)
        width = random.uniform(60, 120)
        tw = (start, start + width)
        demand = round(random.uniform(0.5, 2.0), 2)
        service = random.uniform(0.5, 2.0)
        customers.append(Node(cid, demand=demand, tw=tw, service=service, x=x, y=y))
    return customers

# ----------------------------
# 主流程
# ----------------------------
if __name__ == "__main__":
    OUTDIR = "bp_test_logs"
    logger = setup_logger(OUTDIR)

    # ★ 确保 bp_logs 目录存在
    os.makedirs("bp_logs", exist_ok=True)

    # 1) 构造问题：1个仓库 + 20 客户
    depot = Node(0, demand=0.0, tw=(0, 180), service=0.0, x=0.0, y=0.0)
    cust_nodes = gen_customers(n=40, seed=42)
    nodes = [depot] + cust_nodes
    customers = [n.id for n in cust_nodes]

    drone = DroneSpec(capacity=8.0, endurance=60.0, speed=5)
    prob = Problem(nodes=nodes, customers=customers, drone=drone)

    # 2) BP 参数
    rmp_outdir = os.path.join("bp_logs", "rmp")
    bp_params = BPParams(
        label_setting_params=LabelSettingParams(
            max_len=5,
            depot_idx=0,
            logger=logger,
            K_per_sig=15,
            eps=0.05,
            duals=True,               # 有对偶
            time_bucket=30.0,
            require_return=True,
            lambda_route=0.0,
            seed=42,
            outdir=os.path.join("bp_logs", "label_setting")
        ),
        rmp_params=RMPParams(
            relax=True,
            lambda_uncovered=1e4,
            lambda_route=0.0,
            time_limit=60,
            mip_gap=None,
            outdir=rmp_outdir,
            solver_log=True,
            export_lp=True,
            log_level=logging.INFO
        ),
        enable_branch=True,
        branch_strategy="most_fractional",
        max_nodes=50,
        time_limit=300,
        no_improve_rounds=3,
        outdir="bp_logs",
        log_level=logging.INFO,
        dump_every_iter=True,

    )

    # 3) 初始列生成
    logger.info("Generating initial columns (single-customer round trips)...")
    init_cols = generate_init_columns(prob, depot_idx=0, feasible_only=True, logger=logger)
    logger.info("Init columns generated: %d", len(init_cols))

    # 4) 运行 BP
    logger.info("Starting Branch-and-Price...")
    bp = BPSolver(problem=prob, params=bp_params, logger=logger)
    result = bp.solve(init_columns=init_cols)

    # 5) 汇总输出
    logger.info("BP status: %s", result.status)
    logger.info("Best objective: %.6f", result.obj_value)

    picked = result.selected_columns
    logger.info("Picked columns: %d", len(picked))
    for c in picked[:10]:
        logger.info("  id=%s, cost=%.3f, served=%s, path=%s",
                    c.id, c.cost, sorted(c.served_set), c.path)

    # 保存最终解（stats 已经在 _record 写到 CSV）
    sol_path = os.path.join(OUTDIR, "bp_solution.json")
    final_sol = {
        "status": result.status,
        "objective": result.obj_value,
        "num_picked": len(picked),
        "picked_ids": [c.id for c in picked],
        "picked": [
            {
                "id": c.id,
                "cost": c.cost,
                "served": sorted(map(int, c.served_set)),
                "path": c.path,
                "meta": c.meta
            } for c in picked
        ]
    }
    with open(sol_path, "w", encoding="utf-8") as f:
        json.dump(final_sol, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print(f"Test log:       {os.path.join(OUTDIR, 'test_bp.log')}")
    print(f"BP events:      {os.path.join('bp_logs', 'bp_events.log')}")
    print(f"Stats CSV:      {os.path.join('bp_logs', 'stats.csv')}")
    print(f"Final solution: {sol_path}")
