# test_lbbp.py
import os
import json
import random
import logging
from typing import List
import matplotlib.pyplot as plt  # 预留后续可视化使用

from data_model import Node, DroneSpec, Problem
from bp_drones import BPSolver, BPParams
from label_setting import LabelSettingParams
from rmp import RMPParams
from init_column_generator import generate_init_columns
from logging_utils import init_logging


# ----------------------------
# 随机生成客户
# ----------------------------
def gen_customers(n=20, seed=42) -> List[Node]:
    random.seed(seed)
    customers = []
    for cid in range(1, n + 1):
        x = random.uniform(0, 50)
        y = random.uniform(0, 50)
        # 时间窗：离散化起点 + 固定宽度，保证可行性但带约束性
        start = random.choice([0, 60, 120, 180])
        width = 60
        tw = (start, start + width)

        demand = round(random.uniform(0.5, 2.0), 2)
        service = random.uniform(0.5, 2.0)

        customers.append(Node(
            cid,
            demand=demand,
            tw=tw,
            service=service,
            x=x,
            y=y
        ))
    return customers


# ----------------------------
if __name__ == "__main__":
    OUTDIR = "lbbp_test_logs"
    os.makedirs(OUTDIR, exist_ok=True)

    logger = init_logging(OUTDIR, name="test_lbbp")
    os.makedirs("bp_logs", exist_ok=True)

    # ================
    # 构造问题实例
    # ================
    depot = Node(
        0,
        demand=0.0,
        tw=(0, 180),
        service=0.0,
        x=0.0,
        y=0.0
    )
    cust_nodes = gen_customers(n=20, seed=42)
    nodes = [depot] + cust_nodes
    customers = [n.id for n in cust_nodes]

    # 无人机参数
    drone = DroneSpec(
        capacity=8.0,   # 载重上限
        endurance=60.0, # 最大可飞行时间
        speed=5.0       # 飞行速度 距离/时间
    )

    prob = Problem(
        nodes=nodes,
        customers=customers,
        drone=drone
    )

    # ================
    # BP / LBBP 参数
    # ================
    rmp_outdir = os.path.join("bp_logs", "rmp")
    label_outdir = os.path.join("bp_logs", "label_setting")

    # 这里延续你之前 test_bp 的写法：BPParams 中包含 label_setting_params 和 rmp_params
    # 注意：RMPParams 现在多了 bigM_truck，用一个保守的大数
    bp_params = BPParams(
        label_setting_params=LabelSettingParams(
            max_len=20,
            depot_idx=0,
            logger=logger,
            K_per_sig=10,
            require_return=True,
            lambda_route=20.0,
            outdir=label_outdir,
            order_strategy="random"
        ),
        rmp_params=RMPParams(
            relax=True,
            lambda_uncovered=1e4,
            lambda_route=20.0,
            max_routes=None,
            bigM_truck=1e5,          # ★ θ_truck 的 Big-M，后续可以按实例调小
            time_limit=300,           # Gurobi RMP 求解单次时限
            mip_gap=None,
            solver_log=True,
            export_lp=True,
            outdir=rmp_outdir,
            log_level=logging.INFO
        ),
        # 下面这些字段与我们 BPSolver.__init__ / _column_generation_loop / 分支逻辑对应
        outdir="bp_logs",
        log_level=logging.INFO,

        max_nodes=50,
        max_iterations=50,
        time_limit=30000,            # LBBP 总时间上限（秒）

        rc_tolerance=-1e-6,
        pricing_batch=2,

        stabilize_duals=False,
        stabilize_alpha=0.5,

        cut_violation_tol=1e-6,
        cut_time_limit_per_round=0.1,
        cut_max_add_per_round=5,

        enable_branch=True,
        branch_strategy="auto",

        # 静态cut / 动态cut 候选
        # 先给空，保持最小可运行；后续可根据地理聚类添加
        sri_static=[],
        clique_static=[],
        sri_candidates=[],
        clique_candidates=[]
    )

    # LabelSettingParams 是给 LabelSettingGenerator 用的。
    # 你的 LabelSettingGenerator 在我们BPSolver里是用 self.params 来拿这些信息。
    # 如果你的实现需要手动把这个对象塞进 bp_params，可以在这里赋值：
  

    # ================
    # 初始列生成（单客户往返）
    # ================
    logger.info("Generating initial columns (single-customer round trips)...")
    init_cols = generate_init_columns(
        prob,
        depot_idx=0,
        feasible_only=True,
        logger=logger
    )
    logger.info("Init columns generated: %d", len(init_cols))

    # ================
    # 运行 LBBP
    # ================
    logger.info("Starting Logic-Based Benders Branch-and-Price...")
    bp = BPSolver(problem=prob, params=bp_params, logger=logger)
    result = bp.solve(init_columns=init_cols)

    # ================
    # 汇总输出
    # ================
    logger.info("LBBP status: %s", result.status)
    logger.info("Best objective (global incumbent): %.6f", result.obj_value)

    picked = result.selected_columns
    logger.info("Picked columns: %d", len(picked))
    for c in picked[:10]:
        logger.info("  id=%s, cost=%.3f, served=%s, path=%s",
                    c.id, c.cost, sorted(c.served_set), c.path)

    # ================
    # 保存最终解
    # ================
    sol_path = os.path.join(OUTDIR, "lbbp_solution.json")
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
        ],
        # 未来可以把卡车调度结果也放进来（来自 TruckSolverCheck）
    }
    with open(sol_path, "w", encoding="utf-8") as f:
        json.dump(final_sol, f, ensure_ascii=False, indent=2)

    logger.info("Final solution saved to %s", sol_path)
    for fmt, path in result.snapshot_paths.items():
        logger.info("Snapshot (%s) saved to %s", fmt, path)

    # 也可以把 BP 的收敛曲线信息 stats 存下来，后面 plot_bp_convergence 用
    stats_path = os.path.join(OUTDIR, "lbbp_stats.json")
    stats_dump = [
        {
            "iteration": st.iteration,
            "node_id": st.node_id,
            "obj_value": st.obj_value,
            "best_bound": st.best_bound,
            "gap": st.gap,
            "num_columns": st.num_columns,
            "num_new_columns": st.num_new_columns,
            "time_elapsed": st.time_elapsed,
            "rc_min": st.rc_min,
            "cut_count": st.cut_count,
            "truck_feasible": st.truck_feasible,
        }
        for st in result.stats
    ]
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_dump, f, ensure_ascii=False, indent=2)
    logger.info("Stats saved to %s", stats_path)
