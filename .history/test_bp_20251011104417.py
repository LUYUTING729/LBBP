# test_bp.py
import os
import json
import random
import logging
from typing import List
import matplotlib.pyplot as plt

from data_model import Node, DroneSpec, Problem
from bp_drones import BPSolver, BPParams
from label_setting import LabelSettingParams
from rmp import RMPParams
from init_column_generator import generate_init_columns

# ----------------------------
# 日志工具
# ----------------------------
def setup_logger(outdir: str, name: str = "TEST_BP") -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(ch)
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
        start = random.choice([0, 60, 120, 180])
        width = 60
        tw = (start, start + width)
        demand = round(random.uniform(0.5, 2.0), 2)
        service = random.uniform(0.5, 2.0)
        customers.append(Node(cid, demand=demand, tw=tw, service=service, x=x, y=y))
    return customers

# ----------------------------
# 可视化：把选中的列画成路径
# ----------------------------
def plot_solution(problem: Problem, picked_columns, out_png: str, title: str = "Drone Routes"):
    nodes = problem.nodes
    # 坐标数组
    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_title(title)

    # 画所有客户点（蓝点）和仓库（红星，假定 id=0 存在）
    # 找到 depot: 假定为 id=0
    depot = nodes[0] if len(nodes) > 0 and nodes[0].id == 0 else None

    # 客户散点
    cust_ids = set(problem.customers)
    cust_x = [nodes[i].x for i in cust_ids]
    cust_y = [nodes[i].y for i in cust_ids]
    ax.scatter(cust_x, cust_y, s=30, label="customers")

    # 仓库
    if depot is not None:
        ax.scatter([depot.x], [depot.y], marker="*", s=150, c="r", label="depot (0)")

    # 标注节点 id
    for n in nodes:
        ax.text(n.x + 0.15, n.y + 0.15, str(n.id), fontsize=8, alpha=0.8)

    # 画每条路径
    for k, col in enumerate(picked_columns, 1):
        path = getattr(col, "path", [])
        if not path or len(path) == 1:
            # 单点列：也标出一个小圈
            i = path[0]
            ax.scatter([nodes[i].x], [nodes[i].y], s=60, facecolors="none", edgecolors="k", linewidths=1.0)
            continue

        # 连线
        px = [nodes[i].x for i in path]
        py = [nodes[i].y for i in path]
        ax.plot(px, py, linewidth=1.8, label=f"route {k}: {path}")

    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
# ----------------------------
# 主流程
# ----------------------------
if __name__ == "__main__":
    OUTDIR = "bp_test_logs"
    logger = setup_logger(OUTDIR)
    os.makedirs("bp_logs", exist_ok=True)

    # 构造问题
    depot = Node(0, demand=0.0, tw=(0, 180), service=0.0, x=0.0, y=0.0)
    cust_nodes = gen_customers(n=40, seed=42)
    nodes = [depot] + cust_nodes
    customers = [n.id for n in cust_nodes]
    drone = DroneSpec(capacity=8.0, endurance=60.0, speed=5)
    prob = Problem(nodes=nodes, customers=customers, drone=drone)

    # BP参数
    rmp_outdir = os.path.join("bp_logs", "rmp")
    label_outdir = os.path.join("bp_logs", "label_setting")
    bp_params = BPParams(
        label_setting_params=LabelSettingParams(
            max_len=5,
            depot_idx=0,
            logger=logger,
            K_per_sig=10,
            eps=0.05,
            duals=dict(),
            time_bucket=30.0,
            require_return=True,
            # 下面参数如无特殊处理可去掉
            lambda_route=20.0,
            # seed=42,
            outdir=label_outdir
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
        time_limit=30000,
        no_improve_rounds=3,
        outdir="bp_logs",
        log_level=logging.INFO,
        dump_every_iter=True,
    )

    # 初始列生成
    logger.info("Generating initial columns (single-customer round trips)...")
    init_cols = generate_init_columns(prob, depot_idx=0, feasible_only=True, logger=logger)
    logger.info("Init columns generated: %d", len(init_cols))

    # 运行 BP
    logger.info("Starting Branch-and-Price...")
    bp = BPSolver(problem=prob, params=bp_params, logger=logger)
    result = bp.solve(init_columns=init_cols)

    # 汇总输出
    logger.info("BP status: %s", result.status)
    logger.info("Best objective: %.6f", result.obj_value)
    picked = result.selected_columns
    logger.info("Picked columns: %d", len(picked))
    for c in picked[:10]:
        logger.info("  id=%s, cost=%.3f, served=%s, path=%s",
                    c.id, c.cost, sorted(c.served_set), c.path)

    # 保存最终解
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
    
    
    # 6) 可视化保存
    fig_path = os.path.join(OUTDIR, "routes.png")
    try:
        plot_solution(prob, picked, fig_path, title="Selected Drone Routes")
        logger.info("Route figure saved to %s", fig_path)
    except Exception as e:
        logger.exception("Failed to plot routes: %s", e)

    print("\n=== DONE ===")
    print(f"Test log:       {os.path.join(OUTDIR, 'test_bp.log')}")
    print(f"BP events:      {os.path.join('bp_logs', 'bp_events.log')}")
    print(f"Stats CSV:      {os.path.join('bp_logs', 'stats.csv')}")
    print(f"Final solution: {sol_path}")

