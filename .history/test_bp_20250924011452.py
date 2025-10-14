# test_bp.py
import os
import json
import random
import logging
from typing import List

from data_model import Node, DroneSpec, Problem
from bp_drones import BPSolver, BPParams
from rmp import RMPParams

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
# 随机生成 20 个客户
# ----------------------------
def gen_customers(n=20, seed=42) -> List[Node]:
    random.seed(seed)
    customers = []
    for cid in range(1, n + 1):
        x = random.uniform(-12, 12)
        y = random.uniform(-12, 12)
        # 时间窗：适度宽松但有差异
        start = random.uniform(0, 60)
        width = random.uniform(120, 240)
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

    # 1) 构造问题：1个可选仓库 + 20 客户
    depot = Node(0, demand=0.0, tw=(0, 180), service=0.0, x=0.0, y=0.0)
    cust_nodes = gen_customers(n=20, seed=42)
    nodes = [depot] + cust_nodes
    customers = [n.id for n in cust_nodes]

    # 无人机参数（可按需要调整）
    drone = DroneSpec(capacity=8.0, endurance=60.0, speed=1.5)
    prob = Problem(nodes=nodes, customers=customers, drone=drone)

    # 2) BP 参数
    #    - 默认开启分支
    #    - RMP 使用 Gurobi，日志与模型文件输出到 bp_logs/rmp
    rmp_outdir = os.path.join("bp_logs", "rmp")
    bp_params = BPParams(
        # 列生成策略
        max_iterations=30,
        rc_tolerance=-1e-5,
        pricing_batch=8,
        stabilize_duals=False,
        stabilize_alpha=0.7,

        # label-setting 超参
        max_label_len=4,
        k_per_sig=5,
        eps_dom=0.05,
        depot_idx=0,  # -1 表示不尝试回仓

        # RMP（Gurobi）参数
        rmp_params=RMPParams(
            relax=True,                 # 列生成阶段先 LP
            lambda_uncovered=1e4,       # 未覆盖惩罚
            lambda_route=0.0,           # 想惩罚路线数可设 >0
            time_limit=60,              # RMP 单次求解时间（秒）
            mip_gap=None,               # LP 下无意义
            outdir=rmp_outdir,
            solver_log=True,
            export_lp=True,
            log_level=logging.INFO
        ),

        # 分支设置
        enable_branch=True,
        branch_strategy="most_fractional",
        max_nodes=50,                  # 为测试限定节点数，必要时增大

        # 终止
        time_limit=300,                # BP 全局时间限制（秒）
        no_improve_rounds=3,

        # 日志
        outdir="bp_logs",
        log_level=logging.INFO,
        dump_every_iter=True
    )

    # 3) 运行 BP
    logger.info("Starting Branch-and-Price...")
    bp = BPSolver(problem=prob, params=bp_params, logger=None)  # 让 BP 自建独立日志
    result = bp.solve()

    # 4) 汇总输出
    logger.info("BP status: %s", result.status)
    logger.info("Best objective: %.6f", result.obj_value)

    # 选中列概览
    picked = result.selected_columns
    logger.info("Picked columns: %d", len(picked))
    for c in picked[:10]:
        logger.info("  id=%s, cost=%.3f, served=%s, meta=%s",
                    c.id, c.cost, sorted(c.served_set), json.dumps(c.meta, ensure_ascii=False))

    # 统计每轮信息
    stats_path = os.path.join(OUTDIR, "bp_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump([s.__dict__ for s in result.stats], f, ensure_ascii=False, indent=2)

    # 导出最终解（便于外部分析）
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
    print(f"Test log:           {os.path.join(OUTDIR, 'test_bp.log')}")
    print(f"BP events:          {os.path.join('bp_logs', 'bp_events.log')}")
    print(f"RMP outdir:         {rmp_outdir}")
    print(f"Stats JSON:         {stats_path}")
    print(f"Final solution:     {sol_path}")

