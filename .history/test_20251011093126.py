import os
import json
import random
import logging
from typing import List

# 你自己的模块
from data_model import Node, DroneSpec, Problem
from label_setting import label_setting
from rmp import RMP, RMPParams

# ----------------------------
# 基本日志：label_setting 过程日志
# ----------------------------
def setup_logger(outdir: str, name: str = "E2E") -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # 文件
    fh = logging.FileHandler(os.path.join(outdir, "label_setting.log"), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    return logger

# ----------------------------
# 随机生成 20 个客户
# ----------------------------
def gen_customers(n=200, seed=42) -> List[Node]:
    random.seed(seed)
    customers = []
    for cid in range(1, n + 1):
        x = random.uniform(-12, 12)
        y = random.uniform(-12, 12)

        # 时间窗：适度宽松但有差异
        start = random.uniform(0, 60)     # 最早时间
        width = random.uniform(60, 120)   # 窗口宽度
        tw = (start, start + width)

        demand = round(random.uniform(0.5, 2.0), 2)
        service = random.uniform(0.5, 2.0)

        customers.append(Node(cid, demand=demand, tw=tw, service=service, x=x, y=y))
    return customers

# ----------------------------
# 主流程
# ----------------------------
if __name__ == "__main__":
    OUTDIR = "e2e_logs"
    logger = setup_logger(OUTDIR)

    # 1) 数据：单仓 + 20 客户（注意：算法不强制从仓库出发/结束）
    depot = Node(0, demand=0.0, tw=(0, 180), service=0.0, x=0.0, y=0.0)
    cust_nodes = gen_customers(n=200, seed=42)
    nodes = [depot] + cust_nodes
    customers = [n.id for n in cust_nodes]

    # 无人机参数（可按需要调整）
    # 续航以飞行时间计（分钟），速度为 距离单位/分钟
    drone = DroneSpec(capacity=8.0, endurance=60.0, speed=1.5)
    prob = Problem(nodes=nodes, customers=customers, drone=drone)

    # 2) Label-setting：生成候选路径（列）
    #    为控制规模，max_len 不宜太大，建议 3~5。
    logger.info("Running label_setting ...")
    labels = label_setting(
        problem=prob,
        max_len=4,               # 每条路径最多 4 个客户
        depot_idx=0,             # 可用于尝试回仓（算法内部会尝试）
        logger=logger,           # 详细过程写入 label_setting.log
        K_per_sig=5,             # k-best per signature，保持多样性
        eps=0.05                 # ε-支配容忍度
    )
    logger.info("Label-setting finished. labels=%d", len(labels))

    # 将 labels 概要写个 JSON 方便审计
    quick_dump = [{
        "path": getattr(l, "path", []),
        "cost": getattr(l, "cost", 0.0),
        "served_count": getattr(l, "served_count", 0),
        "latest_departure": getattr(l, "latest_departure", 0.0)
    } for l in labels[:50]]
    with open(os.path.join(OUTDIR, "labels_sample.json"), "w", encoding="utf-8") as f:
        json.dump(quick_dump, f, ensure_ascii=False, indent=2)

    # 3) 转换为列
    cols = RMP.make_columns_from_labels(labels, id_prefix="it0_")
    if not cols:
        logger.error("No columns generated from labels. Try increasing endurance/speed or time windows.")
        raise SystemExit(1)

    # 4) 构建并求解 RMP（Gurobi）
    params = RMPParams(
        relax=True,                 # 列生成阶段先用 LP
        lambda_uncovered=1e4,       # 未覆盖惩罚
        lambda_route=0.0,           # 若想压缩路线数可设 >0
        time_limit=60,              # Gurobi 时间限制（秒）
        mip_gap=None,               # LP 下无意义
        outdir=os.path.join(OUTDIR, "rmp"),  # RMP 的日志目录
        solver_log=True,            # 输出 gurobi_rmp.log
        export_lp=True,             # 导出 model.lp 便于调试
        log_level=logging.INFO
    )
    rmp = RMP(customers=customers, params=params)  # rmlp 内部会建立独立日志文件

    # 5) 加列并求解
    rmp.add_columns(cols)
    obj = rmp.solve()
    logger.info("RMP solved. Obj=%.6f", obj)

    # 6) 取对偶价喂给定价器（下一轮可用，这里仅打印前几个）
    duals = rmp.get_duals()
    logger.info("Duals sample (first 10): %s", list(sorted(duals.items()))[:10])

    # 7) 查看被选列（x>1e-6）
    picked = rmp.get_selected_columns()
    logger.info("Picked columns: %d", len(picked))
    for c in picked[:10]:
        logger.info("  id=%s, cost=%.3f, served=%s, meta=%s",
                    c.id, c.cost, sorted(c.served_set), json.dumps(c.meta, ensure_ascii=False))

    print("\n=== Done ===")
    print(f"Label logs:        {os.path.join(OUTDIR, 'label_setting.log')}")
    print(f"RMP outdir:        {params.outdir}")
    print(f"- columns.csv:     {os.path.join(params.outdir, 'columns.csv')}")
    print(f"- selected.csv:    {os.path.join(params.outdir, 'selected_columns.csv')}")
    print(f"- duals.csv:       {os.path.join(params.outdir, 'coverage_duals.csv')}")
    print(f"- solution.json:   {os.path.join(params.outdir, 'solution.json')}")
    print(f"- solver log:      {os.path.join(params.outdir, 'gurobi_rmp.log')}")
    print(f"- model lp:        {os.path.join(params.outdir, 'model.lp')}")
