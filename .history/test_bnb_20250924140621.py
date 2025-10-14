
# test_bnb.py
import os
import math
import heapq
import random
import logging
from typing import List

# ---- 依赖你的现有模块 ----
from data_model import Node, DroneSpec, Problem

# RMP 兼容导入（你可能叫 RMP 或 RestrictedMasterProblem）
try:
    from rmp import RMP as RMPClass
except ImportError:
    try:
        from rmp import RMP as RMPClass
    except ImportError as e:
        raise RuntimeError("请确保 rmp.py 中提供 RMP(problem) 或 rmlp.py 中提供 RestrictedMasterProblem(problem)") from e

from init_column_generator import generate_init_columns
from branch_and_bound import BranchEngine, BnBParams, BranchFix


# ----------------------------
# 构造单无人机 Problem（20 客户）
# ----------------------------
def build_problem_20(seed: int = 42) -> Problem:
    random.seed(seed)
    # depot = 0
    depot = Node(id=0, x=0.0, y=0.0, demand=0.0, tw=(0.0, 180.0), service=0.0)
    customers: List[Node] = []
    for i in range(1, 21):
        x = random.uniform(-12, 12)
        y = random.uniform(-12, 12)
        start = random.uniform(0, 60)
        width = random.uniform(60, 120)
        tw = (start, start + width)
        demand = round(random.uniform(0.5, 2.0), 2)
        service = random.uniform(0.5, 2.0)
        customers.append(Node(id=i, x=x, y=y, demand=demand, tw=tw, service=service))

    nodes = [depot] + customers
    cust_ids = [n.id for n in customers]
    drone = DroneSpec(speed=1.5, endurance=60.0, capacity=8.0)

    return Problem(nodes=nodes, customers=cust_ids, drone=drone)


# ----------------------------
# B&B 节点
# ----------------------------
class NodeBB:
    __slots__ = ("id", "depth", "rmp", "fixed_one", "fixed_zero")
    def __init__(self, node_id: int, depth: int, rmp_obj):
        self.id = node_id
        self.depth = depth
        self.rmp = rmp_obj
        self.fixed_one = set()   # {var_id}
        self.fixed_zero = set()  # {var_id}

    def __lt__(self, other):  # 供 heapq 使用（按深度浅优先；可改为按LB）
        return (self.depth, self.id) < (other.depth, other.id)


# ----------------------------
# 将分支修复应用到 RMP (Gurobi)
# ----------------------------
def apply_fixed_bounds(node: NodeBB, logger: logging.Logger):
    be = getattr(node.rmp, "backend", None)
    if be is None or not hasattr(be, "x_vars"):
        raise RuntimeError("RMP.backend.x_vars 未暴露，请在 rmp 中提供 backend.x_vars 映射到列变量")

    x_vars = be.x_vars  # dict[var_id -> gurobi.Var]

    # 应用 1/0 修复
    for vid in node.fixed_one:
        v = x_vars.get(vid)
        if v is not None:
            v.LB, v.UB = 1.0, 1.0
        else:
            logger.warning("fixed_one 变量未在模型中找到: %s", vid)
    for vid in node.fixed_zero:
        v = x_vars.get(vid)
        if v is not None:
            v.LB, v.UB = 0.0, 0.0
        else:
            logger.warning("fixed_zero 变量未在模型中找到: %s", vid)


# ----------------------------
# 入口：构建 RMP(problem) → 加 init 列 → solve → B&B 分支若干节点
# ----------------------------
def main():
    # 基础日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("TEST_BNB")

    outdir = "bp_logs"
    os.makedirs(outdir, exist_ok=True)

    # 1) 构造问题
    prob = build_problem_20(seed=42)

    # 2) 构建 RMP(problem)
    rmp = RMPClass(prob, outdir=os.path.join(outdir, "rmp"))  # 要求 RMP 支持 RMP(problem, outdir=...)

    # 3) 初始列 → 加入 RMP → 求解
    logger.info("Generating initial columns...")
    init_cols = generate_init_columns(prob, depot_idx=0, feasible_only=True, logger=logger)
    if not init_cols:
        logger.error("初始列为空：请放宽 endurance/speed/timewindow 或检查 init_column_generator")
        return

    rmp.add_columns(init_cols)
    lb_root = rmp.solve()
    logger.info("Root RMP solved. LB = %.6f", lb_root)

    # 4) 分支引擎
    bparams = BnBParams(
        strategy="auto",              # "auto" | "strong" | "pseudocost" | "most_fractional"
        strong_top_k=10,
        strong_time_limit=1.0,
        strong_stop_early=True,
        use_pseudocost=True,
        pseudocost_min_obs=2,
        outdir=os.path.join(outdir, "branch"),
        log_level=logging.INFO
    )
    bengine = BranchEngine(params=bparams, logger=logger)

    # 5) B&B 循环（仅测试分支，目标≥20个节点）
    root = NodeBB(node_id=0, depth=0, rmp_obj=rmp)
    pq = [root]
    heapq.heapify(pq)
    node_visits = 0
    target_nodes = 20
    incumbent = float("inf")  # 若你有整数解可更新

    while pq and node_visits < target_nodes:
        node = heapq.heappop(pq)
        node_visits += 1
        logger.info("=== Visit Node %d (depth=%d) ===", node.id, node.depth)

        # 提出分支决策（不会修改模型；只返回修复建议）
        decision = bengine.propose(node, incumbent_obj=incumbent)
        if decision is None or decision.chosen is None:
            logger.info("[Node %d] No branching decision (integer/pruned).", node.id)
            continue

        # 左右子节点
        # 要求 RMP 提供 clone()：深/浅拷贝一个独立模型（变量名相同）
        left_rmp = node.rmp.clone()
        right_rmp = node.rmp.clone()
        left = NodeBB(node_id=node.id * 2 + 1, depth=node.depth + 1, rmp_obj=left_rmp)
        right = NodeBB(node_id=node.id * 2 + 2, depth=node.depth + 1, rmp_obj=right_rmp)

        # 应用修复
        for fx in decision.left_fixes:
            if isinstance(fx, BranchFix) and fx.lb == 1.0 and fx.ub == 1.0:
                left.fixed_one.add(fx.var_id)
            else:
                left.fixed_zero.add(fx.var_id)
        for fx in decision.right_fixes:
            if isinstance(fx, BranchFix) and fx.lb == 1.0 and fx.ub == 1.0:
                right.fixed_one.add(fx.var_id)
            else:
                right.fixed_zero.add(fx.var_id)

        apply_fixed_bounds(left, logger)
        apply_fixed_bounds(right, logger)

        # 求解左右子节点（LP）
        lb_left = left.rmp.solve()
        lb_right = right.rmp.solve()
        logger.info("[Node %d] Left LB=%.6f | Right LB=%.6f", node.id, lb_left, lb_right)

        # 伪成本更新（在两个子节点都有 LB 后）
        bengine.update_pseudocost(
            var_id=decision.chosen.var_id,
            parent_lb=node.rmp.get_objective_value(),
            left_lb=lb_left,
            right_lb=lb_right,
            x_value=decision.chosen.x_value
        )

        # 推入队列（这里简单用深度优先/广度可自行改造）
        heapq.heappush(pq, left)
        heapq.heappush(pq, right)

    print("\n✅ test_bnb 完成")
    print(f"输出目录：{outdir}/branch  （branch_decisions.csv、node_*.json、branch.log）")
    print(f"RMP 日志目录：{outdir}/rmp")


if __name__ == "__main__":
    output_dir = "logs_test_bnb"
    main()
