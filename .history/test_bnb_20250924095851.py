# test_bnb.py
import os
import heapq
import logging
from branch_and_bound import BranchEngine, BranchParams
from rmlp import RestrictedMasterProblem
from init_column import generate_init_columns  # 你已有的初始列生成器

class Node:
    def __init__(self, node_id, depth, rmp):
        self.id = node_id
        self.depth = depth
        self.rmp = rmp
        self.fixed_one = set()
        self.fixed_zero = set()

def apply_fixed_bounds(node: Node):
    """在 RMP 中应用分支修复"""
    for var in node.rmp.model.getVars():
        if var.VarName in node.fixed_one:
            var.lb, var.ub = 1.0, 1.0
        elif var.VarName in node.fixed_zero:
            var.lb, var.ub = 0.0, 0.0

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # 参数
    outdir = "logs_test_bnb"
    os.makedirs(outdir, exist_ok=True)
    params = BranchParams(outdir=outdir, strategy="most_fractional")
    engine = BranchEngine(params)

    # 1. 初始化 root 节点
    inst = ...  # 你的 ProblemInstance
    init_cols = generate_init_columns(inst)
    rmp = RestrictedMasterProblem(inst, outdir=outdir)
    rmp.add_columns(init_cols)
    rmp.solve()
    root = Node(0, 0, rmp)

    # 2. B&B 循环
    pq = [root]
    node_count = 0

    while pq and node_count < 20:
        node = heapq.heappop(pq)

        # 分支决策
        decision = engine.propose(node, incumbent_obj=99999)
        engine._dump_decision(node, decision)
        if decision.chosen is None:
            continue

        # 创建左右子节点
        for side, fixes in [("L", decision.left_fixes), ("R", decision.right_fixes)]:
            child_rmp = node.rmp.clone()  # 需要你在 rmlp.py 里加一个 clone() 方法
            child = Node(node.id * 2 + (1 if side == "L" else 2), node.depth + 1, child_rmp)

            # 应用分支修复
            for fx in fixes:
                if fx.lb == 1.0 and fx.ub == 1.0:
                    child.fixed_one.add(fx.var_id)
                else:
                    child.fixed_zero.add(fx.var_id)
            apply_fixed_bounds(child)

            # 解 LP
            obj = child.rmp.solve()

            # 更新伪成本
            engine.update_pseudocost(
                var_id=decision.chosen.var_id,
                parent_lb=node.rmp.get_objective_value(),
                left_lb=obj if side == "L" else None,
                right_lb=obj if side == "R" else None,
                x_value=decision.chosen.x_value
            )

            heapq.heappush(pq, child)

        node_count += 2

    print(f"✅ Test B&B 完成，共生成 {node_count} 个节点")
    print(f"请查看 {outdir}/branch_decisions.csv 与 node_xxx_decision.json")

if __name__ == "__main__":
    main()
