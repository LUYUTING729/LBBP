# test_bnb.py
import os
import heapq
import logging
from branch_and_bound import BranchEngine, BnBParams
from rmp import RMP
from init_column_generator import generate_init_columns  # 你已有的初始列生成器

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

from data_model import Problem, Node  # 假设你的数据类定义在这里

def build_toy_instance():
    # 例子：一个 depot + 5 个客户
    depot = Node(id=0, x=0.0, y=0.0, demand=0, tw=(0, 999))
    customers = [
        Node(id=i, x=i*10, y=i*5, demand=1, tw=(0, 999))
        for i in range(1, 6)
    ]
    nodes = [depot] + customers

    # 构造 ProblemInstance
    return ProblemInstance(
        N=nodes,
        C=customers,
        R=[],               # 如果不用 rendezvous，可以留空
        K_max=1,
        D_max=1,
        tT={(i,j): 1.0 for i in nodes for j in nodes if i!=j},
        tD={(i,j): 1.0 for i in nodes for j in nodes if i!=j},
        cT=1.0,
        cD=1.0,
        F=0.0,
        U_d=10.0,
        Q_d=5.0,
        e={n: 0 for n in nodes},
        l={n: 999 for n in nodes},
        g={n: n.demand for n in nodes},
    )

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # 参数
    outdir = "logs_test_bnb"
    os.makedirs(outdir, exist_ok=True)
    params = BnBParams(outdir=outdir, strategy="most_fractional")
    engine = BranchEngine(params)

    # 1. 初始化 root 节点
    inst = build_toy_instance()
    init_cols = generate_init_columns(inst)
    rmp = RMP(inst, outdir=outdir)
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
