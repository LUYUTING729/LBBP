# test_bnb.py
import os
import random
import logging
from branch_and_bound import BranchEngine, BnBParams

# ==== Fake RMP ====
class FakeRMP:
    def __init__(self, var_count=10, obj=100.0):
        self.vars = {f"lambda_{i}": random.uniform(0, 1) for i in range(var_count)}
        self._obj = obj

    def model(self):
        return self

    def getVars(self):
        # 模拟 gurobi 的接口
        return [type("v", (), {"VarName": k, "x": v}) for k, v in self.vars.items()]

    def get_objective_value(self):
        return self._obj

# ==== Fake Node ====
class FakeNode:
    def __init__(self, node_id, depth, rmp):
        self.id = node_id
        self.depth = depth
        self.rmp = rmp
        self.fixed_one = set()
        self.fixed_zero = set()

# ==== 测试主循环 ====
def branch_and_bound_demo():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    params = BnBParams(outdir="logs_test_bnb", strategy="most_fractional")
    engine = BranchEngine(params)

    # 初始化根节点
    root = FakeNode(0, 0, FakeRMP(var_count=6, obj=100.0))
    stack = [root]
    node_count = 0

    while stack and node_count < 20:  # 至少生成 20 个节点
        node = stack.pop()
        decision = engine.propose(node, incumbent_obj=9999)
        engine._dump_decision(node, decision)

        if decision.chosen is None:
            continue

        # 生成左右子节点，模拟目标值和解
        obj_left = node.rmp.get_objective_value() + random.uniform(1, 5)
        obj_right = node.rmp.get_objective_value() + random.uniform(1, 5)

        left = FakeNode(node.id * 2 + 1, node.depth + 1, FakeRMP(var_count=6, obj=obj_left))
        right = FakeNode(node.id * 2 + 2, node.depth + 1, FakeRMP(var_count=6, obj=obj_right))

        # 更新伪成本
        engine.update_pseudocost(
            var_id=decision.chosen.var_id,
            parent_lb=node.rmp.get_objective_value(),
            left_lb=obj_left,
            right_lb=obj_right,
            x_value=decision.chosen.x_value
        )

        stack.append(left)
        stack.append(right)
        node_count += 2

    print("✅ 分支测试完成，请查看 logs_test_bnb/branch_decisions.csv 和 JSON 文件")

if __name__ == "__main__":
    branch_and_bound_demo()
