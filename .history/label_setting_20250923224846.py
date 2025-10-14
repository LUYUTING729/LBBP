from dataclasses import dataclass
from typing import List
from data_model import Problem
import logging


@dataclass
class Label:
    node: int                 # 当前所在节点
    time: float               # 当前节点服务开始时刻（已包含等待）
    load: float               # 已累计载重
    energy: float             # 剩余续航（以飞行时间计）
    path: List[int]           # 已访问节点序列
    cost: float               # 累计时间成本 = Σ(飞行 + 等待 + 服务)
    latest_departure: float   # 路径整体右移的最晚可行出发时间 Δ
    served_count: int   # 新增：已服务客户数
    covered_set: frozenset   # 新增: 已覆盖的客户集合

# pruning.py

def pareto_dominance(newL: Label, existing: List[Label]) -> tuple[bool, str]:
    """
    Pareto 前沿保留：
    如果已有标签 L 在以下所有维度都 >= newL，则 newL 被支配：
      - served_count
      - -cost
      - -latest_departure
    否则保留 newL。
    """
    for L in existing:
        if (L.served_count >= newL.served_count and
            L.cost <= newL.cost and
            L.latest_departure >= newL.latest_departure):
            # 完全支配
            return True, "Pareto dominated"
    return False, ""

def diversity_filter(newL: Label, existing: List[Label]) -> Tuple[bool, str]:
    """
    增强覆盖：如果 newL 的 covered_set 已被某个 existing 完全包含，
    且 cost 没有明显优势，则丢弃。
    """
    for L in existing:
        if newL.covered_set.issubset(L.covered_set) and newL.cost >= L.cost * 0.95:
            return True, "No new coverage"
    return False, ""


def feasibility_pruning(
    label: Label, problem: Problem, depot_idx: int = 0, shift_limit: float = 120.0
) -> tuple[bool, str]:
    """
    快速可行性剪枝。
    返回 (True, "reason") 表示不可行，应丢弃。
    """
    if label.latest_departure <= 0:
        return True, "latest_departure <= 0"
    if label.load > problem.drone.capacity:
        return True, "Capacity exceeded"
    if label.time > shift_limit:
        return True, f"Shift time {label.time:.1f} exceeds {shift_limit}"
    if 0 <= depot_idx < len(problem.nodes) and label.node != depot_idx:
        back_time = problem.flight_time(label.node, depot_idx)
        if label.energy < back_time:
            return True, "Not enough energy to return depot"
    return False, ""

def label_setting(
    problem: Problem,
    max_len: int = 5,
    depot_idx: int = 0,
    logger: logging.Logger = None
) -> List[Label]:
    nodes = problem.nodes
    Q = problem.drone.capacity
    E = problem.drone.endurance

    solutions: List[Label] = []
    stack: List[Label] = []
    labels_at_node: Dict[int, List[Label]] = {i: [] for i in range(len(nodes))}

    # 初始化：从所有客户出发
    for s in problem.customers:
        ns = nodes[s]
        start_time_at_s = max(0.0, ns.tw[0])
        wait_s = start_time_at_s - 0.0
        cost0 = wait_s + ns.service
        slack0 = ns.tw[1] - start_time_at_s

        init = Label(
            node=s,
            time=start_time_at_s,
            load=ns.demand,
            energy=E,
            path=[s],
            cost=cost0,
            latest_departure=slack0,
            served_count=1,
            covered_set=frozenset({s})
        )

        feas, reason = feasibility_pruning(init, problem, depot_idx)
        if feas:
            if logger: logger.debug(f"Init pruned {init.path}: {reason}")
            continue
        dom, reason = pareto_dominance(init, labels_at_node[s])
        if dom:
            if logger: logger.debug(f"Init pruned {init.path}: {reason}")
            continue
        div, reason = diversity_filter(init, labels_at_node[s])
        if div:
            if logger: logger.debug(f"Init pruned {init.path}: {reason}")
            continue

        stack.append(init)
        labels_at_node[s].append(init)

    while stack:
        L = stack.pop()
        solutions.append(L)

        if L.served_count >= max_len:
            continue

        for j in problem.customers:
            if j in L.path:
                continue
            dj = nodes[j].demand
            if L.load + dj > Q:
                continue
            fly = problem.flight_time(L.node, j)
            if L.energy < fly:
                continue

            arr = L.time + nodes[L.node].service + fly
            aj, bj = nodes[j].tw
            start_j = max(arr, aj)
            if start_j > bj:
                continue

            wait_j = start_j - arr
            delta_cost = fly + wait_j + nodes[j].service
            new_latest_dep = min(L.latest_departure, bj - start_j)

            newL = Label(
                node=j,
                time=start_j,
                load=L.load + dj,
                energy=L.energy - fly,
                path=L.path + [j],
                cost=L.cost + delta_cost,
                latest_departure=new_latest_dep,
                served_count=L.served_count + 1,
                covered_set=L.covered_set | {j}
            )

            feas, reason = feasibility_pruning(newL, problem, depot_idx)
            if feas:
                if logger: logger.debug(f"Pruned {newL.path}: {reason}")
                continue
            dom, reason = pareto_dominance(newL, labels_at_node[j])
            if dom:
                if logger: logger.debug(f"Pruned {newL.path}: {reason}")
                continue
            div, reason = diversity_filter(newL, labels_at_node[j])
            if div:
                if logger: logger.debug(f"Pruned {newL.path}: {reason}")
                continue

            stack.append(newL)
            labels_at_node[j].append(newL)

    # 不做单目标排序，而是返回 Pareto front 解集合
    return solutions