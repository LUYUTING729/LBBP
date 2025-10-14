from dataclasses import dataclass
from typing import List
from data_model import Problem

@dataclass
class Label:
    node: int                 # 当前所在节点
    time: float               # 当前节点服务开始时刻（已包含等待）
    load: float               # 已累计载重
    energy: float             # 剩余续航（以飞行时间计）
    path: List[int]           # 已访问节点序列
    cost: float               # 累计时间成本 = Σ(飞行 + 等待 + 服务)
    latest_departure: float   # 路径整体右移的最晚可行出发时间 Δ

# pruning.py
from typing import List
from label_setting import Label
from data_model import Problem

def dominance_check(new_label: Label, labels_at_node: List[Label]) -> bool:
    """
    判断 new_label 是否被 labels_at_node 中某个 label 支配。
    支配条件：已有 L 使得：
      L.time <= new_label.time
      L.load <= new_label.load
      L.energy >= new_label.energy
      L.cost <= new_label.cost
      L.latest_departure >= new_label.latest_departure
    返回 True 表示 new_label 被支配，应当剪枝。
    """
    for L in labels_at_node:
        if (L.time <= new_label.time and
            L.load <= new_label.load and
            L.energy >= new_label.energy and
            L.cost <= new_label.cost and
            L.latest_departure >= new_label.latest_departure):
            return True
    return False


def feasibility_pruning(label: Label, problem: Problem, depot_idx: int = 0) -> bool:
    """
    快速可行性剪枝：
      - latest_departure <= 0 → 已经没有整体右移余量
      - 载重超过 capacity
      - 剩余能量不足以回仓（可选）
    返回 True 表示 label 不可行，应当丢弃。
    """
    # latest departure 无余量
    if label.latest_departure <= 0:
        return True
    
    # 载重超限
    if label.load > problem.drone.capacity:
        return True
    
    # 如果要求必须能回仓，则做额外检查
    if 0 <= depot_idx < len(problem.nodes) and label.node != depot_idx:
        back_time = problem.flight_time(label.node, depot_idx)
        if label.energy < back_time:
            return True

    return False

from dataclasses import dataclass
from typing import List, Dict
from data_model import Problem
from pruning import dominance_check, feasibility_pruning

@dataclass
class Label:
    node: int
    time: float
    load: float
    energy: float
    path: List[int]
    cost: float
    latest_departure: float

def label_setting(problem: Problem, max_len: int = 5, depot_idx: int = 0) -> List[Label]:
    """
    升级版 Label-setting:
      - 起点 = 所有客户
      - 终点 = 任意节点（同时尝试回仓）
      - 增强：dominance check + feasibility pruning
    """
    nodes = problem.nodes
    Q = problem.drone.capacity
    E = problem.drone.endurance

    solutions: List[Label] = []
    stack: List[Label] = []

    # 存储每个节点的 labels，用于 dominance check
    labels_at_node: Dict[int, List[Label]] = {i: [] for i in range(len(nodes))}

    # 初始化起点（所有客户）
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
            latest_departure=slack0
        )

        # 剪枝
        if feasibility_pruning(init, problem, depot_idx):
            continue
        if dominance_check(init, labels_at_node[s]):
            continue

        stack.append(init)
        labels_at_node[s].append(init)

    def count_customers_in(path: List[int]) -> int:
        return sum(1 for v in path if v in problem.customers)

    while stack:
        L = stack.pop()

        # (A) 当前节点可作为终点
        solutions.append(L)

        # (A.1) 尝试回仓
        if 0 <= depot_idx < len(nodes) and L.node != depot_idx:
            fly_back = problem.flight_time(L.node, depot_idx)
            if L.energy >= fly_back:
                arr0 = L.time + nodes[L.node].service + fly_back
                a0, b0 = nodes[depot_idx].tw
                start0 = max(arr0, a0)
                if start0 <= b0:
                    wait0 = start0 - arr0
                    cost_add = fly_back + wait0 + nodes[depot_idx].service
                    latest_dep0 = min(L.latest_departure, b0 - start0)
                    newL = Label(
                        node=depot_idx,
                        time=start0,
                        load=L.load,
                        energy=L.energy - fly_back,
                        path=L.path + [depot_idx],
                        cost=L.cost + cost_add,
                        latest_departure=latest_dep0
                    )

                    if not feasibility_pruning(newL, problem, depot_idx) and \
                       not dominance_check(newL, labels_at_node[depot_idx]):
                        solutions.append(newL)
                        labels_at_node[depot_idx].append(newL)

        # (B) 扩展到下一个客户
        if count_customers_in(L.path) >= max_len:
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
                latest_departure=new_latest_dep
            )

            # 剪枝
            if feasibility_pruning(newL, problem, depot_idx):
                continue
            if dominance_check(newL, labels_at_node[j]):
                continue

            stack.append(newL)
            labels_at_node[j].append(newL)

    return solutions

