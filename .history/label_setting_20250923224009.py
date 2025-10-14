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
# pruning.py

def dominance_check(new_label: Label, labels_at_node: List[Label]) -> tuple[bool, str]:
    """
    改进版支配规则：
    Label A dominates B if at same node:
      - A.served_count >= B.served_count
      - A.time <= B.time
      - A.load <= B.load
      - A.energy >= B.energy
      - A.cost <= B.cost
      - A.latest_departure >= B.latest_departure
    """
    for L in labels_at_node:
        if (L.served_count >= new_label.served_count and
            L.time <= new_label.time and
            L.load <= new_label.load and
            L.energy >= new_label.energy and
            L.cost <= new_label.cost and
            L.latest_departure >= new_label.latest_departure):
            return True, "Dominated by existing label"
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
    """
    升级版 Label-setting:
      - 目标 = 最大化服务客户数，次目标 = 最小化 cost
      - 剪枝 = dominance + feasibility
    """
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
            served_count=1
        )

        feas, reason = feasibility_pruning(init, problem, depot_idx)
        if feas:
            if logger: logger.debug(f"Init pruned {init.path}: {reason}")
            continue
        dom, reason = dominance_check(init, labels_at_node[s])
        if dom:
            if logger: logger.debug(f"Init pruned {init.path}: {reason}")
            continue

        stack.append(init)
        labels_at_node[s].append(init)

    while stack:
        L = stack.pop()
        solutions.append(L)

        # (A) 回仓尝试
        if depot_idx != -1 and L.node != depot_idx:
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
                        latest_departure=latest_dep0,
                        served_count=L.served_count
                    )
                    feas, reason = feasibility_pruning(newL, problem, depot_idx)
                    if feas:
                        if logger: logger.debug(f"Pruned {newL.path}: {reason}")
                    else:
                        dom, reason = dominance_check(newL, labels_at_node[depot_idx])
                        if dom:
                            if logger: logger.debug(f"Pruned {newL.path}: {reason}")
                        else:
                            solutions.append(newL)
                            labels_at_node[depot_idx].append(newL)

        # (B) 扩展新客户
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
                served_count=L.served_count + 1
            )

            feas, reason = feasibility_pruning(newL, problem, depot_idx)
            if feas:
                if logger: logger.debug(f"Pruned {newL.path}: {reason}")
                continue
            dom, reason = dominance_check(newL, labels_at_node[j])
            if dom:
                if logger: logger.debug(f"Pruned {newL.path}: {reason}")
                continue

            stack.append(newL)
            labels_at_node[j].append(newL)

    # 最终排序：先按 served_count 降序，再按 cost 升序
    solutions.sort(key=lambda L: (-L.served_count, L.cost))
    return solutions