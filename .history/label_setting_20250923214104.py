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

def label_setting(problem: Problem, max_len: int = 5, depot_idx: int = 0) -> List[Label]:
    """
    生成可行路径（起点=所有客户，终点=任意节点；并额外尝试“回仓终止”）。
    cost = 累计(飞行 + 等待 + 服务)；latest_departure = min_k (b_k - start_k)。
    """
    nodes = problem.nodes
    Q = problem.drone.capacity
    E = problem.drone.endurance

    solutions: List[Label] = []
    stack: List[Label] = []

    # 初始化多个起点（所有客户）
    for s in problem.customers:
        ns = nodes[s]
        start_time_at_s = max(0.0, ns.tw[0])  # 以 dep=0 参考系
        wait_s = start_time_at_s - 0.0
        cost0 = wait_s + ns.service
        slack0 = ns.tw[1] - start_time_at_s  # 路径整体右移的最大余量

        init = Label(
            node=s,
            time=start_time_at_s,
            load=ns.demand,
            energy=E,                # 起步未飞行
            path=[s],
            cost=cost0,
            latest_departure=slack0
        )
        stack.append(init)

    def count_customers_in(path: List[int]) -> int:
        return sum(1 for v in path if v in problem.customers)

    while stack:
        L = stack.pop()

        # (A) 当前节点即可作为终点（任意节点终止）
        solutions.append(L)

        # (A.1) 额外生成“回仓终止”的路径（如果存在 depot 且可行）
        if 0 <= depot_idx < len(nodes) and L.node != depot_idx:
            fly_back = problem.flight_time(L.node, depot_idx)
            if L.energy >= fly_back:
                arr0 = L.time + nodes[L.node].service + fly_back
                a0, b0 = nodes[depot_idx].tw
                start0 = max(arr0, a0)  # 到仓也视为“服务开始”
                if start0 <= b0:
                    wait0 = start0 - arr0
                    cost_add = fly_back + wait0 + nodes[depot_idx].service
                    latest_dep0 = min(L.latest_departure, b0 - start0)
                    solutions.append(
                        Label(
                            node=depot_idx,
                            time=start0,
                            load=L.load,  # 回仓不改变载重
                            energy=L.energy - fly_back,
                            path=L.path + [depot_idx],
                            cost=L.cost + cost_add,
                            latest_departure=latest_dep0
                        )
                    )

        # 达到客户上限则不再扩展
        if count_customers_in(L.path) >= max_len:
            continue

        # 扩展到尚未服务的客户
        for j in problem.customers:
            if j in L.path:
                continue  # 不重复服务
            dj = nodes[j].demand
            if L.load + dj > Q:
                continue  # 载重超限
            fly = problem.flight_time(L.node, j)
            if L.energy < fly:
                continue  # 续航不足

            arr = L.time + nodes[L.node].service + fly
            aj, bj = nodes[j].tw
            start_j = max(arr, aj)
            if start_j > bj:
                continue  # 时间窗不可行

            wait_j = start_j - arr
            delta_cost = fly + wait_j + nodes[j].service
            new_latest_dep = min(L.latest_departure, bj - start_j)

            stack.append(
                Label(
                    node=j,
                    time=start_j,
                    load=L.load + dj,
                    energy=L.energy - fly,
                    path=L.path + [j],
                    cost=L.cost + delta_cost,
                    latest_departure=new_latest_dep
                )
            )

    return solutions
