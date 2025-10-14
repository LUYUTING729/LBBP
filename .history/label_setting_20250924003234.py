from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
from collections import defaultdict
from data_model import Problem

@dataclass
class Label:
    """
    标签类，用于存储路径搜索过程中的状态信息
    node: 当前所在节点编号
    time: 当前累计时间
    load: 当前累计负载
    energy: 当前剩余电量
    path: 已访问节点序列
    cost: 当前累计成本
    latest_departure: 最晚出发时间（时间窗约束）
    served_count: 已服务客户数量
    covered_set: 已覆盖的客户集合
    """
    node: int
    time: float
    load: float
    energy: float
    path: List[int]
    cost: float
    latest_departure: float
    served_count: int
    covered_set: frozenset   # 已覆盖客户集合


# ----------------------------
# 支配与剪枝工具
# ----------------------------

def signature(label: Label, time_bucket: float = 30.0) -> tuple:
    """
    计算标签的特征签名，用于分组比较
    参数:
        label: 待计算的标签
        time_bucket: 时间分桶大小(默认30分钟)
    返回:
        tuple: (当前节点, 已服务数量, 时间分桶号)
    """
    tbin = int(label.time // time_bucket)
    return (label.node, label.served_count, tbin)

def relaxed_pareto_dominance(newL: Label, pool: List[Label], eps: float = 0.05) -> tuple:
    """
    松弛的帕累托支配判断
    参数:
        newL: 新标签
        pool: 已有标签池
        eps: 支配容忍度(默认5%)
    返回:
        (bool, str): (是否被支配, 原因说明)
    判断条件:
        1. 已服务客户数更多
        2. 成本更低(允许eps容忍度)
    """
    for L in pool:
        if (L.served_count >= newL.served_count and
            L.cost <= newL.cost * (1 - eps)):
            return True, "Relaxed Pareto dominated"
    return False, ""


def feasibility_pruning(
    label: Label,
    problem: Problem,
    depot_idx: int = 0,
    require_return: bool = False,
    shift_limit: float = 120.0
) -> tuple:
    """
    可行性检查与剪枝
    参数:
        label: 待检查的标签
        problem: 问题实例
        depot_idx: 仓库节点编号
        require_return: 是否要求返回仓库
        shift_limit: 班次时间限制
    返回:
        (bool, str): (是否需要剪枝, 剪枝原因)
    检查项:
        1. 最晚出发时间是否合法
        2. 载重是否超限
        3. 总时间是否超限
        4. 剩余电量是否足够返回
    """
    if label.latest_departure <= 0:
        return True, "latest_departure <= 0"
    if label.load > problem.drone.capacity:
        return True, "Capacity exceeded"
    if label.time > shift_limit:
        return True, f"Shift time {label.time:.1f} exceeds {shift_limit}"
    if require_return and depot_idx >= 0 and label.node != depot_idx:
        back = problem.flight_time(label.node, depot_idx)
        if label.energy < back:
            return True, "Not enough energy to return depot"
    return False, ""


# ----------------------------
# 主算法
# ----------------------------

def label_setting(
    problem: Problem,
    max_len: int = 5,
    depot_idx: int = 0,
    logger: Optional[logging.Logger] = None,
    K_per_sig: int = 5,
    eps: float = 0.05
) -> List[Label]:
    """
    标签设置算法主体
    参数:
        problem: 问题实例
        max_len: 单条路径最大访问节点数
        depot_idx: 仓库节点编号
        logger: 日志记录器
        K_per_sig: 每个特征签名保留的最优标签数
        eps: 支配判断容忍度
    执行流程:
        1. 初始化：从每个客户点生成初始标签
        2. 主循环：不断扩展标签直到栈空
           - 检查访问数是否达到上限
           - 尝试扩展到每个未访问客户
           - 进行可行性检查和支配检查
           - 保留每个签名组最优的K个标签
        3. 终选：使用覆盖度量选择多样化路径集
    """
    nodes = problem.nodes
    Q = problem.drone.capacity
    E = problem.drone.endurance

    stack: List[Label] = []
    solutions: List[Label] = []
    labels_by_sig: Dict[tuple, List[Label]] = defaultdict(list)

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
        sig = signature(init)
        dom, reason = relaxed_pareto_dominance(init, labels_by_sig[sig], eps)
        if dom:
            if logger: logger.debug(f"Init pruned {init.path}: {reason}")
            continue

        labels_by_sig[sig].append(init)
        stack.append(init)

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
            sig = signature(newL)
            dom, reason = relaxed_pareto_dfrom dataclasses import dataclass
from typing import List, Dict, Optional
import logging
from collections import defaultdict
from data_model import Problem

@dataclass
class Label:
    """
    标签类，用于存储路径搜索过程中的状态信息
    node: 当前所在节点编号
    time: 当前累计时间
    load: 当前累计负载
    energy: 当前剩余电量
    path: 已访问节点序列
    cost: 当前累计成本
    latest_departure: 最晚出发时间（时间窗约束）
    served_count: 已服务客户数量
    covered_set: 已覆盖的客户集合
    """
    node: int
    time: float
    load: float
    energy: float
    path: List[int]
    cost: float
    latest_departure: float
    served_count: int
    covered_set: frozenset   # 已覆盖客户集合


# ----------------------------
# 支配与剪枝工具
# ----------------------------

def signature(label: Label, time_bucket: float = 30.0) -> tuple:
    """
    计算标签的特征签名，用于分组比较
    参数:
        label: 待计算的标签
        time_bucket: 时间分桶大小(默认30分钟)
    返回:
        tuple: (当前节点, 已服务数量, 时间分桶号)
    """
    tbin = int(label.time // time_bucket)
    return (label.node, label.served_count, tbin)

def relaxed_pareto_dominance(newL: Label, pool: List[Label], eps: float = 0.05) -> tuple:
    """
    松弛的帕累托支配判断
    参数:
        newL: 新标签
        pool: 已有标签池
        eps: 支配容忍度(默认5%)
    返回:
        (bool, str): (是否被支配, 原因说明)
    判断条件:
        1. 已服务客户数更多
        2. 成本更低(允许eps容忍度)
    """
    for L in pool:
        if (L.served_count >= newL.served_count and
            L.cost <= newL.cost * (1 - eps)):
            return True, "Relaxed Pareto dominated"
    return False, ""


def feasibility_pruning(
    label: Label,
    problem: Problem,
    depot_idx: int = 0,
    require_return: bool = False,
    shift_limit: float = 120.0
) -> tuple:
    """
    可行性检查与剪枝
    参数:
        label: 待检查的标签
        problem: 问题实例
        depot_idx: 仓库节点编号
        require_return: 是否要求返回仓库
        shift_limit: 班次时间限制
    返回:
        (bool, str): (是否需要剪枝, 剪枝原因)
    检查项:
        1. 最晚出发时间是否合法
        2. 载重是否超限
        3. 总时间是否超限
        4. 剩余电量是否足够返回
    """
    if label.latest_departure <= 0:
        return True, "latest_departure <= 0"
    if label.load > problem.drone.capacity:
        return True, "Capacity exceeded"
    if label.time > shift_limit:
        return True, f"Shift time {label.time:.1f} exceeds {shift_limit}"
    if require_return and depot_idx >= 0 and label.node != depot_idx:
        back = problem.flight_time(label.node, depot_idx)
        if label.energy < back:
            return True, "Not enough energy to return depot"
    return False, ""


# ----------------------------
# 主算法
# ----------------------------

def label_setting(
    problem: Problem,
    max_len: int = 5,
    depot_idx: int = 0,
    logger: Optional[logging.Logger] = None,
    K_per_sig: int = 5,
    eps: float = 0.05
) -> List[Label]:
    """
    标签设置算法主体
    参数:
        problem: 问题实例
        max_len: 单条路径最大访问节点数
        depot_idx: 仓库节点编号
        logger: 日志记录器
        K_per_sig: 每个特征签名保留的最优标签数
        eps: 支配判断容忍度
    执行流程:
        1. 初始化：从每个客户点生成初始标签
        2. 主循环：不断扩展标签直到栈空
           - 检查访问数是否达到上限
           - 尝试扩展到每个未访问客户
           - 进行可行性检查和支配检查
           - 保留每个签名组最优的K个标签
        3. 终选：使用覆盖度量选择多样化路径集
    """
    nodes = problem.nodes
    Q = problem.drone.capacity
    E = problem.drone.endurance

    stack: List[Label] = []
    solutions: List[Label] = []
    labels_by_sig: Dict[tuple, List[Label]] = defaultdict(list)

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
        sig = signature(init)
        dom, reason = relaxed_pareto_dominance(init, labels_by_sig[sig], eps)
        if dom:
            if logger: logger.debug(f"Init pruned {init.path}: {reason}")
            continue

        labels_by_sig[sig].append(init)
        stack.append(init)

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
            sig = signature(newL)
            dom, reason = relaxed_pareto_d