from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
from collections import defaultdict
from data_model import Problem

@dataclass
class Label:
    node: int
    time: float
    load: float
    energy: float
    path: List[int]
    cost: float
    latest_departure: float
    served_count: int
    covered_set: frozenset   # 已覆盖客户集合
    red_cost: float = 0.0   # ★新增：累计约简成本（cost - sum(pi)）

@dataclass
class LabelSettingParams:
    """
    控制 label_setting 行为的参数类
    """
    max_len: int = 5
    depot_idx: int = 0
    logger: Optional[logging.Logger] = None
    K_per_sig: int = 5
    eps: float = 0.05
    duals: Optional[Dict[int, float]] = None
    time_bucket: float = 30.0
    require_return: bool = False

# label_setting.py 内部新增工具函数
def _delta_red_cost(problem, i: int, j: int, arrive_time_i: float,
                    duals: Optional[Dict[int, float]]) -> float:
    """
    从 i 扩展到客户 j 的约简成本增量：
      Δrc = (飞行 + 等待 + 服务_j) - π_j
    若 duals is None → 退化为 Δcost。
    """
    nodes = problem.nodes
    fly = problem.flight_time(i, j)
    arr = arrive_time_i + fly
    start_j = max(arr, nodes[j].tw[0])
    if start_j > nodes[j].tw[1]:
        return float("inf")  # 时间窗不可行，外层会拦截
    wait = start_j - arr
    pure_inc_cost = fly + wait + nodes[j].service
    if duals is None:
        return pure_inc_cost
    pi_j = duals.get(j, 0.0)
    return pure_inc_cost - pi_j

# ----------------------------
# 支配与剪枝工具
# ----------------------------

def signature(label: Label, time_bucket: float = 120.0) -> tuple:
    """对标签分桶，避免跨域比较"""
    tbin = int(label.time // time_bucket)
    return (label.node, label.served_count, tbin)

# 以当前文件已内联的支配/剪枝为基础，替换为 “relaxed ε-支配 + red_cost 优先”
def _primary_value(L: Label, duals: Optional[Dict[int, float]]) -> float:
    # 用于排序/比较的主指标：有 duals→red_cost，否则→cost
    return L.red_cost if duals is not None else L.cost

def relaxed_pareto_dominance(newL: Label, pool: List[Label], eps: float,
                             duals: Optional[Dict[int, float]]) -> tuple:
    """
    在同 signature 桶内做 ε-支配。
    指标：served_count 大优；primary_value (red_cost/cost) 小优。
    若已有 L 满足：
      L.served_count >= newL.served_count
      且 primary(L) <= primary(newL)*(1 - eps)
    则 newL 被支配。
    """
    pv_new = _primary_value(newL, duals)
    for L in pool:
        pv = _primary_value(L, duals)
        if (L.served_count >= newL.served_count) and (pv <= pv_new * (1 - eps)):
            return True, "Relaxed Pareto dominated (primary)"
    return False, ""

def _kbest_insert(labels_by_sig: Dict[tuple, List[Label]], sig: tuple,
                  L: Label, K: int, duals: Optional[Dict[int, float]], logger=None):
    bucket = labels_by_sig[sig]
    bucket.append(L)
    # 按 primary 排序（red_cost 优先）
    bucket.sort(key=lambda x: (_primary_value(x, duals), -x.latest_departure))
    if len(bucket) > K:
        removed = bucket.pop(-1)
        if logger: logger.debug(f"Removed {removed.path} by K-best (primary worse)")

import random

def _expansion_order(customers: List[int],
                     duals: Optional[Dict[int, float]],
                     shuffle_if_no_dual: bool = True) -> List[int]:
    if duals is None:
        if shuffle_if_no_dual:
            tmp = customers[:]
            random.shuffle(tmp)
            return tmp
        return customers
    # dual-aware：高 π 优先
    return sorted(customers, key=lambda j: duals.get(j, 0.0), reverse=True)

def feasibility_pruning(
    label: Label,
    problem: Problem,
    depot_idx: int = 0,
    require_return: bool = False,
    shift_limit: float = 120.0
) -> tuple:
    """硬约束剪枝，返回 (bool, str)"""
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
    params: Optional[LabelSettingParams] = None,
    # 兼容旧参数方式
    max_len: int = 5,
    depot_idx: int = 0,
    logger: Optional[logging.Logger] = None,
    K_per_sig: int = 5,
    eps: float = 0.05,
    duals: Optional[Dict[int, float]] = None,
    time_bucket: float = 30.0,
    require_return: bool = False,
) -> List[Label]:
    """
    主入口推荐使用 params: LabelSettingParams
    旧参数方式仍兼容，但建议逐步迁移
    """
    # 优先使用 params
    if params is not None:
        max_len = params.max_len
        depot_idx = params.depot_idx
        logger = params.logger
        K_per_sig = params.K_per_sig
        eps = params.eps
        duals = params.duals
        time_bucket = params.time_bucket
        require_return = params.require_return

    nodes = problem.nodes
    Q = problem.drone.capacity
    E = problem.drone.endurance

    def signature(L: Label) -> tuple:
        tbin = int(L.time // time_bucket)
        return (L.node, L.served_count, tbin)

    from collections import defaultdict
    labels_by_sig: Dict[tuple, List[Label]] = defaultdict(list)
    solutions: List[Label] = []
    stack: List[Label] = []

    # --- 初始化：从所有客户起步 ---
    for s in problem.customers:
        ns = nodes[s]
        start_s = max(0.0, ns.tw[0])
        wait_s = start_s - 0.0
        cost0 = wait_s + ns.service
        rc0 = cost0 - (duals.get(s, 0.0) if duals is not None else 0.0)

        init = Label(
            node=s, time=start_s, load=ns.demand, energy=E,
            path=[s], cost=cost0, latest_departure=ns.tw[1] - start_s,
            served_count=1, covered_set=frozenset({s}), red_cost=rc0
        )

        feas, reason = feasibility_pruning(init, problem, depot_idx, require_return=require_return)
        if feas:
            if logger: logger.debug(f"Init pruned {init.path}: {reason}")
            continue
        sig = signature(init)
        dom, reason = relaxed_pareto_dominance(init, labels_by_sig[sig], eps, duals)
        if dom:
            if logger: logger.debug(f"Init pruned {init.path}: {reason}")
            continue

        _kbest_insert(labels_by_sig, sig, init, K_per_sig, duals, logger)
        stack.append(init)

    # --- 深/宽混合搜索 ---
    order = _expansion_order(problem.customers, duals, shuffle_if_no_dual=True)

    while stack:
        L = stack.pop()
        solutions.append(L)

        if L.served_count >= max_len:
            continue

        for j in order:
            if j in L.path:  # 不重复客户
                continue
            dj = nodes[j].demand
            if L.load + dj > Q:
                continue

            # 可行性初筛（续航/时间窗）
            fly = problem.flight_time(L.node, j)
            if L.energy < fly:
                continue
            arr_i = L.time + nodes[L.node].service
            # 约简增量（同时检查时间窗）
            d_rc = _delta_red_cost(problem, L.node, j, arr_i, duals)
            if d_rc == float("inf"):
                continue

            # 真正的时间窗计算
            arr = arr_i + fly
            aj, bj = nodes[j].tw
            start_j = max(arr, aj)
            if start_j > bj:
                continue
            wait_j = start_j - arr
            d_cost = fly + wait_j + nodes[j].service

            newL = Label(
                node=j,
                time=start_j,
                load=L.load + dj,
                energy=L.energy - fly,
                path=L.path + [j],
                cost=L.cost + d_cost,
                latest_departure=min(L.latest_departure, bj - start_j),
                served_count=L.served_count + 1,
                covered_set=L.covered_set | {j},
                red_cost=L.red_cost + d_rc
            )

            feas, reason = feasibility_pruning(newL, problem, depot_idx, require_return=require_return)
            if feas:
                if logger: logger.debug(f"Pruned {newL.path}: {reason}")
                continue
            sig = signature(newL)
            dom, reason = relaxed_pareto_dominance(newL, labels_by_sig[sig], eps, duals)
            if dom:
                if logger: logger.debug(f"Pruned {newL.path}: {reason}")
                continue

            _kbest_insert(labels_by_sig, sig, newL, K_per_sig, duals, logger)
            stack.append(newL)

    # 注：不再做单目标排序，solutions 保留 Pareto 风格多样性

    # ★ 新增：打印 Top-5 候选（按 red_cost 升序）并分解成本
    if logger is not None:
        cands = sorted(solutions, key=lambda L: L.red_cost)[:5]
        for z, L in enumerate(cands, 1):
            pisum = sum(duals.get(j, 0.0) for j in getattr(L, "covered_set", [])) if duals else 0.0
            logger.info(
                "PRICING TOP%-2d rc=%.6f  cost=%.4f  pi_sum=%.4f  served=%d  path=%s",
                z, L.red_cost, L.cost, pisum, getattr(L, "served_count", 0), getattr(L, "path", [])
            )

    return solutions

def select_diverse_routes(solutions: List[Label], K_pick: int = 100,
                          lam_cov: float = 1.0, lam_cost: float = 0.1) -> List[Label]:
    """
    贪心选择能带来最大覆盖增益的路径
    """
    picked = []
    covered = set()
    sols = solutions.copy()

    while len(picked) < K_pick and sols:
        best_gain, best = -1e9, None
        for lab in sols:
            new_cov = len(lab.covered_set - covered)
            gain = lam_cov * new_cov - lam_cost * lab.cost
            if gain > best_gain:
                best_gain, best = gain, lab
        if best is None or (best_gain <= 0 and picked):
            break
        picked.append(best)
        covered |= best.covered_set
        sols.remove(best)

    return picked
