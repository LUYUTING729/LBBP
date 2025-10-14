# label_setting.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterable, Any, Tuple
import logging
import math
from collections import defaultdict
import random

# ====== 你的外部数据结构（假定已存在） ======
# from data_model import Problem   # 需包含 nodes/customers/drone/flight_time

# ====== Label 与参数 ======
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
    covered_set: frozenset
    red_cost: float

@dataclass
class LabelSettingParams:
    max_len: int = 5
    depot_idx: int = 0
    logger: Optional[logging.Logger] = None
    K_per_sig: int = 10
    eps: float = 0.05
    duals: Optional[Dict[int, float]] = None          # 覆盖约束对偶 π_i（key=客户id）
    time_bucket: float = 30.0                         # 时间分桶，影响 signature
    require_return: bool = False                      # 是否要求可回仓可行
    lambda_route: float = 0.0                         # 每条路径一次性惩罚项（与 RMP 一致）
    seed: Optional[int] = 42      
    disable_dominance: bool = False                    # 无对偶时的随机扰动、
    outdir: str = "LS_logs"

# ====== 可行性剪枝（返回 pruned?） ======
def feasibility_pruning(
    L: Label,
    problem,                      # Problem
    depot_idx: int,
    require_return: bool
) -> Tuple[bool, str]:
    """返回 (pruned, reason)。只做与当前标签独立的快速检查。"""
    # 负能量/负载等基本检查（扩展时已控，这里兜底）
    if L.energy < -1e-9:
        return True, "energy<0"

    return False, ""

# ====== ε-支配：主指标 red_cost（越小越好），辅指标 served_count（越大越好），再看 time（越小越好） ======
def relaxed_pareto_dominance(
    cand: Label,
    pool: List[Label],
    eps: float,
    duals: Optional[Dict[int, float]],
) -> Tuple[bool, str]:
    """
    返回 (dominated?, reason)
    cand 被 pool 中的任何一个以 ε-放宽强支配，则剔除。
    """
    # 主：red_cost（min），辅：served_count（max），三：time（min）
    rc_c, sc_c, t_c = cand.red_cost, cand.served_count, cand.time
    # 放宽量：对 red_cost 做 eps-放宽，避免微小数值导致误杀
    tol = abs(rc_c) * eps + 1e-9

    for L in pool:
        rc, sc, t = L.red_cost, L.served_count, L.time
        # L 在所有目标上不差于 cand，且至少一项严格好（放宽）
        not_worse = (rc <= rc_c + tol) and (sc >= sc_c) and (t <= t_c + 1e-9)
        strictly_better = (rc < rc_c - tol) or (sc > sc_c) or (t < t_c - 1e-9)
        if not_worse and strictly_better:
            return True, f"dominated_by(rc={rc:.4f},sc={sc},t={t:.2f})"
    return False, ""

# ====== K-best 插入：按 (rc, -served, time) 排序 ======
def _kbest_insert(
    labels_by_sig: Dict[tuple, List[Label]],
    sig: tuple,
    L: Label,
    K: int,
):
    bucket = labels_by_sig[sig]
    bucket.append(L)
    # 以 rc 为主键，served_count 次键（降序），time 次次键（升序）
    bucket.sort(key=lambda z: (round(z.red_cost, 8), -z.served_count, z.time))
    if len(bucket) > K:
        del bucket[K:]

# ====== 扩展顺序：有对偶时按 π 大到小；无对偶时扰动随机顺序 ======
def _expansion_order(customers: Iterable[int], duals: Optional[Dict[int, float]], seed: Optional[int]) -> List[int]:
    order = list(customers)
    if duals is None:
        if seed is not None:
            random.Random(seed).shuffle(order)
    else:
        order.sort(key=lambda j: duals.get(j, 0.0), reverse=True)
    return order

# ====== 主函数 ======
def label_setting(
    problem,
    params: Optional[LabelSettingParams] = None,

) -> List[Label]:
    """
    生成标签（候选路径），与 RMP 的 rc 定义严格一致：
        rc = cost + lambda_route - sum_{i in covered} pi_i
    注意：lambda_route 为每条路径的常数项，仅在初始化时计入一次；扩展时只累加 d_cost - pi_j。
    """
    # -------- 参数归一化 --------
    if params is not None:

        depot_idx     = params.depot_idx
        logger        = params.logger
        K_per_sig     = params.K_per_sig
        eps           = params.eps
        duals         = params.duals
        time_bucket   = params.time_bucket
        require_return= params.require_return
        lambda_route  = params.lambda_route
        seed          = params.seed

    nodes = problem.nodes
    Q = problem.drone.capacity
    E = problem.drone.endurance

    def signature(L: Label) -> tuple:
        # node + served_count + time-bucket
        tbin = int(L.time // time_bucket)
        return (L.node, L.served_count, tbin)

    labels_by_sig: Dict[tuple, List[Label]] = defaultdict(list)
    stack: List[Label] = []
    solutions: List[Label] = []

    def try_dominance(L, sig_bucket):
        if params and getattr(params, "disable_dominance", False):
            return False, ""  # 诊断轮：完全跳过支配
        return relaxed_pareto_dominance(L, sig_bucket, eps, duals)

    # -------- 初始化：从所有客户起步（路径常数项一次性计入） --------
    for s in problem.customers:
        ns = nodes[s]
        start_s = max(0.0, ns.tw[0])
        # 等待到最早服务时刻（初始时刻 t=0）
        wait_s = start_s - 0.0
        cost0 = wait_s + ns.service

        # rc0 = cost0 + lambda_route - pi_s
        pi_s = duals.get(s, 0.0) if duals is not None else 0.0
        rc0 = (cost0 + lambda_route) - pi_s

        init = Label(
            node=s, time=start_s, load=ns.demand, energy=E,
            path=[s], cost=cost0, latest_departure=ns.tw[1] - start_s,
            served_count=1, covered_set=frozenset({s}), red_cost=rc0
        )
        
        pruned, reason = feasibility_pruning(init, problem, depot_idx, require_return)
        if pruned:
            if logger: logger.debug(f"[INIT-PRUNE] {init.path} -> {reason}")
            continue

        sig = signature(init)
        dom, reason = try_dominance(init, labels_by_sig[sig])
        if dom:
            if logger: logger.debug(f"[INIT-DOM] {init.path} -> {reason}")
            continue

        _kbest_insert(labels_by_sig, sig, init, K_per_sig)
        stack.append(init)

    # -------- 扩展顺序 --------
    order = _expansion_order(problem.customers, duals, seed)

    # -------- 深/宽混合搜索 --------
    while stack:
        L = stack.pop()
        solutions.append(L)


        # 当前节点 i
        i = L.node
        service_i = nodes[i].service
        arr_i = L.time + service_i  # 从 i 出发时基准时刻（已服务 i）

        for j in order:
            if j in L.path:       # 不重复客户
                continue
            dj = nodes[j].demand
            if L.load + dj > Q:   # 载重
                continue

            # 能量（只扣飞行时间）
            fly_ij = problem.flight_time(i, j)
            if L.energy + 1e-9 < fly_ij:
                continue

            # 到达 j 的时间窗推进
            arr = arr_i + fly_ij
            aj, bj = nodes[j].tw
            start_j = max(arr, aj)
            if start_j > bj + 1e-9:
                continue
            wait_j = start_j - arr

            # 成本增量（与 RMP 一致的物理成本）
            d_cost = fly_ij + wait_j + nodes[j].service
            # 对偶收益（只对新增 j 扣 π_j；路径常数项已在初始化时计过）
            pi_j = duals.get(j, 0.0) if duals is not None else 0.0
            d_rc  = d_cost - pi_j

            newL = Label(
                node=j,
                time=start_j,
                load=L.load + dj,
                energy=L.energy - fly_ij,
                path=L.path + [j],
                cost=L.cost + d_cost,
                latest_departure=min(L.latest_departure, bj - start_j),
                served_count=L.served_count + 1,
                covered_set=L.covered_set | {j},
                red_cost=L.red_cost + d_rc
            )

            pruned, reason = feasibility_pruning(newL, problem, depot_idx, require_return)
            if pruned:
                if logger: logger.debug(f"[PRUNE] {newL.path} -> {reason}")
                continue

            sig = signature(newL)
            dom, reason = try_dominance(newL, labels_by_sig[sig])
            if dom:
                if logger: logger.debug(f"[DOM] {newL.path} -> {reason}")
                continue

            _kbest_insert(labels_by_sig, sig, newL, K_per_sig)
            stack.append(newL)

    # （可选）日志摘要
    if logger:
        total_buckets = sum(len(v) for v in labels_by_sig.values())
        logger.info(
            "LabelSetting done: kept=%d, buckets=%d, K=%d, eps=%.3f, require_return=%s",
            len(solutions), len(labels_by_sig), K_per_sig, eps, require_return
        )
    return solutions
