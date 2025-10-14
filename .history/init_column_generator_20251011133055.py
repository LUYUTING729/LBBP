from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Iterable
import logging
import math
import random
import itertools

from data_model import Problem
from rmp import Column


# =========================
# 参数
# =========================

@dataclass
class InitColGenParams:
    depot_idx: int = 0                    # 仓库索引
    feasible_only: bool = True            # 仅输出严格可行列
    log_level: int = logging.INFO
    out_log_file: Optional[str] = None

    # ---- 新增：多客户初始化控制 ----
    enable_pairs: bool = True             # 生成双客户往返 列
    enable_triples: bool = False          # 生成三客户往返 列（谨慎打开）
    max_pairs_per_customer: int = 5       # 每个客户与其最近的 K 个邻居配对
    max_triples_sample: int = 300         # 全局最多尝试的三元组样本数（随机采样）
    consider_both_orders: bool = True     # (i,j) 与 (j,i) 两种顺序都尝试
    random_seed: int = 42                 # 随机数种子（用于三元组采样）

    # 成本/可行性：与 label-setting 保持一致（飞行时间=续航消耗；速度在 Problem.flight_time 内部体现）
    # 初始化阶段不引入额外惩罚，避免偏置


# =========================
# 生成器
# =========================

class InitColumnGenerator:
    """
    初始化列生成：
      1) 单客户往返：[0,i,0]
      2) （可选）双客户往返：[0,i,j,0] & [0,j,i,0]
      3) （可选）三客户往返：[0,i,j,k,0] 的若干样本

    目标：给 RMP 初期提供“组合服务”的可行列，降低早期对偶偏置。
    """

    def __init__(self, problem: Problem, params: Optional[InitColGenParams] = None, logger: Optional[logging.Logger] = None):
        self.P = problem
        self.params = params or InitColGenParams()
        self.logger = logger or logging.getLogger("InitColGen")
        self.logger.setLevel(self.params.log_level)
        if self.params.out_log_file:
            fh = logging.FileHandler(self.params.out_log_file, mode="w", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)
        random.seed(self.params.random_seed)

    # ------------ Public API ------------
    def generate(self, id_prefix: str = "init_") -> List[Column]:
        depot = self.params.depot_idx
        self._check_depot(depot)

        cols: List[Column] = []
        skipped_single: Dict[int, str] = {}
        skipped_pairs: List[Tuple[Tuple[int,int], str]] = []
        skipped_triples: List[Tuple[Tuple[int,int,int], str]] = []

        # ---- 1) 单客户往返 ----
        for i in self.P.customers:
            ok, col_or_reason = self._build_roundtrip_single(depot, i)
            if ok:
                c = col_or_reason
                c.id = f"{id_prefix}c{i}"
                cols.append(c)
            else:
                skipped_single[i] = str(col_or_reason)

        # ---- 2) 双客户往返（可选）----
        if self.params.enable_pairs and len(self.P.customers) >= 2:
            pairs = self._nearest_pairs(self.P.customers, self.params.max_pairs_per_customer)
            for (i, j) in pairs:
                built_any = False
                orders = [(i, j)]
                if self.params.consider_both_orders:
                    orders.append((j, i))
                for (a, b) in orders:
                    ok, col_or_reason = self._build_roundtrip_pair(depot, a, b)
                    if ok:
                        c = col_or_reason
                        c.id = f"{id_prefix}c{a}_{b}"
                        cols.append(c)
                        built_any = True
                if not built_any:
                    skipped_pairs.append(((i, j), "infeasible_all_orders"))

        # ---- 3) 三客户往返（可选）----
        if self.params.enable_triples and len(self.P.customers) >= 3:
            triples = self._sample_triples(self.P.customers, self.params.max_triples_sample)
            for (i, j, k) in triples:
                # 尝试若干顺序：启发式只试 (i,j,k) 与 (k,j,i)
                tried_orders = [(i, j, k), (k, j, i)]
                success = False
                for order in tried_orders:
                    ok, col_or_reason = self._build_roundtrip_triple(depot, order)
                    if ok:
                        c = col_or_reason
                        c.id = f"{id_prefix}c{order[0]}_{order[1]}_{order[2]}"
                        cols.append(c)
                        success = True
                        break
                if not success:
                    skipped_triples.append(((i, j, k), "infeasible_sample"))

        # ---- 汇总日志 ----
        self._log_summary(cols, skipped_single, skipped_pairs, skipped_triples)
        return cols

    # ------------ 单客户往返 ------------
    def _build_roundtrip_single(self, depot: int, i: int) -> Tuple[bool, Column | str]:
        P = self.P
        nodes = P.nodes
        a0, b0 = nodes[depot].tw

        # 载重
        if nodes[i].demand > P.drone.capacity:
            return False, "capacity_exceeded"

        t1 = P.flight_time(depot, i)
        t2 = P.flight_time(i, depot)
        energy = t1 + t2
        if energy > P.drone.endurance:
            return False, "endurance_insufficient"

        ai, bi = nodes[i].tw
        start_i = max(ai, a0 + t1)
        s_dep = start_i - t1
        r_dep = start_i + nodes[i].service + t2

        if start_i > bi or r_dep > b0:
            return (False, "time_window_infeasible") if self.params.feasible_only else \
                   (True, self._mk_col([depot, i, depot], {i}, s_dep, r_dep, energy, nodes[depot].service + nodes[i].service, {}))

        # 成本=飞行+服务
        cost = t1 + nodes[i].service + t2 + nodes[depot].service
        duration = r_dep - s_dep
        meta = {
            "type": "single_customer_roundtrip",
            "depot": depot,
            "customers": [i],
            "s_dep": s_dep,
            "arrive_i": start_i,
            "return_time": r_dep,
        }
        return True, self._mk_col([depot, i, depot], {i}, s_dep, r_dep, energy, duration, meta, cost)

    # ------------ 双客户往返 ------------
    def _build_roundtrip_pair(self, depot: int, i: int, j: int) -> Tuple[bool, Column | str]:
        P = self.P
        nodes = P.nodes
        a0, b0 = nodes[depot].tw

        # 载重
        if nodes[i].demand + nodes[j].demand > P.drone.capacity:
            return False, "capacity_exceeded"

        # 路段飞行时间
        t0i = P.flight_time(depot, i)
        tij = P.flight_time(i, j)
        tj0 = P.flight_time(j, depot)
        energy = t0i + tij + tj0
        if energy > P.drone.endurance:
            return False, "endurance_insufficient"

        # 时间窗调度（到达即服务，依次处理 i -> j）
        ai, bi = nodes[i].tw
        aj, bj = nodes[j].tw

        # 先到 i
        start_i = max(ai, a0 + t0i)
        s_dep = start_i - t0i
        finish_i = start_i + nodes[i].service

        # 再到 j
        arrive_j = finish_i + tij
        start_j = max(aj, arrive_j)
        wait_j = start_j - arrive_j
        finish_j = start_j + nodes[j].service

        # 回仓
        r_dep = finish_j + tj0

        if start_i > bi or start_j > bj or r_dep > b0:
            return (False, "time_window_infeasible") if self.params.feasible_only else \
                   (True, self._mk_col([depot, i, j, depot], {i, j}, s_dep, r_dep, energy,
                                       r_dep - s_dep, {"note": "infeasible_relaxed"}))

        # 成本：飞行 + 两客户服务 + depot服务（与单点一致）
        cost = (t0i + tij + tj0) + (nodes[i].service + nodes[j].service) + nodes[depot].service
        duration = r_dep - s_dep
        meta = {
            "type": "two_customer_roundtrip",
            "depot": depot,
            "customers": [i, j],
            "s_dep": s_dep,
            "arrive_i": start_i,
            "arrive_j": arrive_j,
            "start_j": start_j,
            "wait_j": wait_j,
            "return_time": r_dep
        }
        return True, self._mk_col([depot, i, j, depot], {i, j}, s_dep, r_dep, energy, duration, meta, cost)

    # ------------ 三客户往返（样本化） ------------
    def _build_roundtrip_triple(self, depot: int, order: Tuple[int,int,int]) -> Tuple[bool, Column | str]:
        P = self.P
        nodes = P.nodes
        a0, b0 = nodes[depot].tw
        i, j, k = order

        # 载重
        if nodes[i].demand + nodes[j].demand + nodes[k].demand > P.drone.capacity:
            return False, "capacity_exceeded"

        # 飞行
        t0i = P.flight_time(depot, i)
        tij = P.flight_time(i, j)
        tjk = P.flight_time(j, k)
        tk0 = P.flight_time(k, depot)
        energy = t0i + tij + tjk + tk0
        if energy > P.drone.endurance:
            return False, "endurance_insufficient"

        # 调度
        ai, bi = nodes[i].tw
        aj, bj = nodes[j].tw
        ak, bk = nodes[k].tw

        start_i = max(ai, a0 + t0i)
        s_dep = start_i - t0i
        finish_i = start_i + nodes[i].service

        arrive_j = finish_i + tij
        start_j = max(aj, arrive_j)
        finish_j = start_j + nodes[j].service

        arrive_k = finish_j + tjk
        start_k = max(ak, arrive_k)
        finish_k = start_k + nodes[k].service

        r_dep = finish_k + tk0

        if start_i > bi or start_j > bj or start_k > bk or r_dep > b0:
            return (False, "time_window_infeasible") if self.params.feasible_only else \
                   (True, self._mk_col([depot, i, j, k, depot], {i, j, k}, s_dep, r_dep, energy,
                                       r_dep - s_dep, {"note": "infeasible_relaxed"}))

        cost = (t0i + tij + tjk + tk0) + (nodes[i].service + nodes[j].service + nodes[k].service) + nodes[depot].service
        duration = r_dep - s_dep
        meta = {
            "type": "three_customer_roundtrip",
            "depot": depot,
            "customers": [i, j, k],
            "s_dep": s_dep,
            "arrive_i": start_i,
            "arrive_j": arrive_j,
            "arrive_k": arrive_k,
            "return_time": r_dep
        }
        return True, self._mk_col([depot, i, j, k, depot], {i, j, k}, s_dep, r_dep, energy, duration, meta, cost)

    # ------------ 工具：最近邻配对 ------------
    def _nearest_pairs(self, customers: Iterable[int], k: int) -> List[Tuple[int,int]]:
        """
        针对每个 i，找距离最近的 k 个客户 j（i!=j），生成不重复的无序对 (min, max)。
        """
        cust = list(customers)
        # 预计算几何距离（或用 flight_time 近似）
        def dist(a: int, b: int) -> float:
            na, nb = self.P.nodes[a], self.P.nodes[b]
            dx, dy = na.x - nb.x, na.y - nb.y
            return math.hypot(dx, dy)

        pair_set = set()
        for i in cust:
            neigh = sorted((j for j in cust if j != i), key=lambda j: dist(i, j))[:k]
            for j in neigh:
                a, b = (i, j) if i < j else (j, i)
                pair_set.add((a, b))
        return sorted(pair_set)

    # ------------ 工具：三元组采样 ------------
    def _sample_triples(self, customers: Iterable[int], max_samples: int) -> List[Tuple[int,int,int]]:
        cust = list(customers)
        all_cnt = len(cust) * (len(cust) - 1) * (len(cust) - 2) // 6
        if all_cnt <= max_samples:
            return list(itertools.combinations(cust, 3))
        # 随机不重复采样
        seen = set()
        triples = []
        while len(triples) < max_samples:
            i, j, k = random.sample(cust, 3)
            triple = tuple(sorted((i, j, k)))
            if triple not in seen:
                seen.add(triple)
                triples.append(triple)
        return triples

    # ------------ Column 构造 ------------
    def _mk_col(self, path: List[int], served: Iterable[int],
                s_dep: float, r_dep: float, energy: float,
                duration: float, meta: Dict, cost: Optional[float] = None) -> Column:
        col = Column(
            id="tmp",  # 生成后会被覆盖
            path=path,
            served_set=frozenset(set(served)),
            cost=float(cost) if cost is not None else (duration),  # 兜底：用时长代价
            duration=float(duration),
            energy=float(energy),
            dep_window=(s_dep, s_dep),  # 这里记录一个可行出发时刻；你也可存 (a0, latest_dep)
            meta=meta
        )
        return col

    # ------------ 校验 ------------
    def _check_depot(self, depot: int):
        if not (0 <= depot < len(self.P.nodes)):
            raise ValueError(f"Invalid depot_idx={depot}. Node count={len(self.P.nodes)}")


# =============== 便捷函数（保持旧接口兼容） ===============

def generate_init_columns(problem: Problem,
                          depot_idx: int = 0,
                          feasible_only: bool = True,
                          logger: Optional[logging.Logger] = None,
                          enable_pairs: bool = True,
                          enable_triples: bool = False,
                          max_pairs_per_customer: int = 5,
                          max_triples_sample: int = 300,
                          consider_both_orders: bool = True,
                          random_seed: int = 42) -> List[Column]:
    """
    便捷函数：一行生成初始列
    - 默认：单客户 + 双客户（近邻配对）初始化
    - 可选：三客户样本
    """
    params = InitColGenParams(
        depot_idx=depot_idx,
        feasible_only=feasible_only,
        enable_pairs=enable_pairs,
        enable_triples=enable_triples,
        max_pairs_per_customer=max_pairs_per_customer,
        max_triples_sample=max_triples_sample,
        consider_both_orders=consider_both_orders,
        random_seed=random_seed
    )
    gen = InitColumnGenerator(problem, params=params, logger=logger)
    return gen.generate(id_prefix="init_")
