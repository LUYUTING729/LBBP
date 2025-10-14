from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import logging

from data_model import Problem
from rmlp import Column


@dataclass
class InitColGenParams:
    """
    初始列生成的可调参数
    """
    depot_idx: int = 0                 # 仓库节点索引
    feasible_only: bool = True         # True: 仅输出严格可行的往返列
    log_level: int = logging.INFO      # 日志等级
    out_log_file: Optional[str] = None # 若提供路径，则把日志写入文件


class InitColumnGenerator:
    """
    从 depot 出发、单客户往返的初始列生成器。
    - 每位客户 j 生成一条路径: [depot, j, depot]
    - 成本定义与 label-setting 对齐：cost = t(depot->j) + service(j) + t(j->depot) + service(depot)
      （等待时间为 0，因为会选择“到达即服务”的最优出发时刻）
    - 可行性检查：载重、续航、时间窗（depot 与客户）全满足
    - 产出：List[Column]，可直接喂给 RMP
    """

    def __init__(self, problem: Problem, params: Optional[InitColGenParams] = None, logger: Optional[logging.Logger] = None):
        self.problem = problem
        self.params = params or InitColGenParams()
        self.logger = logger or logging.getLogger("InitColGen")
        self.logger.setLevel(self.params.log_level)
        if self.params.out_log_file:
            fh = logging.FileHandler(self.params.out_log_file, mode="w", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)

    # ------------ 公共接口 ------------
    def generate(self, id_prefix: str = "init_") -> List[Column]:
        P = self.problem
        nodes = P.nodes
        depot = self.params.depot_idx

        if not (0 <= depot < len(nodes)):
            raise ValueError(f"Invalid depot_idx={depot}. Node count={len(nodes)}")

        cols: List[Column] = []
        skipped: Dict[int, str] = {}

        a0, b0 = nodes[depot].tw
        s0_dep = a0  # 最早可出发

        for j in P.customers:
            if j == depot:
                continue

            dj = nodes[j].demand
            if dj > P.drone.capacity:
                skipped[j] = "capacity_exceeded"
                continue

            t1 = P.flight_time(depot, j)
            t2 = P.flight_time(j, depot)
            energy_need = t1 + t2
            if energy_need > P.drone.endurance:
                skipped[j] = "endurance_insufficient"
                continue

            # 选择“到达即服务”的出发时刻，避免等待成本：
            # start_j = max(a_j, a0 + t1), s_dep = start_j - t1  (>= a0)
            aj, bj = nodes[j].tw
            start_j = max(aj, a0 + t1)
            s_dep = start_j - t1  # 出发时刻（>= a0）

            # 时间窗可行：客户服务不晚于 bj；回仓不晚于 b0
            r_dep = start_j + nodes[j].service + t2
            if start_j > bj or r_dep > b0:
                skipped[j] = "time_window_infeasible"
                if self.params.feasible_only:
                    continue

            # 成本：飞行 + 服务（不含等待；回仓等待为0，因为 s_dep >= a0）
            route_cost = t1 + nodes[j].service + t2 + nodes[depot].service

            # 计算“可行最晚出发时间” margin（便于后续调度）
            # s_late = min(bj - t1, b0 - nodes[j].service - t2 - t1)
            s_late = min(bj - t1, b0 - nodes[j].service - t2 - t1)
            latest_departure_margin = max(0.0, s_late - s_dep)

            col = Column(
                id=f"{id_prefix}c{j}",
                path=[depot, j, depot],
                served_set=frozenset({j}),
                cost=route_cost,
                duration=(r_dep - s_dep),
                energy=energy_need,
                dep_window=(s0_dep, s_late),  # 从最早可出发到最晚可出发
                meta={
                    "type": "single_customer_roundtrip",
                    "depot": depot,
                    "customer": j,
                    "s_dep": s_dep,
                    "start_at_customer": start_j,
                    "return_time": r_dep,
                    "latest_departure_margin": latest_departure_margin,
                    "skipped_reason": None if j not in skipped else skipped[j]
                }
            )
            if self.params.feasible_only and j in skipped:
                # 理论不会到这（上面 continue 了），防御性检查
                continue

            cols.append(col)

        self._log_summary(cols, skipped)
        return cols

    # ------------ 内部工具 ------------
    def _log_summary(self, cols: List[Column], skipped: Dict[int, str]) -> None:
        self.logger.info("Init columns generated: %d (feasible_only=%s)", len(cols), self.params.feasible_only)
        if skipped:
            reasons: Dict[str, int] = {}
            for _, r in skipped.items():
                reasons[r] = reasons.get(r, 0) + 1
            self.logger.info("Skipped customers: %d | reasons=%s", len(skipped), reasons)
            for cid, reason in list(skipped.items())[:10]:
                self.logger.debug("  skipped cid=%d: %s", cid, reason)
        # 打印前若干列示例
        for c in cols[:10]:
            self.logger.debug("  col id=%s | path=%s | cost=%.3f | dep_window=%s | energy=%.3f",
                              c.id, c.path, c.cost, c.dep_window, c.energy)


# ------------ 便捷函数 ------------
def generate_init_columns(problem: Problem,
                          depot_idx: int = 0,
                          feasible_only: bool = True,
                          logger: Optional[logging.Logger] = None) -> List[Column]:
    """
    便捷函数：一行生成初始列
    """
    gen = InitColumnGenerator(problem, InitColGenParams(depot_idx=depot_idx,
                                                        feasible_only=feasible_only),
                              logger=logger)
    return gen.generate(id_prefix="init_")
