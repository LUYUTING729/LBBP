# truck_solver.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable
import math
import logging

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    gp = None
    GRB = None


@dataclass
class TruckParams:
    truck_speed: float = 1.0              # 卡车速度（距离/时间）
    truck_cost_per_time: float = 1.0      # 每单位时间成本
    bigM_time: float = 1e5                # 时间窗/到达时间约束用的大 M
    time_limit: Optional[float] = 5.0     # 小问题求解时限（秒）
    mip_gap: Optional[float] = 0.01       # MIP gap（可选）
    log_to_console: bool = False          # 是否在控制台打印 Gurobi 日志


@dataclass
class TruckResult:
    feasible: bool
    cost: float
    route: List[int]                      # 访问顺序（含 depot）
    arrival: Dict[int, float]             # 到达时间
    conflicts: List[int]                  # 若不可行：建议的冲突节点集合（用于可行性割）
    msg: str = ""


class TruckSolver:
    """
    给定当前选中的无人机列（sorties），抽取所有“首尾节点”形成集合 W，
    在 V = {depot} ∪ W 上建立一个小型 VRPTW（单车），最小化卡车行驶时间成本，
    返回卡车成本 cost 和具体访问顺序 route。若不可行，返回 conflicts 用于 LB-Benders 可行性割。
    """
    def __init__(self, problem, depot_idx: int = 0, params: Optional[TruckParams] = None, logger: Optional[logging.Logger] = None):
        if gp is None:
            raise RuntimeError("需要 gurobipy。请安装并配置 Gurobi（pip install gurobipy）")
        self.pb = problem
        self.depot = int(depot_idx)
        self.params = params or TruckParams()
        self.logger = logger or logging.getLogger("TruckSolver")

        # 预计算欧氏距离（如果问题没有卡车时间矩阵）
        self._precompute_dist()

    # ---------- 公共入口 ----------
    def evaluate(self,
                 selected_columns: Iterable,          # RMP.selected_columns 列对象集合
                 use_path_endpoints: bool = True      # 从列的 path 抽取首尾；若没有则尝试 meta['start','end']
                 ) -> TruckResult:

        W = self._collect_endpoints(selected_columns, use_path_endpoints)
        V = [self.depot] + sorted(W)
        if len(V) == 1:
            # 没有任何端点：卡车不需要出车，成本为 0
            return TruckResult(feasible=True, cost=0.0, route=[self.depot], arrival={self.depot: self.pb.nodes[self.depot].tw[0]},
                               conflicts=[], msg="Empty endpoints; zero truck cost.")

        try:
            model, y, T = self._build_vrptw(V)
            model.optimize()
        except Exception as e:
            return TruckResult(False, float("inf"), [], {}, list(W), f"Build/solve error: {e}")

        status = model.Status
        if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            # 取到 incumbent 就认为可行（TIME_LIMIT 有可行解）
            if model.SolCount >= 1:
                route, arrival = self._extract_solution(V, y, T)
                cost = float(model.ObjVal)
                return TruckResult(True, cost, route, arrival, [], "")
            else:
                return TruckResult(False, float("inf"), [], {}, list(W), "No incumbent solution.")
        else:
            # 不可行或其他
            return TruckResult(False, float("inf"), [], {}, list(W), f"Gurobi status={status}")

    # ---------- 内部：抽取端点 ----------
    def _collect_endpoints(self, selected_columns: Iterable, use_path_endpoints: bool) -> set:
        """
        从列集合中抽取首尾节点：W = {s_1, e_1, s_2, e_2, ...} \ {depot}
        优先从 path[0], path[-1] 获取；若 path 不可靠，尝试 meta["start"], meta["end"]。
        """
        W = set()
        for col in selected_columns:
            s = None; e = None
            if use_path_endpoints and getattr(col, "path", None):
                p = [int(v) for v in col.path if v is not None]
                if len(p) >= 1:
                    s = int(p[0])
                    e = int(p[-1])
            # 备用：从 meta 获取
            s = s if s is not None else int(col.meta.get("start", self.depot))
            e = e if e is not None else int(col.meta.get("end", self.depot))

            if s != self.depot:
                W.add(s)
            if e != self.depot:
                W.add(e)
        return W

    # ---------- 内部：VRPTW 构建 ----------
    def _build_vrptw(self, V: List[int]):
        """
        单车 VRPTW：
            变量：
                y_{i,j} ∈ {0,1}    i≠j，边是否选
                T_i ≥ 0            到达 i 的时间
            目标：
                min ∑ c_{i,j} y_{i,j}   （c = t_ij * truck_cost_per_time）
            约束：
                1) 出入度：对所有 i∈V\{depot}，∑_j y_{i,j} = 1，∑_j y_{j,i} = 1
                   depot：∑_j y_{depot,j} = 1，∑_j y_{j,depot} = 1        （单条回路）
                2) 时间窗：e_i ≤ T_i ≤ l_i
                3) 序约束：T_j ≥ T_i + s_i + t_ij - M (1 - y_{i,j})
        说明：
            - 单车 + 时间窗的“时间传播”常可防止子环。若担心，可加 MTZ 辅助（此处暂不加，保持轻量）
        """
        m = gp.Model("TruckVRPTW")
        if not self.params.log_to_console:
            m.Params.OutputFlag = 0
        if self.params.time_limit is not None:
            m.Params.TimeLimit = self.params.time_limit
        if self.params.mip_gap is not None:
            m.Params.MIPGap = self.params.mip_gap
        m.Params.Method = 1        # Dual Simplex 更友好（小模型也可保留默认）

        # 索引与参数准备
        idx = {i: k for k, i in enumerate(V)}
        depot = self.depot
        e = {i: float(self.pb.nodes[i].tw[0]) for i in V}
        l = {i: float(self.pb.nodes[i].tw[1]) for i in V}
        s = {i: float(self.pb.nodes[i].service) for i in V}
        t = {(i, j): self._truck_time(i, j) for i in V for j in V if i != j}
        c = {(i, j): t[(i, j)] * self.params.truck_cost_per_time for i, j in t.keys()}

        # 变量
        y = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"y[{i},{j}]") for i in V for j in V if i != j}
        T = {i: m.addVar(lb=e[i], ub=l[i], vtype=GRB.CONTINUOUS, name=f"T[{i}]") for i in V}

        # 目标
        m.setObjective(gp.quicksum(c[(i, j)] * y[(i, j)] for (i, j) in y.keys()), GRB.MINIMIZE)

        # 出度/入度
        for i in V:
            if i == depot:
                m.addConstr(gp.quicksum(y[(i, j)] for j in V if j != i) == 1, name=f"out[{i}]")
                m.addConstr(gp.quicksum(y[(j, i)] for j in V if j != i) == 1, name=f"in[{i}]")
            else:
                m.addConstr(gp.quicksum(y[(i, j)] for j in V if j != i) == 1, name=f"out[{i}]")
                m.addConstr(gp.quicksum(y[(j, i)] for j in V if j != i) == 1, name=f"in[{i}]")

        # 时间传播 + 时间窗
        M = float(self.params.bigM_time)
        for i in V:
            for j in V:
                if i == j:
                    continue
                m.addConstr(T[j] >= T[i] + s[i] + t[(i, j)] - M * (1 - y[(i, j)]),
                            name=f"time[{i},{j}]")

        m.update()
        return m, y, T

    # ---------- 内部：解读取 ----------
    def _extract_solution(self, V: List[int], y, T) -> Tuple[List[int], Dict[int, float]]:
        depot = self.depot
        # 从 depot 出发沿 y=1 的边恢复序列
        next_map = {i: None for i in V}
        for (i, j), var in y.items():
            if var.X > 0.5:
                next_map[i] = j
        route = [depot]
        seen = set([depot])
        cur = depot
        while True:
            nxt = next_map.get(cur, None)
            if nxt is None:
                break
            route.append(nxt)
            if nxt == depot:
                break
            if nxt in seen:
                # 环异常，直接退出
                break
            seen.add(nxt)
            cur = nxt
        arrival = {i: float(T[i].X) for i in V}
        return route, arrival

    # ---------- 内部：距离/时间 ----------
    def _precompute_dist(self) -> None:
        # 若 problem 自带 tT 矩阵可直接用；否则用欧氏距离
        self._has_tT = hasattr(self.pb, "tT") and self.pb.tT is not None

    def _truck_time(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        # 优先使用问题给定卡车路网时间
        if self._has_tT and (i, j) in self.pb.tT:
            return float(self.pb.tT[(i, j)])
        # 否则用欧氏距离 / truck_speed
        ni = self.pb.nodes[i]
        nj = self.pb.nodes[j]
        d = math.hypot(ni.x - nj.x, ni.y - nj.y)
        return d / max(1e-9, float(self.params.truck_speed))
