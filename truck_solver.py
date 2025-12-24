# truck_solver.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterable
import re
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
    truck_speed: float =5.0              # 卡车速度（距离/时间）
    truck_cost_per_time: float = 1.0      # 每单位时间成本
    bigM_time: float = 1e5                # 时间窗/到达时间约束用的大 M
    time_limit: Optional[float] = 100.0     # 小问题求解时限（秒）
    iis_on_infeasible: bool = False         # 不可行时尝试提取 IIS
    iis_mode: str = "tw"                    # "tw" | "all"
    iis_max_nodes: int = 20                 # IIS 节点上限




@dataclass
class TruckResult:
    feasible: bool
    cost: float
    route: List[int]                      # 访问顺序（含 depot）
    arrival: Dict[int, float]             # 到达时间
    service_start: Dict[int, float] = field(default_factory=dict)  # 服务开始时间
    wait_time: Dict[int, float] = field(default_factory=dict)      # 等待时间
    conflicts: List[int] = field(default_factory=list)             # 若不可行：建议的冲突节点集合（用于可行性割）
    msg: str = ""


class TruckSolver:
    """
    给定当前选中的无人机列（sorties），抽取所有"首尾节点"形成集合 W，
    在 V = {depot} ∪ W 上建立一个小型 VRPTW（单车），最小化卡车行驶时间成本，
    支持卡车在节点处等待。返回卡车成本、访问顺序、到达时间和等待时间。
    若不可行，返回 conflicts 用于 LB-Benders 可行性割。
    """
    def __init__(self, problem, depot_idx: int = 0, params: Optional[TruckParams] = None, logger: Optional[logging.Logger] = None):
        if gp is None:
            raise RuntimeError("需要 gurobipy。请安装并配置 Gurobi（pip install gurobipy）")
        self.pb = problem
        self.depot = int(depot_idx)
        self.params = params or TruckParams()
        self.logger = logger or logging.getLogger("TruckSolver")
        self.start_depot = self.depot
        self.end_depot = None


    # ---------- 公共入口 ----------
    def evaluate(self,
                 selected_columns: Iterable,          # RMP.selected_columns 列对象集合
                 use_path_endpoints: bool = True      # 从列的 path 抽取首尾
                 ) -> TruckResult:

        W_tw = self._collect_endpoints(selected_columns, use_path_endpoints)
        start_depot = self.start_depot

        # 客户集合（不含 depot）
        customers = sorted(W_tw.keys())     # W_tw 里应该只有客户

        # 构造一个新的终点 depot 索引：保证和所有节点都不冲突
        # 这里简单用 "最大节点编号 + 1" 作为 end depot
      
        self.end_depot = 21
        end_depot = self.end_depot

        # 最终的节点集合 V：起点 + 客户 + 终点
        V = [start_depot] + customers + [end_depot]

        
        # 详细日志：数据提取结果
        self.logger.info(f"[TruckSolver.evaluate] Extracted endpoints W: {sorted(W_tw.keys())}")
        self.logger.info(f"[TruckSolver.evaluate] Endpoint time windows:")
        for node_id in sorted(W_tw.keys()):
            earliest, latest = W_tw[node_id]
            self.logger.info(f"  Node {node_id}: TW=[{earliest:.4f}, {latest:.4f}]")
        self.logger.info(f"[TruckSolver.evaluate] Vertices V (depot + endpoints): {V}")
        self.logger.info(f"[TruckSolver.evaluate] Total vertices: {len(V)}")
        
        if len(V) == 1:
            # 没有任何端点：卡车不需要出车，成本为 0
            self.logger.info("[TruckSolver.evaluate] No endpoints found, returning zero cost solution")
            return TruckResult(feasible=True, cost=0.0, route=[self.depot], arrival={self.depot: self.pb.nodes[self.depot].tw[0]},
                               conflicts=[], msg="Empty endpoints; zero truck cost.")

        try:
            self.logger.info(f"[TruckSolver.evaluate] Building VRPTW model with {len(V)} vertices...")
            model, y, T, S, W_var = self._build_vrptw(V, W_tw)
            
            # 导出LP文件用于调试
            import os
            lp_dir = "truck_logs"
            os.makedirs(lp_dir, exist_ok=True)
            lp_file = os.path.join(lp_dir, "truck_vrptw.lp")
            model.write(lp_file)
            self.logger.info(f"[TruckSolver.evaluate] LP file exported to: {lp_file}")
            
            self.logger.info("[TruckSolver.evaluate] Starting Gurobi optimization...")
            model.optimize()
            log_file = os.path.join(lp_dir, "gurobi_truck.log")
            self.logger.info(f"[TruckSolver.evaluate] Gurobi log file written to: {log_file}")
            self.logger.info(f"[TruckSolver.evaluate] Optimization finished, Status={model.Status}")
        except Exception as e:
            self.logger.error(f"[TruckSolver.evaluate] Build/solve error: {e}", exc_info=True)
            return TruckResult(False, float("inf"), [], {}, msg=f"Build/solve error: {e}")

        status = model.Status
        self.logger.info(f"[TruckSolver.evaluate] Gurobi status: {status}")
        self.logger.info(f"[TruckSolver.evaluate] Model SolCount: {model.SolCount}")
        self.logger.info(f"[TruckSolver.evaluate] Model ObjVal: {model.ObjVal if model.SolCount > 0 else 'N/A'}")
        self.logger.info(f"[TruckSolver.evaluate] Model ObjBound: {model.ObjBound}")
        
        if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            # 取到 incumbent 就认为可行（TIME_LIMIT 有可行解）
            if model.SolCount >= 1:
                route, arrival, service_start, wait_time = self._extract_solution(V, y, T, S, W_var)
                cost = float(model.ObjVal)
                self.logger.info(f"[TruckSolver.evaluate] Solution found: feasible=True, cost={cost:.6f}, route={route}")
                self.logger.info(f"[TruckSolver.evaluate] Arrival times: {arrival}")
                self.logger.info(f"[TruckSolver.evaluate] Service start times: {service_start}")
                self.logger.info(f"[TruckSolver.evaluate] Wait times: {wait_time}")
                return TruckResult(True, cost, route, arrival, service_start, wait_time, [], "")
            else:
                self.logger.warning("[TruckSolver.evaluate] Status is OPTIMAL/TIME_LIMIT but no incumbent solution")
                return TruckResult(False, float("inf"), [], {}, msg="No incumbent solution.")
        else:
            # 不可行或其他
            self.logger.error(f"[TruckSolver.evaluate] Gurobi returned infeasible or error status={status}")
            conflicts: List[int] = []
            if status == GRB.INFEASIBLE and getattr(self.params, "iis_on_infeasible", False):
                conflicts = self._compute_iis_nodes(model)
            return TruckResult(False, float("inf"), [], {}, conflicts=conflicts, msg=f"Gurobi status={status}")

    def _compute_iis_nodes(self, model) -> List[int]:
        try:
            model.computeIIS()
        except Exception as e:
            self.logger.warning("[TruckSolver.evaluate] computeIIS failed: %s", e)
            return []

        mode = getattr(self.params, "iis_mode", "tw")
        max_nodes = int(getattr(self.params, "iis_max_nodes", 0) or 0)

        nodes = set()
        pat = re.compile(r"\[([0-9]+)(?:,([0-9]+))?\]")
        for con in model.getConstrs():
            try:
                if not con.IISConstr:
                    continue
            except Exception:
                continue
            name = getattr(con, "ConstrName", "") or ""

            if mode == "tw":
                if not (name.startswith("tw_lower[") or
                        name.startswith("tw_upper[") or
                        name.startswith("service_after_arrival[")):
                    continue

            m = pat.search(name)
            if not m:
                continue
            nodes.add(int(m.group(1)))
            if m.group(2):
                nodes.add(int(m.group(2)))
            if max_nodes > 0 and len(nodes) >= max_nodes:
                break

        return sorted(nodes)

    # ---------- 内部：抽取端点 ----------
    def _collect_endpoints(self, selected_columns: Iterable, use_path_endpoints: bool) -> Dict[int, Tuple[float, float]]:
        """
        从列集合中抽取首尾节点，并重新计算其时间窗。
        
        对于每个列（无人机路径）：
          - 路径成本 = cost（包括飞行时间和所有服务时间）
          - 端点 start = path[1]（第一个服务节点）
          - 端点 end = path[-2]（最后一个服务节点）
          
        时间窗计算：
          - 端点是无人机路径的首尾节点，卡车需要在这些点接/送无人机
          - 对于起点：卡车最早到达时间 = 无人机可以从该点出发的最早时间 = meta.latest_departure + col_cost + service_times_sum
          - 对于终点：卡车最晚到达时间 = 原时间窗上界（简化）
        
        返回：{node_id: (earliest, latest), ...}
        """
        W = {}  # {node_id: (earliest, latest)}
        columns_list = list(selected_columns)
        self.logger.info(f"[_collect_endpoints] Processing {len(columns_list)} columns")
        
        for idx, col in enumerate(columns_list):
            col_info = f"Column {idx}: id={getattr(col, 'id', 'unknown')}"
            
            # 抽取路径和meta信息
            s = None; e = None
            path = getattr(col, "path", None)
            served_set = getattr(col, "served", frozenset())
            col_cost = float(getattr(col, "cost", 0.0))
            meta = getattr(col, "meta", {})
            
            self.logger.debug(f"{col_info}, path={path}, cost={col_cost:.4f}")
            
            # 从path提取首尾节点
            if path is not None:
                p = [int(v) for v in path if v is not None]
                if len(p) >= 2:
                    # path 格式: [depot, node1, node2, ..., depot]
                    # 首节点：第一个非depot的节点
                    # 尾节点：最后一个非depot的节点
                    for node in p:
                        if node != self.depot:
                            if s is None:
                                s = int(node)
                            e = int(node)
            
            # 如果从 path 没有获取到，尝试从 served_set 获取
            if s is None and served_set:
                served_list = sorted(list(served_set))
                if served_list:
                    s = int(served_list[0])
                    e = int(served_list[-1])
            
            self.logger.info(f"{col_info}, extracted endpoints: s={s}, e={e}")
            
            if s is not None and s != self.depot:
                # 计算起点的时间窗
                # 最早到达0,最晚=meta.latest_departure
                latest_departure = float(meta.get("latest_departure", self.pb.nodes[s].tw[0]))
                
                # 计算路径中所有节点的service时间
                service_time_sum = 0.0
                if served_set:
                    for node_id in served_set:
                        if node_id != self.depot:
                            service_time_sum += float(self.pb.nodes[node_id].service)
                
                earliest_arrival = 0 
                latest_arrival = latest_departure
                
                W[s] = (earliest_arrival, latest_arrival)
                self.logger.info(f"  Start endpoint {s}: TW=[{earliest_arrival:.4f}, {latest_arrival:.4f}]")
            
            if e is not None and e != self.depot and e != s:
                # 计算终点的时间窗
                # 终点与起点有相同的时间窗约束（都需要在路径的时间内服务）
                latest_departure = float(meta.get("latest_departure", self.pb.nodes[e].tw[0]))
                
                service_time_sum = 0.0
                if served_set:
                    for node_id in served_set:
                        if node_id != self.depot:
                            service_time_sum += float(self.pb.nodes[node_id].service)
                
                earliest_arrival =max(col_cost + service_time_sum, float(self.pb.nodes[e].tw[0])    ) 
                latest_arrival = float(self.pb.nodes[e].tw[1])
                
                W[e] = (earliest_arrival, latest_arrival)
                self.logger.info(f"  End endpoint {e}: TW=[{earliest_arrival:.4f}, {latest_arrival:.4f}]")
            elif e is not None and e == s:
                # 如果只有一个served节点，只需要一个endpoint
                self.logger.info(f"  Single node endpoint {s}, already added")
        
        self.logger.info(f"[_collect_endpoints] Final endpoint set W: {sorted(W.keys())}")
        return W

    # ---------- 内部：VRPTW 构建（支持等待）----------
    def _build_vrptw(self, V: List[int], W_tw: Dict[int, Tuple[float, float]]):
        """
        单车 VRPTW，支持等待：
            变量：
                y_{i,j} ∈ {0,1}    i≠j，边是否选
                T_i ≥ 0            到达 i 的时间
                S_i ≥ 0            开始服务 i 的时间
                W_i ≥ 0            在 i 处的等待时间
            目标：
                min ∑ c_{i,j} y_{i,j}   （c = t_ij * truck_cost_per_time）
            约束：
                1) 出入度：对所有 i∈V，∑_j y_{i,j} = 1，∑_j y_{j,i} = 1（单条回路）
                2) 时间窗：e_i ≤ T_i, S_i ≤ l_i 
                3) 等待：W_i = S_i - T_i ≥ 0
                4) 序约束：T_j ≥ T_i + W_i + service_i + t_ij - M (1 - y_{i,j})
        说明：
            - 卡车可以在任何节点等待，直到可以服务为止
            - 对于endpoint节点，时间窗由meta信息和列的成本确定
        """
        m = gp.Model("TruckVRPTW_with_wait")

        if self.params.time_limit is not None:
            m.Params.TimeLimit = self.params.time_limit

        m.Params.Method = 1

        start_depot = self.start_depot
        end_depot = self.end_depot
        # 对于depot，使用原始时间窗；对于endpoint，使用W_tw中的时间窗
        e = {}
        l = {}
        for i in V:
            if i == start_depot:
                e[i] = float(self.pb.nodes[i].tw[0])
                l[i] = float(self.pb.nodes[i].tw[1])
            else:
                e_val, l_val = W_tw.get(i, (self.pb.nodes[i].tw[0], self.pb.nodes[i].tw[1]))
                e[i] = float(e_val)
                l[i] = float(l_val)
        
        s = {i: float(self.pb.nodes[i].service) for i in V}
        t = {(i, j): self._truck_time(i, j) for i in V for j in V if i != j}
        c = {(i, j): t[(i, j)] * self.params.truck_cost_per_time for i, j in t.keys()}

        # 详细日志：参数信息
        self.logger.info(f"[_build_vrptw] Building VRPTW model with {len(V)} vertices (with wait support)")
        self.logger.info(f"[_build_vrptw] Vertices: {V}")
        self.logger.info(f"[_build_vrptw] Time windows and service times:")
        for node in V:
            self.logger.info(f"  Node {node}: TW=[{e[node]:.2f}, {l[node]:.2f}], service={s[node]:.2f}")
        
        self.logger.info(f"[_build_vrptw] Travel times (sample - first 5 edges):")
        for (i, j) in list(t.keys())[:5]:
            self.logger.info(f"  Edge ({i}, {j}): time={t[(i, j)]:.4f}, cost={c[(i, j)]:.4f}")

        # 变量
        y = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"y[{i},{j}]") for i in V for j in V if i != j}
        T = {i: m.addVar(lb=e[i], ub=l[i], vtype=GRB.CONTINUOUS, name=f"T[{i}]") for i in V}
        S = {i: m.addVar(lb=e[i], ub=l[i], vtype=GRB.CONTINUOUS, name=f"S[{i}]") for i in V}
        W_var = {i: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"W[{i}]") for i in V}
        
        self.logger.info(f"[_build_vrptw] Created {len(y)} binary edge variables, {len(T)} arrival vars, {len(S)} service start vars, {len(W_var)} wait vars")

        # 目标：最小化旅行成本（与等待时间无关）
        m.setObjective(gp.quicksum(c[(i, j)] * y[(i, j)] for (i, j) in y.keys()), GRB.MINIMIZE)
        self.logger.info(f"[_build_vrptw] Set objective function")

        # 出度/入度
        constr_count = 0
        for i in V:
            if i == start_depot:
                # 起点：只能发车，不能回
                m.addConstr(
                    gp.quicksum(y[(i, j)] for j in V if j != i) == 1,
                    name=f"out[{i}]"
                )
                m.addConstr(
                    gp.quicksum(y[(j, i)] for j in V if j != i) == 0,
                    name=f"in[{i}]"
                )
                constr_count += 2

            elif i == end_depot:
                # 终点：只能到达，不能再走
                m.addConstr(
                    gp.quicksum(y[(i, j)] for j in V if j != i) == 0,
                    name=f"out[{i}]"
                )
                m.addConstr(
                    gp.quicksum(y[(j, i)] for j in V if j != i) == 1,
                    name=f"in[{i}]"
                )
                constr_count += 2

            else:
                # 中间客户：保持 in = 1, out = 1
                m.addConstr(
                    gp.quicksum(y[(i, j)] for j in V if j != i) == 1,
                    name=f"out[{i}]"
                )
                m.addConstr(
                    gp.quicksum(y[(j, i)] for j in V if j != i) == 1,
                    name=f"in[{i}]"
                )
                constr_count += 2

        self.logger.info(f"[_build_vrptw] Added {constr_count} degree constraints")
        # 等待时间约束：W_i = S_i - T_i
        wait_constr_count = 0
        for i in V:
            m.addConstr(W_var[i] == S[i] - T[i], name=f"wait[{i}]")
            wait_constr_count += 1
        
        self.logger.info(f"[_build_vrptw] Added {wait_constr_count} wait time constraints")

        # 服务开始时间的时间窗约束：e_i <= S_i <= l_i
        tw_constr_count = 0
        for i in V:
            m.addConstr(S[i]>=T[i], name=f"service_after_arrival[{i}]"  )
            m.addConstr(S[i] >= e[i], name=f"tw_lower[{i}]")
            m.addConstr(S[i] <= l[i], name=f"tw_upper[{i}]")
            tw_constr_count += 2
        
        self.logger.info(f"[_build_vrptw] Added {tw_constr_count} time window constraints for service start times")

        # 时间传播 + 时间窗约束
        # T_j >= T_i + W_i + service_i + t_ij - M (1 - y_{i,j})
        M = float(self.params.bigM_time)
        time_constr_count = 0
        for i in V:
            for j in V:
                if i == j:
                    continue
                m.addConstr(T[j] >= T[i] + W_var[i] + s[i] + t[(i, j)] - M * (1 - y[(i, j)]),
                            name=f"time[{i},{j}]")
                time_constr_count += 1
        
        self.logger.info(f"[_build_vrptw] Added {time_constr_count} time propagation constraints")
        self.logger.info(f"[_build_vrptw] BigM value: {M}")

        m.update()
        self.logger.info(f"[_build_vrptw] Model updated. Total constraints: {m.NumConstrs}, Variables: {m.NumVars}")
        return m, y, T, S, W_var

    # ---------- 内部：解读取 ----------
    def _extract_solution(self, V: List[int], y, T, S, W_var) -> Tuple[List[int], Dict[int, float], Dict[int, float], Dict[int, float]]:
        self.logger.info(f"[_extract_solution] Extracting solution from model")
        depot = self.depot
        # 从 depot 出发沿 y=1 的边恢复序列
        next_map = {i: None for i in V}
        edge_count = 0
        for (i, j), var in y.items():
            if var.X > 0.5:
                next_map[i] = j
                edge_count += 1
                self.logger.debug(f"[_extract_solution] Selected edge: ({i}, {j}), var.X={var.X:.4f}")
        
        self.logger.info(f"[_extract_solution] Selected {edge_count} edges from {len(y)} total edges")
        
        route = [depot]
        seen = set([depot])
        cur = depot
        while True:
            nxt = next_map.get(cur, None)
            if nxt is None:
                self.logger.warning(f"[_extract_solution] No outgoing edge from node {cur}")
                break
            route.append(nxt)
            self.logger.debug(f"[_extract_solution] Route: {route}")
            if nxt == depot:
                break
            if nxt in seen:
                # 环异常，直接退出
                self.logger.warning(f"[_extract_solution] Cycle detected at node {nxt}, breaking")
                break
            seen.add(nxt)
            cur = nxt
        
        arrival = {i: float(T[i].X) for i in V}
        service_start = {i: float(S[i].X) for i in V}
        wait_time = {i: float(W_var[i].X) for i in V}
        
        self.logger.info(f"[_extract_solution] Final route: {route}")
        self.logger.info(f"[_extract_solution] Arrival times: {arrival}")
        self.logger.info(f"[_extract_solution] Service start times: {service_start}")
        self.logger.info(f"[_extract_solution] Wait times: {wait_time}")
        return route, arrival, service_start, wait_time


    def _truck_time(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        # 否则用欧氏距离 / truck_speed
        ni = self.pb.nodes[i]
        nj = self.pb.nodes[j]
        d = math.hypot(ni.x - nj.x, ni.y - nj.y)
        return d / max(1e-9, float(self.params.truck_speed))
