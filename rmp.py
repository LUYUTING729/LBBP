# rmlp.py
from __future__ import annotations

from dataclasses import dataclass, field
from data_model import Problem
from typing import List, Dict, Optional, Any, Sequence, Iterable, Tuple, Union
from collections import defaultdict
import logging
import os
import json
import csv
import time

# =========================
# 数据结构（与定价器/列池对齐）
# =========================

@dataclass
class Column:
    """
    一个可行路径（列）。
    - id: 唯一标识
    - path: 节点序列（用于审计与可视化）
    - served_set: 覆盖的客户集合（用于覆盖约束）
    - cost: 列成本（飞行+等待+服务等，纯无人机）
    - duration, energy: 诊断字段
    - dep_window: 可行发车区间（供后续并发/时序资源约束扩展）
    - meta: 任意元信息，比如 latest_departure、served_count 等
    """
    id: str
    path: List[int]
    served_set: frozenset
    cost: float
    duration: float = 0.0
    energy: float = 0.0
    dep_window: Optional[tuple[float, float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RMPParams:
    """
    RMP 超参数与求解器设置
    """
    # 建模
    relax: bool = True                  # True: LP 放松；False: 0-1 MIP
    lambda_uncovered: float = 1e4       # 未覆盖罚系数（越大→逼近硬覆盖）
    lambda_route: float = 10            # 选路正则（可为 0）
    max_routes: Optional[int] = None    # 预留：总路线数上限（当前通过正则化控制）

    # Logic-based Benders 相关
    bigM_truck: float = 1e5             # θ 割用的 Big-M 上界（需按实例尺度调）

    # 求解器
    time_limit: Optional[float] = None  # 秒
    mip_gap: Optional[float] = None     # 仅 MIP 时生效
    log_level: int = logging.INFO
    solver_log: bool = True             # 输出 gurobi_rmp.log
    export_lp: bool = True              # 导出 model.lp 以便调试

    # 审计输出
    outdir: str = "rmp_logs"            # 日志与产出目录
    events_log_name: str = "rmp_events.log"
    solver_log_name: str = "gurobi_rmp.log"
    columns_csv: str = "columns.csv"
    selected_csv: str = "selected_columns.csv"
    duals_csv: str = "coverage_duals.csv"
    solution_json: str = "solution.json"


# =========================
# Gurobi 后端
# =========================

class GurobiBackend:
    """
    负责与 gurobipy 交互的后端。
    - 增量加列：add_columns()
    - 对偶读取：get_duals_coverage()（仅 relax=True 时）
    - add_sri() / add_clique(): 增量加入结构性不等式
    - add_conflict_cut(): 逻辑可行性割（互斥）
    - add_theta_cut(): LBB 最优性割（下界 θ_truck）
    """
    def __init__(self):
        self.model = None
        self.customers: List[int] = []
        self.params: Optional[RMPParams] = None
        self.logger = logging.getLogger("RMP.Gurobi")

        # 主问题变量
        self.x_vars: Dict[str, Any] = {}      # 列变量 x_r
        self.s_vars: Dict[int, Any] = {}      # 覆盖松弛 s_i
        self.theta_truck = None               # ### NEW: 全局卡车成本承诺值 θ_truck

        # 约束
        self.cover_constr: Dict[int, Any] = {}  # 覆盖约束 cover_i: s_i == 1 - sum a_ir x_r，等价地我们写成 s_i == 1 并补系数

        # SRI / Clique 存根
        # SRI: key -> {'S': set(int), 'beta': float, 'con': gurobi.Constr}
        self.sri_defs: Dict[str, Dict[str, Any]] = {}
        # Clique: key -> {'Q': set(int), 'rhs': float, 'con': gurobi.Constr}
        self.clique_defs: Dict[str, Dict[str, Any]] = {}

    def build(self, customers: Sequence[int], params: RMPParams, logger: logging.Logger):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception as e:
            raise RuntimeError(
                "需要 gurobipy。请安装 Gurobi 并确保有有效许可（pip install gurobipy）。"
            ) from e

        os.makedirs(params.outdir, exist_ok=True)
        self.logger = logger or self.logger
        self.params = params
        self.customers = list(customers)

        self.model = gp.Model("RMP")
        if not params.solver_log:
            self.model.Params.OutputFlag = 0
        else:
            self.model.Params.OutputFlag = 1
            self.model.Params.LogFile = os.path.join(params.outdir, params.solver_log_name)

        if params.time_limit is not None:
            self.model.Params.TimeLimit = params.time_limit
        if (not params.relax) and (params.mip_gap is not None):
            self.model.Params.MIPGap = params.mip_gap

        self.model.ModelSense = GRB.MINIMIZE

        # ========== 变量部分 ==========

        # 覆盖松弛 s_i (>=0)，目标系数 lambda_uncovered
        for i in self.customers:
            s = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"s[{i}]")
            s.Obj = params.lambda_uncovered
            self.s_vars[i] = s

        # ### NEW: θ_truck 变量（卡车全局成本下界）
        # 连续非负，初始没有任何约束下界它，后续用 add_theta_cut() 逼它 >= 真正卡车成本
        self.theta_truck = self.model.addVar(lb=0.0,
                                             vtype=GRB.CONTINUOUS,
                                             name="theta_truck")
        # 注意：我们不直接给 theta_truck.Obj = 1.0 吗？——要的
        # 因为目标是 sum(x_r cost) + sum(lambda * s_i) + theta_truck
        self.theta_truck.Obj = 1.0

        # ========== 约束部分 ==========

        # 覆盖约束：目前是 s_i == 1.0，
        # 后续我们动态向该等式中添加 x_r 系数（chgCoeff），
        # 变成 s_i - sum_{r覆盖i} x_r = 1 - (那堆x_r)。
        # 也就是 sum a_ir x_r + s_i == 1。
        for i in self.customers:
            c = self.model.addConstr(self.s_vars[i] == 1.0, name=f"cover[{i}]")
            self.cover_constr[i] = c

        self.model.update()

        # 初始化完变量和等式后，刷新一次目标（线性组合）
        self._refresh_objective()
        self.logger.debug("GurobiBackend built: customers=%d", len(self.customers))

    # ---------- 内部工具：刷新目标函数 ----------
    def _refresh_objective(self):
        """
        把当前所有变量的 Obj 系数聚合成一个新的线性目标。
        目标 = Σ_r (x_r.Obj * x_r) + Σ_i (lambda_uncovered * s_i) + 1.0 * theta_truck
        注意：Gurobi 允许直接设置 var.Obj 后用 setObjective(sum var.Obj * var)，
        但为了安全/一致性，我们集中构建表达式。
        """
        import gurobipy as gp

        expr = gp.LinExpr()

        # x_r 部分 (每个x已经在 add_columns() 里设置了 x.Obj)
        for cid, var in self.x_vars.items():
            expr.addTerms(var.Obj, var)

        # s_i 部分 (s.Obj 在 build() 中设为 lambda_uncovered)
        for i, s in self.s_vars.items():
            expr.addTerms(s.Obj, s)

        # theta_truck
        if self.theta_truck is not None:
            expr.addTerms(self.theta_truck.Obj, self.theta_truck)

        self.model.setObjective(expr)

    # ---------- 公用：计算列在 S 中的覆盖数 κ_Sr ----------
    @staticmethod
    def _kappa_served_in_set(served_set: frozenset, S: set) -> int:
        if not S:
            return 0
        return len(served_set.intersection(S))

    # ---------- 公用：列与 Q 的交集是否达到 2（Clique） ----------
    @staticmethod
    def _covers_ge2_in_Q(served_set: frozenset, Q: set) -> int:
        if not Q:
            return 0
        return 1 if len(served_set.intersection(Q)) >= 2 else 0

    # ---------- 新增：SRI ----------
    def add_sri(self, key: str, S: Iterable[int], beta: Union[int, float]) -> None:
        """
        添加/覆盖一条 SRI: sum_r κ_{S,r} * x_r <= beta
        - key: 唯一名称（重复调用同 key 会删除旧约束再建）
        - S: 客户子集
        - beta: 上界
        会对现有所有列变量填充系数；后续 add_columns() 会自动为新增列补系数。
        """
        import gurobipy as gp

        # 如果已有同名约束，先 remove
        if key in self.sri_defs:
            try:
                self.model.remove(self.sri_defs[key]['con'])
            except Exception:
                pass
            del self.sri_defs[key]

        S = set(int(i) for i in S)
        con = self.model.addConstr(gp.LinExpr(0.0) <= float(beta), name=f"SRI[{key}]")

        # 为现有列补系数
        for col_id, var in self.x_vars.items():
            served = getattr(var, "_served_set", frozenset())
            kappa = self._kappa_served_in_set(served, S)
            if kappa:
                self.model.chgCoeff(con, var, float(kappa))

        self.model.update()
        self.sri_defs[key] = {"S": S, "beta": float(beta), "con": con}
        self.logger.info("Added SRI[%s]: |S|=%d, beta=%s", key, len(S), str(beta))

        # 目标没结构变化，不必刷新

    # ---------- 新增：Clique ----------
    def add_clique(self, key: str, Q: Iterable[int], rhs: Union[int, float] = 1) -> None:
        """
        添加/覆盖一条 Clique: sum_{r: |r ∩ Q| >= 2} x_r <= rhs
        - key: 唯一名称
        - Q: 客户子集
        """
        import gurobipy as gp

        if key in self.clique_defs:
            try:
                self.model.remove(self.clique_defs[key]['con'])
            except Exception:
                pass
            del self.clique_defs[key]

        Q = set(int(i) for i in Q)
        con = self.model.addConstr(gp.LinExpr(0.0) <= float(rhs), name=f"CLQ[{key}]")

        for col_id, var in self.x_vars.items():
            served = getattr(var, "_served_set", frozenset())
            coeff = self._covers_ge2_in_Q(served, Q)
            if coeff:
                self.model.chgCoeff(con, var, float(coeff))

        self.model.update()
        self.clique_defs[key] = {"Q": Q, "rhs": float(rhs), "con": con}
        self.logger.info("Added CLQ[%s]: |Q|=%d, rhs=%s", key, len(Q), str(rhs))
        # 目标没结构变化，不必刷新

    # ---------- NEW: LBB 可行性割 (冲突互斥) ----------
    def add_conflict_cut(self, col_ids_conflict: List[str]) -> None:
        """
        对一组列 C: sum_{r in C} x_r <= |C|-1
        用于卡车子问题判定不可同时调度的 sorties 组合。
        """
        import gurobipy as gp

        # 过滤只保留仍然存在的变量
        vars_list = [self.x_vars[cid] for cid in col_ids_conflict if cid in self.x_vars]
        if len(vars_list) <= 1:
            # 只有1个或0个，没必要建约束
            return

        lhs = gp.quicksum(vars_list)
        cname = f"conflict_{hash(tuple(sorted(col_ids_conflict)))}"
        self.model.addConstr(lhs <= len(vars_list) - 1, name=cname)
        self.model.update()

        self.logger.info("Added conflict cut on cols=%s", col_ids_conflict)
        # 目标没变，不用刷新

    # ---------- NEW: LBB 最优性割 (θ_truck 下界) ----------
    def add_theta_cut(self,
                      selected_col_ids: List[str],
                      truck_cost: float,
                      bigM: Optional[float] = None) -> None:
        """
        θ_truck + M * sum_{r in S} x_r >= truck_cost + M*|S|
        其中 S = selected_col_ids 是当前整数解中 x_r=1 的列集合，
        truck_cost 是卡车子问题对这套 S 的真实最优成本。
        bigM 取 params.bigM_truck，应该是一个足够大的常数。
        """
        import gurobipy as gp

        if bigM is None:
            if self.params is None:
                raise RuntimeError("No params to infer bigM_truck")
            bigM = float(self.params.bigM_truck)

        S_vars = [self.x_vars[cid] for cid in selected_col_ids if cid in self.x_vars]
        if not S_vars:
            # 空集合就没必要加
            return

        lhs = self.theta_truck + bigM * gp.quicksum(S_vars)
        rhs = float(truck_cost) + bigM * len(S_vars)

        cname = f"thetaCut_{hash((tuple(sorted(selected_col_ids)), round(truck_cost,3)))}"
        self.model.addConstr(lhs >= rhs, name=cname)
        self.model.update()

        self.logger.info(
            "Added theta cut for S(size=%d): truck_cost=%.6f, bigM=%.2f",
            len(S_vars), truck_cost, bigM
        )

        # 目标没有新增变量，但 θ_truck 已在目标里，仍可刷新以保持一致
        # （严格讲不需要重建目标，因为目标结构没变；不过保持调用一致没坏处）
        self._refresh_objective()
    
    def get_theta_value(self) -> float:
        try:
            return float(self.theta_truck.X)
        except Exception:
            return 0.0
    def add_columns(self, cols: List[Column], incidence: Dict[int, Dict[str, float]]):
        """
        增量添加列变量 x_r，并把 a_ir 系数 chgCoeff 到覆盖约束；
        同时为 SRI/CLQ 自动补系数。
        """
        if not cols:
            return
        import gurobipy as gp
        from gurobipy import GRB

        # 1) 新增列变量
        for col in cols:
            if col.id in self.x_vars:
                continue
            vtype = GRB.CONTINUOUS if self.params.relax else GRB.BINARY
            x = self.model.addVar(lb=0.0,
                                  ub=1.0 if self.params.relax else 1.0,
                                  vtype=vtype,
                                  name=f"x[{col.id}]")
            # 目标系数：无人机列成本 + lambda_route
            # 注意：我们此处仍然没有把卡车整体成本直接分摊到列，
            # 因为真实卡车成本是全局组合的函数，不是逐列可加的。
            x.Obj = float(col.cost) + float(self.params.lambda_route)

            # 给 SRI / CLQ 用的影子属性
            setattr(x, "_served_set", col.served_set)

            self.x_vars[col.id] = x

        self.model.update()

        # 2) 覆盖约束填系数
        # cover[i]: s_i == 1.0 目前。
        # chgCoeff(con, x_var, coeff) 会把约束改成
        #   s_i - sum_{r覆盖i} x_r = 1 - ...
        # => 等价于 sum a_ir x_r + s_i == 1
        for i in self.customers:
            con = self.cover_constr[i]
            row = incidence.get(i, {})
            for col in cols:
                coeff = row.get(col.id, 0.0)
                if coeff != 0.0:
                    self.model.chgCoeff(con, self.x_vars[col.id], coeff)

        # 3) SRI/CLQ 对新增列补系数
        if self.sri_defs:
            for key, meta in self.sri_defs.items():
                S = meta["S"]
                con = meta["con"]
                for col in cols:
                    var = self.x_vars[col.id]
                    served = getattr(var, "_served_set", frozenset())
                    kappa = self._kappa_served_in_set(served, S)
                    if kappa:
                        self.model.chgCoeff(con, var, float(kappa))

        if self.clique_defs:
            for key, meta in self.clique_defs.items():
                Q = meta["Q"]
                con = meta["con"]
                for col in cols:
                    var = self.x_vars[col.id]
                    served = getattr(var, "_served_set", frozenset())
                    coeff = self._covers_ge2_in_Q(served, Q)
                    if coeff:
                        self.model.chgCoeff(con, var, float(coeff))

        self.model.update()

        # 每次列集改变后，目标线性式结构并没有变（还是 sum var.Obj * var + ...）
        # 但为了稳态一致，我们调用一次刷新
        self._refresh_objective()

        self.logger.debug("Added %d columns. Total=%d", len(cols), len(self.x_vars))

    def solve(self) -> None:
        t0 = time.time()
        self.model.optimize()
        self._solve_time = time.time() - t0

    def get_objective_value(self) -> float:
        return float(self.model.ObjVal)

    def get_primal_values(self) -> Dict[str, float]:
        return {cid: float(var.X) for cid, var in self.x_vars.items()}

    def get_duals_coverage(self) -> Dict[int, float]:
        # 只有 LP 放松(continuous)才能稳定读取对偶
        duals = {}
        try:
            for i, con in self.cover_constr.items():
                duals[i] = float(con.Pi)
        except Exception:
            # MIP or infeasible etc.
            duals = {i: 0.0 for i in self.customers}
        return duals

    def warm_start(self, x0: Dict[str, float]) -> None:
        # MIP warm start（LP 下可忽略）
        for cid, val in x0.items():
            v = self.x_vars.get(cid)
            if v is not None:
                v.Start = float(val)

    # 导出 LP/MPS
    def export_lp(self, path: str):
        try:
            self.model.write(path)
        except Exception as e:
            self.logger.warning("Failed to export LP: %s", e)

    @property
    def solve_time(self) -> float:
        return getattr(self, "_solve_time", 0.0)


# =========================
# RMP 封装
# =========================

class RMP:
    """
    受限主问题（列生成的主问题）：
      - add_columns(): 增量加入列
      - add_sri() / add_clique(): 结构性不等式
      - add_conflict_cut(): LBB 可行性割
      - add_theta_cut(): LBB 最优性割 (下界 θ_truck)
      - solve(): 求解并刷新 x/π
      - get_duals(): 覆盖约束对偶价（π[i]）
      - dump_state(): 导出结构化日志文件
    """
    def __init__(self,
                 problem: Problem,
                 params: Optional[RMPParams] = None,
                 outdir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        - problem: 当前问题（含 nodes / customers / drone）
        - params: 超参数与日志设置
        - outdir: 覆盖 params.outdir
        - logger: 可重用外部 logger；不传则建文件日志
        """
        self.problem: Problem = problem
        self.customers: List[int] = list(problem.customers)

        # 参数与输出目录
        self.params = params or RMPParams()
        if outdir is not None:
            self.params.outdir = outdir

        os.makedirs(self.params.outdir, exist_ok=True)

        # 业务日志
        self.logger = logger or logging.getLogger(f"RMP[{id(self)}]")
        self.logger.setLevel(self.params.log_level)
        need_file_handler = True
        for h in self.logger.handlers:
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "").endswith(self.params.events_log_name):
                need_file_handler = False
                break
        if need_file_handler:
            fh = logging.FileHandler(
                os.path.join(self.params.outdir, self.params.events_log_name),
                mode="w",
                encoding="utf-8"
            )
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)

        # 覆盖矩阵
        self.columns: Dict[str, Column] = {}
        self.incidence: Dict[int, Dict[str, float]] = {i: {} for i in self.customers}

        # 后端
        self.backend = GurobiBackend()
        self.backend.build(self.customers, self.params, self.logger)

        # 状态量
        self._objective_value: float = 0.0
        self._x_values: Dict[str, float] = {}
        self._duals: Dict[int, float] = {}

        self.logger.info("RMP initialized. customers=%d, relax=%s, outdir=%s",
                         len(self.customers), self.params.relax, self.params.outdir)

    # ---------- 业务接口：增量加入结构性不等式 ----------
    def add_sri(self, key: str, S: Iterable[int], beta: Union[int, float]) -> None:
        self.backend.add_sri(key, S, beta)
        if self.params.export_lp:
            self.backend.export_lp(os.path.join(self.params.outdir, "model.lp"))

    def add_clique(self, key: str, Q: Iterable[int], rhs: Union[int, float] = 1) -> None:
        self.backend.add_clique(key, Q, rhs)
        if self.params.export_lp:
            self.backend.export_lp(os.path.join(self.params.outdir, "model.lp"))

    # ---------- NEW: LBB cuts ----------
    def add_conflict_cut(self, col_ids_conflict: List[str]) -> None:
        """
        可行性割：sum_{r in C} x_r <= |C|-1
        """
        self.backend.add_conflict_cut(col_ids_conflict)
        if self.params.export_lp:
            self.backend.export_lp(os.path.join(self.params.outdir, "model.lp"))

    def add_theta_cut(self,
                      selected_col_ids: List[str],
                      truck_cost: float,
                      bigM: Optional[float] = None) -> None:
        """
        最优性割（Benders optimality cut）：
        θ_truck + M * sum_{r in S} x_r >= truck_cost + M|S|
        """
        self.backend.add_theta_cut(
            selected_col_ids=selected_col_ids,
            truck_cost=truck_cost,
            bigM=bigM if bigM is not None else self.params.bigM_truck
        )
        if self.params.export_lp:
            self.backend.export_lp(os.path.join(self.params.outdir, "model.lp"))

    # ---------- 列操作 ----------
    def add_columns(self, cols: List[Column]) -> None:
        if not cols:
            return
        new_cols = []
        for c in cols:
            if c.id in self.columns:
                continue
            self.columns[c.id] = c
            for i in self.customers:
                if i in c.served_set:
                    self.incidence[i][c.id] = 1.0
            new_cols.append(c)

        if new_cols:
            self.backend.add_columns(new_cols, self.incidence)
            # 写入 columns.csv（以 "w" 打开保持你原来的行为）
            self._append_columns_csv(new_cols)
            self.logger.info("Added %d new columns. Total=%d",
                             len(new_cols), len(self.columns))

    def solve(self) -> float:
        self.logger.info("Solving RMP ...")
        if self.params.export_lp:
            self.backend.export_lp(os.path.join(self.params.outdir, "model.lp"))

        self.backend.solve()
        self._objective_value = self.backend.get_objective_value()
        self._x_values = self.backend.get_primal_values()
        self._duals = self.backend.get_duals_coverage()

        # ---- 诊断信息：对偶 key 一致性 ----
        missing = set(self.customers) - set(self._duals.keys())
        extra   = set(self._duals.keys()) - set(self.customers)
        if missing or extra:
            self.logger.error(
                "[DIAG] Dual-key mismatch. missing=%s extra=%s",
                sorted(missing), sorted(extra)
            )
        else:
            avg_dual = sum(self._duals.values())/max(1, len(self._duals))
            self.logger.info("[DIAG] Dual-key OK. |duals|=%d avg=%.6f",
                             len(self._duals), avg_dual)

        # ---- 诊断信息：RC 一致性抽检 ----
        try:
            model = getattr(self.backend, "model", None)
            xmap  = getattr(self.backend, "x_vars", {})
            if model and xmap:
                sample = list(self.columns.items())[:5]
                for col_id, col in sample:
                    var = xmap.get(col_id)
                    if var is None:
                        continue
                    pisum = sum(self._duals.get(i, 0.0) for i in col.served_set)
                    rc_theory = (col.cost + self.params.lambda_route) - pisum
                    rc_model  = float(getattr(var, "RC", 0.0))
                    self.logger.info("[DIAG] RC col=%s theory=%.6f model=%.6f diff=%.2e",
                                     col_id, rc_theory, rc_model,
                                     abs(rc_theory-rc_model))
        except Exception as e:
            self.logger.warning("[DIAG] RC check skipped: %s", e)

        # ---- 基础统计 & 导出 ----
        picked = {cid: x for cid, x in self._x_values.items() if x > 1e-8}
        cover_avg_dual = sum(self._duals.values()) / max(1, len(self._duals))
        self.logger.info(
            "Solved: obj=%.6f, picked=%d/%d, runtime=%.3fs, avg_dual=%.4f",
            self._objective_value, len(picked), len(self.columns),
            self.backend.solve_time, cover_avg_dual
        )

        self._write_selected_csv(picked)
        self._write_duals_csv()
        self._write_solution_json(picked)

        return self._objective_value

    # ---------- 导出 ----------
    def dump_state(self) -> None:
        """
        手动触发完整导出（可在每轮列生成之后调用）
        """
        picked = {cid: x for cid, x in self._x_values.items() if x > 1e-8}
        self._write_selected_csv(picked)
        self._write_duals_csv()
        self._write_solution_json(picked)
        self.logger.info("State dumped to %s", self.params.outdir)

    def get_duals(self) -> Dict[int, float]:
        if not self.params.relax:
            self.logger.warning(
                "Model is integer; duals are not meaningful. Returning zeros."
            )
        return dict(self._duals)

    def get_selected_columns(self, threshold: float = 1e-6) -> List[Column]:
        return [self.columns[cid]
                for cid, x in self._x_values.items()
                if x > threshold]

    def get_solution_vector(self) -> Dict[str, float]:
        return dict(self._x_values)

    def get_theta_truck(self) -> float:
        """
        Return current θ_truck value; safe fallback to 0.0 if unavailable.
        """
        try:
            return float(self.backend.get_theta_value())
        except Exception:
            try:
                theta = getattr(self.backend, "theta_truck", None)
                return float(theta.X) if theta is not None else 0.0
            except Exception:
                return 0.0

    def get_objective_value(self) -> float:
        return float(self._objective_value)

    def warm_start(self, x0: Dict[str, float]) -> None:
        self.backend.warm_start(x0)

    # ---------- 工具：从 label 生成列 ----------
    @staticmethod
    def make_columns_from_labels(
        labels: Sequence[Any],
        id_prefix: str = "r"
    ) -> List[Column]:
        cols: List[Column] = []
        for k, lab in enumerate(labels):
            served = set(getattr(lab, "covered_set", set())) \
                     or {v for v in getattr(lab, "path", []) if v != 0}
            cols.append(Column(
                id=f"{id_prefix}{k}",
                path=list(getattr(lab, "path", [])),
                served_set=frozenset(served),
                cost=float(getattr(lab, "cost", 0.0)),
                duration=float(getattr(lab, "duration", 0.0))
                         if hasattr(lab, "duration") else 0.0,
                energy=float(getattr(lab, "energy", 0.0))
                       if hasattr(lab, "energy") else 0.0,
                dep_window=getattr(lab, "dep_window", None),
                meta={
                    "latest_departure": getattr(lab, "latest_departure", None),
                    "served_count": getattr(lab, "served_count", len(served))
                }
            ))
        return cols

    # =========================
    # 内部：结构化日志导出
    # =========================

    def _append_columns_csv(self, cols: List[Column]) -> None:
        path = os.path.join(self.params.outdir, self.params.columns_csv)
        exists = os.path.exists(path)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow([
                    "id",
                    "cost",
                    "len_path",
                    "served_size",
                    "served",
                    "duration",
                    "energy",
                    "dep_window",
                    "meta"
                ])
            for c in cols:
                w.writerow([
                    c.id,
                    f"{c.cost:.6f}",
                    len(c.path),
                    len(c.served_set),
                    ";".join(map(str, sorted(c.served_set))),
                    f"{c.duration:.6f}",
                    f"{c.energy:.6f}",
                    "" if c.dep_window is None else
                    f"[{c.dep_window[0]:.3f},{c.dep_window[1]:.3f}]",
                    json.dumps(c.meta, ensure_ascii=False)
                ])

    def _write_selected_csv(self, picked: Dict[str, float]) -> None:
        path = os.path.join(self.params.outdir, self.params.selected_csv)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "x_value", "cost", "served_size", "served", "meta"])
            for cid, x in sorted(picked.items(), key=lambda kv: -kv[1]):
                c = self.columns[cid]
                w.writerow([
                    cid,
                    f"{x:.6f}",
                    f"{c.cost:.6f}",
                    len(c.served_set),
                    ";".join(map(str, sorted(c.served_set))),
                    json.dumps(c.meta, ensure_ascii=False)
                ])

    def _write_duals_csv(self) -> None:
        path = os.path.join(self.params.outdir, self.params.duals_csv)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["customer", "dual_pi"])
            for i in sorted(self._duals.keys()):
                w.writerow([i, f"{self._duals[i]:.6f}"])

    def _write_solution_json(self, picked: Dict[str, float]) -> None:
        sol = {
            "objective": self._objective_value,
            "picked": [
                {
                    "id": cid,
                    "x": float(x),
                    "cost": float(self.columns[cid].cost),
                    "served": sorted(map(int, self.columns[cid].served_set)),
                    "meta": self.columns[cid].meta
                }
                for cid, x in picked.items()
            ],
            "duals": {int(k): float(v) for k, v in self._duals.items()}
        }
        path = os.path.join(self.params.outdir, self.params.solution_json)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sol, f, ensure_ascii=False, indent=2)
