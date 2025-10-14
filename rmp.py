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
    - cost: 列成本（飞行+等待+服务等）
    - dep_window: 可选的可行发车区间（供并发/时段约束扩展）
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
    relax: bool = True                 # True: LP 放松；False: 0-1 MIP
    lambda_uncovered: float = 1e4      # 未覆盖罚系数（越大→逼近硬覆盖）
    lambda_route: float = 0.0          # 选路正则（可为 0）
    max_routes: Optional[int] = None   # 预留：总路线数上限（当前通过正则化控制）

    # 求解器
    time_limit: Optional[float] = None # 秒
    mip_gap: Optional[float] = None    # 仅 MIP 时生效
    log_level: int = logging.INFO
    solver_log: bool = True            # 输出 gurobi_rmp.log
    export_lp: bool = True             # 导出 model.lp 以便调试

    # 审计输出
    outdir: str = "rmp_logs"           # 日志与产出目录
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
    """
    def __init__(self):
        self.model = None
        self.customers: List[int] = []
        self.params: Optional[RMPParams] = None
        self.logger = logging.getLogger("RMP.Gurobi")
        self.x_vars: Dict[str, Any] = {}   # 列变量 x_r
        self.s_vars: Dict[int, Any] = {}   # 覆盖松弛 s_i
        self.cover_constr: Dict[int, Any] = {}  # 覆盖约束 cover_i: sum a_ir x_r + s_i >= 1

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

        # 变量：覆盖松弛 s_i
        for i in self.customers:
            s = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"s[{i}]")
            s.Obj = params.lambda_uncovered
            self.s_vars[i] = s

        # 覆盖约束：s_i >= 1  （后续通过 chgCoeff 动态加入 x_r 系数）
        for i in self.customers:
            c = self.model.addConstr(self.s_vars[i] >= 1.0, name=f"cover[{i}]")
            self.cover_constr[i] = c

        # 保持目标只由变量的 Obj 系数构成，无需 setObjective
        self.model.update()

        self.logger.debug("GurobiBackend built: customers=%d", len(self.customers))

    def add_columns(self, cols: List[Column], incidence: Dict[int, Dict[str, float]]):
        """增量添加列变量 x_r，并把 a_ir 系数 chgCoeff 到覆盖约束"""
        if not cols: 
            return
        import gurobipy as gp
        from gurobipy import GRB

        for col in cols:
            if col.id in self.x_vars:
                continue
            vtype = GRB.CONTINUOUS if self.params.relax else GRB.BINARY
            x = self.model.addVar(lb=0.0, ub=1.0 if self.params.relax else 1.0,
                                  vtype=vtype, name=f"x[{col.id}]")
            # 目标：cost + lambda_route
            x.Obj = float(col.cost) + float(self.params.lambda_route)
            self.x_vars[col.id] = x

        self.model.update()
        # 填系数
        for i in self.customers:
            con = self.cover_constr[i]
            row = incidence.get(i, {})
            for col in cols:
                coeff = row.get(col.id, 0.0)
                if coeff != 0.0:
                    self.model.chgCoeff(con, self.x_vars[col.id], coeff)

        self.model.update()
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
        # 只有 LP 放松才能读取对偶
        duals = {}
        try:
            for i, con in self.cover_constr.items():
                duals[i] = float(con.Pi)
        except Exception:
            # 非 LP 或未最优等情况
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
        新签名：
          - problem: 当前问题（含 nodes / customers / drone）
          - params: 仍可传入；不传则用默认
          - outdir: 若提供，则覆盖 params.outdir，用于该 RMP 的日志与产出目录
          - logger: 可复用外部 logger；不传则创建文件日志到 outdir/events_log_name
        """
        # 保存 problem 并展开客户集合
        self.problem: Problem = problem
        self.customers: List[int] = list(problem.customers)

        # 参数与输出目录
        self.params = params or RMPParams()
        if outdir is not None:
            self.params.outdir = outdir  # ★ 允许从构造器覆盖输出目录

        os.makedirs(self.params.outdir, exist_ok=True)

        # 业务日志
        # 注意避免重复添加同一 FileHandler（如果创建多个 RMP 实例）
        self.logger = logger or logging.getLogger(f"RMP[{id(self)}]")
        self.logger.setLevel(self.params.log_level)
        need_file_handler = True
        for h in self.logger.handlers:
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "").endswith(self.params.events_log_name):
                need_file_handler = False
                break
        if need_file_handler:
            fh = logging.FileHandler(os.path.join(self.params.outdir, self.params.events_log_name), mode="w", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)

        # 覆盖矩阵（稀疏）
        self.columns: Dict[str, Column] = {}
        self.incidence: Dict[int, Dict[str, float]] = {i: {} for i in self.customers}

        # 后端（Gurobi）
        self.backend = GurobiBackend()
        # 仍沿用“以 customers + params + logger 创建模型”的接口；如你后端需要 Problem，可自行扩展
        self.backend.build(self.customers, self.params, self.logger)

        # 状态量
        self._objective_value: float = 0.0
        self._x_values: Dict[str, float] = {}
        self._duals: Dict[int, float] = {}

        self.logger.info("RMP initialized. customers=%d, relax=%s, outdir=%s",
                         len(self.customers), self.params.relax, self.params.outdir)
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
            # 写入 columns.csv（追加）
            self._append_columns_csv(new_cols)
            self.logger.info("Added %d new columns. Total=%d", len(new_cols), len(self.columns))

    def solve(self) -> float:
        self.logger.info("Solving RMP ...")
        if self.params.export_lp:
            self.backend.export_lp(os.path.join(self.params.outdir, "model.lp"))

        self.backend.solve()
        self._objective_value = self.backend.get_objective_value()
        self._x_values = self.backend.get_primal_values()
        self._duals = self.backend.get_duals_coverage()
        # --- 放在 RMP.solve() 获取 self._duals 之后 ---
        # 1) dual keys 一致性
        missing = set(self.customers) - set(self._duals.keys())
        extra   = set(self._duals.keys()) - set(self.customers)
        if missing or extra:
            self.logger.error("[DIAG] Dual-key mismatch. missing=%s extra=%s", sorted(missing), sorted(extra))
        else:
            self.logger.info("[DIAG] Dual-key OK. |duals|=%d avg=%.6f", len(self._duals), sum(self._duals.values())/max(1,len(self._duals)))

        # 2) RC 一致性抽检（Gurobi.Var.RC vs 手算 rc）
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
                    self.logger.info("[DIAG] RC col=%s theory=%.6f model=%.6f diff=%.2e", col_id, rc_theory, rc_model, abs(rc_theory-rc_model))
        except Exception as e:
            self.logger.warning("[DIAG] RC check skipped: %s", e)


        # 基础统计
        picked = {cid: x for cid, x in self._x_values.items() if x > 1e-8}
        cover_avg_dual = sum(self._duals.values()) / max(1, len(self._duals))
        self.logger.info(
            "Solved: obj=%.6f, picked=%d/%d, runtime=%.3fs, avg_dual=%.4f",
            self._objective_value, len(picked), len(self.columns), self.backend.solve_time, cover_avg_dual
        )
        # 导出结果
        self._write_selected_csv(picked)
        self._write_duals_csv()
        self._write_solution_json(picked)
        return self._objective_value

    # ---------- 导出 ----------
    def dump_state(self) -> None:
        """
        手动触发完整导出（可在每轮列生成之后调用）
        """
        self._write_selected_csv({cid: x for cid, x in self._x_values.items() if x > 1e-8})
        self._write_duals_csv()
        self._write_solution_json({cid: x for cid, x in self._x_values.items() if x > 1e-8})
        self.logger.info("State dumped to %s", self.params.outdir)

    def get_duals(self) -> Dict[int, float]:
        if not self.params.relax:
            self.logger.warning("Model is integer; duals are not meaningful. Returning zeros.")
        return dict(self._duals)

    def get_selected_columns(self, threshold: float = 1e-6) -> List[Column]:
        return [self.columns[cid] for cid, x in self._x_values.items() if x > threshold]

    def get_solution_vector(self) -> Dict[str, float]:
        return dict(self._x_values)

    def get_objective_value(self) -> float:
        return float(self._objective_value)

    def warm_start(self, x0: Dict[str, float]) -> None:
        self.backend.warm_start(x0)

    # ---------- 工具：从 label 生成列 ----------
    @staticmethod
    def make_columns_from_labels(labels: Sequence[Any], id_prefix: str = "r") -> List[Column]:
        cols: List[Column] = []
        for k, lab in enumerate(labels):
            served = set(getattr(lab, "covered_set", set())) or {v for v in getattr(lab, "path", []) if v != 0}
            cols.append(Column(
                id=f"{id_prefix}{k}",
                path=list(getattr(lab, "path", [])),
                served_set=frozenset(served),
                cost=float(getattr(lab, "cost", 0.0)),
                duration=float(getattr(lab, "duration", 0.0)) if hasattr(lab, "duration") else 0.0,
                energy=float(getattr(lab, "energy", 0.0)) if hasattr(lab, "energy") else 0.0,
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
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["id", "cost", "len_path", "served_size", "served",
                            "duration", "energy", "dep_window", "meta"])
            for c in cols:
                w.writerow([
                    c.id,
                    f"{c.cost:.6f}",
                    len(c.path),
                    len(c.served_set),
                    ";".join(map(str, sorted(c.served_set))),
                    f"{c.duration:.6f}",
                    f"{c.energy:.6f}",
                    "" if c.dep_window is None else f"[{c.dep_window[0]:.3f},{c.dep_window[1]:.3f}]",
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
                } for cid, x in picked.items()
            ],
            "duals": {int(k): float(v) for k, v in self._duals.items()}
        }
        path = os.path.join(self.params.outdir, self.params.solution_json)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sol, f, ensure_ascii=False, indent=2)
