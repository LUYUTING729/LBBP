# rmlp.py - 受限主问题(Restricted Master Problem)求解器
# 主要用于列生成算法中的主问题求解,包含以下核心功能:
# 1. 数据结构定义(Column类和RMPParams类)
# 2. Gurobi求解器后端封装
# 3. RMP类作为主要接口

from __future__ import annotations

from dataclasses import dataclass, field
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
    列/路径数据结构,代表一条可行的配送路径
    
    属性说明:
    - id: 列的唯一标识符
    - path: 访问节点的序列,例如[0,1,2,0]表示从0出发访问1、2后返回0
    - served_set: 该路径服务的客户集合,用frozenset存储以保证不可变
    - cost: 路径的总成本,包括行驶成本、等待成本等
    - duration: 路径的总时长
    - energy: 路径的总能耗
    - dep_window: 路径的可行发车时间窗,格式为(earliest, latest)
    - meta: 额外的元数据字典,可以存储如最晚出发时间等信息
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
    RMP求解器的参数配置类
    
    建模参数:
    - relax: 是否采用LP松弛。True表示求解LP，False表示求解整数规划
    - lambda_uncovered: 未覆盖客户的惩罚系数,越大越趋近于硬约束
    - lambda_route: 路径数量的正则化系数,用于控制使用的路径数量
    - max_routes: 可选的路径数量上限约束
    
    求解器设置:
    - time_limit: 求解时间限制(秒)
    - mip_gap: MIP求解的收敛间隙
    - log_level: 日志输出级别
    - solver_log: 是否输出求解器日志
    - export_lp: 是否导出LP文件用于调试
    
    输出设置:
    - outdir: 输出文件目录
    - events_log_name: 事件日志文件名
    - solver_log_name: 求解器日志文件名 
    - columns_csv: 列信息CSV文件名
    - selected_csv: 被选中列的CSV文件名
    - duals_csv: 对偶值CSV文件名
    - solution_json: 完整解JSON文件名
    """
    # ...existing code...

# =========================
# Gurobi 后端
# =========================

class GurobiBackend:
    """
    Gurobi求解器的封装类,处理与Gurobi的所有交互
    
    主要功能:
    1. 构建和维护Gurobi模型
    2. 动态添加列(路径变量)
    3. 求解模型并获取结果
    4. 提取对偶值(仅LP模式)
    
    关键组件:
    - model: Gurobi模型对象
    - x_vars: 列变量字典 {col_id: gurobi_var}
    - s_vars: 覆盖松弛变量字典 {customer_id: gurobi_var}  
    - cover_constr: 覆盖约束字典 {customer_id: gurobi_constr}
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
    受限主问题(RMP)的主接口类,提供完整的问题求解功能
    
    核心功能:
    1. 列管理:
       - add_columns(): 动态添加新列
       - make_columns_from_labels(): 从路径标签构建列
    
    2. 求解控制:
       - solve(): 求解当前RMP
       - warm_start(): 提供热启动解
       
    3. 结果获取:
       - get_duals(): 获取覆盖约束的对偶值
       - get_selected_columns(): 获取被选中的列
       - get_solution_vector(): 获取决策变量值
       - get_objective_value(): 获取目标函数值
       
    4. 结果输出:
       - dump_state(): 导出完整的求解状态
       - 自动生成多个CSV/JSON格式的详细日志文件
    
    关键属性:
    - columns: 所有列的字典 {col_id: Column}
    - incidence: 覆盖关系矩阵(稀疏表示) {customer_id: {col_id: coef}}
    - backend: Gurobi求解器后端
    - params: 求解器参数配置
    """
    def __init__(self, customers: Iterable[int], params: Optional[RMPParams] = None, logger: Optional[logging.Logger] = None):
        self.customers = list(customers)
        self.params = params or RMPParams()
        os.makedirs(self.params.outdir, exist_ok=True)

        # 业务日志
        self.logger = logger or logging.getLogger("RMP")
        self.logger.setLevel(self.params.log_level)
        # 独立文件日志
        fh = logging.FileHandler(os.path.join(self.params.outdir, self.params.events_log_name), mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(fh)

        # 覆盖矩阵（稀疏）
        self.columns: Dict[str, Column] = {}
        self.incidence: Dict[int, Dict[str, float]] = {i: {} for i in self.customers}

        # 后端（Gurobi）
        self.backend = GurobiBackend()
        self.backend.build(self.customers, self.params, self.logger)

        self._objective_value: float = 0.0
        self._x_values: Dict[str, float] = {}
        self._duals: Dict[int, float] = {}

        self.logger.info("RMP initialized. customers=%d, relax=%s", len(self.customers), self.params.relax)

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
