# bp_drones.py
from __future__ import annotations

import os
import time
import json
import heapq
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterable, Sequence, Tuple, Set

# 依赖你的现有模块
from data_model import Problem
from label_setting import label_setting
from rmp import RMP, RMPParams, Column
from column_pool import ColumnPool

import os, json, time, math, csv

# =========================
# 参数、状态、结果
# =========================

@dataclass
class BPParams:
    # ---- 列生成（CG）参数 ----
    max_iterations: int = 30
    rc_tolerance: float = -1e-6          # 负约简成本阈值（< 0 即有改进）
    pricing_batch: int = 1               # 每轮最多“批次”列生成（多次过滤后再解 RMP）
    stabilize_duals: bool = False        # 对偶稳定化开关
    stabilize_alpha: float = 0.7         # 稳定化权重 (π^st = α π + (1-α) π_anchor)

    # ---- Label-setting 相关（定价器内部使用）----
    max_label_len: int = 4
    k_per_sig: int = 5
    eps_dom: float = 0.05
    depot_idx: int = 0                   # -1 表示不尝试回仓

    # ---- RMP 参数（Gurobi 后端）----
    rmp_params: RMPParams = field(default_factory=RMPParams)

    # ---- 分支设置 ----
    enable_branch: bool = True
    branch_strategy: str = "most_fractional"  # "most_fractional" / "first_fractional"
    max_nodes: int = 100

    # ---- 终止条件 ----
    time_limit: Optional[float] = None
    no_improve_rounds: int = 3

    # ---- 日志 ----
    outdir: str = "bp_logs"
    log_level: int = logging.INFO
    dump_every_iter: bool = True         # 每轮导出结构化快照


@dataclass
class BPStatus:
    iteration: int
    node_id: int
    obj_value: float
    best_bound: float
    gap: float
    num_columns: int
    num_new_columns: int
    time_elapsed: float
    rc_min: float
    # ---- 新增扩展指标 ----
    coverage_rate: float               # 已覆盖客户占比
    avg_dual: float                    # 对偶均值
    max_dual: float                    # 对偶最大值
    dual_entropy: float                # 对偶分布熵，用于衡量分布均匀性


@dataclass
class BPResult:
    status: str                                   # "optimal" / "stopped" / "time_limit"
    obj_value: float
    solution: Dict[str, float]                    # 列解向量 x_r
    selected_columns: List[Column]
    duals: Dict[int, float]
    stats: List[BPStatus]


# =========================
# 定价器（基于 label_setting）
# =========================

class LabelSettingGenerator:
    """
    Column Generator（定价器）：
      - 调用 label_setting 生成候选路径
      - 计算 reduced cost = 原始 cost - sum(π_i * a_ir) (+ λ_route 若计入RMP目标)
      - 过滤出 rc < rc_tolerance 的列，最多返回 budget 个
      - 维护去重缓存（避免重复路径）
    """
    def __init__(self, problem: Problem, params: BPParams, logger: logging.Logger):
        self.problem = problem
        self.params = params
        self.logger = logger
        self._seen_keys: Set[tuple] = set()   # 路径去重：以 path tuple 作为 key
        self._gen_call = 0                    # 调用计数用于列 id 前缀

    def _labels_to_columns(self, labels: Sequence[Any], id_prefix: str) -> List[Column]:
        # 复用 RMP 的工具函数
        cols = RMP.make_columns_from_labels(labels, id_prefix=id_prefix)
        return cols

    def _reduced_cost(self, col: Column, duals: Dict[int, float], lambda_route: float) -> float:
        pi_sum = sum(duals.get(i, 0.0) for i in col.served_set)
        rc = (col.cost + lambda_route) - pi_sum
        return rc

    def generate(self, duals: Dict[int, float], budget: int, lambda_route: float) -> List[Column]:
        self._gen_call += 1
        id_prefix = f"it{self._gen_call}_"
        # 1) 生成候选路径（不带对偶偏置，靠后筛选）
        labels = label_setting(
            problem=self.problem,
            max_len=self.params.max_label_len,
            depot_idx=self.params.depot_idx,
            logger=self.logger,
            K_per_sig=self.params.k_per_sig,
            eps=self.params.eps_dom,
            duals=duals,               # 这里不带对偶（靠后筛选）
        )
        # 2) 转列并计算约简成本
        candidates: List[Tuple[float, Column]] = []
        for col in self._labels_to_columns(labels, id_prefix=id_prefix):
            key = tuple(col.path)
            if key in self._seen_keys:
                continue
            rc = self._reduced_cost(col, duals, lambda_route)
            if rc < self.params.rc_tolerance:
                col.meta["rc"] = float(rc)
                candidates.append((rc, col))
        # 3) 选出最有希望的若干（按 rc 从小到大）
        candidates.sort(key=lambda x: x[0])
        picked: List[Column] = []
        for rc, col in candidates:
            picked.append(col)
            self._seen_keys.add(tuple(col.path))
            if len(picked) >= budget:
                break
        self.logger.info("Pricing: generated %d labels, %d negative-rc candidates, return %d.",
                         len(labels), len(candidates), len(picked))
        return picked


# =========================
# 分支节点
# =========================

@dataclass(order=True)
class BranchNode:
    # 优先队列按 lower_bound 排序（小优先）
    lower_bound: float
    id: int = field(compare=False)
    depth: int = field(compare=False, default=0)
    fixed_one: Set[str] = field(compare=False, default_factory=set)   # x_r = 1 的列 id
    fixed_zero: Set[str] = field(compare=False, default_factory=set)  # x_r = 0 的列 id
    rmp: Optional[RMP] = field(compare=False, default=None)
    parent_obj: float = field(compare=False, default=float("inf"))


# =========================
# 主控器：BPSolver
# =========================

class BPSolver:
    """
    分支定价主循环：
      - Best-bound 节点选择
      - 节点内执行：RMP ↔ 定价（列生成）直到无负 rc
      - 若解非整数 → 分支（对变量 x_r：=1 / =0）
      - 记录过程与导出日志
    """
    def __init__(self, problem: Problem, params: Optional[BPParams] = None, logger: Optional[logging.Logger] = None):
        self.problem = problem
        self.params = params or BPParams()
        os.makedirs(self.params.outdir, exist_ok=True)
        self.logger = self._init_logger(logger, self.params)
        self.cg = LabelSettingGenerator(problem, self.params, self.logger)
        self.node_counter = 0
        self.stats: List[BPStatus] = []
        self.global_best_obj = float("inf")
        self.global_best_solution: Dict[str, float] = {}
        self.global_best_cols: List[Column] = []
        self.dual_anchor: Optional[Dict[int, float]] = None
        self.start_time = time.time()
        self.pool = ColumnPool(logger=self.logger)

    # ---- 外部入口 ----
    def solve(self, init_columns: Optional[List[Column]] = None) -> BPResult:
        # 初始根节点
        root = self._make_node(depth=0)
        # 可选：先加入一批初始列（比如来自启发式）
        if init_columns:
            root.rmp.add_columns(init_columns)

        # 优先队列（best bound）
        pq: List[BranchNode] = []
        heapq.heappush(pq, root)

        explored = 0
        iteration = 0
        while pq and explored < self.params.max_nodes:
            node = heapq.heappop(pq)
            explored += 1
            iteration += 1
            self.logger.info("=" * 60)
            self.logger.info("[Node %d | depth %d] Start. fixed_one=%d, fixed_zero=%d",
                             node.id, node.depth, len(node.fixed_one), len(node.fixed_zero))

            # 在该节点做列生成迭代
            obj, rc_min, new_cols_total = self._column_generation_loop(node)

            # 记录/导出
            self._record(iteration, node, obj, rc_min, new_cols_total)

            # 时间/终止检查
            if self._timed_out():
                self.logger.warning("Time limit reached. Stop.")
                break

            # 检查整数性
            x_vals = node.rmp.get_solution_vector()
            fractional = [(cid, x) for cid, x in x_vals.items() if 1e-6 < x < 1 - 1e-6]
            slacks = [v for k, v in node.rmp.get_duals().items()]  # 对偶不用于整数校验，这里仅保留接口

            if not fractional:
                # 整数可行（在 LP 放松下，x ∈ {0,1}）
                if obj < self.global_best_obj:
                    self.global_best_obj = obj
                    self.global_best_solution = x_vals
                    self.global_best_cols = node.rmp.get_selected_columns()
                self.logger.info("[Node %d] Integer solution found. obj=%.6f", node.id, obj)
                continue

            # 若没有负 rc 且非整数 → 分支
            if rc_min >= self.params.rc_tolerance:
                if not self.params.enable_branch:
                    self.logger.info("[Node %d] No negative rc and branching disabled. Prune.", node.id)
                    continue
                # 选分支变量
                cid, x = self._choose_branch_var(fractional)
                self.logger.info("[Node %d] Branch on variable %s = %.4f.", node.id, cid, x)
                # 左子树：x_cid = 1
                left = self._make_child(node, fix_one={cid})
                # 右子树：x_cid = 0
                right = self._make_child(node, fix_zero={cid})
                heapq.heappush(pq, left)
                heapq.heappush(pq, right)

        status = "optimal" if self.global_best_obj < float("inf") else "stopped"
        return BPResult(
            status=status,
            obj_value=self.global_best_obj if status == "optimal" else float("inf"),
            solution=self.global_best_solution,
            selected_columns=self.global_best_cols,
            duals={},  # 最终节点对偶通常不再关注，这里可留空或填最后节点的
            stats=self.stats
        )

    # ---- 节点构建与复制 ----
    def _make_node(self, depth: int, fixed_one: Optional[Iterable[str]] = None,
                   fixed_zero: Optional[Iterable[str]] = None) -> BranchNode:
        self.node_counter += 1
        node_id = self.node_counter
        # 每个节点独立 RMP（共享列池由调用方决定，这里节点内自己累加）
        rmp = RMP(customers=self.problem.customers, params=self.params.rmp_params, logger=self.logger)
        node = BranchNode(lower_bound=float("inf"), id=node_id, depth=depth,
                          fixed_one=set(fixed_one or []), fixed_zero=set(fixed_zero or []), rmp=rmp)
        # 施加分支修正（若对应列存在则限界，不存在则等其出现时再限界）
        self._apply_fixed_bounds(node)
        return node

    def _make_child(self, parent: BranchNode, fix_one: Optional[Iterable[str]] = None,
                    fix_zero: Optional[Iterable[str]] = None) -> BranchNode:
        child = self._make_node(depth=parent.depth + 1)
        # 继承父节点的列（为加速，也可只继承被选列；此处全部继承父节点已加入列）
        if parent.rmp.columns:
            child.rmp.add_columns(list(parent.rmp.columns.values()))
        # 合并约束
        child.fixed_one = set(parent.fixed_one) | set(fix_one or [])
        child.fixed_zero = set(parent.fixed_zero) | set(fix_zero or [])
        self._apply_fixed_bounds(child)
        # 初始下界设为父节点目标，利于 best-bound 排序
        child.lower_bound = parent.rmp.get_objective_value()
        return child

    def _apply_fixed_bounds(self, node: BranchNode) -> None:
        """
        在该节点的 RMP 上应用 x_r 的上下界：
          - fixed_one: LB=UB=1；若变量不存在，等待其被添加后再设置（在 add_columns 后也会设置）
          - fixed_zero: UB=0
        """
        # 直接访问后端 x_vars（Gurobi 变量），以避免改 rmlp 接口
        backend = node.rmp.backend
        # 现有变量立即设置
        for cid in list(backend.x_vars.keys()):
            if cid in node.fixed_one:
                var = backend.x_vars[cid]; var.LB = 1.0; var.UB = 1.0
            if cid in node.fixed_zero:
                var = backend.x_vars[cid]; var.UB = 0.0
        # 在后续 add_columns 之后，也需要再次调用本方法（由 _column_generation_loop 内处理）

    # ---- 列生成循环（节点内）----
    def _column_generation_loop(self, node: BranchNode) -> tuple[float, float, int]:
        new_cols_total = 0
        rc_min_global = 0.0
        no_improve = 0
        iter_in_node = 0

        while iter_in_node < self.params.max_iterations:
            iter_in_node += 1
            # 1) 解 RMP
            obj = node.rmp.solve()
            node.lower_bound = obj
            duals = node.rmp.get_duals()

            # 稳定化
            if self.params.stabilize_duals:
                if self.dual_anchor is None:
                    self.dual_anchor = dict(duals)
                stab_duals = self._stabilize_duals(duals, self.dual_anchor, self.params.stabilize_alpha)
                used_duals = stab_duals
            else:
                used_duals = duals

            # 2) 定价若干批次
            rc_min_round = 0.0
            round_cols = 0
            for b in range(self.params.pricing_batch):
                # (a) 定价器生成 → 入池
                gen_cols = self.cg.generate(used_duals, budget=200, lambda_route=self.params.rmp_params.lambda_route)
                self.pool.add_columns(gen_cols)

                # (b) 从池子里筛出负列 → 入 RMP
                neg_cols = self.pool.check_negative_rc(
                    used_duals,
                    lambda_route=self.params.rmp_params.lambda_route,
                    tol=self.params.rc_tolerance,
                    budget=50
                )
                if not neg_cols:
                    break
                node.rmp.add_columns(neg_cols)
                # 分支修正：新加的列要立即应用 fixed_one/fixed_zero


                self._apply_fixed_bounds(node)
                # 统计 rc
                rc_min_round = min(rc_min_round, min(c.meta.get("rc", 0.0) for c in neg_cols))
                new_cols_total += len(neg_cols)
                round_cols += len(neg_cols)

            rc_min_global = min(rc_min_global, rc_min_round)

            # 3) 停止条件：无负 rc 或者节点内无改进
            if round_cols == 0:
                break

            if obj + 1e-6 < self.global_best_obj:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.params.no_improve_rounds:
                    break

            if self._timed_out():
                break

        return node.rmp.get_objective_value(), rc_min_global, new_cols_total

    # ---- 分支变量选择 ----
    def _choose_branch_var(self, fractional: List[tuple[str, float]]) -> tuple[str, float]:
        if self.params.branch_strategy == "first_fractional":
            return fractional[0]
        # most fractional：离 0.5 最近
        cid, x = min(fractional, key=lambda kv: abs(kv[1] - 0.5))
        return cid, x

    # ---- 对偶稳定化 ----
    @staticmethod
    def _stabilize_duals(duals: Dict[int, float], anchor: Dict[int, float], alpha: float) -> Dict[int, float]:
        return {i: alpha * duals.get(i, 0.0) + (1 - alpha) * anchor.get(i, 0.0) for i in set(anchor) | set(duals)}

    # ---- 记录/导出 ----


    def _record(self, iteration: int, node: BranchNode, obj: float, rc_min: float, new_cols: int) -> None:
        elapsed = time.time() - self.start_time
        gap = max(0.0, (obj - self.global_best_obj) / max(1.0, abs(self.global_best_obj))) \
            if self.global_best_obj < float("inf") else float("inf")

        # ---- 计算扩展指标 ----
        duals = node.rmp.get_duals()
        customers = self.problem.customers

        covered = set()
        for col in node.rmp.get_selected_columns():
            covered.update(col.served_set)
        coverage_rate = len(covered) / max(1, len(customers))

        dual_vals = list(duals.values()) if duals else []
        avg_dual = sum(dual_vals) / len(dual_vals) if dual_vals else 0.0
        max_dual = max(dual_vals) if dual_vals else 0.0
        s = sum(dual_vals)
        dual_entropy = 0.0
        if s > 1e-9:
            probs = [v/s for v in dual_vals]
            dual_entropy = -sum(p * math.log(p + 1e-12) for p in probs)

        # ---- 状态对象 ----
        st = BPStatus(
            iteration=iteration,
            node_id=node.id,
            obj_value=obj,
            best_bound=obj,   # 暂用 obj 作为 bound，可拓展
            gap=gap,
            num_columns=len(node.rmp.columns),
            num_new_columns=new_cols,
            time_elapsed=elapsed,
            rc_min=rc_min,
            coverage_rate=coverage_rate,
            avg_dual=avg_dual,
            max_dual=max_dual,
            dual_entropy=dual_entropy
        )
        self.stats.append(st)

        # ---- 控制台/日志输出 ----
        self.logger.info(
            "[Iter %d][Node %d] obj=%.6f, new_cols=%d, rc_min=%.6g, "
            "elapsed=%.2fs, gap=%s, cov=%.1f%%",
            iteration, node.id, obj, new_cols, rc_min, elapsed,
            f"{gap:.4%}" if gap < float("inf") else "NA",
            coverage_rate * 100
        )

        # ---- 写 CSV 累计 ----
        csv_path = os.path.join(self.params.outdir, "stats.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=st.__dict__.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(st.__dict__)

        # ---- 每轮快照 ----
        if self.params.dump_every_iter:
            snap_dir = os.path.join(self.params.outdir, f"iter_{iteration}_node_{node.id}")
            os.makedirs(snap_dir, exist_ok=True)
            summary = {
                "iteration": iteration,
                "node_id": node.id,
                "obj": obj,
                "rc_min": rc_min,
                "num_columns": len(node.rmp.columns),
                "elapsed": elapsed,
                "gap": gap,
                "coverage_rate": coverage_rate,
                "avg_dual": avg_dual,
                "max_dual": max_dual,
                "dual_entropy": dual_entropy
            }
            with open(os.path.join(snap_dir, "bp_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

    def _timed_out(self) -> bool:
        return self.params.time_limit is not None and (time.time() - self.start_time) >= self.params.time_limit

    # ---- 日志初始化 ----
    def _init_logger(self, logger: Optional[logging.Logger], params: BPParams) -> logging.Logger:
        os.makedirs(params.outdir, exist_ok=True)
        if logger is not None:
            return logger
        lg = logging.getLogger("BP")
        lg.setLevel(params.log_level)
        # 控制台
        ch = logging.StreamHandler()
        ch.setLevel(params.log_level)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        lg.addHandler(ch)
        # 文件
        fh = logging.FileHandler(os.path.join(params.outdir, "bp_events.log"), mode="w", encoding="utf-8")
        fh.setLevel(params.log_level)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        lg.addHandler(fh)
        return lg
