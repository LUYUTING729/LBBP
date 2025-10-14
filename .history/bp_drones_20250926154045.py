# bp_drones.py
from __future__ import annotations

import os
import time
import json
import heapq
import logging
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Any, Iterable, Sequence, Tuple, Set

# 依赖你的现有模块
from data_model import Problem
from label_setting import label_setting, LabelSettingParams
from rmp import RMP, RMPParams, Column
from column_pool import ColumnPool
from branch_and_bound import BranchEngine, BnBParams

import os, json, time, math, csv

# =========================
# 参数、状态、结果
# =========================

@dataclass
class BPParams:

    max_iterations: int = 100
    depot_idx: int = 0
    rc_tolerance: float = 1e-6
    stabilize_duals: bool = True
    # ---- Label-setting 相关（定价器内部使用）----
    label_setting_params: Optional[LabelSettingParams] = field(default_factory=LabelSettingParams)
    enable_diagnostic_round: bool = True
    # 诊断轮放宽参数（只覆盖 label_setting 的关键项）
    diag_K_per_sig: int = 50
    diag_eps: float = 0.15
    diag_disable_dominance: bool = True
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
      - 计算 reduced cost（与 RMP 一致）
      - 过滤 rc < rc_tol 的列，返回至多 budget 个
      - 维护路径去重缓存
      - 规范化日志与 CSV 输出逻辑
    """
    def __init__(self, problem: Problem, params: BPParams, logger: logging.Logger):
        self.problem = problem
        self.params  = params
        self.logger  = logger
        self._seen_keys: Set[tuple] = set()
        self._gen_call: int = 0

        self.label_params = self._resolve_label_params(params, logger)
        self.label_outdir = self._prepare_label_outdir(self.label_params, getattr(params, "outdir", ""))

        # 统一生成一个独立的 label logger（文件可选）
        self.label_logger = self._prepare_label_logger(self.label_params, self.label_outdir, logger)

    # ---------- 参数收敛 ----------
    def _resolve_label_params(self, bp_params: BPParams, default_logger: logging.Logger) -> LabelSettingParams:
        if getattr(bp_params, "label_params", None) is not None:
            params = replace(bp_params.label_params)
        else:
            params = LabelSettingParams(
                max_len    = getattr(bp_params, "max_label_len", 5),
                depot_idx  = getattr(bp_params, "depot_idx", 0),
                logger     = default_logger,
                K_per_sig  = getattr(bp_params, "k_per_sig", 10),
                eps        = getattr(bp_params, "eps_dom", 0.05),
                outdir     = getattr(bp_params, "outdir", "LS_logs"),
            )
        if params.logger is None:
            params.logger = default_logger
        return params

    # ---------- 输出目录规范化 ----------
    def _prepare_label_outdir(self, label_params: LabelSettingParams, base_outdir: str) -> str:
        """
        返回一个**存在的**绝对路径目录；若 label_params.outdir 是相对路径，则相对 base_outdir 拼。
        """
        outdir = label_params.outdir or "LS_logs"
        if base_outdir and not os.path.isabs(outdir):
            outdir = os.path.join(base_outdir, outdir)
        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)
        label_params.outdir = outdir
        return outdir


    def _prepare_label_logger(self, label_params: LabelSettingParams, outdir: str, fallback: logging.Logger) -> logging.Logger:
        """
        简单可靠版：永远把日志写到 outdir/label_setting.log。
        若已添加过同一个文件的 Handler，则不重复添加。
        """
        logger = label_params.logger or fallback
        logfile = os.path.join(outdir, "label_setting.log")
        abs_logfile = os.path.abspath(logfile)

        # 已有同路径的 FileHandler 就不再加
        has_same = False
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == abs_logfile:
                has_same = True
                break

        if not has_same:
            fh = logging.FileHandler(abs_logfile, mode="w", encoding="utf-8")
            fh.setLevel(getattr(label_params, "log_level", logging.INFO))
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(fh)

        # 不冒泡到父 logger，避免重复打印到控制台
        logger.propagate = False
        label_params.logger = logger
        return logger


    def _maybe_dump_labels(self, labels: Sequence[Any], iteration: int) -> None:
        """
        极简导出：把本轮最好的 Top-K（按 red_cost 升序）追加到 outdir/labels.csv。
        首次写入会写表头。
        """
        outdir = self.label_outdir  # 由 _prepare_label_outdir 保证存在
        path = os.path.join(outdir, "labels.csv")
        top_k = getattr(self.label_params, "dump_top_k", 50)

        # 选 Top-K
        top = sorted(labels, key=lambda L: getattr(L, "red_cost", getattr(L, "cost", 0.0)))[:top_k]

        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["iteration", "rank", "path", "served", "cost", "red_cost", "time", "energy", "latest_departure"])
            for rank, lab in enumerate(top, 1):
                w.writerow([
                    iteration,
                    rank,
                    ";".join(map(str, getattr(lab, "path", []))),
                    getattr(lab, "served_count", 0),
                    f"{getattr(lab, 'cost', 0.0):.6f}",
                    f"{getattr(lab, 'red_cost', 0.0):.6f}",
                    f"{getattr(lab, 'time', 0.0):.6f}",
                    f"{getattr(lab, 'energy', 0.0):.6f}",
                    f"{getattr(lab, 'latest_departure', 0.0):.6f}",
                ])

    def _labels_to_columns(self, labels: Sequence[Any], id_prefix: str) -> List[Column]:
        # 复用 RMP 的工具函数
        cols = RMP.make_columns_from_labels(labels, id_prefix=id_prefix)
        return cols

    def _reduced_cost(self, col: Column, duals: Dict[int, float], lambda_route: float) -> float:
        pi_sum = sum(duals.get(i, 0.0) for i in col.served_set)
        rc = (col.cost + lambda_route) - pi_sum
        return rc

    from dataclasses import replace

    def generate(self, duals: Dict[int, float], budget: int, lambda_route: float) -> List[Column]:
        """
        常规定价 → 若无负列且允许诊断轮 → 放宽参数再定价一次。
        返回至多 budget 条负列。
        """
        # 记录迭代序号
        self._gen_call = getattr(self, "_gen_call", 0) + 1
        iteration = self._gen_call
        id_prefix = f"it{iteration}_"

        # 确保有诊断轮标记
        if not hasattr(self, "_diag_used"):
            self._diag_used = False

        def _one_pass(params_for_ls: LabelSettingParams, pass_tag: str) -> List[Column]:
            """单次定价封装：跑 label_setting → 转列 → 计算 rc → 选前 budget 条"""
            # 注入对偶、lambda_route（与 RMP 一致）
            params_for_ls = replace(params_for_ls, duals=duals, lambda_route=lambda_route)

            labels = label_setting(problem=self.problem, params=params_for_ls)
            # 导出本轮 label（带上 tag）
            self._maybe_dump_labels(labels, iteration if pass_tag == "normal" else f"{iteration}-diag")

            candidates: List[Tuple[float, Column]] = []
            for col in self._labels_to_columns(labels, id_prefix=id_prefix if pass_tag == "normal" else f"{id_prefix}D_"):
                key = tuple(col.path)
                if key in self._seen_keys:
                    continue
                rc = self._reduced_cost(col, duals, lambda_route)
                if rc < self.params.rc_tolerance:
                    col.meta["rc"] = float(rc)
                    candidates.append((rc, col))

            candidates.sort(key=lambda x: x[0])
            picked: List[Column] = []
            for rc, col in candidates:
                picked.append(col)
                self._seen_keys.add(tuple(col.path))
                if len(picked) >= budget:
                    break

            self.label_logger.info(
                "Pricing(%s): labels=%d, neg-rc=%d, return=%d.",
                pass_tag, len(labels), len(candidates), len(picked),
            )
            return picked

        # ---------- 常规一轮 ----------
        picked = _one_pass(self.label_params, pass_tag="normal")
        if picked:
            return picked

        # ---------- 诊断轮（只触发一次） ----------
        enable_diag = getattr(self.params, "enable_diagnostic_round", True)
        if (not picked) and enable_diag and (not self._diag_used):
            self._diag_used = True
            # 放宽参数：K_per_sig↑、eps↑、关闭支配
            diag_K = getattr(self.params, "diag_K_per_sig", max(self.label_params.K_per_sig * 4, 50))
            diag_eps = getattr(self.params, "diag_eps", max(self.label_params.eps * 3, 0.15))
            diag_disable_dom = getattr(self.params, "diag_disable_dominance", True)

            diag_params = replace(
                self.label_params,
                K_per_sig=diag_K,
                eps=diag_eps,
                # 需要 label_setting 支持该开关；若没加，请在支配调用处判断后跳过
                disable_dominance=diag_disable_dom  # 新增字段：LabelSettingParams里加上，默认False
            )

            self.label_logger.info(
                "[DIAG] No negative columns this round. Try relaxed params: K=%d, eps=%.3f, disable_dom=%s",
                diag_K, diag_eps, diag_disable_dom
            )
            picked_diag = _one_pass(diag_params, pass_tag="diag")
            if picked_diag:
                self.label_logger.info("[DIAG] Diagnostic round found %d negative columns.", len(picked_diag))
                return picked_diag

            self.label_logger.info("[DIAG] Diagnostic round still found NO negative columns.")
            return []  # 诊断轮也没找到 → 返回空，外层据此终止

        # 没开诊断轮或已经用过，直接返回
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
        self.branch_engine = BranchEngine(BnBParams(
            strategy="auto",              # 你也可设成 "strong"/"pseudocost"/"most_fractional"
            strong_top_k=10,
            strong_time_limit=1.0,
            strong_stop_early=True,
            use_pseudocost=True,
            pseudocost_min_obs=2,
            outdir=os.path.join(self.params.outdir, "branch"),
            log_level=self.params.log_level
        ), logger=self.logger)

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
                decision = self.branch_engine.propose(node, incumbent_obj=self.global_best_obj)
                if decision is None:
                    self.logger.info("[Node %d] Pruned by BnB decision (no candidates or integer).", node.id)
                    continue

                # 左子树（x=1）与右子树（x=0）
                left = self._make_child(node)
                right = self._make_child(node)

                # 应用修复（仍沿用你现有的 fixed_one/fixed_zero + _apply_fixed_bounds）
                for fx in decision.left_fixes:
                    left.fixed_one.add(fx.var_id) if fx.lb == 1.0 and fx.ub == 1.0 else left.fixed_zero.add(fx.var_id)
                for fx in decision.right_fixes:
                    right.fixed_one.add(fx.var_id) if fx.lb == 1.0 and fx.ub == 1.0 else right.fixed_zero.add(fx.var_id)

                # 立刻把修复同步到变量 LB/UB（避免后续 add_columns 后忘记）
                self._apply_fixed_bounds(left)
                self._apply_fixed_bounds(right)

                # 3) 先解左右子节点的 RMP，得到各自的 LB
                obj_left, rc_min_left, _ = self._column_generation_loop(left)
                obj_right, rc_min_right, _ = self._column_generation_loop(right)

                # 4) 更新伪成本信息（此处调用）
                self.branch_engine.update_pseudocost(
                    var_id=decision.chosen.var_id,
                    parent_lb=node.rmp.get_objective_value(),
                    left_lb=obj_left,
                    right_lb=obj_right,
                    x_value=decision.chosen.x_value
                )
                # 推入队列
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
        rmp = RMP(problem=self.problem,params=self.params.rmp_params, logger=self.logger)
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



            # 2) 定价若干批次
            rc_min_round = 0.0
            round_cols = 0
            for _ in range(self.params.pricing_batch):
                # (a) 定价器生成 → 入池
                gen_cols = self.cg.generate(
                    used_duals,
                    budget=200,
                    lambda_route=self.params.rmp_params.lambda_route
                )
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
                self._apply_fixed_bounds(node)  # 分支修正

                # 统计
                rc_min_round = min(rc_min_round, min(c.meta.get("rc", 0.0) for c in neg_cols))
                added = len(neg_cols)
                new_cols_total += added
                round_cols += added

            # === 诊断轮：本轮完全没负列时触发一次放宽 ===
            if round_cols == 0 and (not getattr(self, "_diag_used", False)) and self.params.enable_diagnostic_round:
                self._diag_used = True
                self.logger.info("[DIAG] No negative columns. Trigger a RELAXED diagnostic round ...")

                try:
                    # 构造放宽版 label 参数（只覆盖关键字段）
                    diag_lp = replace(
                        self.cg.label_params,
                        K_per_sig=self.params.diag_K_per_sig,
                        eps=self.params.diag_eps,
                        disable_dominance=True
                    )
                except Exception:
                    # 若没有 replace 或字段不全，退化为直接 new（避免依赖）
                    diag_lp = LabelSettingParams(
                        max_len=self.cg.label_params.max_len,
                        depot_idx=self.cg.label_params.depot_idx,
                        logger=self.cg.label_params.logger,
                        K_per_sig=self.params.diag_K_per_sig,
                        eps=self.params.diag_eps,
                        duals=None,  # generate 会覆盖
                        time_bucket=self.cg.label_params.time_bucket,
                        require_return=self.cg.label_params.require_return,
                        lambda_route=self.cg.label_params.lambda_route,
                        seed=self.cg.label_params.seed,
                        outdir=self.cg.label_params.outdir
                    )
                    diag_lp.disable_dominance = True  # 动态属性也可

                # 仅此一轮使用放宽参数进行一次定价
                gen_cols_relaxed = self.cg.generate(
                    used_duals,
                    budget=400,  # 适度增大预算
                    lambda_route=self.params.rmp_params.lambda_route,
                    override_params=diag_lp
                )
                self.pool.add_columns(gen_cols_relaxed)

                neg_cols_relaxed = self.pool.check_negative_rc(
                    used_duals,
                    lambda_route=self.params.rmp_params.lambda_route,
                    tol=self.params.rc_tolerance,
                    budget=80
                )
                if neg_cols_relaxed:
                    node.rmp.add_columns(neg_cols_relaxed)
                    self._apply_fixed_bounds(node)
                    rc_min_round = min(rc_min_round, min(c.meta.get("rc", 0.0) for c in neg_cols_relaxed))
                    added = len(neg_cols_relaxed)
                    new_cols_total += added
                    round_cols += added
                    self.logger.info("[DIAG] Diagnostic round found %d negative columns.", added)
                else:
                    self.logger.info("[DIAG] Diagnostic round still found NO negative columns.")

            rc_min_global = min(rc_min_global, rc_min_round)

            # 3) 停止条件：无负 rc 或节点内无改进/超时
            if round_cols == 0:
                self.logger.info("[STOP] No negative columns (even after diagnostic if enabled).")
                break

            if obj + 1e-6 < self.global_best_obj:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.params.no_improve_rounds:
                    self.logger.info("[STOP] No improvement for %d rounds.", no_improve)
                    break

            if self._timed_out():
                self.logger.info("[STOP] Timed out in node loop.")
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
