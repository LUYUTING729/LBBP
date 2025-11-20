# bp_drones.py
# --- at top of bp_solver.py ---
from __future__ import annotations

import heapq
import time
import math
import os
import logging
import csv
from dataclasses import dataclass, field, replace
from typing import List, Dict, Optional, Iterable, Tuple, Any, Sequence, Set, Union
from truck_solver import TruckSolver, TruckParams
from rmp import RMP, RMPParams, Column
from data_model import Problem
from label_setting import LabelSettingParams, label_setting
from column_pool import ColumnPool
from branch_and_bound import BranchEngine, BnBParams

# =========================
# BP 参数 / 统计 / 结果（按你给的定义）
# =========================
@dataclass
class BPParams:
    # 主循环
    max_iterations: int = 30
    rc_tolerance: float = -1e-4          # 负约简成本阈值（< 0 即有改进）
    pricing_batch: int = 1               # 每轮最多“批次”列生成（多次过滤后再解 RMP）
    stabilize_duals: bool = True        # 对偶稳定化开关
    stabilize_alpha: float = 0.7         # 稳定化权重 (π^st = α π + (1-α) π_anchor)

    # ---- Label-setting 相关（定价器内部使用）----
    label_setting_params: Optional[LabelSettingParams] = field(default_factory=LabelSettingParams)

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

    # ========= 新增：不等式（SRI / Clique）配置 =========
    # 静态：在每个分支节点初始化时就加入
    sri_static: List[Tuple[str, Iterable[int], Union[int, float]]] = field(default_factory=list)
    clique_static: List[Tuple[str, Iterable[int], Union[int, float]]] = field(default_factory=list)  # (key, Q, rhs)

    # 候选：供分离器检查是否被当前分数解违反（可选）
    sri_candidates: List[Tuple[str, Iterable[int], Union[int, float]]] = field(default_factory=list)
    clique_candidates: List[Tuple[str, Iterable[int], Union[int, float]]] = field(default_factory=list)

    # 分离控制
    cut_time_limit_per_round: float = 0.5     # 每轮用于分离的时间预算（秒）
    cut_max_add_per_round: int = 30           # 每轮最多新增的不等式数量（总数，含 SRI+Clique）
    cut_violation_tol: float = 1e-6           # 违反阈值（> beta + tol 视为违反）


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
        label_params = getattr(bp_params, "label_setting_params", None)
        if label_params is not None:
            params = replace(label_params)
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
        outdir = label_params.outdir or "LS_logs"
        if base_outdir and not os.path.isabs(outdir):
            outdir = os.path.join(base_outdir, outdir)
        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)
        label_params.outdir = outdir
        return outdir

    def _prepare_label_logger(self, label_params: LabelSettingParams, outdir: str, fallback: logging.Logger) -> logging.Logger:
        logger = label_params.logger or fallback
        logfile = os.path.join(outdir, "label_setting.log")
        abs_logfile = os.path.abspath(logfile)
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
        logger.propagate = False
        label_params.logger = logger
        return logger

    def _maybe_dump_labels(self, labels: Sequence[Any], iteration: int) -> None:
        outdir = self.label_outdir
        path = os.path.join(outdir, "labels.csv")
        top_k = getattr(self.label_params, "dump_top_k", 50)
        top = sorted(labels, key=lambda L: getattr(L, "red_cost", getattr(L, "cost", 0.0)))[:top_k]
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["iteration", "rank", "path", "served", "cost", "red_cost", "time", "energy", "latest_departure"])
            for rank, lab in enumerate(top, 1):
                w.writerow([
                    iteration, rank,
                    ";".join(map(str, getattr(lab, "path", []))),
                    getattr(lab, "served_count", 0),
                    f"{getattr(lab, 'cost', 0.0):.6f}",
                    f"{getattr(lab, 'red_cost', 0.0):.6f}",
                    f"{getattr(lab, 'time', 0.0):.6f}",
                    f"{getattr(lab, 'energy', 0.0):.6f}",
                    f"{getattr(lab, 'latest_departure', 0.0):.6f}",
                ])

    def _labels_to_columns(self, labels: Sequence[Any], id_prefix: str) -> List[Column]:
        cols = RMP.make_columns_from_labels(labels, id_prefix=id_prefix)
        return cols

    def _reduced_cost(self, col: Column, duals: Dict[int, float], lambda_route: float) -> float:
        pi_sum = sum(duals.get(i, 0.0) for i in col.served_set)
        rc = (col.cost + lambda_route  ) - pi_sum
        return rc

    def generate(self, duals: Dict[int, float], budget: int, lambda_route: float) -> List[Column]:
        self._gen_call += 1 
        iteration = self._gen_call 
        id_prefix = f"it{iteration}_" 
        self.label_params.duals = duals 
        labels = label_setting(problem=self.problem, params=self.label_params) 
        self._maybe_dump_labels(labels, iteration) 
        candidates: List[Tuple[float, Column]] = [] 
        for col in self._labels_to_columns(labels, id_prefix=id_prefix): 
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
        self.label_logger.info("Pricing: generated %d labels, %d negative-rc candidates, return %d.",
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



# =========================================================
# TruckSolverCheck
#   - Logic-Based Benders 的卡车子问题检查器
#   - 给定当前整数解里被选的列集合 S，判断卡车调度是否可行
#   - 如果可行：返回 (True, truck_cost, [])
#   - 如果不可行：返回 (False, None, [conflict_sets...])
#
#   注意：这是对接点。你需要用真实的卡车VRPTW(带时间窗)启发式/MIP替换 evaluate() 内容。
# =========================================================

class TruckSolverCheck:
    """
    Logic-Based Benders 的卡车子问题检查器：
    - 给定当前被选择的无人机列（Column）集合，抽取其首尾节点
    - 调用 TruckSolver 构建并求解单车 VRPTW
    - 返回可行性、成本和冲突集合
    """
    def __init__(self, truck_solver: TruckSolver, logger: logging.Logger):
        self.truck_solver = truck_solver
        self.logger = logger

    def evaluate(self, selected_columns: List[Column]) -> Tuple[bool, Optional[float], List[List[str]]]:
        """
        真实实现：
        1. 从列对象提取首尾节点（endpoints）
        2. 调用 TruckSolver.evaluate() 求解 VRPTW
        3. 返回 (可行性, 成本, 冲突集合)
        
        Args:
            selected_columns: 被选中的 Column 对象列表
            
        Returns:
            (feasible, truck_cost, conflicts)
            - feasible: bool，卡车调度是否可行
            - truck_cost: float，卡车成本（若不可行为 None）
            - conflicts: List[List[str]]，冲突的列id集合（用于可行性割）
        """
        if not selected_columns:
            # 没有任务，自然可行且cost=0
            self.logger.info("[TruckCheck] No columns selected. Feasible=True, cost=0.0")
            return True, 0.0, []

        try:
            # 调用 TruckSolver 来评估这组列
            tres = self.truck_solver.evaluate(
                selected_columns=selected_columns,
                use_path_endpoints=True
            )
            
            feasible = tres.feasible
            truck_cost = tres.cost if feasible else None
            
            # 从冲突端点集合转换为列id冲突集合（简化：若不可行则把所有列标记为冲突）
            if not feasible:
                # 可以更精细地处理：根据 tres.conflicts 中的节点id找到相关列
                conflict_sets = [
                    [col.id for col in selected_columns]  # 简化：整个集合作为一个冲突
                ]
            else:
                conflict_sets = []
            
            self.logger.info(
                "[TruckCheck] S=%d feasible=%s cost=%s conflicts=%d | msg='%s'",
                len(selected_columns),
                feasible,
                f"{truck_cost:.6f}" if truck_cost is not None else "N/A",
                len(conflict_sets),
                tres.msg or "OK"
            )
            
            return feasible, truck_cost, conflict_sets
            
        except Exception as e:
            self.logger.error("[TruckCheck] Exception during truck evaluation: %s", e)
            # 发生异常时认为不可行，返回冲突集合为整个选定列集合
            conflict_sets = [[col.id for col in selected_columns]]
            return False, None, conflict_sets


# =========================================================
# BPSolver 主类（含 LB-Benders 集成）
# =========================================================




class BPSolver:
    """
    分支定价主循环（含 Logic-Based Benders 钩子）：
      - best-bound 取节点
      - 节点内部: RMP ↔ 定价(列生成) ↔ cut分离
      - 无负列后:
          * 若解(当前RMP)整数，则跑卡车子问题：
                - 不可行 => 加可行性割 (conflict cut)，节点回队列继续
                - 可行   => 加 θ_truck 最优性割，更新下界/ incumbent
          * 若解分数 => 分支
    """

    def __init__(self,
                 problem: Problem,
                 params: Optional[BPParams] = None,
                 logger: Optional[logging.Logger] = None):
        self.problem = problem
        self.params = params or BPParams()

        os.makedirs(self.params.outdir, exist_ok=True)

        self.logger = self._init_logger(logger, self.params)

        # 子问题定价器（label setting / pricing）
        self.cg = LabelSettingGenerator(problem, self.params, self.logger)

        # 列池（缓存所有生成过的列，筛负 reduced cost 列）
        self.pool = ColumnPool(logger=self.logger)

        # 分支引擎（选择分支变量、维护伪成本）
        self.branch_engine = BranchEngine(BnBParams(
            strategy="auto",
            strong_top_k=10,
            strong_time_limit=1.0,
            strong_stop_early=True,
            use_pseudocost=True,
            pseudocost_min_obs=2,
            outdir=os.path.join(self.params.outdir, "branch"),
            log_level=self.params.log_level
        ), logger=self.logger)

        # 卡车子问题求解器（真实的 VRPTW 求解）
        # TruckParams 配置：可根据实际问题调整
        truck_solver = TruckSolver(
            problem=problem,
            depot_idx=0,  # depot 在 problem.customers[0]
            params=TruckParams(
                truck_speed=1.0,            # 卡车速度（若有时间矩阵则不用）
                truck_cost_per_time=1.0,    # 单位时间成本
                bigM_time=1e5,              # 时间约束用的大 M
                time_limit=5.0,             # 每个VRPTW求解的时间限制
                mip_gap=0.01,               # MIP gap
                log_to_console=False        # 不打印 Gurobi 日志
            ),
            logger=self.logger
        )

        # 卡车子问题检查器（Logic-Based Benders）
        self.truck_checker = TruckSolverCheck(
            truck_solver=truck_solver,
            logger=self.logger
        )

        # 全局计数/记录
        self.node_counter = 0
        self.stats: List[BPStatus] = []
        self.global_best_obj = float("inf")
        self.global_best_solution: Dict[str, float] = {}
        self.global_best_cols: List[Column] = []
        self.dual_anchor: Optional[Dict[int, float]] = None
        self.start_time = time.time()

    # -----------------------------------------------------
    # solve(): B&P 主循环
    # -----------------------------------------------------
    def solve(self, init_columns: Optional[List[Column]] = None) -> BPResult:
        # 初始化根节点
        root = self._make_node(depth=0)
        if init_columns:
            root.rmp.add_columns(init_columns)

        # best-bound 优先队列
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
                             node.id, node.depth,
                             len(node.fixed_one), len(node.fixed_zero))

            # 在该节点做列生成与cut分离循环
            obj, rc_min, new_cols_total = self._column_generation_loop(node)

            # 记录一发当前状态
            self._record(iteration, node, obj, rc_min, new_cols_total)

            # 超时检查
            if self._timed_out():
                self.logger.warning("Time limit reached. Stop.")
                break

            # 取当前 RMP 解
            x_vals = node.rmp.get_solution_vector()
            fractional = [(cid, x) for cid, x in x_vals.items() if 1e-6 < x < 1 - 1e-6]

            # =====================================================
            # CASE A: 当前RMP解是整数解 (LP解基本全0/1)
            #         -> 卡车子问题检验 & LB-Benders 切割
            # =====================================================
            if not fractional:
                selected_col_ids = [cid for cid, x in x_vals.items() if x > 0.5]
                # 取"列对象"以便 TruckChecker 评估
                picked_cols: List[Column] = [
                    node.rmp.columns[cid] for cid in selected_col_ids if cid in node.rmp.columns
                ]

                feasible, truck_cost, conflict_sets = self.truck_checker.evaluate(picked_cols)

                if not feasible:
                    # 卡车调度不可行：
                    #   -> 加可行性割来禁止这个配置
                    #   基于 conflict_sets 来做更精细的冲突割
                    self.logger.info("[Node %d] Truck infeasible. Add conflict cut and reprocess.",
                                     node.id)
                    node.rmp.add_conflict_cut(selected_col_ids)
                    # 重新 push 当前节点，因 RMP 被加强，继续探索
                    heapq.heappush(pq, node)
                    continue

                else:
                    # 卡车可行：
                    #   -> 对该集合 S 加 θ_truck 最优性割
                    bigM = getattr(self.params.rmp_params, "bigM_truck", 1e5)
                    self.logger.info("[Node %d] Truck feasible. Add theta cut with cost=%.6f.",
                                     node.id, float(truck_cost))

                    node.rmp.add_theta_cut(
                        selected_col_ids=selected_col_ids,
                        truck_cost=float(truck_cost),
                        bigM=bigM
                    )
                    refined_obj = node.rmp.solve()
                    node.lower_bound = refined_obj

                    # 更新全局最好整数上界 (incumbent)
                    if refined_obj < self.global_best_obj:
                        self.global_best_obj = refined_obj
                        self.global_best_solution = node.rmp.get_solution_vector()
                        self.global_best_cols = node.rmp.get_selected_columns()
                        # 打印一次成本拆分（若你在 RMP 实现了 get_cost_breakdown）
                        try:
                            parts = node.rmp.get_cost_breakdown()
                            self.logger.info(
                                "[Node %d] Cost breakdown: truck=%.6f, drone_flt=%.6f, startup=%.6f | total_parts=%.6f, rmp_obj=%.6f",
                                node.id,
                                parts.get("truck_cost", 0.0),
                                parts.get("drone_flight_cost", 0.0),
                                parts.get("drone_startup_cost", 0.0),
                                parts.get("total_by_parts", 0.0),
                                parts.get("rmp_objective", refined_obj)
                            )
                        except Exception:
                            pass

                    theta_now = 0.0
                    try:
                        theta_now = node.rmp.get_theta_truck()
                    except Exception:
                        pass
                    self.logger.info("[Node %d] Integer+Feasible incumbent updated. θ_truck=%.6f, total_obj=%.6f",
                                     node.id, theta_now, refined_obj)
                    # 整数可行节点 -> 不再分支，继续处理队列的下一个节点
                    continue

            # =====================================================
            # CASE B: 解是分数解（有fractional）
            # =====================================================

            # 若已无负 reduced cost 且解仍非整数 => 分支
            if rc_min >= self.params.rc_tolerance:
                if not self.params.enable_branch:
                    self.logger.info("[Node %d] No negative rc and branching disabled. Prune.",
                                     node.id)
                    continue

                # 用分支引擎选分支变量
                decision = self.branch_engine.propose(node, incumbent_obj=self.global_best_obj)
                if decision is None:
                    self.logger.info("[Node %d] Pruned by BnB decision (no candidates or integer).",
                                     node.id)
                    continue

                # 生成左右子树
                left = self._make_child(node)
                right = self._make_child(node)

                # 应用修复(=固定变量)
                for fx in decision.left_fixes:
                    if fx.lb == 1.0 and fx.ub == 1.0:
                        left.fixed_one.add(fx.var_id)
                    else:
                        left.fixed_zero.add(fx.var_id)
                for fx in decision.right_fixes:
                    if fx.lb == 1.0 and fx.ub == 1.0:
                        right.fixed_one.add(fx.var_id)
                    else:
                        right.fixed_zero.add(fx.var_id)

                # 把 LB/UB 约束同步到新子节点
                self._apply_fixed_bounds(left)
                self._apply_fixed_bounds(right)

                # 立即列生成一轮, 获取左右子节点的下界
                obj_left, rc_min_left, _ = self._column_generation_loop(left)
                obj_right, rc_min_right, _ = self._column_generation_loop(right)

                # 更新伪成本学习
                self.branch_engine.update_pseudocost(
                    var_id=decision.chosen.var_id,
                    parent_lb=node.rmp.get_objective_value(),
                    left_lb=obj_left,
                    right_lb=obj_right,
                    x_value=decision.chosen.x_value
                )

                # 将子节点放回队列
                heapq.heappush(pq, left)
                heapq.heappush(pq, right)

            # 否则：还有负列没加完，或者时间到了 -> 下个节点
            # （这里不push当前node，以免死循环；当前node已在_column_generation_loop中拉满）

        status = "optimal" if self.global_best_obj < float("inf") else "stopped"
        return BPResult(
            status=status,
            obj_value=self.global_best_obj if status == "optimal" else float("inf"),
            solution=self.global_best_solution,
            selected_columns=self.global_best_cols,
            duals={},  # 可按需回填最终节点的对偶
            stats=self.stats
        )

    # -----------------------------------------------------
    # 节点构建
    # -----------------------------------------------------
    def _make_node(self,
                   depth: int,
                   fixed_one: Optional[Iterable[str]] = None,
                   fixed_zero: Optional[Iterable[str]] = None) -> BranchNode:
        """
        创建一个新的分支节点：
          - 新建 RMP
          - 注入静态 cuts (SRI / Clique)
          - 应用 (已知的) 分支修复上下界
        """
        self.node_counter += 1
        node_id = self.node_counter

        rmp = RMP(problem=self.problem,
                  params=self.params.rmp_params,
                  logger=self.logger)

        node = BranchNode(
            lower_bound=float("inf"),
            id=node_id,
            depth=depth,
            fixed_one=set(fixed_one or []),
            fixed_zero=set(fixed_zero or []),
            rmp=rmp
        )

        # 先加静态 cuts
        self._apply_static_cuts(node)

        # 应用分支上下界
        self._apply_fixed_bounds(node)

        return node

    def _make_child(self,
                    parent: BranchNode,
                    fix_one: Optional[Iterable[str]] = None,
                    fix_zero: Optional[Iterable[str]] = None) -> BranchNode:
        """
        生成子节点：
          - 复制父节点已有列到子节点RMP
          - 合并父节点的上下界修复
          - 同步上下界到子RMP变量
        """
        child = self._make_node(depth=parent.depth + 1)
        # 继承父节点的列
        if parent.rmp.columns:
            child.rmp.add_columns(list(parent.rmp.columns.values()))

        # 合并上下界固定
        child.fixed_one = set(parent.fixed_one) | set(fix_one or [])
        child.fixed_zero = set(parent.fixed_zero) | set(fix_zero or [])

        # 把这些 bound 应用在子节点
        self._apply_fixed_bounds(child)

        # 初始下界可设为父RMP当前目标
        child.lower_bound = parent.rmp.get_objective_value()
        return child

    def _apply_fixed_bounds(self, node: BranchNode) -> None:
        """
        把 node.fixed_one / fixed_zero 的上下界直接施加到 RMP backend 里的 x_vars。
        """
        backend = node.rmp.backend
        for cid in list(backend.x_vars.keys()):
            if cid in node.fixed_one:
                var = backend.x_vars[cid]
                var.LB = 1.0
                var.UB = 1.0
            if cid in node.fixed_zero:
                var = backend.x_vars[cid]
                var.UB = 0.0

    # -----------------------------------------------------
    # 静态 cuts 注入
    # -----------------------------------------------------
    def _apply_static_cuts(self, node: BranchNode) -> None:
        """
        在节点初始化后，立即加入固定不等式：
          - SRI
          - Clique
        """
        for key, S, beta in getattr(self.params, "sri_static", []):
            try:
                node.rmp.add_sri(key=key, S=S, beta=beta)
            except Exception as e:
                self.logger.warning("Add static SRI[%s] failed: %s", key, e)

        for key, Q, rhs in getattr(self.params, "clique_static", []):
            try:
                node.rmp.add_clique(key=key, Q=Q, rhs=rhs)
            except Exception as e:
                self.logger.warning("Add static Clique[%s] failed: %s", key, e)

    # -----------------------------------------------------
    # Cut 分离器（SRI / Clique 的在线分离）
    # -----------------------------------------------------
    def _separate_and_add_cuts(self,
                               node: BranchNode,
                               x_vals: Dict[str, float]) -> int:
        """
        当前节点的 RMP 分数解 x_vals 下，尝试分离违反的 SRI / Clique，
        若发现违反则增量加入相应 cut。
        返回新增 cut 的数量。
        """
        t0 = time.time()
        added = 0
        tol = self.params.cut_violation_tol

        # 列 id -> served_set 方便计算
        col_served: Dict[str, frozenset] = {
            cid: node.rmp.columns[cid].served_set
            for cid in x_vals.keys()
            if cid in node.rmp.columns
        }

        # ----- SRI -----
        for base_key, S, beta in getattr(self.params, "sri_candidates", []):
            if (time.time() - t0 > self.params.cut_time_limit_per_round or
                added >= self.params.cut_max_add_per_round):
                break

            S_set = set(int(i) for i in S)
            lhs = 0.0
            for cid, x in x_vals.items():
                if x <= 1e-9:
                    continue
                ss = col_served.get(cid)
                if ss is None:
                    continue
                kappa = len(ss.intersection(S_set))
                if kappa:
                    lhs += kappa * x
            if lhs > float(beta) + tol:
                key = f"{base_key}@n{node.id}@{len(S_set)}_{added}"
                try:
                    node.rmp.add_sri(key=key, S=S_set, beta=beta)
                    added += 1
                    self.logger.info("[Cut] Add SRI key=%s, lhs=%.6f > beta=%.6f, |S|=%d",
                                     key, lhs, float(beta), len(S_set))
                except Exception as e:
                    self.logger.warning("Add SRI[%s] failed: %s", key, e)

        # ----- Clique -----
        for base_key, Q, rhs in getattr(self.params, "clique_candidates", []):
            if (time.time() - t0 > self.params.cut_time_limit_per_round or
                added >= self.params.cut_max_add_per_round):
                break

            Q_set = set(int(i) for i in Q)
            lhs = 0.0
            for cid, x in x_vals.items():
                if x <= 1e-9:
                    continue
                ss = col_served.get(cid)
                if ss is None:
                    continue
                if len(ss.intersection(Q_set)) >= 2:
                    lhs += x

            rhs_val = float(rhs) if rhs is not None else 1.0
            if lhs > rhs_val + tol:
                key = f"{base_key}@n{node.id}@{len(Q_set)}_{added}"
                try:
                    node.rmp.add_clique(key=key, Q=Q_set, rhs=rhs_val)
                    added += 1
                    self.logger.info("[Cut] Add CLQ key=%s, lhs=%.6f > rhs=%.6f, |Q|=%d",
                                     key, lhs, rhs_val, len(Q_set))
                except Exception as e:
                    self.logger.warning("Add Clique[%s] failed: %s", key, e)

        return added

    # -----------------------------------------------------
    # 单节点的列生成循环 (RMP ↔ Pricing ↔ Cut分离)
    # -----------------------------------------------------
    def _column_generation_loop(self, node: BranchNode) -> Tuple[float, float, int]:
        new_cols_total = 0
        last_rc_min = 0.0
        iter_in_node = 0

        while iter_in_node < self.params.max_iterations:
            iter_in_node += 1

            # 1) 解一次 RMP
            obj = node.rmp.solve()
            node.lower_bound = obj
            duals = node.rmp.get_duals()

            # 打印当前 θ_truck（若存在）
            try:
                theta_now = node.rmp.get_theta_truck()
                self.logger.info("[Node %d][it %d] theta_truck=%.6f, RMP obj=%.6f",
                                 node.id, iter_in_node, theta_now, obj)
            except Exception:
                self.logger.debug("[Node %d][it %d] RMP obj=%.6f", node.id, iter_in_node, obj)

            # 1.1 对偶稳定化（可选）
            if self.params.stabilize_duals:
                if self.dual_anchor is None:
                    self.dual_anchor = dict(duals)
                used_duals = self._stabilize_duals(duals,
                                                   self.dual_anchor,
                                                   self.params.stabilize_alpha)
            else:
                used_duals = duals

            # 1.2 分离 cut（SRI / Clique），并在必要时立刻再解一次RMP同步对偶
            try:
                x_vals = node.rmp.get_solution_vector()
                added_cuts = self._separate_and_add_cuts(node, x_vals)
                if added_cuts > 0:
                    obj = node.rmp.solve()
                    node.lower_bound = obj
                    duals = node.rmp.get_duals()
                    used_duals = (self._stabilize_duals(duals,
                                                        self.dual_anchor,
                                                        self.params.stabilize_alpha)
                                  if self.params.stabilize_duals else duals)
            except Exception as e:
                self.logger.warning("Separation step failed: %s", e)

            # 2) 定价（多批次）
            rc_min_round = 0.0
            round_cols = 0
            for _ in range(self.params.pricing_batch):
                # 调用 label-setting 定价器生成候选列
                gen_cols = self.cg.generate(
                    used_duals,
                    budget=200,
                    lambda_route=self.params.rmp_params.lambda_route
                )
                # 加入到列池
                self.pool.add_columns(gen_cols)
                # 挑负rc列
                neg_cols = self.pool.check_negative_rc(
                    used_duals,
                    lambda_route=self.params.rmp_params.lambda_route,
                    tol=self.params.rc_tolerance,
                    budget=50
                )
                if not neg_cols:
                    break

                # 把负列加进当前节点RMP
                node.rmp.add_columns(neg_cols)
                # 分支上下界修正要立刻下放到新列
                self._apply_fixed_bounds(node)

                # 统计
                rc_min_round = min(
                    rc_min_round,
                    min(c.meta.get("rc", 0.0) for c in neg_cols)
                )
                new_cols_total += len(neg_cols)
                round_cols += len(neg_cols)

            last_rc_min = rc_min_round

            # 3) 若本轮没有添加新负列 -> 结束该节点的列生成循环
            if round_cols == 0:
                break

            # 4) 时间终止
            if self._timed_out():
                break

        return node.rmp.get_objective_value(), last_rc_min, new_cols_total

    # -----------------------------------------------------
    # 对偶稳定化
    # -----------------------------------------------------
    @staticmethod
    def _stabilize_duals(duals: Dict[int, float],
                         anchor: Dict[int, float],
                         alpha: float) -> Dict[int, float]:
        """
        平滑当前对偶和历史anchor，缓解对偶震荡：
            π_used = α * π_current + (1-α) * π_anchor
        """
        keys = set(anchor) | set(duals)
        return {
            i: alpha * duals.get(i, 0.0) + (1 - alpha) * anchor.get(i, 0.0)
            for i in keys
        }

    # -----------------------------------------------------
    # 记录日志 & 统计信息
    # -----------------------------------------------------
    def _record(self,
                iteration: int,
                node: BranchNode,
                obj: float,
                rc_min: float,
                new_cols: int) -> None:
        """
        记录一次迭代快照，补充扩展指标：
        - coverage_rate: 选中列覆盖的客户占比
        - avg_dual / max_dual: 覆盖约束对偶统计
        - dual_entropy: 归一化对偶分布的熵（避免 log(0)）
        """
        elapsed = time.time() - self.start_time

        # gap：相对当前全局最优整数解（若还没整数解，则为 inf）
        if self.global_best_obj < float("inf"):
            gap = max(0.0, (obj - self.global_best_obj) / max(1.0, abs(self.global_best_obj)))
        else:
            gap = float("inf")

        # ---- 覆盖率 ----
        x_vals = node.rmp.get_solution_vector()
        picked_ids = [cid for cid, x in x_vals.items() if x > 1e-6]

        covered: Set[int] = set()
        for cid in picked_ids:
            col = node.rmp.columns.get(cid)
            if col is not None:
                covered.update(col.served_set)

        # 仅统计客户（不含 depot 0）
        customers = set(self.problem.customers)
        cov_cnt = len(covered & customers)
        num_cust = len(customers)
        coverage_rate = (cov_cnt / num_cust) if num_cust > 0 else 0.0

        # ---- 对偶统计 + 熵 ----
        duals = node.rmp.get_duals()  # dict: i -> π_i
        if duals:
            vals = [max(0.0, float(v)) for v in duals.values()]  # 防负
            avg_dual = sum(vals) / len(vals)
            max_dual = max(vals)
            s = sum(vals)
            if s <= 0.0:
                dual_entropy = 0.0
            else:
                probs = [v / s for v in vals]
                # 熵 = -∑ p log p；加 1e-12 防 log(0)
                dual_entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        else:
            avg_dual = 0.0
            max_dual = 0.0
            dual_entropy = 0.0

        st = BPStatus(
            iteration=iteration,
            node_id=node.id,
            obj_value=obj,
            best_bound=obj,        # RMP 目标即该节点的 LB
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

        self.logger.info(
            "[Iter %d][Node %d] obj=%.6f, new_cols=%d, rc_min=%.6g, "
            "elapsed=%.2fs, gap=%s | cov=%.1f%% avg_dual=%.3f max_dual=%.3f H=%.3f",
            iteration, node.id, obj, new_cols, rc_min, elapsed,
            f"{gap:.4%}" if gap < float("inf") else "NA",
            100.0 * coverage_rate, avg_dual, max_dual, dual_entropy
        )

    # -----------------------------------------------------
    # 时间限制
    # -----------------------------------------------------
    def _timed_out(self) -> bool:
        if self.params.time_limit is None:
            return False
        return (time.time() - self.start_time) >= self.params.time_limit

    # -----------------------------------------------------
    # 日志初始化
    # -----------------------------------------------------
    def _init_logger(self,
                     logger: Optional[logging.Logger],
                     params: BPParams) -> logging.Logger:
        """
        创建/复用 solver 级别的 logger，并写入到 bp_events.log
        """
        if logger is not None:
            return logger

        lg = logging.getLogger(f"BPSolver[{id(self)}]")
        lg.setLevel(params.log_level)

        # 防止重复 handler
        need_file_handler = True
        for h in lg.handlers:
            if isinstance(h, logging.FileHandler) and \
               getattr(h, "baseFilename", "").endswith("bp_events.log"):
                need_file_handler = False
                break

        if need_file_handler:
            fh = logging.FileHandler(
                os.path.join(params.outdir, "bp_events.log"),
                mode="w",
                encoding="utf-8"
            )
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            lg.addHandler(fh)

        # 同时也打印到控制台（可按需去掉）
        need_stream = True
        for h in lg.handlers:
            if isinstance(h, logging.StreamHandler):
                need_stream = False
                break
        if need_stream:
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            lg.addHandler(sh)

        return lg
