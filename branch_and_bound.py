# branch_and_bound.py
from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from logging_utils import init_logging

# ---------------------------
# 参数与数据类
# ---------------------------

@dataclass
class BnBParams:
    """
    分支定界算法的参数配置类。
    包含分支策略、强分支探测相关参数、伪成本参数、分数化阈值、日志输出等设置。
    """
    # 策略：auto（优先强分支，回退伪成本/最分数化）、strong、pseudocost、most_fractional
    strategy: str = "auto"

    # Strong Branching 探测相关参数
    strong_top_k: int = 10                # 强分支时最多探测的候选变量数
    strong_time_limit: float = 1.0        # 每个子问题求解的时间限制（秒）
    strong_stop_early: bool = True        # 是否提前停止强分支探测（若已出现明显更优的候选）

    # 伪成本（基于历史观测的快速估计）
    use_pseudocost: bool = True
    pseudocost_min_obs: int = 2           # 至少多少次观测后才信任伪成本

    # 分数化阈值
    frac_eps: float = 1e-6                # 判断变量是否分数化的阈值

    # 日志相关参数
    outdir: str = "bp_logs/branch"
    log_level: int = logging.INFO
    csv_name: str = "branch_decisions.csv"


@dataclass
class BranchCandidate:
    """
    分支候选变量的数据结构。
    包含变量ID、当前取值、评分、强分支下的左右子树下界、评估耗时、原因等信息。
    """
    var_id: str
    x_value: float
    # 统一评分：强分支下= min(down_lb, up_lb)，其余策略=越大越优（分数化/伪成本估计）
    score: float
    down_lb: Optional[float] = None
    up_lb: Optional[float] = None
    eval_time: float = 0.0
    reason: str = ""   # "fractional" / "strong:..." / "pseudocost:..."


@dataclass
class BranchFix:
    """
    分支修复（约束）数据结构。
    用于描述在左右子树中对变量的取值修复（如 x=0 或 x=1）。
    """
    var_id: str
    lb: float
    ub: float


@dataclass
class BranchDecision:
    """
    分支决策的数据结构。
    包含被选中的分支候选、左右子树的修复约束、决策原因、所有候选列表。
    """
    chosen: Optional[BranchCandidate]
    left_fixes: List[BranchFix]   # 约定：Left = x=1
    right_fixes: List[BranchFix]  #        Right= x=0
    reason: str
    candidates: List[BranchCandidate]


# ---------------------------
# 分支引擎
# ---------------------------

class BranchEngine:
    """
    分支定界策略执行器：
      - 候选收集（最分数化）
      - 强分支探测（临时改 LB/UB，限时求解，再恢复）
      - 伪成本估计（历史 ΔLB / Δx）
      - 输出左右子树的 LB/UB 修复决策
    低侵入：不改 RMP 接口；Fix 由上层套用到子节点（沿用你已有 _apply_fixed_bounds）
    """
    def __init__(self, params: Optional[BnBParams] = None, logger: Optional[logging.Logger] = None):
        """
        初始化分支引擎，包括参数设置、日志器配置、伪成本字典初始化、决策CSV文件准备等。
        """
        self.params = params or BnBParams()
        os.makedirs(self.params.outdir, exist_ok=True)

        # 日志器
        base_logger = logger or init_logging(
            self.params.outdir,
            name="branch",
            level=self.params.log_level,
            to_console=False,
        )
        self.logger = base_logger.getChild("engine") if logger else base_logger
        self.logger.setLevel(self.params.log_level)

        # 伪成本：var_id -> {"obs": n, "pc_up": val, "pc_down": val}
        self.pseudocost: Dict[str, Dict[str, float]] = {}

        # 决策汇总 CSV
        self._csv_path = os.path.join(self.params.outdir, self.params.csv_name)
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", encoding="utf-8") as f:
                f.write("node_id,depth,chosen_var,x_value,down_lb,up_lb,score,reason,time\n")

    # ---------------------------
    # 对外主入口
    # ---------------------------
    def propose(self, node: Any, incumbent_obj: float) -> Optional[BranchDecision]:
        """
        主分支决策入口。
        输入：
          node：上层的 BranchNode（需有 rmp、id、depth）
          incumbent_obj：当前全局最优整数解（上界）
        输出：
          BranchDecision 或 None（若无需分支/可剪）
        步骤：
          1. 收集分数化候选变量
          2. 根据策略评估候选（强分支/伪成本/分数化）
          3. 构造左右子树修复约束
          4. 记录决策日志
        """
        # 1) 收集候选（分数化）
        cands_frac = self._collect_fractional_candidates(node)
        if not cands_frac:
            self._dump_node_json(node, "no_candidates", [])
            self.logger.info("[Node %d] No fractional variables; pruning or integer.", node.id)
            return None

        # 2) 评估（根据策略）
        if self.params.strategy in ("auto", "strong"):
            # 强分支：对子问题 x=0/1 做限时 LP，评估左右子树下界
            cands_eval = self._strong_branching(node, cands_frac[:self.params.strong_top_k], incumbent_obj)
            chosen = self._choose_strong_or_fallback(cands_eval, cands_frac)
            reason = chosen.reason if chosen and chosen.reason.startswith("strong") else "fractional_fallback"

        elif self.params.strategy == "pseudocost":
            # 伪成本估计：用历史观测估算分支效果
            cands_eval = self._pseudocost_estimate(node, cands_frac)
            chosen = max(cands_eval, key=lambda c: c.score) if cands_eval else cands_frac[0]
            reason = chosen.reason

        else:  # "most_fractional"
            # 仅按分数化选择
            chosen = cands_frac[0]
            reason = "most_fractional"

        # 3) 组装修复
        if not chosen:
            self._dump_node_json(node, "no_choice", cands_frac)
            return None

        left_fixes, right_fixes = self._build_fixes(chosen.var_id)
        decision = BranchDecision(
            chosen=chosen,
            left_fixes=left_fixes,
            right_fixes=right_fixes,
            reason=f"{reason}; x={chosen.x_value:.6f}, score={chosen.score:.6g}, down_lb={chosen.down_lb}, up_lb={chosen.up_lb}",
            candidates=cands_eval if self.params.strategy in ("auto", "strong", "pseudocost") else cands_frac
        )

        # 4) 落盘
        self._dump_decision(node, decision)
        return decision

    # ---------------------------
    # 候选收集
    # ---------------------------
    def _collect_fractional_candidates(self, node: Any) -> List[BranchCandidate]:
        """
        收集当前节点中所有分数化变量作为分支候选。
        分数化变量：取值在 (frac_eps, 1-frac_eps) 之间。
        评分：越接近0.5越优（0.5-|x-0.5|）。
        返回排序后的候选列表。
        """
        x = node.rmp.get_solution_vector()  # {var_id: value}
        cands: List[BranchCandidate] = []
        for vid, val in x.items():
            if self.params.frac_eps < val < 1.0 - self.params.frac_eps:
                # 分数化得分：越接近0.5越高，用 0.5-|x-0.5|
                frac_score = 0.5 - abs(val - 0.5)
                cands.append(BranchCandidate(var_id=vid, x_value=val, score=frac_score, reason="fractional"))
        cands.sort(key=lambda c: c.score, reverse=True)
        # 落盘候选
        self._dump_node_json(node, "candidates", cands[:50])  # 仅保存前50条以免爆文件
        self.logger.info("[Node %d] Fractional candidates: %d (top frac=%.4f)",
                         node.id, len(cands), cands[0].score if cands else -1.0)
        return cands

    # ---------------------------
    # 强分支：对子问题 x=0/1 做限时 LP
    # ---------------------------
    def _strong_branching(self, node: Any, cands: List[BranchCandidate], incumbent_obj: float) -> List[BranchCandidate]:
        """
        对每个候选变量，分别临时固定为0和1，限时求解LP，获得左右子树的下界。
        评分：min(child LB)，越大越好（意味着更强的割界）。
        支持早停：若已出现明显优于当前的候选则提前终止。
        返回评估后的候选列表。
        """
        be = node.rmp.backend
        model = be.model

        # 备份求解参数
        old_tl = getattr(model.Params, "TimeLimit", None)
        old_msg = getattr(model.Params, "OutputFlag", 1)

        # 降低噪声，但仍保留日志到 solver 文件
        model.Params.OutputFlag = 0

        best_score = -float("inf")
        evaluated: List[BranchCandidate] = []

        for c in cands:
            var = be.x_vars.get(c.var_id)
            if var is None:
                continue

            # --- 探测 x=0
            lb0, ub0 = var.LB, var.UB
            t0 = time.time()
            down_lb = self._probe_child_bound(model, var, fix_value=0.0, time_limit=self.params.strong_time_limit)
            # 恢复
            var.LB, var.UB = lb0, ub0

            # --- 探测 x=1
            up_lb = self._probe_child_bound(model, var, fix_value=1.0, time_limit=self.params.strong_time_limit)
            # 恢复
            var.LB, var.UB = lb0, ub0

            c.down_lb = down_lb
            c.up_lb = up_lb
            c.eval_time = time.time() - t0

            # 评分：min(child LB)，越大越好（意味着更强的割界）
            if (down_lb is not None) and (up_lb is not None):
                c.score = min(down_lb, up_lb)
                c.reason = "strong:min_child_lb"
            elif (down_lb is not None) or (up_lb is not None):
                # 半信息：取可用的一侧
                c.score = (down_lb if down_lb is not None else up_lb)
                c.reason = "strong:partial"
            else:
                # 回退到分数化
                c.reason = "strong:fallback"

            evaluated.append(c)
            best_score = max(best_score, c.score if c.score is not None else -float("inf"))

            # 早停：若有明显优于当前的候选且启用 early stop
            if self.params.strong_stop_early and len(evaluated) >= 2:
                # 简单早停准则：当前最好 min(child LB) 已 >= incumbent_obj（或超过当前节点下界较多）
                if incumbent_obj < float("inf") and best_score >= incumbent_obj - 1e-8:
                    self.logger.info("[Node %d] Strong branching early stop (score>=incumbent).", node.id)
                    break

        # 恢复参数
        if old_tl is not None:
            model.Params.TimeLimit = old_tl
        model.Params.OutputFlag = old_msg

        # 排序：强分支优先（有 strong 标记的靠前），其次分数化回退
        evaluated.sort(key=lambda c: (c.reason.startswith("strong"), c.score), reverse=True)
        # 落盘
        self._dump_node_json(node, "strong_eval", evaluated)
        return evaluated

    def _probe_child_bound(self, model: Any, var: Any, fix_value: float, time_limit: float) -> Optional[float]:
        """
        临时固定 var 到 fix_value，限时优化，返回目标值（下界）。
        若不可行则返回 +inf，其他异常返回 None。
        """
        try:
            lb0, ub0 = var.LB, var.UB
            var.LB, var.UB = fix_value, fix_value
            model.Params.TimeLimit = time_limit
            model.optimize()
            # Gurobi 状态：2=OPTIMAL, 11=TIME_LIMIT, 5=INFEASIBLE
            if model.Status in (2, 11):
                return float(model.ObjVal)
            elif model.Status == 5:
                # 不可行 → 下界视为 +inf（非常强）
                return float("inf")
            else:
                return None
        except Exception:
            return None
        finally:
            # 由调用者恢复 LB/UB 与 TimeLimit
            pass

    # ---------------------------
    # 伪成本估计（可选）
    # ---------------------------
    def _pseudocost_estimate(self, node: Any, cands: List[BranchCandidate]) -> List[BranchCandidate]:
        """
        用历史伪成本（pc_up/pc_down）估算分支效果。
        若观测次数不足则回退分数化评分。
        返回估算后的候选列表。
        """
        estd: List[BranchCandidate] = []
        for c in cands:
            pc = self.pseudocost.get(c.var_id)
            if pc and pc.get("obs", 0) >= self.params.pseudocost_min_obs:
                avg_pc = 0.5 * (pc.get("pc_up", 0.0) + pc.get("pc_down", 0.0))
                est_score = (0.5 - abs(c.x_value - 0.5)) * max(0.0, avg_pc)
                estd.append(BranchCandidate(
                    var_id=c.var_id, x_value=c.x_value, score=est_score, reason="pseudocost:avg"))
            else:
                estd.append(c)  # 回退：沿用分数化
        # 记录
        self._dump_node_json(node, "pseudocost_eval", estd)
        return estd

    def update_pseudocost(self, var_id: str, parent_lb: float,
                          left_lb: Optional[float], right_lb: Optional[float],
                          x_value: float):
        """
        在上层“分支求解完子节点”后可调用，更新伪成本观测：
          ΔLB_left  ≈ left_lb  - parent_lb       （x从 ~x_value -> 1）
          ΔLB_right ≈ right_lb - parent_lb       （x从 ~x_value -> 0）
        采用累计平均更新伪成本。
        """
        rec = self.pseudocost.setdefault(var_id, {"obs": 0.0, "pc_up": 0.0, "pc_down": 0.0})
        rec["obs"] = rec.get("obs", 0.0) + 1.0

        if left_lb is not None:
            delta_up = max(0.0, left_lb - parent_lb)
            # 简单累计平均
            rec["pc_up"] = (rec.get("pc_up", 0.0) * (rec["obs"] - 1.0) + delta_up) / rec["obs"]

        if right_lb is not None:
            delta_down = max(0.0, right_lb - parent_lb)
            rec["pc_down"] = (rec.get("pc_down", 0.0) * (rec["obs"] - 1.0) + delta_down) / rec["obs"]

    # ---------------------------
    # 选择策略
    # ---------------------------
    def _choose_strong_or_fallback(self, cands_eval: List[BranchCandidate],
                                   cands_fraction: List[BranchCandidate]) -> Optional[BranchCandidate]:
        """
        根据强分支评估结果选择分支变量。
        若有强分支评分则选分数最高者，否则回退分数化评分。
        """
        if cands_eval:
            # 有强分支评分 → 取 score 最大者
            cands_eval.sort(key=lambda c: (c.reason.startswith("strong"), c.score), reverse=True)
            return cands_eval[0]
        # 回退：最分数化
        return cands_fraction[0] if cands_fraction else None

    # ---------------------------
    # 修复构造（左右子树）
    # ---------------------------
    def _build_fixes(self, var_id: str) -> tuple[list[BranchFix], list[BranchFix]]:
        """
        构造左右子树的修复约束。
        左子树：x=1，右子树：x=0。
        """
        return [BranchFix(var_id, 1.0, 1.0)], [BranchFix(var_id, 0.0, 0.0)]

    # ---------------------------
    # 落盘日志/产物
    # ---------------------------
    def _dump_node_json(self, node: Any, tag: str, cands: List[BranchCandidate]):
        """
        将当前节点的候选变量信息以JSON格式保存到文件，便于调试和分析。
        """
        path = os.path.join(self.params.outdir, f"node_{node.id}_{tag}.json")
        payload = {
            "node_id": node.id,
            "depth": node.depth,
            "lower_bound": node.rmp.get_objective_value(),
            "candidates": [c.__dict__ for c in cands]
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning("Write %s failed: %s", path, e)

    def _dump_decision(self, node: Any, dec: BranchDecision):
        """
        将分支决策结果保存为JSON和CSV文件，并输出到日志。
        包含被选中的变量、左右修复、所有候选等信息。
        """
        # 单节点详单 JSON（保留完整信息）
        path = os.path.join(self.params.outdir, f"node_{node.id}_decision.json")
        payload = {
            "node_id": node.id,
            "depth": node.depth,
            "lower_bound": node.rmp.get_objective_value(),
            "decision": {
                "chosen": dec.chosen.__dict__ if dec.chosen else None,
                "reason": dec.reason,
                "left_fixes": [fx.__dict__ for fx in dec.left_fixes],
                "right_fixes": [fx.__dict__ for fx in dec.right_fixes],
            },
            "candidates": [c.__dict__ for c in dec.candidates]
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning("Write %s failed: %s", path, e)

        # CSV：写入所有候选变量，标记 chosen
        try:
            with open(self._csv_path, "a", encoding="utf-8") as f:
                for cand in dec.candidates:
                    is_chosen = (dec.chosen is not None and cand.var_id == dec.chosen.var_id)
                    f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                        node.id, node.depth,
                        cand.var_id,
                        f"{cand.x_value:.6f}",
                        f"{cand.down_lb:.6f}" if cand.down_lb is not None else "",
                        f"{cand.up_lb:.6f}" if cand.up_lb is not None else "",
                        f"{cand.score:.6f}",
                        cand.reason.replace(",", ";"),
                        f"{cand.eval_time:.4f}",
                        "CHOSEN" if is_chosen else ""
                    ))
        except Exception as e:
            self.logger.warning("Append CSV failed: %s", e)

        # 控制台摘要
        if dec.chosen:
            self.logger.info(
                "[Node %d] Branch on %s (x=%.4f): score=%.6g, down_lb=%s, up_lb=%s | %s",
                node.id, dec.chosen.var_id, dec.chosen.x_value,
                dec.chosen.score,
                (f"{dec.chosen.down_lb:.6f}" if dec.chosen.down_lb is not None else "NA"),
                (f"{dec.chosen.up_lb:.6f}" if dec.chosen.up_lb is not None else "NA"),
                dec.reason
            )
        else:
            self.logger.info("[Node %d] No decision.", node.id)
