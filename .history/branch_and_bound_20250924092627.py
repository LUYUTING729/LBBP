# branch_and_bound.py
from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

# ---------------------------
# 参数与数据类
# ---------------------------

@dataclass
class BnBParams:
    # 策略：auto（优先强分支，回退伪成本/最分数化）、strong、pseudocost、most_fractional
    strategy: str = "auto"

    # Strong Branching 探测
    strong_top_k: int = 10
    strong_time_limit: float = 1.0       # 每个子问题限时（秒）
    strong_stop_early: bool = True       # 若已出现明显更优的候选，提前停止

    # 伪成本（基于历史观测的快速估计）
    use_pseudocost: bool = True
    pseudocost_min_obs: int = 2          # 至少多少次观测后才信任伪成本

    # 分数化阈值
    frac_eps: float = 1e-6

    # 日志
    outdir: str = "bp_logs/branch"
    log_level: int = logging.INFO
    csv_name: str = "branch_decisions.csv"


@dataclass
class BranchCandidate:
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
    var_id: str
    lb: float
    ub: float


@dataclass
class BranchDecision:
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
        self.params = params or BnBParams()
        os.makedirs(self.params.outdir, exist_ok=True)

        # 日志器
        self.logger = logger or logging.getLogger("BnB")
        self.logger.setLevel(self.params.log_level)
        fh = logging.FileHandler(os.path.join(self.params.outdir, "branch.log"), mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(fh)

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
        输入：
          node：上层的 BranchNode（需有 rmp、id、depth）
          incumbent_obj：当前全局最优整数解（上界）
        输出：
          BranchDecision 或 None（若无需分支/可剪）
        """
        # 1) 收集候选（分数化）
        cands_frac = self._collect_fractional_candidates(node)
        if not cands_frac:
            self._dump_node_json(node, "no_candidates", [])
            self.logger.info("[Node %d] No fractional variables; pruning or integer.", node.id)
            return None

        # 2) 评估（根据策略）
        if self.params.strategy in ("auto", "strong"):
            cands_eval = self._strong_branching(node, cands_frac[:self.params.strong_top_k], incumbent_obj)
            chosen = self._choose_strong_or_fallback(cands_eval, cands_frac)
            reason = chosen.reason if chosen and chosen.reason.startswith("strong") else "fractional_fallback"

        elif self.params.strategy == "pseudocost":
            cands_eval = self._pseudocost_estimate(node, cands_frac)
            chosen = max(cands_eval, key=lambda c: c.score) if cands_eval else cands_frac[0]
            reason = chosen.reason

        else:  # "most_fractional"
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
        """临时固定 var 到 fix_value，限时优化，返回目标值；失败则返回 None"""
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
        # 用历史伪成本（pc_up/pc_down）估算 min(child LB)
        # 这里的示例：score = frac_gain * avg(pc_up, pc_down)
        # 若观测不足 → 回退分数化分
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
        # 左：x=1，右：x=0
        return [BranchFix(var_id, 1.0, 1.0)], [BranchFix(var_id, 0.0, 0.0)]

    # ---------------------------
    # 落盘日志/产物
    # ---------------------------
    def _dump_node_json(self, node: Any, tag: str, cands: List[BranchCandidate]):
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
        # 单节点详单
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

        # 汇总 CSV
        try:
            with open(self._csv_path, "a", encoding="utf-8") as f:
                ch = dec.chosen
                f.write("{},{},{},{},{},{},{},{},{}\n".format(
                    node.id, node.depth,
                    (ch.var_id if ch else ""), (f"{ch.x_value:.6f}" if ch else ""),
                    (f"{ch.down_lb:.6f}" if (ch and ch.down_lb is not None) else ""),
                    (f"{ch.up_lb:.6f}" if (ch and ch.up_lb is not None) else ""),
                    (f"{ch.score:.6f}" if ch else ""),
                    dec.reason.replace(",", ";"),
                    (f"{ch.eval_time:.4f}" if ch else "0.0"),
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
