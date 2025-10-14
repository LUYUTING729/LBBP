# column_pool.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import json
import logging
from rmp import Column

class ColumnPool:
    """
    列池：维护所有历史生成过的列（Column）。
    - 去重：按 path(tuple) 唯一
    - 每次可根据当前对偶变量重新计算 reduced cost
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.pool: Dict[Tuple[int, ...], Column] = {}
        self.logger = logger or logging.getLogger("ColumnPool")

    # --- 添加列 ---
    def add_columns(self, cols: List[Column]) -> int:
        added = 0
        for c in cols:
            key = tuple(c.path)
            if key not in self.pool:
                self.pool[key] = c
                added += 1
        self.logger.info("ColumnPool: added=%d, total=%d", added, len(self.pool))
        return added

    # --- 重新计算 reduced cost 并筛选负列 ---
    def check_negative_rc(
        self,
        duals: Dict[int, float],
        lambda_route: float = 0.0,
        tol: float = -1e-6,
        budget: Optional[int] = None
    ) -> List[Column]:
        """
        根据对偶变量重新计算 reduced cost，返回 rc < tol 的列
        - budget: 若不为 None，则只返回 rc 最小的前 k 个
        """
        candidates = []
        for col in self.pool.values():
            pi_sum = sum(duals.get(i, 0.0) for i in col.served_set)
            rc = (col.cost + lambda_route) - pi_sum
            if rc < tol:
                col.meta["rc_check"] = float(rc)
                candidates.append(col)
        candidates.sort(key=lambda c: c.meta["rc_check"])
        if budget:
            candidates = candidates[:budget]
        self.logger.info("ColumnPool: %d negative rc found (tol=%g)", len(candidates), tol)
        return candidates

    # --- 导出池子 ---
    def export_json(self, path: str):
        data = []
        for col in self.pool.values():
            data.append({
                "id": col.id,
                "path": col.path,
                "cost": col.cost,
                "served": sorted(col.served_set),
                "meta": col.meta,
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.logger.info("ColumnPool: exported %d columns to %s", len(self.pool), path)

    # --- 池子统计 ---
    def stats(self) -> Dict[str, int]:
        return {
            "total": len(self.pool),
            "unique_paths": len(self.pool),
        }
