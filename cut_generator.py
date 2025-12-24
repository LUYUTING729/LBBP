from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Union

from data_model import Problem

CutTuple = Tuple[str, Iterable[int], Union[int, float]]


@dataclass
class BucketParams:
    enabled: bool = False
    target: str = "candidates"  # "static" or "candidates"
    num_buckets: int = 4
    field: str = "latest"  # "earliest" or "latest"
    beta_mode: str = "size"  # "size", "ratio", "fixed"
    beta_value: Optional[float] = None
    beta_ratio: Optional[float] = None
    min_size: int = 2
    max_cuts: int = 200
    key_prefix: str = "sri_tw"
    skip_redundant: bool = True


@dataclass
class SRIConfig:
    source: str = "none"  # "none", "manual", "file", "bucket"
    static: List[CutTuple] = field(default_factory=list)
    candidates: List[CutTuple] = field(default_factory=list)
    file_path: Optional[str] = None
    bucket: BucketParams = field(default_factory=BucketParams)


@dataclass
class CliqueConfig:
    source: str = "none"  # "none", "manual", "file"
    static: List[CutTuple] = field(default_factory=list)
    candidates: List[CutTuple] = field(default_factory=list)
    file_path: Optional[str] = None


@dataclass
class CutGenParams:
    enabled: bool = False
    append: bool = True
    sri: SRIConfig = field(default_factory=SRIConfig)
    clique: CliqueConfig = field(default_factory=CliqueConfig)


@dataclass
class CutSets:
    sri_static: List[CutTuple] = field(default_factory=list)
    sri_candidates: List[CutTuple] = field(default_factory=list)
    clique_static: List[CutTuple] = field(default_factory=list)
    clique_candidates: List[CutTuple] = field(default_factory=list)


def build_cut_sets(problem: Problem,
                   params: CutGenParams,
                   logger) -> CutSets:
    cuts = CutSets()
    if not params.enabled:
        return cuts

    if params.sri.source == "manual":
        cuts.sri_static.extend(_normalize_cut_list(params.sri.static))
        cuts.sri_candidates.extend(_normalize_cut_list(params.sri.candidates))
    elif params.sri.source == "file" and params.sri.file_path:
        file_cuts = _load_cuts_from_file(params.sri.file_path, logger)
        cuts.sri_static.extend(file_cuts.sri_static)
        cuts.sri_candidates.extend(file_cuts.sri_candidates)
    elif params.sri.source == "bucket" and params.sri.bucket.enabled:
        bucket_cuts = _generate_sri_buckets(problem, params.sri.bucket, logger)
        if params.sri.bucket.target == "static":
            cuts.sri_static.extend(bucket_cuts)
        else:
            cuts.sri_candidates.extend(bucket_cuts)

    if params.clique.source == "manual":
        cuts.clique_static.extend(_normalize_cut_list(params.clique.static))
        cuts.clique_candidates.extend(_normalize_cut_list(params.clique.candidates))
    elif params.clique.source == "file" and params.clique.file_path:
        file_cuts = _load_cuts_from_file(params.clique.file_path, logger)
        cuts.clique_static.extend(file_cuts.clique_static)
        cuts.clique_candidates.extend(file_cuts.clique_candidates)

    return cuts


def _normalize_cut_list(items: Iterable) -> List[CutTuple]:
    normalized: List[CutTuple] = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            key, sset, rhs = item
        elif isinstance(item, dict):
            key = item.get("key") or item.get("name")
            sset = item.get("set") or item.get("S") or item.get("Q")
            rhs = item.get("beta") if "beta" in item else item.get("rhs")
        else:
            continue
        if key is None or sset is None or rhs is None:
            continue
        normalized.append((str(key), list(int(i) for i in sset), float(rhs)))
    return normalized


def _load_cuts_from_file(path: str, logger) -> CutSets:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Cut file load failed: %s", e)
        return CutSets()

    def get_list(name: str) -> List[CutTuple]:
        return _normalize_cut_list(data.get(name, []))

    return CutSets(
        sri_static=get_list("sri_static"),
        sri_candidates=get_list("sri_candidates"),
        clique_static=get_list("clique_static"),
        clique_candidates=get_list("clique_candidates"),
    )


def _generate_sri_buckets(problem: Problem,
                          params: BucketParams,
                          logger) -> List[CutTuple]:
    customers = list(problem.customers)
    if not customers or params.num_buckets <= 0:
        return []

    def sort_key(cid: int) -> float:
        tw = problem.nodes[cid].tw
        return tw[0] if params.field == "earliest" else tw[1]

    customers.sort(key=sort_key)
    bucket_size = max(1, math.ceil(len(customers) / params.num_buckets))
    buckets = [customers[i:i + bucket_size] for i in range(0, len(customers), bucket_size)]

    cuts: List[CutTuple] = []
    for idx, bucket in enumerate(buckets):
        if len(bucket) < params.min_size:
            continue
        beta = _resolve_beta(len(bucket), params)
        if beta is None:
            continue
        if params.skip_redundant and beta >= len(bucket):
            continue
        key = f"{params.key_prefix}_{idx}"
        cuts.append((key, bucket, beta))
        if len(cuts) >= params.max_cuts:
            break

    if not cuts:
        logger.info("SRI bucket generator produced 0 cuts (check beta settings).")
    return cuts


def _resolve_beta(size: int, params: BucketParams) -> Optional[float]:
    if params.beta_mode == "fixed":
        if params.beta_value is None:
            return None
        return float(params.beta_value)
    if params.beta_mode == "ratio":
        if params.beta_ratio is None:
            return None
        return float(math.floor(size * params.beta_ratio))
    return float(size)
