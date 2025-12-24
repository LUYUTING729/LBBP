"""
test_truck_solver.py
烟雾测试：验证 TruckSolver 在一个小规模实例上可以返回可行解。
"""

import logging
import os
import random
from typing import List

import pytest
import truck_solver as ts

from data_model import Node, DroneSpec, Problem
from rmp import Column
from truck_solver import TruckSolver, TruckParams
from logging_utils import init_logging


def setup_logger(outdir: str = "truck_test_logs") -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    return init_logging(outdir, name="truck_solver_test")


def gen_test_problem(n_customers: int = 3) -> Problem:
    """
    生成一个小规模的 VRPTW 实例，包含固定的终点仓库 id=21（与 TruckSolver.evaluate 内的默认保持一致）。
    """
    random.seed(42)
    start_depot = Node(0, demand=0.0, tw=(0, 180), service=0.0, x=0.0, y=0.0)
    end_depot = Node(21, demand=0.0, tw=(0, 1800), service=0.0, x=0.0, y=0.0)

    customers: List[Node] = []
    for cid in range(1, n_customers + 1):
        x = random.uniform(0, 50)
        y = random.uniform(0, 50)
        start = random.choice([0, 60, 120, 180])
        width = 60
        tw = (start, start + width)
        demand = round(random.uniform(0.5, 2.0), 2)
        service = random.uniform(0.5, 2.0)
        customers.append(Node(cid, demand=demand, tw=tw, service=service, x=x, y=y))

    nodes = [start_depot] + customers + [end_depot]
    customer_ids = [n.id for n in customers]
    drone = DroneSpec(capacity=8.0, endurance=60.0, speed=5)
    truck = TruckParams(truck_speed=5.0, truck_cost_per_time=1.0, bigM_time=1e5, time_limit=30.0)
    return Problem(nodes=nodes, customers=customer_ids, drone=drone, truck=truck)


def test_truck_solver_smoke(tmp_path):
    if ts.gp is None:
        pytest.skip("gurobipy not installed")

    outdir = tmp_path / "truck_logs"
    logger = setup_logger(str(outdir))

    prob = gen_test_problem(n_customers=3)
    # 构造几条简单的列：每条仅服务一个客户，往返仓库。
    columns = [
        Column(
            id=f"c{i}",
            path=[0, cust_id, 0],
            served_set=frozenset({cust_id}),
            cost=1.0,
            duration=0.0,
            energy=0.0,
            meta={"latest_departure": prob.nodes[cust_id].tw[1]},
        )
        for i, cust_id in enumerate(prob.customers[:3], 1)
    ]

    solver = TruckSolver(
        problem=prob,
        depot_idx=0,
        params=TruckParams(truck_speed=5.0, truck_cost_per_time=1.0, bigM_time=1e5, time_limit=30.0),
        logger=logger,
    )

    result = solver.evaluate(columns)

    assert result.feasible
    assert result.cost >= 0.0
    assert result.route[0] == 0
