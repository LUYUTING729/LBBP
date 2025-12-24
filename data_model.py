from dataclasses import dataclass
from typing import List, Tuple
import math

@dataclass
class Node:
    id: int
    demand: float
    tw: Tuple[float, float]  # (earliest, latest)
    service: float = 0.0
    x: float = 0.0
    y: float = 0.0

@dataclass
class DroneSpec:
    capacity: float    # 载重上限（kg）
    endurance: float   # 续航上限（按飞行时间）
    speed: float       # 飞行速度（距离/时间）

@dataclass
class TruckParams:
    truck_speed: float            # 卡车速度（距离/时间）
    truck_cost_per_time: float    # 卡车单位时间成本
    bigM_time: float              # 时间约束用的大 M
    time_limit: float 
    iis_on_infeasible: bool = False
    iis_mode: str = "tw"
    iis_max_nodes: int = 20
    
@dataclass
class Problem:
    nodes: List[Node]       # 0..n；通常 0 可当作 depot（非必须）
    customers: List[int]    # 客户索引集合
    drone: DroneSpec
    truck: TruckParams

    def dist(self, i: int, j: int) -> float:
        ni, nj = self.nodes[i], self.nodes[j]
        return math.hypot(ni.x - nj.x, ni.y - nj.y)

    def flight_time(self, i: int, j: int) -> float:
        return self.dist(i, j) / self.drone.speed
    
    def truck_time(self, i: int, j: int) -> float:
        return self.dist(i, j) / self.truck.truck_speed 
