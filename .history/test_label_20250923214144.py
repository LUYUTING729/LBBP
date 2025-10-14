from data_model import Node, DroneSpec, Problem
from label_setting import label_setting

if __name__ == "__main__":
    # 节点定义（0 可作为 depot，但算法不强制从 0 出发/到 0 结束）
    nodes = [
        Node(0, demand=0, tw=(0, 100), x=0, y=0),        # 可选 depot
        Node(1, demand=1, tw=(0, 50),  x=2, y=2),
        Node(2, demand=1, tw=(0, 60),  x=3, y=1),
        Node(3, demand=2, tw=(10, 80), x=6, y=0),
    ]
    # 无人机：容量3kg，续航20分钟，速度=1 距离单位/分钟
    drone = DroneSpec(capacity=3, endurance=20.0, speed=1.0)
    prob = Problem(nodes=nodes, customers=[1, 2, 3], drone=drone)

    sols = label_setting(prob, max_len=3, depot_idx=0)

    print("Found routes (showing first 10):")
    for lab in sols[:10]:
        # 打印路径、累计成本与可行最晚出发时间
        print(f"path={lab.path}, "
              f"cost={lab.cost:.2f}, "
              f"latest_departure={lab.latest_departure:.2f}, "
              f"end_node={lab.node}, "
              f"served_customers={sum(1 for v in lab.path if v in prob.customers)}")
