import random
from data_model import Node, DroneSpec, Problem
from label_setting import label_setting

def gen_customers(n=20, seed=42):
    """
    生成 n 个客户：
    - 坐标：在 [-12, 12] × [-12, 12] 随机分布
    - 时间窗：仓库运营期内随机，长度 60~120
    - 需求：0.5 ~ 2.0
    """
    random.seed(seed)
    customers = []
    for cid in range(1, n + 1):
        x = random.uniform(-12, 12)
        y = random.uniform(-12, 12)

        # 生成较宽松但有差异的时间窗
        start = random.uniform(0, 60)           # 最早到达 0~60
        width = random.uniform(60, 120)         # 时间窗长度 60~120
        tw = (start, start + width)

        demand = round(random.uniform(0.5, 2.0), 2)
        service = random.uniform(0.5, 2.0)      # 服务时间 0.5~2.0

        customers.append(Node(cid, demand=demand, tw=tw, service=service, x=x, y=y))
    return customers

if __name__ == "__main__":
    # 仓库（可作为回仓点用；也可以不强制使用）
    depot = Node(0, demand=0.0, tw=(0, 180), service=0.0, x=0.0, y=0.0)

    # 生成 20 个客户
    cust_nodes = gen_customers(n=20, seed=42)
    nodes = [depot] + cust_nodes
    customer_ids = [n.id for n in cust_nodes]

    # 无人机参数（可根据实例难度调整）
    # - 容量 8 kg
    # - 续航 60 分钟（飞行时间）
    # - 速度 1.5 距离单位/分钟
    drone = DroneSpec(capacity=8.0, endurance=60.0, speed=1.5)

    # 构建问题
    prob = Problem(nodes=nodes, customers=customer_ids, drone=drone)

    # 运行 label-setting
    # 注意：为了避免组合爆炸，这里限制每条路径最多访问 4 个客户
    sols = label_setting(prob, max_len=4, depot_idx=0)

    # 统计
    print(f"Total labels (solutions) generated: {len(sols)}")

    # 过滤出“以仓库为终点”的解，便于观察回仓路径
    sols_to_depot = [lab for lab in sols if lab.path and lab.path[-1] == 0]
    print(f"Solutions ending at depot: {len(sols_to_depot)}")

    # 展示：按 cost 升序选取前 15 条
    sols_sorted = sorted(sols, key=lambda L: L.cost)
    show_k = min(15, len(sols_sorted))
    print(f"\nTop {show_k} solutions by cost:")
    for lab in sols_sorted[:show_k]:
        served = sum(1 for v in lab.path if v in prob.customers)
        print(f"  path={lab.path}, "
              f"served={served}, "
              f"cost={lab.cost:.2f}, "
              f"latest_departure={lab.latest_departure:.2f}, "
              f"end_node={lab.node}")

    # 展示：按 latest_departure 降序选取 10 条（便于观察可延迟空间大的路径）
    sols_by_latest = sorted(sols, key=lambda L: L.latest_departure, reverse=True)
    show_k2 = min(10, len(sols_by_latest))
    print(f"\nTop {show_k2} solutions by latest_departure:")
    for lab in sols_by_latest[:show_k2]:
        served = sum(1 for v in lab.path if v in prob.customers)
        print(f"  path={lab.path}, "
              f"served={served}, "
              f"cost={lab.cost:.2f}, "
              f"latest_departure={lab.latest_departure:.2f}, "
              f"end_node={lab.node}")

    # 展示：随机抽样 10 条“回仓”解
    sample_k = min(10, len(sols_to_depot))
    if sample_k > 0:
        random.seed(123)
        sample = random.sample(sols_to_depot, sample_k)
        print(f"\nRandom {sample_k} depot-ending solutions:")
        for lab in sample:
            served = sum(1 for v in lab.path if v in prob.customers)
            print(f"  path={lab.path}, "
                  f"served={served}, "
                  f"cost={lab.cost:.2f}, "
                  f"latest_departure={lab.latest_departure:.2f})")
