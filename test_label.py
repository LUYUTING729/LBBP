import random
import logging
from data_model import Node, DroneSpec, Problem, TruckParams
from label_setting import label_setting

# 配置日志
logging.basicConfig(
    filename="label_setting.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("LabelSettingTest")

def gen_customers(n=20, seed=42):
    random.seed(seed)
    customers = []
    for cid in range(1, n + 1):
        x = random.uniform(-12, 12)
        y = random.uniform(-12, 12)
        start = random.uniform(0, 60)
        width = random.uniform(60, 120)
        tw = (start, start + width)
        demand = round(random.uniform(0.5, 2.0), 2)
        service = random.uniform(0.5, 2.0)
        customers.append(Node(cid, demand, tw, service, x, y))
    return customers

if __name__ == "__main__":
    depot = Node(0, demand=0.0, tw=(0, 180), service=0.0, x=0.0, y=0.0)
    cust_nodes = gen_customers(n=20, seed=42)
    nodes = [depot] + cust_nodes
    customer_ids = [n.id for n in cust_nodes]
    truck = TruckParams(
        truck_speed=4,            # 卡车速度（距离/时间）         
        truck_cost_per_time=1.0,    # 卡车单位时间成本
        bigM_time=1e5,              # 时间约束用的大 M
        time_limit=300              # 每个 VRPTW 求解的时间限制
    )
    drone = DroneSpec(capacity=8.0, endurance=60.0, speed=1.5)
    prob = Problem(nodes=nodes, customers=customer_ids, drone=drone, truck=truck)
    logger.info("Problem instance created with %d customers.", len(customer_ids))   
    sols = label_setting(prob)

    logger.info(f"Total solutions found: {len(sols)}")
    depot_sols = [lab for lab in sols if lab.path and lab.path[-1] == 0]
    logger.info(f"Depot-ending solutions: {len(depot_sols)}")

    # 输出前 10 个解
    for lab in sols[:10]:
        logger.info(
            f"Path={lab.path}, cost={lab.cost:.2f}, "
            f"latest_departure={lab.latest_departure:.2f}, load={lab.load:.2f}, "
            f"energy={lab.energy:.2f}"
        )

    print("运行完成，详情请查看 label_setting.log 日志文件。")
