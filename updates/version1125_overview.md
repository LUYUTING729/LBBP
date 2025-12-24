# TruckSolver v1125 改动说明（定位版）

## 修改日期
2025-11-25

## 改动目标
- 解决 TruckSolver 在 LB-Benders 流程中频繁判定 infeasible 的问题。
- 放松端点时间窗、显式建模等待时间，确保有解时不被误报不可行。
- 强化日志与调试产物，便于快速定位真实不可行的根因。

## 主要改动与定位

### 1) 端点时间窗重算（`truck_solver.py`）
- **位置**：`TruckSolver._collect_endpoints()` 一节。
- **内容**：
  - 从列的 `path/served_set` 自动抽取首尾节点；首节点时间窗放宽为 `[0, latest_departure]`，终节点最早到达改为 `col_cost + service_sum` 与原时间窗上界对齐，避免窗口过紧。
  - 无选中列时直接返回可行零成本解，避免空集也跑 MIP。
- **目的**：减少因窗口构造过窄导致的伪 infeasible。

### 2) VRPTW 模型支持等待（`truck_solver.py`）
- **位置**：`TruckSolver._build_vrptw()` 与序约束部分。
- **内容**：
  - 新增到达 `T_i`、服务开始 `S_i`、等待 `W_i`，约束 `W_i = S_i - T_i`、`S_i >= T_i`，允许卡车提前到达后等待。
  - 序约束调整为 `T_j ≥ T_i + W_i + service_i + t_ij - M(1 - y_ij)`，`bigM` 由 `TruckParams.bigM_time` 控制。
  - 引入独立终点仓（默认 id=21）：起点只出不入、终点只入不出，避免必须回到起点造成假不可行。
- **目的**：让可等待的时间窗问题有解时能被模型捕捉，而不是被硬性时间传播卡死。

### 3) 结果读取与冲突返回（`truck_solver.py`）
- **位置**：`TruckSolver.evaluate()` 收尾与 `_extract_solution()`。
  - 可行时返回真实成本与路径；不可行或异常时返回 `feasible=False`，供上层产生冲突割。
  - 统一日志：求解状态、目标值、SolCount；无 incumbent 时直接标记不可行。

### 4) LB-Benders 对接（`bp_drones.py`）
- **位置**：`TruckSolverCheck.evaluate()`。
- **内容**：
  - 直接消费新的 `TruckSolver.evaluate` 结果：可行则用真实卡车成本加 θ 割；不可行则仍把当前列集合作为冲突加入可行性割。
  - 日志包含列数、可行性、卡车成本与冲突数，便于节点级定位。
- **目的**：确保误报 infeasible 的概率降低后，节点能继续用 θ 割收紧下界。

### 5) 调试与可复现性（`truck_solver.py`）
- **位置**：`TruckSolver.evaluate()` 内部。
- **内容**：
  - 每次构建模型都导出 LP 至 `truck_logs/truck_vrptw.lp`，Gurobi 日志写入 `truck_logs/gurobi_truck.log`。
  - 详细打印端点集合、时间窗、变量规模与前几条边的旅行时间，辅助重现场景。

### 6) 测试脚本覆盖（`test_truck_solver.py`, `test_lbbp.py`）
- **内容**：
  - `test_truck_solver_with_csv` 聚合 CSV 列求卡车可行性，输出冲突节点与成本，便于针对性复现。
  - LBBP 端到端测试沿用新的 `TruckParams`，可验证整体流程在随机实例下不再因卡车误判中断。

## 参数与调优建议
- `TruckParams.bigM_time`：若仍出现不必要的等待放松，可减小；若序约束仍过紧，可适度增大。
- `TruckParams.time_limit`：单次 VRPTW 求解时限；调低可加快分支节点迭代，调高可提升可行性发现概率。
- `TruckParams.truck_speed`：若距离单位或尺度变化，请同步调整，以免 travel time 偏大导致时间窗冲突。

## 排查指引
1. 复现：使用触发 infeasible 的列集合，查看 `truck_logs/truck_vrptw.lp` 与 `gurobi_truck.log`。
2. 观察：端点时间窗是否明显过窄（尤其终点最早到达值）；travel time 是否异常大。
3. 调整：按需修改 `bigM_time`、`truck_speed` 或节点时间窗后重跑 `test_truck_solver_with_csv`。

## 回归测试建议
- 运行 `python test_truck_solver.py`（需先生成 `bp_logs/rmp/selected_columns.csv`），确认：
  - 可行场景不再误报 infeasible；
  - 可行时返回合理成本、路径长度与到达时间。
- 运行 `python test_lbbp.py`，检查分支节点在可行时能产生 θ 割并更新全局上界，不再因卡车子问题阻塞。

## 兼容性
- 与现有 RMP 接口保持一致：`add_conflict_cut()`、`add_theta_cut()` 均未改签名。
- `Problem` 仍通过 `truck: TruckParams` 传入卡车参数；无额外配置需求。


## 后续工作
- 目前lb割仅执行一轮，需检查。
- 进一步优化 VRPTW 求解效率，缩短单节点求解时间。
- 数值实验