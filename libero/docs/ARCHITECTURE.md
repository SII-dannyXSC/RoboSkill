# 项目架构

## 核心模型

Gateway 将 Run 作为一次可比较评测，将 Session 作为一次 Episode 和资源隔离单位：

```text
POST /v2/runs（锁定配置并开始计时）
       ↓
可信启动器启动 Agent
       ↓
POST /v2/sessions
       ↓
认证与配置校验
       ↓
GPU Slot Admission Controller
       ↓
启动一个独立 LIBERO 子进程
       ↓
立即返回 session_id；客户端轮询至 ready
```

不存在固定 Worker 池。没有 Session 时没有仿真子进程；有 N 个活跃 Session 时按需存在 N 个子进程。

服务端仍设置资源硬上限。例如 4 张 GPU、每卡最多 2 个 Session 时，总容量为 8。客户端只请求 Session，不选择 GPU 或进程。

## Session 生命周期

```text
starting → ready → running → completed → closed
                         ↘ failed → closed
```

- `create`：原子检查身份配额并占用 slot，立即返回，再在后台启动仿真。
- `poll`：`GET /v2/sessions/{id}` 查询 starting/ready/failed；幂等键防止重试重复创建。
- `reset`：每个 Session 只允许一次。
- `step`：同一 Session 串行执行。
- `result`：返回服务端判定的状态、成功与步数。
- `close`：结束进程并释放 slot。
- 空闲超时、子进程死亡和仿真超时也会触发清理。

## Run 生命周期与首次成功

```text
running/measuring → running/first_success_recorded → finished
                  ↘ finished_without_success
                  ↘ interrupted（网关重启）
```

一个 Run 固定 task、Episode 长度、Observation Level、bbox scope 和最大尝试次数。所有 Session
必须绑定 Run 且配置相同；同一身份一次只能有一个运行中的 Run。Run 开始时间在服务端接受可信启动
请求时采样。成功时间在仿真子进程执行 `check_success()` 后立即采样，早于 Observation 序列化；
Gateway 使用 SQLite 原子 compare-and-set，仅保留第一次成功并在响应前提交。

## 用户可配置项

创建 Session 时可以设置：

- `task.benchmark`
- `task.task_id`，省略时随机采样
- `episode_length`
- `observation.level`
- `observation.bbox_scope`
- `seed`，仅获得固定 seed 权限的调试 Token 可选

用户可先调用 `GET /v2/tasks/{benchmark}` 获取该任务集中的 `task_id` 和自然语言指令，再选择任务；客户端不能提交任意 BDDL 或代码。

服务端返回 `session_config`，明确记录本次评测的任务、长度和观测条件。正式对比时应只比较配置完全相同的结果。

## Observation Levels

Level 是累进的，且创建后锁定。

### Level 1

- `agentview_rgb`
- `wrist_rgb`
- 关节位置、夹爪位置
- 末端位置和四元数

### Level 2

在 Level 1 上增加任务相关实体的可见区域 2D bbox。bbox 来自服务端内部实例分割渲染，但只返回框，不返回分割图、数字实例 ID 或 3D 物体状态。

`bbox_scope`：

- `initial`：只在 Reset 帧返回。
- `every_frame`：每帧更新，信息更强，结果必须单独统计。

### Level 3

在 Level 2 上增加：

- 关节和夹爪速度。
- 控制器施加的关节命令力矩。
- 末端力、力矩及六维速度。

### Level 4

在 Level 3 上增加：

- 米制 float32 深度图。
- 3×3 相机内参。
- OpenCV 坐标约定下的 4×4 `camera_to_world`。

手腕相机外参随机器人运动，因此每帧更新。

不返回 MuJoCo 接触列表、物体受力真值或求解器内部状态。

## 权限与隔离

每个 Token 包含：

- 可访问 benchmark。
- 最大并发 Session 数。
- 最大 Observation Level。
- 最大 Episode 步数。
- 是否允许固定 seed（默认不允许）。

Session 所有权由 Token 决定。Agent 不能通过请求体声明或切换身份，也不能操作其他 Token 创建的 Session。
Run 同样绑定 Token；跨 Token 查询返回 404。Run 期间未绑定的 v1 Session 被拒绝，避免旁路试跑。

服务端仅序列化对应 Level 的白名单字段。Level 1/2 不会因仿真原始 observation 中存在深度、相机或物体状态而意外返回这些数据。

## 公平性

以下配置会显著改变任务难度，结果不能混合比较：

- Observation Level 或 bbox scope。
- Episode 长度。
- Task、seed 或初始状态。
- 图像分辨率、控制频率和环境版本。

建议正式评测由外部实验配置固定这些参数；开发调试时再允许自由选择。
