# LIBERO Agent HTTP API

本原型不提供 LIBERO Harness Tool。可信控制器已经创建唯一 Run 和 Session，并在 Workspace 根目录写入 `.libero-endpoint.json`。远程 Run ID、Session ID 和 Token 不提供给 Agent。

{{LIBERO_GATE_POLICY}}

## 推荐 Python SDK

```python
import json
from pathlib import Path
from libero_sdk import LiberoClient

client = LiberoClient.from_workspace()
initial = json.loads(Path(
    "libero-artifacts/episodes/episode-0001/observations/frame-00000_reset.json"
).read_text())
state = client.step(policy(initial))  # 根据当前 profile 的准入规则执行
while client.result()["success"] is not True:
    action = policy(state["observation"])
    state = client.step(action)
    # state 已经是该动作执行后的最新状态；无需额外刷新。
```

SDK 是普通 Python HTTP 封装，Agent 可以阅读、修改或再次封装。它不创建或关闭远程 Session，也不持有远程 Token。

## 本地 HTTP 接口

Endpoint 文件：

```json
{
  "schema_version": 1,
  "endpoint": "http://127.0.0.1:19000/leases/<opaque>"
}
```

允许调用：

```http
POST {endpoint}/reset
POST {endpoint}/step
GET  {endpoint}/result
GET  {endpoint}/healthz
```

禁止创建、列举、关闭或切换 Session。Endpoint 只绑定当前实验。

### Reset

```http
POST {endpoint}/reset
Content-Type: application/json
Idempotency-Key: reset-<unique>

{}
```

Reset 不接收 seed；它复用创建 Session 时的 seed。

### Step

```http
POST {endpoint}/step
Content-Type: application/json
Idempotency-Key: step-<unique>

{"action":[0.1,-0.2,0,0,0.1,0,1]}
```

动作顺序为 `[dx,dy,dz,drx,dry,drz,gripper]`，必须恰好包含七个有限的 `[-1,1]` 数值。平移使用 MuJoCo world frame；`±1` 会把 OSC 的当前位置目标偏移最多 `±0.05 m`，不承诺该 Step 的实际 EEF 位移为 `±0.05 m`。旋转是 world-frame rotation vector；`±1` 对应 `±0.5 rad`。夹爪负值打开、正值闭合、零保持命令。

`client.step(action)` 会等待动作执行和本地持久化完成，然后直接返回该动作之后的最新状态。最新 Observation 位于返回值的 `state["observation"]`，其中的图片路径、允许的本体信息、BBox 或深度均对应这一次 Step 完成后的 Frame。下一动作必须直接根据这个返回值计算。

```python
state = client.step(action)
observation = state["observation"]  # 当前最新 Observation
next_action = policy(observation)
```

不需要、也不存在 `client.refresh()`。不要在 Step 后额外请求或自行实现 refresh；否则既不会获得比 `step()` 更新的 Observation，还可能让控制流程错误地脱离严格的 `reset → 串行 step → result` 顺序。

Step 返回值的主要结构如下；实际 Observation 字段取决于当前 Level：

```json
{
  "observation": {},
  "reward": 0.0,
  "terminated": false,
  "truncated": false,
  "step_index": 1,
  "local_artifacts": {
    "persisted": true,
    "frame_ref": "episode-0001/frame-00001_step",
    "frame_index": "/absolute/attempt/libero-artifacts/frames.jsonl"
  }
}
```

`client.reset()` 同样直接返回 Reset 后的最新状态和 Observation。只有在需要主动开始一个新 Episode 时才再次调用 `reset()`；读取最新状态不是 Reset 的用途。

SDK 和本地代理自动处理串行执行及网络恢复。`expected_step_index` 是代理内部字段，Agent 不得传入；代理会把 Step 归入最近一次成功 Reset 建立的 Episode。Agent只需根据最新 Observation 计算并提交下一动作，不需要在 Prompt、计划或工作日志中讨论幂等键、Step index 或重连实现。

### OSC 目标不是单步实际位移

每个 Step 只给 robosuite `OSC_POSE` 控制器 0.05 秒追踪目标。实际 EEF 位移受初始速度、动力学、碰撞、奇异位形和关节限制影响，通常小于名义目标偏移，也可能在 Reset 后短暂反向运动。下一 Step 的 delta 又从当时的实际 EEF 位姿计算。

因此禁止使用 `action × 0.05 m × Step 数` 推算实际位置；禁止因为固定次数循环结束就声明“没有接触”或“目标不可达”。距离、完成和停滞必须使用每个响应中的最新 `robot0_eef_pos`。

### `move_until()`：反馈闭环直线运动

SDK 的 `move_until()` 会根据真实 EEF 位姿修正横向漂移，并识别调用方停止条件、测量距离边界、停滞和 Episode 结束。代理仍会在 helper 返回每个 Step 前自动持久化完整证据。

```python
result = client.move_until(
    initial_observation,
    direction=(0.0, 0.0, -1.0),   # MuJoCo world frame；内部归一化
    max_distance_m=0.08,           # 根据测得的 EEF 位移判断
    command_magnitude=0.1,         # OSC 目标幅值，不是实际步长
    stop_when=contact_detector,    # 由调用方读取当前 Observation 判断
    settle_steps=10,               # 只用于紧接 Reset 的第一次运动
    max_steps=120,                 # 单个 primitive 的局部预算
)
observation = result.observation
```

1000-Step Episode 中不要全程使用很小的 `command_magnitude`。空旷远距离通常使用 `0.1–0.2`，进入物体附近或准备接触时再降到 `0.02–0.05`。`settle_steps=10` 只用于 Reset 后第一次运动；后续稳定状态设为 `0`。默认 `max_steps=120` 防止单个 primitive 吞掉整个 Episode，只有在仍有测量进展且总预算足够时才提高。

主要返回原因：

| `reason` | 含义 | 正确处理 |
|---|---|---|
| `condition_met` | 调用方条件成立，例如检测到接触 | 用最新视觉和本体状态判断 operation |
| `distance_limit` | 测得的方向位移达到上限 | 不得声称已经发生接触 |
| `stalled` | 当前路径窗口内进展不足 | 观察、换姿态/路径、恢复或 Reset；不等于全局不可达 |
| `cross_track_limit` | 显式设置的横向误差上限被触发 | 停止并重新规划 |
| `step_limit` | 局部 Step 预算耗尽但仍有进展 | 核对 Episode 剩余预算后决定是否继续同一 operation |
| `episode_terminated` / `episode_truncated` | Episode 已结束 | 禁止继续 Step并读取 Result |

### Result

```http
GET {endpoint}/result
```

只有服务器返回 `success=true` 才算成功。`terminated` 或 `truncated` 为真时立即停止 Step 并读取 Result。

## Observation Level 1–4

四个 Level 是严格累进的：高 Level 包含低 Level 的公开信息。Level 在创建实验时由可信控制器固定，Agent 不得修改或尝试获取更高 Level 的数据。

| Level | 在上一级基础上增加 | 主要用途 | 当前 Level 没有什么 |
|---:|---|---|---|
| 1 | 两路 RGB 与基础本体感知 | 用视觉和机器人自身状态做闭环控制 | BBox、扩展本体/力觉、深度、相机标定 |
| 2 | 任务实体的二维 BBox | 缩小视觉搜索区域，完成初始对准 | 扩展本体/力觉、深度、相机标定、物体三维真值 |
| 3 | 扩展机械臂运动状态 | 根据机械臂关节和末端速度判断动作结果 | 夹爪本体、力/力矩、深度、相机标定、物体三维真值 |
| 4 | 两路米制深度与相机标定 | 使用真实深度和当前帧标定做三维几何推理 | API 未返回的仿真隐藏状态 |

所有 Level 都不提供 MuJoCo state、XML、BDDL、服务器源码、对象三维 ground truth 或其他 Agent 的结果。

### Level 1：RGB + 基础本体感知

`observation.images` 包含 `agentview_rgb` 和 `wrist_rgb`；`observation.proprioception` 包含：

- `robot0_joint_pos[7]`：Panda 七个机械臂关节角；最后一维仍是机械臂 joint 7，不是夹爪；
- `robot0_eef_pos`：末端在 MuJoCo world frame 中的位置；
- `robot0_eef_quat=[qx,qy,qz,qw]`：EEF local/body frame 相对 MuJoCo world frame 的姿态。

将四元数归一化后，局部坐标到 world 的旋转矩阵为：

```text
R_eef_to_world =
[[1-2(qy²+qz²), 2(qx*qy-qz*qw), 2(qx*qz+qy*qw)],
 [2(qx*qy+qz*qw), 1-2(qx²+qz²), 2(qy*qz-qx*qw)],
 [2(qx*qz-qy*qw), 2(qy*qz+qx*qw), 1-2(qx²+qy²)]]
```

`R_eef_to_world` 的三列分别是 EEF 局部 `+x`、`+y`、`+z` 轴在 world frame 中的单位方向。使用
`v_world = R_eef_to_world @ v_eef` 和 `v_eef = R_eef_to_world.T @ v_world`。如果希望沿 EEF 局部方向平移或旋转，先把局部平移向量或局部 rotation vector 乘当前帧的 `R_eef_to_world`，再分别按 `0.05 m` 或 `0.5 rad` 编码到 Step 的 world-frame action。姿态变化后必须用最新四元数重算，不能永久假设局部 `+z` 等于 world 向下。

当前 Panda 夹爪的 EEF 结构语义为：

- 局部 `+z` 从掌部指向指尖，是名义接近和插入方向；
- 两根手指沿局部 `±y` 相互远离以张开，并向中心闭合；物体希望被夹紧的相对表面法向应尽量与这条张合轴对齐；
- 局部 `+x` 补全右手坐标系，与接近轴和张合轴正交；
- 绕局部 `+z` 旋转会改变接触平面内的夹持方向，绕局部 `+x/+y` 旋转会改变倾斜和接近角度；
- 手指开合只由 action 第七维 `gripper` 控制；沿 Cartesian `y` 移动整个 EEF 不等于闭合夹爪。

以上是 EEF 局部语义；实际 world 方向始终由当前帧 `R_eef_to_world` 决定。

`wrist_rgb` 使用独立的 OpenCV camera frame。它的当前 `camera_to_world` 用于视觉和深度射线；它不是
EEF local/body frame，不能把相机 `+z` 光轴直接当作 EEF `+z`。

此外每帧都有 `frame_id` 和 `step_index`。空间关系不确定时，只能通过最新 RGB 与机器人运动反馈逐步校正。

### Level 2：Level 1 + 任务实体 BBox

`observation.annotations.object_bboxes` 增加任务相关物体、容器和可交互部件在相机图像中的二维框。每个框包含 `label`、像素坐标 `xyxy`、`normalized_xyxy` 和 `visible_pixels`。

- `bbox_scope=initial`：BBox 只描述 Reset 初始帧；物体移动后必须根据最新 RGB 重新判断。
- `bbox_scope=every_frame`：每帧返回最新 BBox。
- BBox 只是二维先验，不是深度、米制距离或物体三维坐标。

### Level 3：Level 2 + 扩展机械臂运动状态（Strict）

`observation.proprioception` 还包含机械臂关节速度和末端线速度、角速度。本实验的固定 profile 为 `l4-no-tactile-no-gripper-proprio-v1`。

所有 `robot0_gripper_*`、`robot0_joint_commanded_torque`、`robot0_eef_force`、`robot0_eef_torque` 以及任何其他 gripper/force/torque/wrench/tactile 字段都不会返回或落盘。缺失不代表零值，不得尝试从其他接口恢复。Agent 仍可通过 action 第七维控制夹爪，但接触和抓稳判断只能使用 RGB、机械臂/末端运动等可见证据。Level 3 仍然没有深度和相机标定。

### Level 4：Level 3 + 米制深度与相机标定

`observation.depth` 为两路相机的米制 `float32` 深度，编码是 `float32-zlib-base64`，载荷同时给出 `shape`、`dtype` 和 `unit=meter`。`observation.camera_calibration` 按相机给出：

- `intrinsic`：相机内参矩阵；
- `camera_to_world`：从相机坐标系到 world frame 的外参；
- `convention=opencv`：投影约定。

手腕相机会跟随末端运动，所以必须使用当前帧的 `camera_to_world`。深度必须按 API 编码解码为原始米制 `float32` 数组，不能用 RGB 伪造，也不能只把伪彩色图当作原始深度。

本地代理会解压 Level 4 深度，将米制值量化为 16-bit 后无损编码成
`<frame>.depth.png`。默认量化单位是 `0.0001 m`；如果一帧的最大深度超过
`6.5535 m`，会自动增大该帧的 scale，避免截断。Observation 中的 Base64
会替换为描述符，其中包括 `encoding=png-u16-linear`、`path`、`shape`、
`scale_m_per_unit`、`offset_m`、`bytes` 和校验值。读取时统一使用 SDK，返回值
仍是以米为单位的 `float32` 数组：

```python
from libero_sdk import load_depth

descriptor = state["observation"]["depth"]["agentview_rgb"]
depth_m = load_depth(descriptor)
```

`load_depth()` 同时兼容历史 `.float32.bin` 描述符。PNG 在量化以后是无损的；
默认 scale 下，相比原始 float32 的最大量化误差为 `0.00005 m`（0.05 mm）。

## Observation 与图片

代理把 Gateway 返回的 JPEG Base64 保存为原始 `.jpg`，并在返回给 Agent 的 Observation 中用以下描述替换 Base64：

```json
{
  "images": {
    "agentview_rgb": {
      "path": "/absolute/path/to/frame.jpg",
      "bytes": 12345,
      "sha256": "..."
    }
  }
}
```

代理采用“先持久化，后返回”：Reset/Step 成功返回时，`local_artifacts` 会包含：

```json
{
  "persisted": true,
  "frame_ref": "episode-0001/frame-00042_step",
  "frame_index": "/absolute/attempt/libero-artifacts/frames.jsonl"
}
```

`persisted=true` 表示图片、Observation、可选 Action、Frame metadata 和 `frames.jsonl` 提交记录已经落盘。Agent 不需要调用日志 API。

每个 Reset 和 Step 自动写入：

```text
libero-artifacts/
├── manifest.json
├── frames.jsonl
├── result.json
├── closed.json
└── episodes/episode-NNNN/
    ├── images/<camera>/frame-NNNNN_{reset|step}.jpg
│   ├── depth/<camera>/frame-NNNNN_{reset|step}.depth.png
    ├── observations/frame-NNNNN_{reset|step}.json
    ├── actions/frame-NNNNN_step.json
    └── frame-metadata/frame-NNNNN_{reset|step}.json
```

Token 永远不写入这些文件。Agent 编写的控制程序和派生图像应放在 `agent/`，不要覆盖原始产物。

固定相对路径为 `./libero-artifacts/frames.jsonl`、`./libero-artifacts/result.json` 和 `./libero-artifacts/closed.json`。当前图片使用 Observation 中的绝对 `path`。只按需读取最近帧，不要把整份 `frames.jsonl` 打印进模型上下文。

## 生命周期

Harness Turn、浏览器断线和 SSH 短时中断不会关闭 LIBERO 环境。可信控制器在成功、硬截止、用户停止或控制器退出时执行 Result、Session Close 和 Run Finish。Agent 不直接清理远程环境。
