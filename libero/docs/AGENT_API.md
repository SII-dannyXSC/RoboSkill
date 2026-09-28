# LIBERO Agent API v2

## 认证

除 `/healthz` 外，请求需要：

```http
Authorization: Bearer <LIBERO_API_TOKEN>
Content-Type: application/json
```

## 查询能力

```http
GET /v2/capabilities
```

返回 Token 可用 benchmark、最高 Observation Level、最大 Episode 长度和并发配额。

查询某个 benchmark 的可选任务：

```http
GET /v2/tasks/libero_10
```

响应只包含稳定的 `task_id` 和自然语言 `instruction`，不包含 BDDL、初始状态或服务端路径。

## 创建 Evaluation Run（由可信启动器调用）

正式计时必须在模型进程启动前创建 Run：

```http
POST /v2/runs
Idempotency-Key: <唯一随机值>
```

```json
{
  "label": "example-agent",
  "task": {"benchmark": "libero_10", "task_id": 1},
  "episode_length": 1000000,
  "observation": {"level": 2, "bbox_scope": "initial"},
  "max_attempts": 100
}
```

响应返回 `run_id`、不透明的 `agent_identity`、一次性明文 `agent_token` 和服务端锁定的
`config`。如果省略 `task_id`，以响应中的随机 task ID 为准。启动器将 `run_id` 注入
`LIBERO_RUN_ID`，并只把 `agent_token` 作为 `LIBERO_API_TOKEN` 交给模型进程。每个 Run 都使用
独立访问者身份；模型和 Level 只是 Run 配置，不参与权限主键。Agent 不应调用 Run 创建/结束接口。

`agent_token` 只有 `sessions:operate` scope，强绑定该 `run_id`，不会写入 SQLite。Run 完成时
立即撤销；同一个 Idempotency-Key 重试时会返回原 Run 并轮换新的 Token，旧 Token 随即失效。

```http
GET  /v2/runs/{run_id}
POST /v2/runs/{run_id}/finish
```

`time_to_first_success_seconds` 从服务端接受 Run 开始，到任意绑定 Episode 首次由 LIBERO
`check_success()` 判定成功为止。成功时刻在仿真子进程中、Observation 编码之前采样，随后在成功
响应返回前由 SQLite 事务持久化。它不是客户端上报时间。`finish` 会回收该 Run 遗留的 Session。

## 创建 Session

```http
POST /v2/sessions
Idempotency-Key: <本次创建请求的唯一随机值>
```

```json
{
  "run_id": "run_...",
  "task": {
    "benchmark": "libero_10",
    "task_id": 1
  },
  "episode_length": 1000000,
  "observation": {
    "level": 2,
    "bbox_scope": "initial"
  }
}
```

字段规则：

- `run_id` 必填，来自可信启动器。
- task、`episode_length`、Observation Level 和 bbox scope 必须与 Run 的 `config` 完全一致。
- `observation.level` 为 1–4。
- `bbox_scope` 为 `initial` 或 `every_frame`，仅 Level 2 及以上生效。
- `seed` 默认不可指定；服务端生成且不返回。只有显式授权的调试 Token 可传 uint32 seed。
- 请求不能包含 BDDL 路径、GPU ID、Worker 数或服务端代码。

服务立即返回 HTTP 202 和 `session_id`，`state` 通常为 `starting`。使用同一
`Idempotency-Key` 重试同一个请求只会得到同一个 Session；把同一个 Key 用于不同请求会返回
`IDEMPOTENCY_KEY_REUSED`。

轮询：

```http
GET /v2/sessions/{session_id}
```

等到 `state=ready` 后再 Reset；此时响应包含自然语言 `instruction`、动作规格、Observation
规格和最终 `session_config`。`state=failed` 时关闭 Session 并报告 `error_code`。如果客户端等待
超时，Session 仍存在；可以继续轮询或 `DELETE` 取消，不能换新 Key 盲目重建。

响应中的 `timing.startup_seconds` 是 Session create 到环境 ready 的冷启动时间；成功 Episode 的
`timing.time_to_success_seconds` 是该 Session create 到成功判定的时间。跨多个 Episode 的正式指标
使用 Run 的 `time_to_first_success_seconds`。

## Observation Levels

| Level | 内容 |
|---:|---|
| 1 | RGB + 基础本体感知 |
| 2 | Level 1 + 任务实体 bbox |
| 3 | Level 2 + 扩展本体感知 |
| 4 | Level 3 + 深度和相机标定 |

### BBox

```json
{
  "annotations": {
    "object_bboxes": {
      "agentview_rgb": [
        {
          "label": "butter_1",
          "xyxy": [34, 52, 61, 91],
          "normalized_xyxy": [0.2656, 0.4063, 0.4766, 0.7109],
          "visible_pixels": 487
        }
      ]
    }
  }
}
```

`xyxy` 使用右下角不包含的半开区间。完全不可见实体不返回框。BBox 是与 JPEG 像素对齐的
独立 JSON 坐标，服务端不会把框画进 JPEG。

### 深度

深度采用 `float32-zlib-base64`，单位为米：

```json
{
  "encoding": "float32-zlib-base64",
  "dtype": "float32",
  "shape": [128, 128],
  "unit": "meter",
  "base64": "..."
}
```

解码顺序：Base64 → zlib → little-endian float32 array。

### 相机标定

```json
{
  "intrinsic": [[0,0,0],[0,0,0],[0,0,1]],
  "camera_to_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "convention": "opencv"
}
```

## Episode 流程

```text
POST   /v2/sessions
GET    /v2/sessions/{session_id}   # 轮询到 ready
POST   /v2/sessions/{session_id}/reset
POST   /v2/sessions/{session_id}/step
GET    /v2/sessions/{session_id}/result
DELETE /v2/sessions/{session_id}
```

`Reset` 可以在同一个仍然活跃的 Session 中重复调用。每次调用会恢复该
Session 的干净初始场景、把 Episode 步数归零，并返回递增的
`episode_index`。Session ID、GPU slot、任务配置和服务端 Run 计时保持不变：

```json
{
  "episode_index": 2,
  "terminated": false,
  "truncated": false,
  "observation": {"frame_id": 0, "step_index": 0}
}
```

Python 客户端使用 `client.reset_episode(session_id)` 读取完整响应；兼容接口
`client.reset(session_id)` 仍只返回 Observation。Agent 判断当前策略不可恢复时应优先
Reset 当前 Session，不要关闭后创建新 Session。成功后的 Session 不允许再次 Reset。

Step 请求：

```json
{"action":[0.1,-0.2,0,0,0.1,0,1]}
```

动作格式为 `[dx,dy,dz,drx,dry,drz,gripper]`，必须是 7 个 `[-1,1]` 内的有限数值：

- `dx,dy,dz`：MuJoCo world frame 中的末端位置增量，`±1` 对应 `±0.05 m`。
- `drx,dry,drz`：world frame 中的 rotation vector（axis × angle），不是欧拉角；`±1` 对应 `±0.5 rad`。
- `gripper < 0` 打开，`> 0` 闭合，`= 0` 保持当前命令；非零幅值不是绝对开度或速度。
- 控制频率为 20 Hz；一次 Step 对应 0.05 秒仿真时间。

每次成功 Step 后必须使用响应中的新 Observation。当 `terminated` 或 `truncated` 为真时停止 Step。

无论成功、失败或客户端异常，都要在 `finally` 中关闭 Session。

基础本体感知约定：

- `robot0_joint_pos`：关节角，rad。
- `robot0_gripper_qpos`：夹爪关节位置，rad。
- `robot0_eef_pos=[x,y,z]`：world frame，m。
- `robot0_eef_quat=[qx,qy,qz,qw]`：`xyzw` 顺序，EEF local/body frame 相对 world frame 的姿态。

Level 3 扩展字段在 `optional_proprioception_keys` 中；传感器在某帧无有限值时会省略。
`robot0_joint_commanded_torque` 是控制器命令力矩（N·m），不是测得的外部关节力矩；
`robot0_eef_force` 单位 N，`robot0_eef_torque` 单位 N·m。

## 错误码

| HTTP | code | 含义 |
|---:|---|---|
| 401 | `UNAUTHENTICATED` | Token 无效 |
| 403 | `TASK_FORBIDDEN` | Task/benchmark 未授权 |
| 403 | `OBSERVATION_LEVEL_FORBIDDEN` | 请求等级超过 Token 权限 |
| 404 | `SESSION_NOT_FOUND` | 不存在、已关闭或属于其他身份 |
| 404 | `RUN_NOT_FOUND` | Run 不存在或属于其他身份 |
| 409 | `SESSION_BUSY` | 同一 Session 正在执行另一请求 |
| 409 | `SESSION_NOT_READY` | 仿真仍在后台初始化，继续轮询 |
| 409 | `IDEMPOTENCY_KEY_REUSED` | 同一个创建 Key 被用于不同配置 |
| 409 | `INVALID_SESSION_STATE` | 生命周期顺序错误 |
| 409 | `ACTIVE_RUN_EXISTS` | 当前身份已有未结束 Run |
| 409 | `RUN_CONFIG_MISMATCH` | Session 参数与 Run 锁定配置不同 |
| 409 | `RUN_ATTEMPT_LIMIT_REACHED` | 已达到 Run 的最大尝试次数 |
| 409 | `RUN_ALREADY_SUCCEEDED` | Run 已记录首次成功，不再接受新 Session |
| 409 | `RUN_HAS_ACTIVE_SESSIONS` | Run 结束时仍有正在操作且暂时无法回收的 Session |
| 409 | `EVALUATION_RUN_REQUIRED` | 活跃 Run 期间试图创建未绑定 Session |
| 403 | `SEED_SELECTION_FORBIDDEN` | Token 不允许选择初始状态 seed |
| 422 | `TASK_NOT_FOUND` | Task ID 不存在 |
| 422 | `MAX_STEPS_OUT_OF_RANGE` | Episode 长度超限 |
| 422 | `INVALID_REQUEST` | JSON 或动作无效 |
| 429 | `AGENT_SESSION_QUOTA_EXCEEDED` | 本 Token 配额已满，应关闭旧 Session |
| 503 | `GLOBAL_CAPACITY_EXHAUSTED` | GPU slot 全部占用，可退避重试 |
| 504 | `SIMULATION_START_TIMEOUT` | 环境初始化超时 |
| 504 | `SIMULATION_TIMEOUT` | 操作超时，Session 已失败，不要重放动作 |
| 500 | `EVALUATION_PERSISTENCE_FAILED` | 服务端未能安全持久化计时；该次成功不会作为已确认结果返回 |

## Python 示例

```python
import os

created = client.create_configured_session(
    os.environ["LIBERO_BENCHMARK"],
    task_id=int(os.environ["LIBERO_TASK_ID"]),
    episode_length=int(os.environ["LIBERO_EPISODE_LENGTH"]),
    observation_level=int(os.environ["LIBERO_OBSERVATION_LEVEL"]),
    bbox_scope=os.environ["LIBERO_BBOX_SCOPE"],
    run_id=os.environ["LIBERO_RUN_ID"],
)
session_id = created["session_id"]
try:
    obs = client.reset(session_id)
    while True:
        response = client.step(session_id, policy(obs, created["instruction"]))
        obs = response["observation"]
        if response["terminated"] or response["truncated"]:
            break
    result = client.result(session_id)
finally:
    client.close(session_id)
```

`/v1` 仍可用于没有正式 Run 的旧客户端。活跃 Run 期间，未绑定 Run 的 v1 创建会被拒绝；正式
评测必须使用 v2。
