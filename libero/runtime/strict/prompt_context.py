"""Build the level-specific LIBERO context injected into a Harness prompt."""

from __future__ import annotations

from typing import Any


CONTEXT_MARKER = "{{LIBERO_EXPERIMENT_CONTEXT}}"
DEFAULT_DEADLINE_SECONDS = 4 * 60 * 60
_BBOX_SCOPES = {"initial", "every_frame"}
NO_TACTILE_PROFILE = "l4-no-tactile-no-gripper-proprio-v1"


def _observation_config(experiment: dict[str, Any]) -> tuple[int, str, str]:
    observation = experiment.get("observation", {})
    if not isinstance(observation, dict):
        raise ValueError("experiment.observation must be an object")
    level = observation.get("level", 1)
    if type(level) is not int or level not in {1, 2, 3, 4}:
        raise ValueError("experiment.observation.level must be one of 1, 2, 3, 4")
    bbox_scope = observation.get("bbox_scope", "initial")
    if not isinstance(bbox_scope, str) or bbox_scope not in _BBOX_SCOPES:
        raise ValueError("experiment.observation.bbox_scope must be initial or every_frame")
    profile = observation.get("profile")
    if level != 4 or profile != NO_TACTILE_PROFILE:
        raise ValueError(
            f"this frozen runtime requires Level 4 and profile={NO_TACTILE_PROFILE}"
        )
    return level, bbox_scope, profile


def build_experiment_context(
    experiment: dict[str, Any],
    *,
    instruction: str,
    deadline_seconds: int = DEFAULT_DEADLINE_SECONDS,
    workflow_variant: str = "operation-ledger",
) -> str:
    """Return the authoritative task and information available at the selected level."""

    level, bbox_scope, profile = _observation_config(experiment)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("the ready LIBERO Session must provide a task instruction")
    instruction = instruction.strip()
    if type(deadline_seconds) is not int or deadline_seconds < 1:
        raise ValueError("deadline_seconds must be a positive integer")
    if workflow_variant not in {
        "plain",
        "operation-ledger",
        "trusted-step-liveness",
        "episode-ledger",
        "action-ledger",
        "research-explorer",
        "research-executor",
    }:
        raise ValueError("unknown workflow_variant")
    task = experiment.get("task", {})
    if not isinstance(task, dict):
        raise ValueError("experiment.task must be an object")
    benchmark = task.get("benchmark", "server-selected")
    task_id = task.get("task_id", "server-selected")
    episode_length = experiment.get("episode_length", "server default")
    seed_policy = (
        "本 trial 的 seed 已由控制器固定，Reset 不会改变 seed。"
        if experiment.get("seed") is not None
        else "本 trial 创建时由服务器随机选择 seed，后续 Reset 不会改变该 seed。"
    )

    if workflow_variant in {"plain", "research-explorer", "research-executor"}:
        role_detail = (
            "本 trial 使用 Explorer Research Ledger；Reset/Step 不受账本 gate 限制，"
            "但必须通过多 Episode 真实探索、修订条目并冻结执行账本。"
            if workflow_variant == "research-explorer"
            else (
                "本 trial 使用只读冻结 Execution Ledger；Reset/Step 不受账本 gate 限制，"
                "但必须依据当前 seed 重新验证账本适用性。"
                if workflow_variant == "research-executor"
                else ""
            )
        )
        execution_policy = (
            "Controller 已在 Agent 启动前自动 Reset 一次并持久化初始 Level-4 帧。"
            "Reset 和 Step 从任务开始即直接可用，不需要额外的启动或人工确认。"
            f"{role_detail}"
            f"整个实验共享 `{deadline_seconds}` 秒"
            f"（{deadline_seconds / 3600:g} 小时）的总 wall-clock 上限。"
        )
    elif workflow_variant == "episode-ledger":
        execution_policy = (
            "Controller 已在 PLAN 前自动 Reset 一次并持久化初始 Level-4 帧。PLAN 可读取该场景并使用本地工具，"
            "但 Reset/Step 尚未开放。PLAN 只形成首次可恢复物理操作所需的最小充分场景理解、预期证据、停止条件和恢复路径；"
            "满足后立即提交 `start_task` 开放任务执行，其余不确定性在 ACTIVE 的受保护执行中解决，不要为了穷尽理解延长 PLAN。"
            f"从 PLAN 开始，PLAN 与执行共享 `{deadline_seconds}` 秒"
            f"（{deadline_seconds / 3600:g} 小时）的总 wall-clock 上限。"
        )
    elif workflow_variant == "action-ledger":
        execution_policy = (
            "Controller 已在 PLAN 前自动 Reset 一次并持久化初始 Level-4 帧。PLAN 可读取该场景并使用本地工具，"
            "但 Reset/Step 尚未开放。PLAN 只形成首次可恢复物理操作所需的最小充分场景理解、预期证据、停止条件和恢复路径；"
            "满足后立即提交 `start_task` 进入任务执行，其余不确定性在 ACTIVE 的受保护执行中解决，不要为了穷尽理解延长 PLAN。"
            "每个高层物理操作还必须在首个 Step 前提交 `prepare_action`。从 PLAN 开始，"
            f"PLAN 与执行共享 `{deadline_seconds}` 秒（{deadline_seconds / 3600:g} 小时）的总 wall-clock 上限。"
        )
    else:
        execution_policy = (
            f"Controller 已在 PLAN 前自动 Reset 一次并持久化初始 Level-4 帧。PLAN 可读取该场景并使用本地工具；"
            f"Agent 发起的 Reset/Step 会被 lease 拒绝。规划 10 分钟提示、20 分钟硬停止。Agent 原子提交 "
            f"`start_task_and_prepare` 后，任务时钟才启动，最高运行 `{deadline_seconds}` 秒"
            f"（{deadline_seconds / 3600:g} 小时）。"
        )

    lines = [
        "## 当前实验与 Observation（启动器自动注入）",
        "",
        f"- 唯一任务：`{instruction}`",
        f"- 任务配置：`{benchmark}/task {task_id}`，Episode 上限 `{episode_length}` 步。",
        f"- {execution_policy}",
        f"- 当前 Observation Level：`{level}`。这是控制器锁定的配置，不得修改或切换。",
        f"- 当前 Observation Profile：`{profile}`。所有夹爪本体字段，以及任何 force/torque/wrench/tactile 字段均不可见。",
        f"- {seed_policy}",
        "- 每帧都有 `frame_id`、`step_index`、主视角 `agentview_rgb` 和手腕视角 `wrist_rgb`。",
        "- 基础本体字段只包括机械臂 `robot0_joint_pos`、`robot0_eef_pos` 和 `robot0_eef_quat`；`robot0_joint_pos` 的 7 维均为 Panda 机械臂关节，不包含夹爪。",
        "- `robot0_eef_quat=[qx,qy,qz,qw]` 描述 EEF local/body frame 相对 MuJoCo world frame 的姿态。"
        "归一化四元数得到的 `R_eef_to_world` 把 EEF 局部向量变换到 world；矩阵三列分别是局部 `+x/+y/+z` 轴在 world 中的方向。",
        "- 换算使用 `v_world = R_eef_to_world @ v_eef`、`v_eef = R_eef_to_world.T @ v_world`。"
        "若控制意图以 EEF 局部坐标表达，先把局部平移和局部 rotation vector 分别乘当前帧的 `R_eef_to_world`，"
        "再按 Step 的 world-frame 动作尺度编码；每帧重算，不要把局部 `+z` 永久假定为 world 向下。",
        "- 当前 Panda 夹爪结构中，EEF 局部 `+z` 从掌部指向指尖，是名义接近/插入轴；两根手指沿局部 `±y` 张开、向中心闭合，"
        "局部 `+x` 补全右手系并与接近轴、张合轴正交。绕局部 `+z` 旋转会改变接触平面内的夹持方向。"
        "这些方向仍须用当前 `R_eef_to_world` 转到 world；夹爪开合只使用 action 第七维 `gripper`，不能用 Cartesian `dy` 代替。",
    ]

    if level == 1:
        lines.extend(
            [
                "- Level 1 只提供上述两路 RGB 与基础本体感知。根据最新图像和机器人自身运动反馈逐步校正空间判断。",
                "- 当前没有任务 BBox、扩展速度/力觉、深度或相机标定；不得虚构或尝试读取这些字段。",
                "- 抓取验证主要依靠夹爪开度、试抬前后物体在两路 RGB 中的共同运动和主视角确认；不能声称使用力觉或米制视觉。",
            ]
        )
    else:
        lines.extend(
            [
                "- 本 Level 提供 `observation.annotations.object_bboxes`：任务相关物体、容器和可交互部件的二维框。",
                f"- 当前 `bbox_scope={bbox_scope}`。"
                + (
                    "BBox 只描述 Reset 初始帧；物体或门移动后必须根据最新 RGB 和本体反馈重新判断。"
                    if bbox_scope == "initial"
                    else "每帧使用最新 BBox，但它仍然只表示二维图像范围。"
                ),
                "- BBox 可用于限制搜索范围和初始对准，但不能当作深度、米制距离或物体三维坐标。",
            ]
        )

        if level == 2:
            lines.extend(
                [
                    "- Level 2 没有扩展速度/力觉、深度或相机标定；不得虚构或尝试读取这些字段。",
                    "- 初始 BBox 只用于缩小首次搜索区域。物体被推动、夹起或相机运动后，必须用最新 RGB 和试抬共同运动重新验证，不能继续追逐旧 BBox 中心。",
                ]
            )
        else:
            lines.extend(
                [
                    "- 本实验仅保留机械臂关节速度和末端线/角速度；不提供任何 `robot0_gripper_*`、`robot0_joint_commanded_torque`、`robot0_eef_force`、`robot0_eef_torque` 或其他 force/torque/wrench/tactile 字段。",
                    "- 不得把缺失的夹爪、力或力矩字段当作零值，也不得尝试从其他接口或文件恢复这些隐藏观测。",
                    "- Agent 仍可发送 action 第七维控制夹爪，但不能读取夹爪位置或速度；接触和抓稳只能依据 RGB、深度、末端/机械臂运动以及服务器成功结果等可见证据。",
                ]
            )
            if level == 3:
                lines.extend(
                    [
                        "- Level 3 没有深度或相机标定；不得从 RGB 或 BBox 伪造三维信息。",
                        "- 用少量主动小动作估计局部图像—动作关系即可；腕部模板发生尺度变化、遮挡或跳到其他物体时，立即回到主视角重新定位，不要无限追加标定。",
                    ]
                )
            else:
                lines.extend(
                    [
                        "- Level 4 的 Gateway 源载荷是两路相机的米制 `float32-zlib-base64` 深度；可信代理会将其保存为带 scale 的 16-bit PNG，并在 Observation 中返回文件描述符。用 `libero_sdk.load_depth()` 统一读取为米制 float32 数组。",
                        "- Level 4 提供每路相机的 `intrinsic`、`camera_to_world` 和 `convention=opencv`。手腕相机外参随机器人运动，必须使用当前帧标定。",
                        "- `wrist_rgb` 的 OpenCV camera frame 与 EEF local/body frame 不是同一坐标系；视觉射线用当前 `camera_to_world`，末端局部动作换算用 `R_eef_to_world`，不得混用。",
                        "- 深度必须通过 SDK 按描述符解码并保留米制语义；不能用 RGB 伪造，也不能把伪彩色图当作深度数据。",
                        "- 用深度和当前外参估计候选接触面、障碍高度和接近方向，而不只是求目标点云中心。接近低位后必须重新估计；高位点云中心不能替代真实夹持中心。",
                    ]
                )

    lines.extend(
        [
            "- 所有 Level 都不提供物体三维 ground truth、完整分割、MuJoCo state、XML、BDDL、服务器源码或其他 Agent 的结果。",
            "- 实际返回结构和字段语义以 `AGENT_API.md` 为准。",
            "",
            "## 本地保存路径（启动器自动注入）",
            "",
            "- 原始评测产物：`./libero-artifacts/`",
            "- 帧索引：`./libero-artifacts/frames.jsonl`",
            "- 最新服务端结果：`./libero-artifacts/result.json`",
            "- 远程清理记录：`./libero-artifacts/closed.json`",
            "- 控制器与 Harness 生命周期日志：`./logs/controller.jsonl` 和 `./logs/harness.jsonl`",
            "- Agent 自己的代码、派生文件和进度笔记：`./agent/`",
            "- Reset/Step 只在当前 Frame 已持久化后才返回；`local_artifacts.persisted=true` 是提交确认，`local_artifacts.frame_ref` 是稳定帧引用。",
            "- 当前图片的精确绝对路径在 `observation.images.<camera>.path`。查看历史时只读取 `frames.jsonl` 最近所需的行，不要把整份大日志载入模型上下文。",
        ]
    )
    return "\n".join(lines)


def render_prompt(
    template: str,
    experiment: dict[str, Any],
    *,
    instruction: str,
    deadline_seconds: int = DEFAULT_DEADLINE_SECONDS,
    workflow_variant: str = "operation-ledger",
) -> str:
    """Inject exactly one current-level context block into a task prompt."""

    context = build_experiment_context(
        experiment,
        instruction=instruction,
        deadline_seconds=deadline_seconds,
        workflow_variant=workflow_variant,
    )
    count = template.count(CONTEXT_MARKER)
    if count > 1:
        raise ValueError(f"prompt contains {count} {CONTEXT_MARKER} markers; expected at most one")
    if count == 1:
        return template.replace(CONTEXT_MARKER, context)
    return f"{context}\n\n{template}"
