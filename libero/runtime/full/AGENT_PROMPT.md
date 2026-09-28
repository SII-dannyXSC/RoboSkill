# LIBERO 自主执行

{{LIBERO_EXPERIMENT_CONTEXT}}

运行方已经创建唯一 LIBERO 环境，并完成一次初始 Reset。不得修改实验配置，不得创建、切换或关闭 Session，也不得读取其他 trial。若 Workspace 根目录缺少 `.libero-endpoint.json`，停止并报告控制器未准备好。

先阅读 `AGENT_API.md` 和已持久化的初始 Level-4 帧，然后使用可用工具自主完成唯一任务。每次动作后以返回的最新 Observation 为准；可以在需要时调用 Reset 开始新 Episode。只有 `client.result()` 返回 `success=true` 才算完成；基础设施确认无法继续时才报告阻塞。

原始 Reset/Step 产物由代理自动写入 `libero-artifacts/`；Agent 自己的代码和派生文件写入 `agent/`，不要覆盖或删除原始产物，也不要记录凭据或隐藏思维链。
