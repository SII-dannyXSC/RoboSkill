# 当前单任务经验重新总结流程

本文描述 `libero10-dual-model-experience-growth-r1` 当前实际运行的经验生成机制，即同一个 LIBERO 任务如何从 `E0` 生成 `E1`，再从 `E1` 生成 `E2`。这是一份实现说明，不是未来设计稿。

## 1. 核心结论

当前流程不是由 Harness 根据轨迹自动合成经验，而是：

1. Harness 给执行 Agent 一份只读父经验；
2. Agent 在新 Seed 中完成任务，并在 `agent/` 中形成这次真正运行过的代码；
3. 服务器确认成功后，Harness 禁止继续操作机器人；
4. 同一个 Agent、同一个会话根据自己的完整执行轨迹重新总结经验；
5. Harness 检查文件结构、帧引用和代码来源；
6. 校验通过后，整份新经验被复制成新的只读版本。

因此，经验内容主要由 Agent 自己决定；Harness 目前保证的是“来自一次真实成功执行且文件引用有效”，不是“这份经验一定比父版本更好”。

## 2. 一次经验演化的完整流程

```text
只读父经验 E(n)
        │
        ▼
用新 Seed 启动一个全新 Agent
        │
        ├── 读取 program-index.json
        ├── 按需读取父经验模块
        ├── 复制需要修改的模块到 agent/
        ├── 根据当前 RGB、深度、标定、力和本体重新求值
        └── 自主执行、失败恢复、必要时 Reset
        │
        ▼
client.result().success == true
        │
        ▼
Harness 将环境切到成功后只读状态
禁止 client.step() / client.reset()
        │
        ▼
同一个 Agent、同一个会话执行经验复盘
        │
        ├── Action Ledger
        ├── Keyframes 索引
        ├── 参数 Binding
        ├── 中文 Episode 总结
        └── 从本次实际代码提取层级 Program
        │
        ▼
Harness 结构与事实引用校验
        │
        ├── 不通过：写 validation-error.json
        │             同一会话再修复一次
        │
        └── 通过：复制、冻结并晋级为 E(n+1)
```

## 3. 当前 E0 → E1 → E2 的实验设置

| 新版本 | 使用的父经验 | 优先 Writer Seed | 失败后的备用 Seed | 单任务最多尝试 |
|---|---|---:|---|---:|
| E1 | E0 | 10 | 11、12 | 3 |
| E2 | E1 | 20 | 21、22 | 3 |

每个任务每个版本最终只晋级一份经验。一个 Writer 如果没有完成任务，或者任务成功但经验复盘未通过校验，就不会晋级；调度器会换到下一个备用 Seed。

当前每个模型的生成并发是 4。十个任务全部获得 E1 后才进入 E2；十个任务全部获得 E2 后，才开始 Plain/E0/E1/E2 的正式评测。

正式评测阶段不会继续修改经验，也不会再次触发经验复盘。

## 4. Writer Agent 如何使用父经验

每个生成 Cell 的 Workspace 中只放当前指定的一份父经验，不把 E0、E1、E2 和历史版本同时注入上下文。

Codex 中父经验位于：

```text
experience/
```

Claude Code 中父经验位于：

```text
program-prior/libero_10/task-XX/
```

父经验目录是只读的。Agent 首先读取：

```text
program/program-index.json
```

然后按需读取相关模块。需要使用或修改的代码必须复制到当前 Workspace 的：

```text
agent/
```

Prompt 明确告诉 Agent：

- 父经验是代码和方法的起点，不是当前场景事实；
- 不能直接复用父经验中的绝对坐标、深度、阈值和动作次数；
- 必须使用当前 Reset 的 RGB、深度、相机标定、力反馈和本体状态重新求值；
- 如果经验与当前观测冲突，可以局部修改、替换模块或完全放弃父经验；
- 不能因为经验不适用而停止任务。

## 5. 什么时候开始总结经验

经验总结只能在服务器权威结果满足以下条件后开始：

```text
client.result().success == true
```

Agent 自己声称成功、视觉上看似完成、Episode terminated，均不能单独触发经验总结。

成功后 Harness 会：

1. 保留当前 Codex thread 或 Claude session；
2. 给正在结束的成功回合一小段收尾时间；
3. 将本地 LIBERO 代理切换为只读；
4. 拒绝后续 `step` 和 `reset`；
5. 在同一会话中发送经验复盘 Prompt。

使用同一会话是为了让复盘 Agent保留本次执行中的意图、失败假设、代码修改和工具结果，而不只是看到最终成功帧。

## 6. Agent 总结时允许使用的事实来源

复盘 Prompt 要求只总结自己这个 Task/Seed，不读取其他 Seed。事实来源包括：

```text
当前 Agent 会话
agent/
logs/controller.jsonl
libero-artifacts/frames.jsonl
libero-artifacts/episodes/... 下已经保存的 RGB、深度和 Observation
```

本次 Workspace 中的父经验可以用来判断哪些结构被保留或修正，但新经验必须以本次实际执行及本次 `agent/` 代码为准。

新经验不是把“父经验 + 本次经验”并排拼接成两套程序，而是重新输出一个完整的新版本。

## 7. 新经验包含什么

成功 Writer 必须在 `experience-review/` 中生成以下内容：

```text
experience-review/
├── action-ledger.json
├── keyframes.json
├── binding.json
├── episode-summary.md
├── review-result.json
└── program/
    ├── program-index.json
    └── modules/
        └── *.py
```

### 7.1 `action-ledger.json`

保存实际发生过的高层闭环 Action，而不是每一个环境 Step。每项至少包含：

```text
action_id
purpose
mode
contact_mode
episode
start_step
end_step
method
outcome
observed_effect
failure_mechanism
next_adjustment
reusable_lesson
evidence_refs
```

其中 Episode、Step 和证据帧必须能在 `frames.jsonl` 中找到。

Action Ledger 既可以记录成功动作，也可以记录真正影响后续修正的失败动作。但当前 Harness 不会自动判断某个失败动作是否应该进入未来 Agent 的默认执行路径。

### 7.2 `keyframes.json`

保存关键帧索引：

```text
role
episode
step
frame_ref
reason
```

这里的“保存关键帧”是引用运行过程中已经自动持久化的帧。当前流程不强制把对应 JPEG、深度文件另外复制进经验包，因此完整图像仍主要保存在原 Writer Workspace 中。

### 7.3 `binding.json`

分开保存两类内容：

```text
episode_local_values
```

只适用于本次 Seed/Reset 的坐标、深度、阈值、偏置和其他数值。

```text
recompute_methods
```

在新 Seed 或新 Reset 中如何重新计算上述参数的方法。

这个拆分用于避免把一次成功轨迹中的坐标误当成通用经验。但当前校验器只能检查这两个字段是否存在，不能理解其中的语义是否真的完成了抽象。

### 7.4 `episode-summary.md`

由 Agent 用中文总结：

- 最终成功路径；
- 重要失败；
- 失败机理；
- 如何修正；
- 哪些结论由 RGB、深度、力反馈或本体感知支持；
- 哪些知识可以迁移，哪些数值只能留在本次 Episode。

### 7.5 `program/`

这是给后续 Coding Agent 直接读取和复用的程序经验。

`program-index.json` 描述：

- 根节点；
- 层级 Program 节点；
- 每个节点的目的、子节点、源文件和入口；
- 模块清单与 SHA-256。

`program/modules/` 保存代码文件。这些文件必须逐字复制自本次真正执行过的 `agent/` 源文件，不能在复盘阶段凭空发明一套没有运行过的新算法。

这意味着当前系统保存的是“本次运行验证过的代码”，而不只是自然语言建议。

## 8. Harness 如何校验经验

Agent 写完后，Harness 会执行确定性的校验。

当前会检查：

- 所有必需文件是否存在；
- `review-result.json` 是否为 `status=complete`；
- Action Ledger 和 Keyframes 是否非空；
- Action 的 Episode/Step 范围是否真实存在；
- `evidence_refs` 和关键帧是否引用真实 `frame_ref`；
- `binding.json` 是否同时具有 `episode_local_values` 和 `recompute_methods`；
- Program 是否具有模块、节点和根节点；
- 节点引用的子节点和模块是否存在；
- 模块是否逐字等于对应的 `agent/` 源文件；
- SHA-256 是否一致；
- `episode-summary.md` 是否非空。

Codex 和 Claude 当前在 Program 节点规则上有一个实现差异：

- Codex 校验器要求每个节点都有 `filename` 和 `entrypoint`；
- Claude 校验器允许只组织子节点的 composite node 没有 `filename`，但它必须具有非空 `children`。

第一次校验失败时，Harness 会写入：

```text
experience-review/validation-error.json
```

然后在同一 Agent 会话中发送修复 Prompt。当前最多进行 2 次复盘/修复尝试，复盘总时限为 30 分钟。

## 9. 经验如何晋级

只有同时满足以下条件，生成 Job 才算成功：

```text
任务成功
AND
经验复盘完成
AND
Harness 校验通过
```

校验后的经验先导出到对应生成 Job 的 review 目录，再由总调度器复制到版本目录：

```text
models/<model>/batches/<batch>/experiences/task-XX/e1/
models/<model>/batches/<batch>/experiences/task-XX/e2/
```

晋级后目录被设置为只读，并记录：

- `experience_id`；
- `parent_experience_id`；
- Writer Job；
- Writer Seed；
- 整棵经验目录的 SHA-256。

E2 生成时只读取已经晋级并冻结的 E1，不直接读取 E0 或其他历史版本。

## 10. 当前流程能保证什么

当前 Harness 能保证：

1. 经验来自一次服务器确认成功的任务执行；
2. Action 和关键帧引用真实存在的 Episode/Step；
3. 保存的 Program 模块确实来自本次 `agent/` 中的实际代码；
4. 父经验与新经验以独立只读版本保存；
5. 正式评测不会回写或污染经验；
6. 总结阶段不能继续操作机器人来补造证据。

## 11. 当前流程不能保证什么

当前 Harness 不能保证：

1. 新经验一定优于父经验；
2. 一次成功轨迹在其他 Seed 上仍然有效；
3. Agent 没有把失败脚本或救援流程放进默认主流程；
4. `recompute_methods` 的语义真的足够通用；
5. Program 的层级抽象是最小、清晰且可迁移的；
6. 本次局部常数没有残留在代码中；
7. 经验不会因为重写整个包而丢失父版本中的优秀结构；
8. 关键帧图片会随经验包一起独立迁移。

当前晋级门槛本质上是：

```text
单个 Writer Seed 成功 + 结构/事实引用校验通过
```

目前没有：

- 父版本与新版本的性能对比；
- held-out Seed 回归测试；
- 一次成功率、Episode、Step、耗时的晋级门槛；
- 对默认路径与错误恢复分支的自动检查；
- 对代码中 Seed 特定常数的语义检查。

因此 T3 E2、T9 E2 这类“成功但过度拟合救援过程”的经验仍然能够通过校验并晋级。

## 12. 相关实现

- `experiment.json`：Seed、模型、并发、版本和时限配置；
- `model_orchestrator.py`：E0→E1→E2 调度、重试、复制和冻结；
- Codex `codex_cell.py`：同 thread 执行与成功后复盘；
- Codex `review_support.py`：Codex 经验结构与事实引用校验；
- Claude `claude_cell.py`：同 session 执行、复盘及 Claude 侧校验。
