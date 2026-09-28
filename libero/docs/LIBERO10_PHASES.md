# LIBERO-10 阶段定义

Gateway 为 LIBERO-10 的每个任务定义 4 个阶段，编号表示推荐的任务执行顺序。每个阶段独立判定；一旦在当前 Episode 达成便保持成功，即使物体后来掉落或状态回退。独立判定可正确记录改变执行顺序或通过推动而非抓取完成任务的策略。Reset 会清空全部阶段状态。最终阶段始终对应 LIBERO 原生完整任务成功条件。

| Task | 阶段 1 | 阶段 2 | 阶段 3 | 阶段 4 |
|---:|---|---|---|---|
| 0 | 抓住 alphabet soup | alphabet soup 放进篮子 | 抓住 tomato sauce | 两件物体都在篮子中 |
| 1 | 抓住 cream cheese | cream cheese 放进篮子 | 抓住 butter | 两件物体都在篮子中 |
| 2 | 打开炉灶 | 抓住 moka pot | moka pot 抬高至少 2 cm | moka pot 放在已打开的炉灶上 |
| 3 | 抓住黑碗 | 黑碗抬高至少 2 cm | 黑碗放进底层抽屉 | 黑碗在抽屉内且抽屉关闭 |
| 4 | 抓住白杯 | 白杯放到左盘 | 抓住黄白杯 | 两个杯子分别位于正确盘子上 |
| 5 | 末端进入书本 8 cm 范围 | 抓住书本 | 书本抬高至少 2 cm | 书本放进 caddy 后格 |
| 6 | 抓住白杯 | 白杯放到盘子上 | 抓住 chocolate pudding | 白杯和 pudding 都位于目标位置 |
| 7 | 抓住 alphabet soup | alphabet soup 放进篮子 | 抓住 cream cheese | 两件物体都在篮子中 |
| 8 | 抓住第一只 moka pot | 第一只 moka pot 放到炉灶 | 抓住第二只 moka pot | 两只 moka pot 都在已打开的炉灶上 |
| 9 | 抓住黄白杯 | 杯子放进微波炉 | 放入杯子后关闭微波炉 | 杯子位于关闭的微波炉中 |

放置、开关和终局条件复用 LIBERO BDDL 谓词；抓取使用 robosuite 的双指接触抓取判定；抬高使用物体相对本 Episode 初始高度的 2 cm 位移。Task 5 的接近阶段使用 MuJoCo world frame 中末端与书本中心的欧氏距离。

阶段信息不出现在 Agent 可见的 `tasks`、`reset`、`step`、`result` 或 `DELETE` 响应中。Gateway 在内部持续更新状态；可信 controller 在正常结束、超时、停止或模型断联后的统一清理阶段读取一次结算信息，并写入 `libero-artifacts/task-progress.json`、`libero-artifacts/closed.json` 和 `.libero-controller-result.json`。每个阶段包含 Episode 内累计的 `success` 和首次达成的 Gateway Step；对多个 Episode 的同一阶段统计 `success=true` 比例即可得到阶段成功率。

结算对象包含 `completed_phases`、`total_phases`、`completion_ratio` 和完整 `phases` 列表。`completion_ratio` 是达成阶段数除以总阶段数；它描述任务完成程度，不代替 LIBERO 原生最终 `success`。
