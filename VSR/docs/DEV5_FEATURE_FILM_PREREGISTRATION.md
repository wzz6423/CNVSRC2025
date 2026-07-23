# RSP-VSR dev5 视觉 Feature-FiLM 预注册

## 研究问题

在主干和解码器完全冻结、每 10 条样本最多获得 1 次真值纠正的前提下，
仅适应 CTC 前视觉特征的逐通道缩放与平移，是否能比现有 75,265 参数的
replay adapter 获得更低的 prequential CER，同时不恶化回访遗忘。

Feature-FiLM 使用恒等初始化：

\[
\tilde{h}_{t,d}=h_{t,d}(1+\Delta\gamma_d)+\beta_d,
\qquad \Delta\gamma_d=\beta_d=0.
\]

对 768 维特征，唯一候选共有 1,536 个可更新参数。它复用现有 RSP 的
先预测后更新、可靠性筛选、完整序列反馈目标、特征锚定与事务回滚；
除 adapter 参数化外不改变损失、学习率或阈值。

## 冻结数据

- 数据来自 Chinese-LiPS 训练池，不使用 validation/test 和 holdout2。
- 先排除 dev2 `015/098/133`、dev3 `120/176/183`、dev4 `128/047/202`、
  holdout2 `046/001/093` 以及禁用说话人 `013`。
- 在剩余说话人中，按训练样本数降序、speaker ID 升序确定性选取前三名：
  `071/126/045`，数量为 `229/229/223`。
- 固定 seed 42 在说话人内打乱，构造
  `071(A1=115) -> 126(B=229) -> 045(C=223) -> 071(A2=114)`，共 681 条。
- 原始 manifest 的 `feedback=false`；反馈位置由 periodic-F10 在运行前确定，
  每路恰好 68 次查询。

## 唯一实验矩阵

| 方法 | 更新 | 反馈预算 | 作用 |
| --- | --- | ---: | --- |
| static | 不更新 | 0 | 新流的绝对参照 |
| replay adapter | 现有 bottleneck RSP | 68 | 同协议 incumbent |
| Feature-FiLM | 逐通道 scale/shift RSP | 68 | 唯一新候选 |

三路均使用同一 checkpoint、manifest、decoder、seed 和记分顺序。两条 F10 轨使用完全
相同的 68 个纠正位置，且启用相同的可靠伪标签更新。不追加其他 FiLM 层、
学习率、初始化、seed 或反馈预算搜索。

## 严格验收与门控

每路必须为 attempt 1、681 条连续 index/UID/domain/order、28 条 history
`25,50,...,675,681`、每 100 条 checkpoint、最多 3 个 `.pt`、最终 checkpoint 为
681、配置/manifest/sidecar/checkpoint 哈希一致且结构化错误为 0。static 必须全部
skipped；两条 F10 轨必须各有 68 次策略查询。Feature-FiLM 必须恰好更新
1,536 个参数/2 个 tensor，且 checkpoint 恢复后输出一致。

完成后使用 10,000 次 paired bootstrap、同一 A1/A2 static-corrected forgetting 定义，
并报告更新数、参数量、吞吐、峰值显存和 checkpoint 开销。只有同时满足以下
条件才为 `GO`：

1. `Feature-FiLM - replay adapter` 的整体 CER 差 `<= -0.003`，且 95% CI 全低于 0；
2. `Feature-FiLM - static` 的 CER 差 95% CI 全低于 0；
3. Feature-FiLM 相对 replay adapter 的 static-corrected forgetting 差不大于 0；
4. 两条 F10 轨均恰好使用 68 次查询，所有严格验收通过。

任一条件失败则为 `NO-GO`：holdout2 继续冻结未读，不追加 seed 或扫参。
