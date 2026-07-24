# RSP-VSR dev7 反事实 Beam-Margin 适应预注册

## 研究问题

dev6 证明当前 beam-10 中的正确字符覆盖不足，不能支持直接文本修复。
本阶段不使用语言模型生成文本，而是在真实反馈点把更新前 beam 中
流畅但错误的序列作为视觉反事实负例，检验训练期排序约束能否改善
prequential CER，同时不恶化回访遗忘。

## 方法冻结

对每个真实反馈样本，在更新前的 top-10 中排除与真值相同或 CTC 目标无效的
候选，选择长度归一化 CTC loss 最低的错误候选作为唯一 hard negative。
在现有 full-sequence replay 上增加：

\[
L_{\mathrm{cf}}=\max\left(0,\;m+L_{\mathrm{ctc}}(y^+)-
L_{\mathrm{ctc}}(y^-)\right),
\qquad L=L_{\mathrm{replay}}+\lambda L_{\mathrm{cf}}.
\]

固定 `m=0.2`、`lambda=0.25`，仅在反馈更新中启用。非反馈样本完全沿用
原 replay 的可靠伪标签、特征锚定、一步更新和事务回滚。反事实候选在
当前样本更新前冻结，不从真值后的模型重新解码。若更新后 margin
violation 比更新前增加超过 `1e-6`，则与现有 target-loss、anchor-KL 和
reliability 规则一样回滚。不扫 margin、权重、beam 大小、学习率、seed
或反馈预算。

## 冻结数据与对照

- 仅使用 Chinese-LiPS 训练池，不使用 validation、test 或 holdout2。
- 排除 dev2 `015/098/133`、dev3 `120/176/183`、dev4 `128/047/202`、
  dev5 `071/126/045`、holdout2 `046/001/093` 和禁用 speaker `013`。
- 在剩余训练 speaker 中按样本数降序、ID 升序确定性选取
  `188/011/036`，数量为 `213/206/206`。
- 固定 seed 42 在 speaker 内打乱，构造
  `188(A1=107) -> 011(B=206) -> 036(C=206) -> 188(A2=106)`，共 625 条。
- 两路都使用 beam 12、导出 top-10，在每个完整 10 样本窗口的最后一条
  查询真值，恰好 62 次；最后 5 条不查询。

| 方法 | 反馈预算 | 唯一差异 |
| --- | ---: | --- |
| matched replay | 62 | 现有 full-sequence replay，同样导出 top-10 |
| counterfactual margin | 62 | 只增加冻结的 hard-negative margin loss |

两路使用同一 checkpoint、manifest、sidecar、seed、查询位置、伪更新路径、
回滚规则和计分顺序，并行运行只为节省墙钟时间。

## 完整性与安全门槛

每路必须为 attempt 1、625 条连续且唯一的 index/UID/domain/order、25 条
history、每 100 条 checkpoint、最多 3 个 `.pt`、最终 checkpoint 为 625、
62 次查询、匹配的 commit/config/manifest/sidecar/vocabulary/base checkpoint 哈希、
结构化和日志错误为 0。两路均必须每条导出恰好 10 个有限分数候选。

候选路的 62 个反馈点必须全部产生反事实诊断，至少 56 个（90% 以上）
找到有效错误候选；记录 negative rank/tokens、target/negative loss、gap、violation
及更新前后值。前 10 个反馈点若少于 8 个有效负例，或没有任何正的
pre-update violation，则机制实际不工作，立即早停。温度达到 78°C、磁盘可用
空间不高于 30 GiB、确定性错误、非有限梯度/损失或资源异常时立即停止。
单路总计算预算不超过 6 GPU 小时。

## 分析与决策

两路完整验收后使用 10,000 次 paired bootstrap，报告整体和 A1/B/C/A2 CER、
static-corrected A2--A1 forgetting、更新数、参数量、吞吐、峰值显存、checkpoint
开销，以及反馈点的 margin gap/violation 前后变化。只有同时满足以下条件才为
`COMPONENT_GO`：

1. `counterfactual - replay` 整体 CER 差 `<= -0.003`，且 95% CI 全低于 0；
2. 相对 replay 的 static-corrected forgetting 恶化不超过 `0.002`，且差值 CI
   上界不高于 `0.002`；
3. 在有效负例上，post-update violation 相对 pre-update 至少降低 20%，
   paired bootstrap 差值 CI 全低于 0；
4. 两路各恰好 62 次查询，所有完整性与资源检查通过。

任一条件失败即为 `NO_GO`：不追加 seed、不扫参、不读取 holdout2。
`COMPONENT_GO` 也不直接授权 holdout2，只冻结该训练期组件，并授权在独立
预注册下与证据约束文本纠错器进行联合开发验证。
