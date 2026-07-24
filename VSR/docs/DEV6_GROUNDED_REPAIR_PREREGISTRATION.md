# RSP-VSR dev6 视觉证据约束纠错 Phase 0 预注册

## 研究边界

普通小语言模型润色、N-best 重排、拼音提示、视觉 embedding 和联合训练均已有
直接近邻。本阶段不训练语言模型，只检验当前 replay 轨迹产生的 beam 候选是否
包含足够的可恢复字符证据。若连这一必要条件都不满足，立即停止 EviCo-VSR
路线，不下载模型、不扫参、不读取 holdout2。

## 冻结数据与运行

- 仅复用已消费的 target-dev5 development 流，speaker 为 `071/126/045`，
  A--B--C--A 长度 `115/229/223/114`，共 681 条。
- manifest SHA-256：
  `8c8e967e7076562da70d47f45883ead84c5c2dd2ef47f69412467db2e89ebf56`。
- sidecar SHA-256：
  `27d96cedeba419350abb2daabd1ea9e2be03127d62b0eaf1badd58be7d9421c3`。
- 使用 dev5 replay adapter 的固定 seed 42、68 个 periodic-F10 反馈位置、
  相同基础 checkpoint、可靠伪标签、更新目标和回滚规则。
- 唯一变化是 `decoder.nbest_size=10`；beam size 保持 12。N-best 记录发生在
  当前样本更新前，不得改变 1-best、查询或更新轨迹。
- GPU 总预算不超过 6 小时；温度达到 78°C、磁盘可用空间不高于 30 GiB、
  确定性错误或资源异常时立即停止。

## 完整性验收

运行必须为 attempt 1、681 条连续且唯一的 index/UID、28 条 history、每 100 条
checkpoint、最多 3 个 `.pt`、最终 checkpoint 为 681、68 次反馈、配置/manifest/
sidecar/vocabulary/base checkpoint 哈希一致、结构化和日志错误为 0。每条记录的
N-best rank 必须从 1 连续排列，rank-1 transcript/tokens 必须与原始预测一致。

新的 rank-1 transcript、decoder tokens、feedback query、update status 和累计 CER
必须与已严格验收的 dev5 replay 归档逐条一致。任何不一致都视为实现失败，不能
解释为研究结果。

## Phase 0a 门控

在固定 top-10 上报告 1-best CER、整句 N-best oracle CER、乐观的跨候选字符组合
上界、oracle rank、substitution/deletion coverage、正确 1-best 的错误候选风险和
运行资源。只有同时满足以下条件才为 `BEAM_GO`：

1. 整句 N-best oracle 相对 1-best 至少提供 `0.02` 绝对 CER headroom；
2. 1-best substitution 的正确目标字符 coverage@10 至少为 `0.55`；
3. rank-1 与 dev5 replay 逐条完全复现，且所有完整性验收通过；
4. 总运行时间不超过 6 GPU 小时。

`BEAM_GO` 只授权 Phase 0b：导出拼音 top-k、CTC 时间片证据和视觉单元，验证
候选互补与证据可定位性。它不授权训练 LLM，也不授权读取 holdout2。任一条件
失败即为 `NO_GO`，停止语言纠错路线并回到视觉 encoder/decoder 训练内改进。
