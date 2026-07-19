# RSP-VSR 研究与实验协议

## 1. 研究定位

RSP-VSR（Reliability-gated Structural Plasticity for VSR）研究一个具体问题：

> 中文连续唇读模型部署后遇到新说话人、摄像头、光照或遮挡时，能否不重训主干，
> 只通过受控的参数与结构变化持续适应，同时限制伪标签污染和旧域遗忘？

它不是 RAG 或外部记忆系统。每次被接受的更新都会改变 adapter 参数；稳定的新域还会
导致专家数量增长。主干模型始终冻结，使得这种变化可检查、可保存、可回滚。

必须说清：代码可以构成论文假设，但不能保证录用。只有服务器上的无泄漏对照、
消融、统计显著性和真实场景结果同时成立，才能支撑投稿。
与 PAINT、SDEA/FDD 等最近邻的边界见 [NOVELTY_REVIEW.md](NOVELTY_REVIEW.md)。

## 2. 基座模型选择

上游是 CNVSRC2025 官方 VSR 基线，架构为 3D CNN/ResNet18 视觉前端、12 层
Conformer 编码器和 CTC/双向 Attention 解码器。本项目不改官方主干的 checkpoint key。

| 权重 | 用途 | 原因 |
| --- | --- | --- |
| `model_avg_cncvs_4s_30s.pth` | 主论文的持续适应基座 | 只见过 CN-CVS，可把 CNVSRC-Multi Dev 作为未见目标流 |
| `model_avg_cncvs_2_3_cnvsrc.pth` | 全量官方基线和真实新采集 OOD 实验 | 官方报告 Eval CER `31.55%`，但不能再在它已见的 CNVSRC 数据上宣称未见域适应 |

这是“官方仓库内最强已报告 checkpoint”，不等于宣称它是所有中文唇读工作中的
全局 SOTA。

官方镜像权重校验值：

| 权重 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `model_avg_cncvs_4s_30s.pth` | `1137493817` | `79cb59044e925ce7583b46474ee17c84f938ea088861d33e4340614e1436104f` |
| `model_avg_cncvs_2_3_cnvsrc.pth` | `1137500697` | `577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c` |

## 3. 单一机制闭环

上一版项目的教训是：多个弱模块叠加后，一旦删除消融不变差，创新点就不成立。
RSP-VSR 只保留一个结构可塑性闭环：

```text
冻结基座提取序列特征
  -> 用均值/标准差 + 一/二阶时序差分幅度签名路由专家
  -> 持续新域：隔离观察，不污染旧专家
  -> 偏移确认：增长零初始化残差 adapter
  -> CTC 跨视图 + CTC/AED 一致性筛选伪标签
  -> 候选参数更新
  -> 目标损失/基座锚点/可靠性与 CTC emission 状态验证
  -> 接受，或连同优化器状态一起回滚
```

其中最需要通过消融证明的是“隔离后增长”：普通动态 adapter 在确认新域之前会继续
修改最近的旧专家；RSP-VSR 在偏移证据累积期只做推理。人工纠正与特征新颖性同时出现时，
可以立即确认增长，不必丢掉稀缺的反馈。

零初始化保证新专家刚创建时与冻结基座等价，因此结构增长本身不会立即破坏输出。

运动签名借鉴了 Cued-Bridge 对口型动态的建模思路，但这里只计算编码特征的一、二阶
时序差分幅度，不把它解释成物理速度/加速度，也不把额外 Conv3D 插入官方预测路径。
这样既保留 checkpoint 兼容性，也能通过 `signature_motion_order=0` 做严格删除消融。
Cued-Bridge 对 CTC blank collapse 的监控则
被改造成事务安全条件：如果候选更新前可靠、更新后不再通过 emission-rate 等可靠性约束，
即使聚合分数没有下降也必须回滚。

没有移植其 Visual Phoneme Adapter、CR-CTC、EMA、语义对齐、InterCTC 和手势 prompt：
前两者与现有 adapter/跨视图一致性重复；其余模块依赖离线重训、额外标注或改变 checkpoint
结构。原版 AdaMER 也没有作为独立损失叠加，避免把论文主线重新变成模块堆叠。

## 4. 可证伪假设

- H1：在 speaker-block 未见域流中，RSP-VSR 的 prequential CER 低于静态官方基线。
- H2：在 `A -> B -> A` 循环流中，隔离增长比单一在线 adapter 的返回域遗忘更低。
- H3：在伪标签或人工反馈污染时，可靠性门控和事务回滚使性能退化显著小于无防护版本。
- H4：相对全量微调，动态 adapter 在参数量、更新延迟和存储上足以支持端侧个性化。

H1 不成立时不应仅靠 H3/H4 包装成主方法；H2 不成立时，“结构可塑性”不能作为
主贡献。

## 5. 无泄漏实验协议

1. 主实验从 `model_avg_cncvs_4s_30s.pth` 启动。
2. 阈值和超参数只能用 CN-CVS valid 或独立 calibration stream 选择。
3. CNVSRC-Multi Dev 每条样本必须先预测和记录 CER，再允许用伪标签或预先固定的反馈计划更新。
4. 域标签只用于排序和事后分组评估，不输入路由器。
5. 反馈频率、污染率和随机种子必须在看结果前写入配置。
6. 一条样本只在流中出现一次；不能为了改善当前样本的记录结果而重放。
7. 官方全量 checkpoint 只在真正新采集的说话人/OOD 数据上做持续适应结论。

主流式场景：

| 场景 | 排列 | 回答的问题 |
| --- | --- | --- |
| 未见说话人块 | 每个 speaker 连续出现，speaker 顺序随机 | 能否识别偏移并快速适应 |
| 循环回访 | `A -> B -> C -> A` | 旧域是否被单一 adapter 覆盖 |
| 渐变环境 | 逐步增加亮度、遮挡、压缩和姿态幅度 | 隔离和增长是否对短暂抖动过度反应 |
| 稀疏纠正 | 每 `N` 条提供一次真值 | 人机交互成本与收益的关系 |

## 6. 对照与消融

必做对照：

- 静态官方基线：`plasticity.mode=static`。
- 单一在线 adapter：`plasticity.mode=single_adapter`。
- RSP-VSR 动态专家：默认配置。
- 移植的 TENT/稳定测试时适应方法：同一 backbone、数据顺序和更新预算。
- 离线目标域 adapter 上界：可用标签训练，但必须明确标为 oracle，不与无标签方法混比。

最小消融矩阵：

| 实验 | 配置变化 | 验证点 |
| --- | --- | --- |
| 无结构增长 | `mode=single_adapter` | 增长是否降低返回域遗忘 |
| 无偏移隔离 | `growth_patience=1` | 持续性确认是否抑制短暂偏移误触发 |
| 无运动感知路由 | `signature_motion_order=0` | 时序动态统计是否提高新域识别与路由稳定性 |
| 无可靠性门控 | `reliability_enabled=false` | 错误伪标签的污染程度 |
| 无事务回滚 | `rollback_enabled=false` | 候选更新验证的价值 |
| 反馈不确认增长 | `feedback_confirms_growth=false` | 稀疏人工纠正能否加快新域建模 |
| 去掉 CTC/AED 协同 | `reliability.decoder_agreement_weight=0` | 双解码器一致性是否真正筛出了更好伪标签 |

不要在主方法中继续叠加语音学辅助头、EMA、MixStyle 等模块。如果后续要试，必须先作为
独立候选并通过匹配删除消融。

## 7. 指标和统计

主指标是“更新前预测”的 prequential CER，不允许用当前样本更新后的结果回填。

- 总 CER、分域 CER 和流式分段 CER。
- 返回域遗忘：首段 A 与回访 A 的 CER 差。
- 适应延迟：域切换后恢复到稳态 CER 所需样本数。
- 安全性：更新接受率、回滚率、失败率、伪标签精度和污染反馈下的 CER 增量。
- 结构成本：专家数、可训练参数、显存、单样本更新时间和推理延迟。

每个主结果至少用 3 个随机种子，报告均值、标准差和按 utterance 配对 bootstrap
95% CI。主方法与静态/单 adapter 比较要使用同一流顺序和随机视图种子。

## 8. 发表 go/no-go 线

这些是启动全量投稿写作前的内部标准，不是预先宣称的结果：

- RSP-VSR 相对静态基线的 CER 改善在 95% CI 下仍为正，且至少在两种独立偏移上重现。
- RSP-VSR 在循环流上显著优于单 adapter，否则动态增长不能作为主贡献。
- 可靠性门控和回滚在至少 `10%` 与 `20%` 反馈污染下都要显著降低退化。
- 隔离、增长、门控和回滚各自的匹配消融要能解释一项独立现象；无效部分直接删除。
- 真实新采集数据至少包含多个说话人和两种采集条件，结果不能只有合成扰动。
- 与一个强测试时适应基线比较后仍有收益，且额外参数/延迟有完整报告。

如果只是静态 CER 小幅改善，但循环遗忘、污染鲁棒性或真实场景都不成立，就应该 no-go，
不再重复上一版“持续增加模块”的路线。

## 9. 应用价值

目标应用是本地、隐私保护的个性化静默交互：失语/听障辅助交流、高噪声环境指令输入、
不宜出声场景的终端控制。用户的口型、相机位置和环境会持续变化，而原始视频不需要上传，
只保存小型 adapter 状态。

当前上游许可证限定为非商业比较/基准用途。现阶段只能做研究和应用原型；真正产品化前
必须更换可商用基座或取得单独授权。

## 10. 服务器执行

当前本地是 partial clone，服务器先恢复完整数据清单：

```bash
git sparse-checkout disable
cd VSR
```

按官方 `README.md` 安装 Python 3.10/PyTorch 环境后，下载无泄漏主实验 checkpoint：

```bash
bash scripts/download_checkpoint.sh
```

脚本优先使用 Hugging Face，连接失败时会自动尝试 ModelScope 镜像。

若要下载官方全量 checkpoint：

```bash
bash scripts/download_checkpoint.sh model_avg_cncvs_2_3_cnvsrc.pth
```

清单工具读取 CSV 的前四列，并允许数据集在尾部保留扩展列。生成
speaker-block 流时，域正则要按真实路径调整：

```bash
python scripts/prepare_stream_manifest.py \
  --csv data/cnvsrc-multi/valid.csv \
  --output data/continual/cnvsrc_multi_stream.jsonl \
  --domain-regex '^(?P<domain>[^/]+)' \
  --order domain-block \
  --shuffle-domains \
  --shuffle-within-domain \
  --seed 42
```

先跑同一流的静态基线，再跑 RSP-VSR：

```bash
python continual_adapt.py \
  plasticity.mode=static \
  output_dir=exp/rsp_vsr/static_seed42

python continual_adapt.py \
  plasticity.mode=expert_bank \
  output_dir=exp/rsp_vsr/full_seed42
```

Chinese-LiPS 官方预处理视频和六列 CSV 可以直接生成未见说话人流：

```bash
python scripts/prepare_stream_manifest.py \
  --csv /data/chinese_lips/labels/test.csv \
  --output /data/manifests/chinese_lips_test_seed42.jsonl \
  --domain-regex '(?:^|/)(?P<domain>[0-9]+)_' \
  --order domain-block \
  --shuffle-domains \
  --shuffle-within-domain \
  --seed 42
```

长时实验用 supervisor 绑定单卡并在异常退出或长时无进度时续跑：

```bash
PYTHON_BIN=/path/to/python scripts/run_continual_experiment.sh \
  /data/experiments/chinese_lips/static_seed42 0 \
  data_root_dir=/data/chinese_lips \
  stream_manifest=/data/manifests/chinese_lips_test_seed42.jsonl \
  plasticity.mode=static \
  seed=42

PYTHON_BIN=/path/to/python scripts/run_continual_experiment.sh \
  /data/experiments/chinese_lips/full_seed42 1 \
  data_root_dir=/data/chinese_lips \
  stream_manifest=/data/manifests/chinese_lips_test_seed42.jsonl \
  plasticity.mode=expert_bank \
  seed=42
```

supervisor 默认每 60 秒更新 `supervisor_status.tsv`，连续 45 分钟没有
新结果才判定卡死，并最多自动恢复 3 次。它不会跳过样本或修改实验
超参数；确定性错误连续复现后会停止，避免产生不可比较的结果。

单 adapter 和两个关键消融：

```bash
python continual_adapt.py \
  plasticity.mode=single_adapter \
  output_dir=exp/rsp_vsr/single_adapter_seed42

python continual_adapt.py \
  plasticity.growth_patience=1 \
  output_dir=exp/rsp_vsr/no_quarantine_seed42

python continual_adapt.py \
  plasticity.signature_motion_order=0 \
  output_dir=exp/rsp_vsr/no_motion_signature_seed42

python continual_adapt.py \
  plasticity.reliability_enabled=false \
  output_dir=exp/rsp_vsr/no_reliability_gate_seed42

python continual_adapt.py \
  plasticity.rollback_enabled=false \
  output_dir=exp/rsp_vsr/no_rollback_seed42
```

污染反馈实验必须配合固定反馈频率：

```bash
python continual_adapt.py \
  feedback.every=20 \
  feedback.noise_rate=0.2 \
  output_dir=exp/rsp_vsr/feedback_noise20_seed42
```

每个目录会保存逐样本 `stream_results.jsonl`、汇总 `summary.json` 和可恢复的
`adaptation_state.pt`。默认每 100 条原子刷新一次 checkpoint，其中包含
adapter、优化器、累积指标、RNG 和流清单哈希。从同一输出目录续跑：

```bash
python continual_adapt.py \
  resume_adaptation_checkpoint=exp/rsp_vsr/adaptation_state.pt \
  output_dir=exp/rsp_vsr
```

续跑会丢弃 checkpoint 之后已写入但未持久化的结果尾部，再从确切的
`processed_samples` 继续，避免重放样本。流清单哈希不同时会拒绝恢复。

Apple Silicon 本机会先检查 MPS 是否支持视觉前端的 Conv3D；不支持时
`device=auto` 会回退 CPU。支持时，MPS 未实现的 CTC loss 在 CPU
上计算，其余前向和 adapter 更新保留在 MPS。可用
`RSP_VSR_DEVICE=cpu` 强制 CPU。视频读写需要 PyAV；`imageio-ffmpeg`
会提供跨平台 ffmpeg 回退，也可用 `FFMPEG_BINARY` 指定系统二进制。
本地仍需使用项目声明的 Python 3.10；系统 Python 3.14 不兼容当前 Hydra。
完整服务器实验仍需记录 GPU 型号、依赖版本、commit、随机种子和流清单哈希。
