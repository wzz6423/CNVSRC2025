# Dev6：训练期视觉约束小语言模型与持续纠错调研

更新日期：2026-07-24

## 1. 结论先行

“在 VSR 输出后接一个小语言模型，修正错别字和语序”是合理产品功能，但已经不足以构成顶会或 IEEE TIP/TPAMI 级创新。2023--2026 年的一手工作已经覆盖：

- 从 ASR 的 N-best 文本生成更好的转写（HyPoradise）[S6]；
- 把声学特征、唇动特征或音视频特征注入纠错 LLM（UADF、LipGER、AVGER）[S7][S8][S9]；
- 用视觉离散单元约束 LLM 重排（AVUR-LLM）[S14]；
- 用中文拼音和 N-best 进行 LLM 修正（PY-GEC、VALLR-Pin）[S11][S13]；
- 让 LLM 同时比较 ASR/VSR 两套假设，并接收模态可靠性提示（DualHyp/RelPrompt）[S15]；
- 在视觉和语言两层适配特定说话人（Personalized Lip Reading）[S16]；
- 直接以视觉/音视频特征训练 LLM 解码器（VSP-LLM、Llama-AVSR、MMS-LLaMA）[S3][S4][S5]。

在本次检索范围内，仍然有较清楚且可证伪的组合空白：

> **在未知域边界的数据流中，用固定的少量人工反馈同时持续更新视觉适配器与小型中文纠错模型；每一个语言模型改动都需要可定位的视觉/拼音证据；当证据不足时主动查询而不是自由生成，并显式控制旧域遗忘。**

这个命题把四件目前大多分开研究的事连成闭环：训练期联合优化、细粒度防幻觉、主动反馈和持续适应。它比“后处理纠错”更有论文价值，也与现有 RSP-VSR 的反馈预算、A1/B/C/A2 流式协议和静态校正遗忘指标直接兼容。

但需要正视一个边界：这里只能说“在下列检索范围内未发现同样的完整闭环”，不能据此宣称绝对首创。正式投稿前还应以最终方法名、损失函数关键词和核心图做一次专门的 novelty search。

## 2. 与当前 RSP-VSR 证据的关系

本地已生成的 dev5 分析显示：

- replay adapter CER 为 `0.5721327498`；
- Feature-FiLM CER 为 `0.6004759274`；
- static CER 为 `0.6039179025`；
- Feature-FiLM 相对 replay 劣化 `+0.0283431777`，95% CI `[+0.0229912553,+0.0338018035]`；
- replay 的 68 个反馈样本中，目标字符 2,296 个，替换 1,047 个、缺失 277 个、额外输出 24 个，平均 token error rate 为 `0.5868`。

证据文件：[dev5_feature_film_analysis.json](../analysis/dev5_feature_film_analysis.json)。

这说明下一阶段不能把问题简化成少数错别字。当前输出同时包含大段替换与缺失；文本模型有机会利用上下文，但也很容易凭语义补写未被唇动支持的内容。因此，下一步的核心指标不能只有 CER，还必须统计“有益编辑、错误编辑、无视觉支持编辑、正确文本过度修改”和反馈后的未来窗口收益。

另一个重要结论是：当前流中几乎所有样本本来就有错误。dev5 replay 的 queried true error rate 为 `0.9853`，非查询样本也接近全错。后续主动学习不应继续优化“能否找到错误样本”，而应优化：

> **哪一个标注最可能在后续样本中减少错误。**

也就是从 error detection 转向 expected update value / future-window gain。

## 3. 现有工作到底做到了哪里

### 3.1 外部 LM、融合和 N-best

传统 VSR 已长期使用 beam search 和外部语言模型。Ma 等人的多语言 VSR 工作在 CMLR 配置中显式加载 RNNLM，并设置 beam decoding；论文与官方代码可核验 [S1]。这意味着“加语言模型”不是新贡献。

HyPoradise 将 N-best 到正确转写建模为生成式纠错，证明 LLM 可以生成不在 N-best 内的 token，但其论文也明确讨论了域错配和缺少声学证据的问题 [S6]。SoftCorrect 则用软错误检测和受约束 CTC，只集中修改疑似错误 token，是不依赖大 LLM 的强中文纠错基线 [S10]。

### 3.2 训练期 LLM-VSR/AVSR 已经存在

VSP-LLM 把 AV-HuBERT 视觉表示映射进 LLM 空间，用视觉单元去重，并用 QLoRA 联合训练 VSR/VST [S3]。Llama-AVSR 将冻结的音频/视觉编码器特征投影为 LLM token，训练 projector 和 LoRA [S4]。MMS-LLaMA 进一步压缩多模态 token [S5]。

因此，“不是后处理，而是在训练时把视觉特征送进 LLM”也不能单独作为新意。

### 3.3 多模态纠错和防幻觉已有明显近邻

- UADF 根据 LLM token uncertainty 在自回归解码时动态融合识别模型分布，并报告可迁移到 AVSR [S7]。
- LipGER 用视觉编码器和多模态 adapter 条件化 LLM，从 ASR N-best 生成转写；其目标就是视觉条件纠错 [S8]。
- AVGER 重新编码原始音视频，与 N-best 组成 cross-modal prompt，并使用 logit、utterance 和 representation 三层一致性损失 [S9]。
- MMGER 将声学帧与字符级 1-best 强制对齐，并联合 ASR/口音识别与冻结 LLM [S12]。
- AVUR-LLM 将视觉中层特征量化为离散单元，令 LoRA LLM 对 N-best 做 list-wise 评分；它已经覆盖“视觉单元约束的 LLM 重排”[S14]。
- DualHyp/RelPrompt 让 LLM 同时使用独立 ASR 和 VSR N-best，并用时间对齐的 Clean/Noisy/Mixed mask 告知模态质量 [S15]。
- 2026 年 7 月的 OT-AVSR 已把音频、视觉特征向 LLM 文本 embedding 做 optimal-transport 语义对齐，并以 OT coupling 作为对比学习软标签 [S22]。

所以，新的防幻觉机制必须比“加入视觉特征、置信度、视觉 token 或一致性损失”更细。一个尚有区分度的方向是：**对每次文本编辑给出时间片级证据，并用反事实干预训练模型拒绝流畅但不受视觉支持的改写。**

### 3.4 中文拼音链路也已有先例

PY-GEC 用合成错误训练 Pinyin-enhanced GEC，并用拼音/文本互转多任务对齐特征空间 [S11]。VALLR-Pin 则在中文 VSR 中联合字符、无声调拼音双解码，再用 Qwen3-4B、预测拼音和字符 N-best 做 LoRA 修正 [S13]。

因此“给纠错器增加拼音”不新。可保留的空白是：把拼音作为**可审计的编辑证据和主动查询视图**，并在真实流式反馈下联合适配，而非只作为第二阶段输入。

### 3.5 持续适应和主动学习各自存在，但没有形成上述闭环

Speaker-adaptive VSR 已有 prompt tuning 和 LoRA [S16][S17]。ASR 侧已有单 utterance test-time adaptation、持续 test-time adaptation、rehearsal-free online continual learning 和 unsupervised online continual learning [S18][S19][S20][S21]。Interspeech 2023 的主动 ASR 工作用无监督多粒度 speech units 和 contrastive selection 降低标注成本 [S23]。

这些工作说明“在线适配”“少标注”“视觉与语言两层适配”本身都不是空白；当前可主张的空白应限定为：**中文 VSR 的证据约束纠错模型，在固定反馈预算、未知边界数据流和遗忘评估下，与视觉模型联合持续更新。**

### 3.6 与最接近工作的差异边界

| 近邻 | 已覆盖的能力 | 不能再作为新意的部分 | EviCo-VSR 必须额外证明的部分 |
| --- | --- | --- | --- |
| 传统 LM / N-best / HyPoradise [S1][S6] | beam 候选融合、文本生成式纠错 | “用 LM 改 1-best”或“从 N-best 生成新 token” | 每个编辑绑定视觉时间片；反事实负例与拒改机制降低无证据改写 |
| LipGER [S8] | 唇动编码器和多模态 adapter 条件化 LLM，基于 ASR N-best 生成 | “视觉条件 LLM 纠错” | 中文 VSR 流上的编辑级证据证书、固定反馈预算下联合持续更新和遗忘控制 |
| AVGER [S9] | 重编码音视频、cross-modal prompt、三层一致性损失 | “把原始模态特征注入 LLM”或“一致性约束” | 局部编辑与视觉片段的可审计对应；无证据时 copy/abstain/query，而非句级自由生成 |
| VALLR-Pin [S13] | 中文字符/拼音双解码、N-best + 拼音 LoRA 修正 | “拼音指导中文 VSR 纠错” | 把拼音作为编辑证据和查询视图，并在未知边界流中与视觉 adapter 双时间尺度适配 |
| AVUR-LLM [S14] | 视觉离散单元、LoRA LLM 对 N-best 做 list-wise 重排 | “visual units + N-best rescoring” | 不限于整句候选排序；生成或拒绝每个编辑都受时间片证据和反事实训练约束 |
| DualHyp/RelPrompt [S15] | 独立 ASR/VSR N-best、时间对齐可靠性 mask、LLM 组合 | “多假设 + 模态可靠性提示” | 单一无音频 VSR 的编辑级证据、持续反馈闭环、未来窗口更新价值查询和遗忘评估 |

因此，论文不能把贡献写成“小 LLM、视觉特征、拼音、N-best 或联合训练”中的任意一个。可检验的差异是：**编辑级可定位证据 + 反事实拒改 + 固定预算持续联合适配 + 以未来收益为目标的主动查询**。缺失前两项时，它只是 LipGER/AVGER/AVUR-LLM 的中文复现；缺失后两项时，它只是离线纠错器。

## 4. 候选创新

### 候选 A：EviCo-VSR，编辑级视觉证据约束的持续纠错

优先级：最高，建议作为主论文方法。

#### 核心机制

在现有 VSR + replay adapter 后增加一个小型中文模型，例如 Qwen3-0.6B 或同量级 encoder-decoder，但它不是自由后处理器。输入包含：

1. 字符 decoder 的 N-best、token score 和 CTC 对齐；
2. 独立拼音 decoder 的 top-k 拼音序列；
3. 经 run-length compression 的视觉离散单元或短视觉 segment embedding；
4. 当前候选改动与对应时间片的 evidence score。

纠错器同时输出最终文本、编辑操作和每个编辑对应的视觉时间片。只有 evidence score 超过阈值的改动才落地；否则复制原 token 或触发反馈查询。这里的关键不是生成更流畅的句子，而是让每次替换、插入、删除都有可审计证据。

#### 训练期闭环

联合或交替优化：

`L = L_vsr + λ1 L_corr + λ2 L_edit_support + λ3 L_counterfactual + λ4 L_identity + λ5 L_anchor`

- `L_vsr`：现有 CTC/attention 或 RSP 目标；
- `L_corr`：目标文本的纠错损失；
- `L_edit_support`：编辑 token 与 CTC/视觉 segment 的对齐及证据 margin；
- `L_counterfactual`：真实句子必须胜过“更流畅但视觉不支持”的硬负例；
- `L_identity`：输入已正确时必须复制，抑制过度纠错；
- `L_anchor`：对 source/static teacher 的输出或参数锚定，限制遗忘。

有人工反馈时同时更新视觉 adapter 和纠错 LoRA；无反馈样本只允许高证据的一致性更新，不允许把语言模型自由生成文本反向灌入视觉模型。

#### 必须做的对照与消融

- replay adapter 原始输出；
- 外部 RNNLM/shallow fusion；
- N-best 纯文本重排；
- SoftCorrect 风格非自回归纠错；
- 小 LLM text-only 1-best；
- 小 LLM text-only N-best；
- `N-best + Pinyin`，对应 VALLR-Pin 思路；
- `N-best + visual units`，对应 AVUR-LLM 思路；
- 完整模型去掉 `edit_support`、`counterfactual`、`identity`、joint update 的逐项消融；
- 只后处理 vs 联合训练，直接证明训练期闭环的贡献。

#### 主要失败风险

- CMLR/Chinese-Lips 句子模板或文本重复导致语言模型记忆答案；
- 当前错误很长，原始视觉 posterior 可能没有足够信息恢复缺失内容；
- CTC 对齐在高 CER 下不稳定，证据标签噪声大；
- 小 LLM 可能提高语言流畅性却损害事实转写；
- 联合更新形成 confirmation loop。

#### 计算成本

- 先用 `0.6B` 级模型、4-bit 权重和 LoRA；不要一开始上 4B/7B；
- 视觉 encoder 保持冻结，只训练现有约 75K replay adapter、视觉 connector、拼音 head 和 LM LoRA；
- RTX 3090 24 GiB 上先将单路开发预算限定为 `<= 12 GPU-hours`，峰值显存 `<= 20 GiB`；超预算而无显著收益则停止；
- 最终必须报告参数、吞吐、峰值显存和 checkpoint 开销，而非只报 CER。

#### 预注册 gate

- 相对 replay adapter：CER 至少降低 `0.005`，10,000 次 paired bootstrap 的 95% CI 全低于 0；
- 相对 text-only 小 LLM：CER 至少降低 `0.003`，CI 全低于 0；
- unsupported harmful edit rate 相对 text-only LLM 至少下降 30%，CI 全低于 0；
- 原本正确样本的 over-correction rate `<= 1%`；
- A1/A2 static-corrected forgetting 相对 replay 不恶化超过 `0.002`；
- 单次样本延迟不超过 replay 的 2.5 倍，或提供可复现的 latency/accuracy Pareto。

任何一项未过都不读 holdout2。

### 候选 B：Cross-View Value Query，按“更新价值”而不是错误概率查询

优先级：高，建议和候选 A 联合；单独作为论文偏弱。

#### 核心机制

为每个流样本计算四种不一致：

- 字符 decoder 与拼音 decoder 映射后的分布不一致；
- 视觉-only 与语言纠错 posterior 不一致；
- N-best 候选间的视觉支持排序与 LM 流畅度排序不一致；
- 当前样本与已适配 domain prototype 的表示偏移。

查询分数不预测“这句话错没错”，而预测“给这个样本真值后，未来 H 个样本能减少多少编辑”。每个完整反馈窗口仍只允许固定一次查询，严格保持与 periodic/random 相同预算。

#### 训练信号

- 每次已发生的反馈产生真实 `future-window edit reduction` 标签；
- 用小型 value head 在线拟合 query value；
- 在样本尚未标注时只用当前及历史信息，禁止窥视后续真值；
- 可用 pairwise ranking，让被查询样本的预测收益高于随机候选。

#### 对照与消融

- periodic、random、单一 entropy、现有 uncertainty；
- 仅 char-pinyin disagreement；
- 仅 visual-LM disagreement；
- disagreement 不加 value calibration；
- 完整 value query。

#### 风险与成本

- 反馈很少，value head 易过拟合；
- future-window gain 噪声大，受流顺序影响；
- 拼音与字符视图并不独立，disagreement 可能只是校准差。

大部分 score 可复用已有 forward，新增计算应小于 10%；若需要每个候选都跑 LLM，则先离线缓存 LM score。

#### 预注册 gate

- 固定相同 10% 查询预算；
- 相对 periodic 的整体 CER 至少 `-0.003` 且 CI 全低于 0；
- 相对 random 显著更优；
- 每个 query 的未来 10/25 样本净 edit reduction 显著高于 periodic/random；
- 不再把 queried true error rate 作为主 gate，因为当前流该指标接近饱和；
- 遗忘不恶化。

### 候选 C：双时间尺度的视觉-语言持续适配

优先级：中，适合作为候选 A 的稳定性组件。

#### 核心机制

- fast path：现有视觉 replay adapter 每次可靠反馈后快速更新；
- slow path：小语言模型 LoRA 只在有人工反馈、证据一致且累计到小批次后更新；
- consolidation：以源模型 anchor、EMA/weight averaging 或 Fisher-style penalty 固化稳定知识；
- 未知 domain 边界，不依赖人工切段；
- 可比较无原始视频 replay 与固定极小 buffer 两个协议。

它与 Personalized Lip Reading 的区别必须写清：后者是在目标说话人适配集上做视觉和语言两层 supervised batch adaptation [S16]；这里研究的是未知边界数据流、固定反馈预算、两时间尺度更新和显式遗忘。

#### 训练信号与对照

- fast adapter：现有 supervised feedback + 高可靠 pseudo objective；
- slow LM：纠错、identity、counterfactual loss；
- consolidation：旧域 teacher consistency 或参数锚定。

对照包括 joint-every-step、只视觉、只 LM、相同学习率、去掉 anchor、EMA、无 replay 与小 buffer。

#### 失败风险、成本和 gate

风险是 slow path 更新太少学不到，或 fast/slow 梯度互相抵消。额外成本主要来自反馈点上的 LM backward，可通过累计 8--16 个反馈批量更新控制。

Gate：相对候选 A 的单时间尺度版本，整体 CER 不劣化，A1/A2 forgetting 至少改善 `0.003` 且 CI 全低于 0；如果只改善遗忘但显著损害当前域 CER，则只作为消融结论，不作为最终候选。

### 候选 D：Viseme-CF，中文视觉反事实与编辑证据 benchmark

优先级：高。它可以成为方法之外的第二个论文贡献，也能防止“只是在 CMLR 上调出一个数字”。

#### 核心机制

从真实视频、强制对齐和模型错误中构造最小反事实：

- 同音不同字：拼音相同、语义不同；
- 近 viseme 不同音：语言上都流畅，但只有一个受唇动支持；
- 流畅词序交换；
- 语言模型常见的未说内容插入；
- 已正确转写的 no-edit 样本。

每条样本提供 ground truth、候选编辑、对应视觉时间片、拼音/viseme 类型和“应修改/应保持/应查询”标签。训练时作为 hard-negative ranking，测试时独立报告纠错和证据质量。

#### 指标

- CER；
- edit precision / recall；
- over-correction rate；
- unsupported edit rate；
- evidence localization hit/IoU；
- abstention/query AUROC；
- 按同音、近 viseme、插入、词序分组的 paired CI。

#### 风险、成本和 gate

自动构造的反事实可能不自然，人工审核成本高；同音异字在视觉上本就不可辨，不能错误地声称视觉证据可以解决，必须依赖上下文并允许 abstain。

先做 500--1,000 条开发 benchmark，至少双人审核 200 条并报告一致率。Gate：自动标签与人工审核一致率 `>= 0.9`；完整模型相对 text-only LLM 在 unsupported edit 和 over-correction 上显著更好，同时真实流 CER 不变差。达不到时，不把 benchmark 作为主贡献。

## 5. 推荐的统一论文命题

四个候选不应各自变成零散实验。最有潜力的统一命题是：

> **Evidence-Grounded Continual Correction for Mandarin Visual Speech Recognition under a Fixed Human-Feedback Budget**

建议的三项主贡献：

1. 编辑级视觉/拼音 evidence certificate 和反事实防幻觉训练；
2. 视觉 adapter 与小语言模型的双时间尺度持续联合适配；
3. 以 future update value 为目标的跨视图主动反馈，加上 Viseme-CF 诊断 benchmark。

这比“RSP-VSR + 小 LLM”更接近顶会叙事，因为它回答一个明确问题：语言先验能修复高歧义中文唇读，但如何证明它没有替视频编造内容，并且能在少量用户纠正下越用越准而不忘旧域？

## 6. 最省 GPU 的执行顺序

### Phase 0：先测可恢复上限，不训练 LLM

在 development stream 上重新导出 beam-10 字符 N-best、拼音 top-k、token score、CTC alignment 和视觉单元。只做以下 oracle：

- 1-best CER；
- N-best oracle CER；
- compositional oracle CER；
- 拼音 oracle 可覆盖的 substitution 比例；
- 仅复制、传统 LM 重排和受约束候选重排。

Phase 0 gate：

- N-best/compositional oracle 至少提供 `0.02` 绝对 CER headroom；
- 至少 20% substitution 可由拼音或候选互补解释；
- beam-10 的运行预算 `<= 6 GPU-hours`。

未过 gate 就不要下载或训练 LLM；应回到视觉 encoder/decoder 与数据质量。

### Phase 1：只训练纠错层，验证“视觉证据是否真的有用”

冻结 VSR，对同一错误对依次跑：SoftCorrect、text-only 1-best、text-only N-best、Pinyin+N-best、visual-units+N-best、完整 evidence-gated corrector。先单 seed，全部使用同一训练对和解码预算。

只有完整模型同时通过 CER、unsupported edit 和 over-correction gate，才进入联合训练。

### Phase 2：训练期闭环和主动反馈

固定唯一架构与超参，对比：

- post-process only；
- joint visual + corrector；
- joint + counterfactual；
- joint + counterfactual + value query；
- 完整双时间尺度版本。

单 seed gate 通过后再跑 3 seeds。禁止边看结果边修改 holdout2 配置。

### Phase 3：外部效度与 holdout

顶会级结论不能只依赖一个重复文本明显、说话人数有限的中文集合。至少需要：

- speaker-disjoint 且 sentence/text-disjoint 的开发/测试；
- CMLR 之外一个有合法访问和明确协议的中文 VSR 集，或公开发布的 Viseme-CF benchmark；
- exact text、规范化 text、speaker、视频 hash 和感知 hash 的泄漏审计；
- 固定 dev gate 通过后才能进行一次性 holdout2 验证。

## 7. 必须避免的伪创新和审稿风险

- 只把 Qwen 接在 1-best 后：已有大量 GER/中文拼音纠错。
- 只把视觉 embedding 接入 LLM：LipGER、AVGER、Llama-AVSR 已覆盖。
- 只做拼音双 decoder：VALLR-Pin 已覆盖。
- 只做 visual unit + N-best 重排：AVUR-LLM 已覆盖。
- 只做 audio/VSR 双假设和置信度提示：DualHyp/RelPrompt 已覆盖。
- 只证明 CER 下降：无法排除语料记忆、文本泄漏和语言模型幻觉。
- 在同一固定主题文本上训练和测试纠错器：极易得到漂亮但不可发表的结果。
- 用 holdout 选择 prompt、阈值或 LoRA rank：破坏预注册边界。
- 只和 static 比：至少需要 replay adapter、传统 LM、SoftCorrect、text-only LLM、Pinyin 和视觉约束近邻。

## 8. 一手来源

[S1] Ma, Petridis, Pantic. **Visual speech recognition for multiple languages in the wild**. Nature Machine Intelligence, 2022. [论文 DOI](https://doi.org/10.1038/s42256-022-00550-z)；[官方代码](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)。官方 CMLR 配置包含 RNNLM 与 beam search。

[S2] Shi, Hsu, Mohamed. **Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction (AV-HuBERT)**. ICLR 2022. [OpenReview](https://openreview.net/forum?id=Z1V2A8Zt2Q)；[官方代码](https://github.com/facebookresearch/av_hubert)。

[S3] Yeo et al. **Where Visual Speech Meets Language: VSP-LLM Framework for Efficient and Context-Aware Visual Speech Processing**. Findings of EMNLP 2024. [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.666/)；[官方代码](https://github.com/Sally-SH/VSP-LLM)。

[S4] Cappellazzo et al. **Large Language Models are Strong Audio-Visual Speech Recognition Learners**. ICASSP 2025. [arXiv](https://arxiv.org/abs/2409.12319)；[官方代码](https://github.com/umbertocappellazzo/Llama-AVSR)。

[S5] Yeo et al. **MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens**. Findings of ACL 2025. [ACL Anthology](https://aclanthology.org/2025.findings-acl.1065/)；[官方代码](https://github.com/JeongHun0716/MMS-LLaMA)。

[S6] Chen et al. **HyPoradise: An Open Baseline for Generative Speech Recognition with Large Language Models**. NeurIPS 2023 Datasets and Benchmarks. [arXiv](https://arxiv.org/abs/2309.15701)；[官方代码](https://github.com/Hypotheses-Paradise/Hypo2Trans)；[官方数据](https://huggingface.co/datasets/PeacefulData/HP-v0)。

[S7] Chen et al. **It's Never Too Late: Fusing Acoustic Information into Large Language Models for Automatic Speech Recognition**. ICLR 2024. [arXiv](https://arxiv.org/abs/2402.05457)；[作者发布模型](https://huggingface.co/PeacefulData/UADFusionGER)。

[S8] Ghosh et al. **LipGER: Visually-Conditioned Generative Error Correction for Robust Automatic Speech Recognition**. Interspeech 2024. [正式论文](https://doi.org/10.21437/Interspeech.2024-1918)；[官方代码/数据](https://github.com/Sreyan88/LipGER)。

[S9] Liu, Yuan, Li. **Listening and Seeing Again: Generative Error Correction for Audio-Visual Speech Recognition**. Information Fusion, 2025. [正式论文](https://doi.org/10.1016/j.inffus.2025.103077)；[作者代码](https://github.com/CircleRedRain/AVGER)。

[S10] Leng et al. **SoftCorrect: Error Correction with Soft Detection for Automatic Speech Recognition**. AAAI 2023. [正式论文](https://doi.org/10.1609/aaai.v37i11.26531)；[作者代码仓库](https://github.com/microsoft/NeuralSpeech)。

[S11] Li et al. **Large Language Model Should Understand Pinyin for Chinese ASR Error Correction**. ICASSP 2025. [正式论文](https://doi.org/10.1109/ICASSP49660.2025.10887651)；[arXiv](https://arxiv.org/abs/2409.13262)。

[S12] Mu et al. **MMGER: Multi-Modal and Multi-Granularity Generative Error Correction With LLM for Joint Accent and Speech Recognition**. IEEE Signal Processing Letters, 2024. [正式论文](https://doi.org/10.1109/LSP.2024.3432275)；[arXiv](https://arxiv.org/abs/2405.03152)。

[S13] Sun et al. **VALLR-Pin: Uncertainty-Factorized Visual Speech Recognition for Mandarin with Pinyin Guidance**. arXiv, 2025（本次检索未找到正式接收信息或官方代码）. [arXiv](https://arxiv.org/abs/2512.20032)。

[S14] Su et al. **Robust LLM-based Audio-Visual Speech Recognition with Sparse Modality Alignment and Visual Unit-Guided Refinement**. arXiv, 2026. [arXiv](https://arxiv.org/abs/2603.03811)。

[S15] Kim et al. **Two Heads Are Better Than One: Audio-Visual Speech Error Correction with Dual Hypotheses**. Findings of ACL 2026. [ACL Anthology](https://aclanthology.org/2026.findings-acl.26/)；[官方代码/数据](https://github.com/sungnyun/dualhyp)。

[S16] Yeo et al. **Personalized Lip Reading: Adapting to Your Unique Lip Movements with Vision and Language**. AAAI 2025. [正式论文](https://doi.org/10.1609/aaai.v39i9.33026)；[arXiv](https://arxiv.org/abs/2409.00986)。

[S17] Kim, Kim, Ro. **Prompt Tuning of Deep Neural Networks for Speaker-Adaptive Visual Speech Recognition**. IEEE TPAMI, 2024. [正式论文](https://doi.org/10.1109/TPAMI.2024.3484658)；[arXiv](https://arxiv.org/abs/2302.08102)。

[S18] Lin, Li, Lee. **Listen, Adapt, Better WER: Source-free Single-utterance Test-time Adaptation for Automatic Speech Recognition**. Interspeech 2022. [正式论文](https://doi.org/10.21437/Interspeech.2022-600)。

[S19] Lin, Huang, Lee. **Continual Test-time Adaptation for End-to-end Speech Recognition on Noisy Speech**. EMNLP 2024. [ACL Anthology](https://aclanthology.org/2024.emnlp-main.1116/)。

[S20] Vander Eeckt, Van hamme. **Rehearsal-Free Online Continual Learning for Automatic Speech Recognition**. Interspeech 2023. [正式论文](https://doi.org/10.21437/Interspeech.2023-788)；[arXiv](https://arxiv.org/abs/2306.10860)。

[S21] Vander Eeckt, Van hamme. **Unsupervised Online Continual Learning for Automatic Speech Recognition**. Interspeech 2024. [正式论文](https://doi.org/10.21437/Interspeech.2024-136)。

[S22] Lu et al. **Optimal Transport-based Semantic Alignment for LLM-based Audio-Visual Speech Recognition**. arXiv, 2026. [arXiv](https://arxiv.org/abs/2607.09001)。

[S23] Zheng et al. **Unsupervised Active Learning: Optimizing Labeling Cost-Effectiveness for Automatic Speech Recognition**. Interspeech 2023. [正式论文](https://doi.org/10.21437/Interspeech.2023-614)。

[S24] Gekhman et al. **RED-ACE: Robust Error Detection for ASR using Confidence Embeddings**. EMNLP 2022. [ACL Anthology](https://aclanthology.org/2022.emnlp-main.180/)。

[S25] Qwen Team. **Qwen3**. 可用作小型中文模型候选，但模型选择必须固定并纳入资源对照，而不是作为创新本身。[官方仓库](https://github.com/QwenLM/Qwen3)。

## 9. 检索说明与残余不确定性

- 检索时间截至 2026-07-24；覆盖 arXiv/OpenAlex/Crossref、ACL Anthology、ISCA、IEEE/AAAI DOI 页面和作者官方 GitHub。
- 重点搜索了 VSR/AVSR LLM fusion、N-best rescoring、generative error correction、visual grounding、Pinyin correction、active learning、test-time adaptation、online continual learning。
- Exa 在本机未配置，未作为证据来源；所有写入结论都回到了论文、正式会议/期刊页面或作者仓库。
- “完整闭环尚未出现”是基于已检索文献的保守判断。AVSR+LLM 在 2025--2026 更新很快，投稿前必须再次检索最新 arXiv、CVPR/ICCV/ACL/TIP/TPAMI。
- VALLR-Pin、AVUR-LLM 和 OT-AVSR 当前按 arXiv 预印本对待；不得把它们写成已正式接收论文。
