# RSP-VSR 匹配基线与错误证据驱动更新审计

> 检索截止：2026-07-22。本文只使用论文原文、会议/期刊页面和作者官方仓库。
> “未检索到”只表示在本次检索范围内没有找到直接覆盖项，不等于证明该工作从未存在。

## 1. 结论先行

当前结果同时支持两件事，但它们属于不同监督预算，不能放在同一栏里直接比强弱：

- 无人工反馈的单 adapter 在线适应相对 static 有收益，应放入 **U0：零标签 TTA** 轨道，与
  TENT、ETA/EATA、SAR、CoTTA 比较。
- 每 10 条一次人工纠正的单 adapter 进一步有收益，应放入 **F10：固定 10% 稀疏反馈**
  轨道，与同样能看到这些纠正的 TTA+feedback、HILTTA/ATTA 及 VSR 参数高效个性化移植比较。

动态专家路线已经被 validation 回访证据否定。纯局部 CTC 错误损失也已完成 validation：它比
同质量随机 support 好 2.508 个 CER 点，但比完整序列 replay 差 2.367 点，因此不能进入 test。
下一步不应换一种路由器或继续调纯局部损失，而应：

1. 用 feedback-only 实验把真实纠正收益与无标签 pseudo update 收益拆开；
2. 验证“**完整序列 replay 主目标 + 错误 occupancy 辅助项**”的 hybrid，而不是用局部损失替换完整目标；
3. 在新的、未消费的 development/holdout 上过闸门后，再补参数预算和监督预算匹配的基线；
4. 只有 hybrid 稳定优于整句反馈 CTC 后，才增加“**用后续人工反馈校准无标签准入**”。

这条方向有论文潜力，但不能把 Levenshtein 对齐、CTC forced alignment、focal loss、KL
保持或 sparse-label TTA 中的任何单项写成首创。可能成立的贡献只能是它们在严格
prequential VSR 中形成的、由字符纠错证据闭环控制的参数更新机制。

## 2. 必须分开的比较轨道

| 轨道 | 可用信息 | 应比较的方法 | 主问题 |
| --- | --- | --- | --- |
| U0 | 当前及历史测试视频；无真值、无 source data | Static、现有 no-feedback adapter、TENT、ETA、SAR、CoTTA | 无监督在线适应是否有效 |
| F10-fixed | U0 信息 + 预先固定的每 10 条一次真值纠正 | 整句 feedback adapter、错误证据局部更新、TTA+相同 feedback | 同样人工成本下，怎样利用纠正最有效 |
| F10-active | 总标注数与 F10-fixed 相同，但方法可选择查询位置 | SimATTA、HILTTA、EATTA 等 active TTA 移植 | 主动选样是否优于固定节奏 |
| Offline-oracle | 每位目标说话人的独立 support set | VSR prompt/LoRA/user-dependent padding | 离线个性化上界，不与 prequential 主结果混比 |

所有在线轨道都必须先输出并记录当前样本预测，之后才能读取当前真值或更新参数。active TTA
可以在运行时使用刚查询到的标签，因为这是方法定义的一部分；但候选超参数集合、标注总预算和
选择规则必须在 validation 上冻结，不能看 test 结果后重选。

## 3. 通用 TTA 基线核查

### 3.1 TENT

TENT 在测试时只使用目标数据和模型参数，通过最小化预测熵，在线更新 normalization 的
channel-wise affine 参数和批统计；原设定不需要 source data、标签、teacher 或增强
视图。[论文原文](https://openreview.net/forum?id=uXl3bZLkr3c)；
[官方实现](https://github.com/DequanWang/tent/tree/e9e926a668d85244c66a6d5c006efbd2b82e83e8)
只收集 BatchNorm2d 的 weight/bias，并在每次 forward 内完成一次预测与更新。

- **对当前系统的忠实性**：原版会改变 VSR 主干的 normalization 参数，因此不再满足
  “frozen backbone”。它应命名为 `TENT-Norm`，作为忠实方法基线单独报告可训练参数。
- **预算匹配移植**：把同一 entropy objective 只作用于 75k residual adapter，命名为
  `TENT-Adapter`。这比较的是 TENT 目标函数，不应声称是原版 TENT。
- **序列适配风险**：图像分类的类别熵不能直接照搬到 CTC。必须在 validation 比较
  “全部帧熵”和“排除高 blank 后的非 blank 帧熵”；否则最小熵可能只强化 blank collapse。
- **调参**：只用 validation 选择 learning rate、是否更新统计、熵的 blank 处理和一步更新；
  test 固定。batch size 保持一条 utterance，不得为了 TENT 拼接未来样本。
- **复现**：官方代码为 MIT；不提供任务权重，而是依赖相应 backbone 的预训练权重。

### 3.2 EATA 与 ETA

EATA 在 entropy minimization 前筛掉高熵、不可靠或与历史预测冗余的样本，并用 Fisher
正则限制重要参数偏移。[ICML 论文](https://proceedings.mlr.press/v162/niu22a.html)；
[官方实现](https://github.com/mr-eggplant/EATA/tree/f739b3668cc7617e9b9f1979c1a358497a3472c3)
明确区分 `eta`（无 Fisher）和 `eata`：后者默认从 2,000 条 `original` 样本及伪标签估计
Fisher。SAR 论文也明确指出 EATA 依赖预先收集的 2,000 条 in-distribution 样本
([SAR 论文](https://openreview.net/forum?id=g2YraF75Tj))。

- **资源要求**：ETA 不需 source data、augmentation 或 teacher；完整 EATA-Fisher 需要
  一批 in-distribution 数据，不能标成与当前无源协议等资源。
- **建议主基线**：U0 主表先跑 `ETA-Adapter`；若确有合法、预先声明的 source calibration
  split，再附 `EATA-Fisher-Adapter`，并把 source 样本数写进表头。
- **参数化**：原代码只更新 BatchNorm2d affine；`EATA-Norm` 与 `EATA-Adapter` 必须分名。
- **序列移植**：冗余过滤需要一个 sequence-level posterior。不能直接平均全部 CTC 帧，因为
  blank 会主导余弦相似度；validation 应在非 blank token posterior、CTC collapsed posterior
  和 encoder pooled feature 中预先选一种。
- **调参**：validation 选择 entropy margin、redundancy margin、learning rate，以及是否启用
  Fisher；不要复制 ImageNet 的 `0.4*log(1000)` 到中文字符词表。
- **复现**：官方仓库包含 MIT LICENSE，但 GitHub 元数据未识别 SPDX；无独立权重，依赖
  torchvision ImageNet 模型。

### 3.3 CoTTA

CoTTA 面向连续变化的无标签目标流，使用 EMA teacher、augmentation-averaged prediction
和随机 source-weight restoration；不访问 source 样本，但必须保存 source 参数快照。
[CVPR 论文](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html)；
[官方实现](https://github.com/qinenergy/cotta/tree/c212a204b32be4005092e4323105a24a29ad2952)
的 ImageNet 路径在低 anchor confidence 时生成 32 个增强视图，另保存 EMA teacher 与
anchor model，并对可训练参数做随机恢复。

- **资源要求**：不需 source data 或标签；需要 teacher、anchor、source state 和多视图增强。
- **预算匹配移植**：冻结 VSR 主干，只维护 student adapter、EMA adapter 和零初始化 source
  adapter 快照；明确命名 `CoTTA-Adapter`。它不是论文中“长期适应全网络参数”的忠实参数化。
- **VSR 增强**：同一视频内所有帧必须共享几何变换，不能逐帧随机；水平翻转、强仿射、时间遮挡
  是否保持口型语义必须在 validation 单独消融。
- **prequential 约束**：计分使用 update 前 teacher prediction；当前样本增强与伪标签只能在
  计分后影响后续参数。
- **调参**：validation 选择 teacher EMA、恢复率、增强强度和视图数。视图数优先
  `{2,4,8}`，不能默认 32 后仍宣称计算预算匹配；额外前向必须完整报告。
- **复现**：官方代码为 MIT；无专用 VSR 权重。

### 3.4 SAR

SAR 专门研究 mixed shifts、batch size 1 和在线标签分布不均衡。它先按 entropy 过滤样本，
再用 sharpness-aware minimization 做两次前后向，并在 EMA entropy 触发时恢复 source
模型。[ICLR 论文](https://openreview.net/forum?id=g2YraF75Tj)；
[官方实现](https://github.com/mr-eggplant/SAR/tree/20f6e24b17525f34503510afccedc0629b67b7c4)
支持 BatchNorm、GroupNorm 与 LayerNorm affine 参数。

- **资源要求**：不需 source data、augmentation 或 teacher；需 source 参数快照和 SAM
  两阶段更新，计算量高于单步 TENT/ETA。
- **适配性**：它是四个通用方法中最贴合当前 batch-size-1 流的基线。应同时报告
  `SAR-Norm` 和同参数预算的 `SAR-Adapter`。
- **序列风险**：仍需重新定义 CTC entropy 和 collapse 指标。分类中的 EMA entropy reset
  阈值不能直接搬到大字符词表。
- **调参**：validation 固定 entropy margin、SAM radius、learning rate、reset threshold；
  计算公平表同时列一次样本的 forward/backward 数。
- **复现**：官方代码为 BSD-3-Clause；无专用 VSR 权重。

### 3.5 四个基线的接入结论

| 方法 | Source data | 增强 | Teacher/额外模型 | 原版能否保持主干冻结 | 推荐接入 |
| --- | --- | --- | --- | --- | --- |
| TENT | 否 | 否 | 否 | 否，更新 norm affine/statistics | `TENT-Norm` + `TENT-Adapter` |
| ETA | 否 | 否 | 否 | 否，原版更新 BN affine | `ETA-Adapter` 为 U0 主基线 |
| EATA-Fisher | 是，官方实现默认 2,000 条 ID 样本 | 否 | Fisher/source snapshot | 否 | 资源不等价附表 |
| CoTTA | 否 | 是，官方最多 32 views | EMA teacher + anchor + source state | 否，原文适应全参数 | 预算匹配 `CoTTA-Adapter` |
| SAR | 否 | 否 | source state；SAM 双步 | 否，更新 norm affine | `SAR-Norm` + `SAR-Adapter` |

## 4. 最接近的 VSR 个性化工作

### 4.1 Prompt Tuning for Speaker-Adaptive VSR

[TPAMI 论文](https://arxiv.org/abs/2302.08102)冻结预训练 VSR 权重，只在目标说话人的
1/3/5 分钟 adaptation set 上训练 addition、padding 或 concatenation prompt；论文在
GRID 和 LRW-ID 上使用有标签 support set，且 batch size 分别为 112/55。这证明“冻结 VSR
主干 + 少量目标说话人数据 + 参数高效提示”已存在。

- 它是离线 speaker support-set adaptation，不是逐条先计分再更新的 prequential 协议。
- 可做 `Prompt-FB10` 移植：只用已经到达的反馈训练 prompt，参数量匹配约 75k；这属于
  本项目移植，不是论文原方法。
- 本次在论文、arXiv 元数据和作者名下公开仓库中未定位到官方代码/权重，因此复现风险高；
  不能用第三方实现冒充官方结果。

### 4.2 User-dependent Padding

[ECCV 论文](https://arxiv.org/abs/2208.04498)只学习目标用户的卷积 padding，不修改预训练
权重，也不增加常规模型层；论文同时评估 supervised 与 pseudo-label-based unsupervised
adaptation。这比通用 TTA 更接近 VSR 外观个性化，但仍使用独立 adaptation set，而非动态
stream 中的稀疏纠正。

- 可作为 vision-specific 参数化基线；参数预算应通过插入层数匹配 75k。
- 本次未定位到作者官方代码或权重，重实现需要从论文公式与层配置开始，优先级低于有官方代码
  的 Personalized Lip Reading。

### 4.3 Personalized Lip Reading

[AAAI 论文](https://arxiv.org/abs/2409.00986)对目标说话人同时做视觉与语言适配：视觉端使用
padding prompt + LoRA，语言端使用 input prompt；其定义的 target-speaker adaptation set
包含视频和转写 `(x_i,y_i)`，并实验 1/5/15/30/45 分钟有标签数据。它是当前最接近、且有
公开实现的 sentence-level VSR 个性化工作。

- **公平接入**：当前中文 backbone 没有 Llama 语言端，不能原样移植全文方法。应先移植视觉端
  LoRA/padding prompt，固定相同 feedback10 和 prequential 顺序，命名 `VSR-LoRA-FB10`；
  另用离线 support set 跑论文式 oracle 上界。
- **参数公平**：选择 LoRA rank/插入层，使可训练参数不超过现有 adapter 的 75k，或同时报告
  75k 与论文默认约 0.74M 两档。
- **源码/权重**：[官方仓库](https://github.com/JeongHun0716/Personalized-Lip-Reading/tree/4fa4b788da52d1382d3bc6a090742004153e3244)
  提供训练/评估脚本以及 baseline、vision、vision+language checkpoint 下载链接。
- **许可证**：仓库根许可证为 CC BY-NC 4.0，且代码包含 AV-HuBERT/fairseq 等第三方模块；
  Llama 权重还受其独立许可约束。适合研究比较，不能直接推导商业可用性。

### 4.4 VSR 基线优先级

1. `VSR-LoRA-FB10`：从 Personalized Lip Reading 的视觉 LoRA 思路做参数/反馈预算匹配移植；
2. `Prompt-FB10` 或 user-dependent padding 二选一；
3. 论文式 offline support-set 版本只作 oracle，不进入 prequential 主排名。

## 5. Validation-only 调参与公平性协议

### 5.1 共同冻结项

- base checkpoint、词表、decoder、manifest、样本顺序、随机种子、反馈位置和反馈噪声完全一致；
- 当前样本的 transcript 与 CER 必须来自 update 前状态；
- U0 不可读取任何 test 标签；F10 只能读取预先标记的反馈标签；
- 每种方法保存真实可训练参数、forward/backward 次数、峰值显存、更新时间和 checkpoint 大小；
- canonical norm 版本与 75k adapter 版本分列，禁止只用“参数高效”一词掩盖预算差异。

### 5.2 只在 validation 选择的内容

| 类别 | 候选内容 | 禁止事项 |
| --- | --- | --- |
| 通用 | learning rate、一步更新、weight decay、gradient clip | test CER 反向选参 |
| CTC entropy | blank 是否排除、帧聚合、序列归一化 | 直接照搬 ImageNet `log(C)` 阈值 |
| EATA/ETA | entropy margin、redundancy representation/margin、Fisher 开关 | 将 target test 当 Fisher 数据 |
| CoTTA | EMA、restore rate、views、视频一致增强 | 用 test 选择增强或 32-view 后不报成本 |
| SAR | entropy margin、SAM radius、reset threshold | 把分类 collapse 阈值直接用于 CTC |
| VSR PEFT | LoRA rank/层、prompt 形式、总参数 | 用每个 test speaker 的结果选最佳结构 |
| 稀疏反馈 | 更新权重、反馈噪声策略、查询预算 | 事后移动反馈位置 |

validation 主选择指标是完整流的 prequential CER；同 CER 时优先更低的更新 FLOPs，再看
回访 difference-in-differences 和有害更新率。候选集、选择结果与哈希应在 test 启动前写入
metadata。HILTTA 类方法可在流内用已查询标签选择候选模型，但候选池必须提前由 validation
冻结，这属于在线算法，不是事后 test 调参。

### 5.3 最小可发表比较矩阵

- U0：Static、现有 no-feedback adapter、TENT-Adapter、ETA-Adapter、SAR-Adapter、
  CoTTA-Adapter；canonical norm 版本放附表。
- F10-fixed：整句 feedback adapter、TENT/ETA/SAR/CoTTA-Adapter + 同一 feedback、
  VSR-LoRA-FB10、错误证据局部更新。
- F10-active：相同 10% 总标签预算的 SimATTA 或 HILTTA 移植；query 位置不同，单独成表。
- Offline-oracle：Personalized-VSR visual LoRA/prompt 的固定 support-set 版本。

## 6. “字符错误证据驱动更新”的最近邻

### 6.1 已有技术边界

| 拟用组件 | 最接近的一手证据 | 新颖性判断 |
| --- | --- | --- |
| Levenshtein 定位 substitution/deletion/insertion | CER/WER 的标准编辑对齐 | 工具，不是创新 |
| CTC token/frame forced alignment | [CTC-Segmentation](https://arxiv.org/abs/2007.09127) 从 CTC label probability 求片段；[ICASSP 2024 CTC FA](https://arxiv.org/abs/2406.02560) 讨论 token onset/offset | 已有；标准 CTC 的 peaky posterior 会让细粒度边界不可靠 |
| 难样本加权 CTC | [Focal CTC](https://doi.org/10.1155/2019/9345861) 已把 focal weighting 融入 CTC 以强调低频/难样本 | “focal/error-aware CTC”名称风险高；其目标不是在线编辑位置定位，但加权思想不新 |
| 正确位置保持 teacher posterior | [KL-regularized DNN adaptation](https://doi.org/10.1109/ICASSP.2013.6639201) 已用 source model posterior 约束 ASR 适配；CoTTA 也用 teacher consistency | KL 保持不新；可能的新点是只在 edit-matched CTC occupancy 上保持 |
| 用户纠正驱动 ASR 个性化 | [On-device named-entity personalization](https://arxiv.org/abs/1912.09251) 比较只纠正名称与纠正全部错误；[Gift of Feedback](https://arxiv.org/abs/2310.00141) 用用户纠正做联邦持续学习 | 在线纠正不是空白；但这些不是纯视觉、逐样本本地 adapter prequential VSR |
| 稀疏测试标签 | [SimATTA](https://arxiv.org/abs/2404.05094)、[HILTTA](https://openreview.net/forum?id=P09rAv8UH7)、[EATTA](https://arxiv.org/abs/2503.14564) | sparse-label/active TTA 已形成研究线，不能声称首个 human-in-the-loop TTA |
| 标签校准伪标签/超参数 | HILTTA 用 sparse labels 做在线 model selection；[CPATTA](https://arxiv.org/abs/2509.25692) 用 conformal/pseudo coverage 和 staged human/model-label updates | “反馈校准准入”单独新颖性中低，必须做成 CTC token-level 且证明实效 |

### 6.2 CTC 对齐的关键技术风险

1. **Greedy CTC spike 不是 forced alignment。** `argmax -> collapse -> token span` 只能得到
   模型当前最优路径上的尖峰；ICASSP 2024 证明标准 CTC 的 peaky behavior 会造成不准的
   token onset/offset。诊断原型可以用 greedy mask，但论文不能称其为 forced alignment。
2. **删除没有预测帧。** 真值中缺失的字符必须由 target-conditioned CTC forward-backward 或
   Viterbi path 提供候选帧，不能只看预测 token segment。
3. **插入没有真值 token。** 需要在 hypothesis path 上定位插入 span，并将其推向 blank/邻接真值；
   只加 target CTC 权重不会显式区分是哪一个插入。
4. **CTC 是序列边缘化损失。** 简单给某个 argmax 帧乘权重不再是标准 CTC。论文应称
   `edit-conditioned CTC-alignment loss`，除非真正实现了带权 forward-backward。
5. **当前反馈不能改写当前分数。** 对齐、纠错损失和校准器都只能在预测已经持久化后运行。

## 7. 建议创新方向一：Edit-conditioned local correction

> 状态：纯局部版本已于 2026-07-22 判定 NO-GO。本节保留其预注册定义与失败证据，避免
> 事后重写假设；当前候选是第 7.5 节的 full-replay hybrid。

### 7.1 可实现定义

工作名：**编辑证据条件化的局部纠错 adapter**。它不增加专家，只改变反馈样本的梯度支持域。

1. 用 update 前模型得到 CTC posterior `p0` 和 transcript `y_hat`，先持久化预测和 CER；
2. 反馈到达后，对 `y_hat` 与真值 `y` 做字符 Levenshtein，得到 match/substitution/deletion/insertion；
3. 用 `p0` 与 `y` 做 CTC forward-backward，取得 stop-gradient target occupancy `q(t,u)`；
4. substitution/deletion 对应的 target token occupancy 形成错误学习区域；hypothesis alignment 中的
   insertion span 形成 blank 抑制区域；
5. match token occupancy 只施加 update 前 posterior 的 KL 保持；若无错误，整次反馈更新跳过；
6. 只更新同一 75k residual adapter，主干、decoder 和参数预算不变。

一种清晰的目标分解是：

```text
L = L_error_target
  + lambda_insert * L_insertion_to_blank
  + lambda_keep * KL(p0 || p_adapter) on matched occupancy
  + lambda_feature * feature_anchor
```

其中 `L_error_target` 是在错误 target occupancy 上的 emission loss，而不是把整句真值再次做
均匀 CTC。正确位置的 KL 是保持信号，不是学习新标签。若为了稳定额外保留全句 CTC，必须作为
独立变体报告，因为它违反了“只有错误证据提供纠正梯度”的最强定义。

### 7.2 必做消融

- 整句 feedback CTC（现有强基线）；
- 仅错误区域，不加 KL；
- 仅正确区域 KL + 整句 CTC；
- 错误区域 + 正确 KL（完整方法）；
- 与错误区域相同帧数的随机 mask；
- greedy spike mask 对比 target forward-backward occupancy；
- insertion loss 删除消融；
- 无错误反馈时 skip 对比仍做整句更新。

### 7.3 可证伪指标与 go/no-go

- 主指标：相对整句 feedback adapter 的完整流 prequential CER，paired bootstrap 95% CI；
- 局部指标：反馈后未来 10/25 条的 CER、相同错误字符复发率、原本正确字符变错率；
- 更新质量：每次更新对后续固定窗口真实 CER 的增量、有害更新率、接受更新数；
- 对齐质量：有效 occupancy 比例、插入/删除/替换覆盖率，以及人工抽查的小型 alignment set；
- 资源：每反馈样本额外 forward-backward、延迟和峰值显存。

内部继续线建议：validation 上相对整句 feedback adapter 至少改善 `0.003` 绝对 CER，paired
95% CI 不跨 0，同时原本正确字符变错率下降；否则停止该损失，不进入 test。随机 mask 若与
错误 mask 等效，也应 no-go，因为这说明收益来自稀疏梯度而非错误定位。

### 7.4 新颖性判断

单项新颖性均不足，但“字符编辑操作 -> CTC occupancy -> 错误区域学习/正确区域保持 ->
严格 prequential 后续收益”在本次检索到的 VSR personalization、CTTA 与用户纠正 ASR 中没有
被直接覆盖。它的风险是 **中等**：如果只实现 greedy mask + masked KL，而没有错误区域损失、
insertion/deletion 处理和对齐验证，审稿人会把它视为普通 distillation mask。

### 7.5 Validation 结果与 full-replay hybrid

641 条 validation 回访流上，完整序列 replay CER 为 69.637%，纯局部方法为 72.004%，
same-mass 随机 support 为 74.512%。局部方法显著优于随机 support，但相对 replay 退化
2.367 个 CER 点，paired 95% CI 为 `[+1.757,+3.048]`；其回访 difference-in-differences
也比 replay 差 3.244 点。因此“只在错误区域学习”按预注册规则停止，不跑 test、不追加
seed，也不在这 641 条数据上调权重。

新的独立候选保留 replay 的完整目标，并将错误 token occupancy、insertion-to-blank 与
matched-position KL 作为辅助项：

```text
L_hybrid = L_full_replay
         + lambda_error * L_error_target
         + lambda_insert * L_insertion_to_blank
         + lambda_keep * L_matched_KL
```

feature anchor 已包含在 `L_full_replay`，不得重复计权。其随机 control 只置换 error/insertion
support，保持质量、权重、优化器和反馈位置不变。该候选必须换用新的 hash-locked
development/holdout；development 上相对 replay 至少改善 0.003 且 paired CI 不跨 0，并显著
优于随机 control，冻结后还必须在 holdout 重复通过，才允许进入 test。

## 8. 建议创新方向二：Feedback-calibrated pseudo-label admission

这条路线只在方向一通过 validation 后追加。每个反馈样本不仅训练 adapter，还为当前
reliability features 提供 token-level correctness 标签：CTC confidence/margin、entropy、
跨视图一致性、CTC/AED 一致性、blank rate 与编辑操作。在线校准器只使用此前已到达的反馈，
估计未来无标签 token/sequence 的错误风险，再决定是否允许 pseudo-label 更新。

### 8.1 与已有工作的差异要求

- EATA 已做 entropy/redundancy filtering；固定阈值不是新贡献。
- HILTTA 已用 sparse labels 在线选择超参数/模型；只用反馈选择 learning rate 不新。
- CPATTA 已把 conformal/pseudo coverage 用于 active TTA；简单套 conformal threshold 风险高。
- 可能成立的差异必须落在 **CTC token-level 编辑正确性校准**，且反馈位置固定、无需主动查询，
  校准结果直接控制无标签序列的参数更新而不是只选择模型。

### 8.2 最小实现与验证

- 第一版只做 sequence admission：所有 token 的校准风险低于阈值才允许整句 pseudo CTC；
  不立即实现 partial-token CTC，避免同时引入两个难点；
- 比较固定 reliability gate、EATA-style entropy filter、validation-only static calibrator、
  online feedback calibrator；
- 报告 feedback probe 上的 AUROC、Brier、ECE，接受 pseudo-label 的真实 CER/coverage，以及
  下游 prequential CER；
- 在 10% 反馈字符污染与第二类真实 shift 上复现；
- validation 继续线：校准显著改善 accepted pseudo-label CER，coverage 不塌缩，且相对固定
  gate 的 CER 至少改善 `0.003`、paired CI 不跨 0。否则保留方向一，不增加校准器。

## 9. 实施顺序

1. 完成 no-feedback 三 seed 严格复现，并用 feedback-only 拆分 pseudo 与真实纠正收益；
2. 在新 development 单 seed 比较 replay、hybrid 与 randomized hybrid，通过后冻结到 holdout；
3. 补共同 TTA adapter 接口和 `ETA-Adapter`、`SAR-Adapter`，再评估 `CoTTA-Adapter` 的额外成本；
4. 移植 `VSR-LoRA-FB10`，参数严格匹配 75k；
5. hybrid 与匹配基线过 holdout 后才考虑 test 或第二 seed；
6. 第二类真实 shift、资源表和 paired CI 与方法实现并行补齐；
7. 任一新模块未过 holdout 门槛即停止，不再用已消费 validation 或 test 扫参补救。

## 10. 许可证与复现快照

| 工作 | 官方代码快照 | 许可证 | 权重/复现备注 |
| --- | --- | --- | --- |
| TENT | [`e9e926a`](https://github.com/DequanWang/tent/tree/e9e926a668d85244c66a6d5c006efbd2b82e83e8) | MIT | 示例依赖 torchvision 权重 |
| EATA | [`f739b36`](https://github.com/mr-eggplant/EATA/tree/f739b3668cc7617e9b9f1979c1a358497a3472c3) | MIT LICENSE | 完整 EATA 需 ID Fisher 样本 |
| CoTTA | [`c212a20`](https://github.com/qinenergy/cotta/tree/c212a204b32be4005092e4323105a24a29ad2952) | MIT | 分类/分割代码，无 VSR 权重 |
| SAR | [`20f6e24`](https://github.com/mr-eggplant/SAR/tree/20f6e24b17525f34503510afccedc0629b67b7c4) | BSD-3-Clause | 分类代码，无 VSR 权重 |
| Personalized Lip Reading | [`4fa4b78`](https://github.com/JeongHun0716/Personalized-Lip-Reading/tree/4fa4b788da52d1382d3bc6a090742004153e3244) | CC BY-NC 4.0 | 提供多档 checkpoint；第三方依赖另有许可 |
| SimATTA | [`15e635d`](https://github.com/divelab/ATTA/tree/15e635d9733d3462dc9b92c435aba7813bf8c275) | GPL-3.0 | 图像分类框架，需序列移植 |
| HILTTA | [`0011213`](https://github.com/Yushu-Li/HILTTA/tree/0011213338d05782cf0b4b89a25fc2a0cb0a8a15) | 未发现 LICENSE | 代码公开不等于获得再分发许可 |
| EATTA | [`7c4aed2`](https://github.com/flash1803/EATTA/tree/7c4aed27fea7efd984a5dc652e60b629dd0d27f0) | MIT | CVPR 2025；每批最多一个标注 |
| CTC-Segmentation | [`a30080a`](https://github.com/lumaku/ctc-segmentation/tree/a30080aadd6e606c3a74625ff31b0681fd3496bf) | Apache-2.0 | 可作 alignment 参考，不等于当前 VSR 可直接使用 |
| Label-prior CTC aligner | [`3f98fb9`](https://github.com/huangruizhe/audio/tree/3f98fb96b0c5204d5da878557f46f58fba1336e4) | BSD-2-Clause | 训练 recipe/权重面向音频 forced alignment |

CPATTA 论文声明的 `tingyushi/CPATTA` 仓库在 2026-07-22 查询时返回 404，因此目前只能作为
新颖性最近邻，不能列为已可复现基线。

## 11. 残余不确定性

- 通用 TTA 原论文主要是图像分类/分割；CTC entropy、blank collapse 与单 utterance BN 行为
  必须实测，不能由论文数字外推。
- 标准 CTC posterior 的尖峰使字符边界不可靠；没有人工帧标注时，局部纠错收益可能来自随机
  稀疏化而非正确定位，因此 random-mask control 是硬要求。
- Personalized Lip Reading 的语言端依赖英文 Llama/tokenization，不能公平移植到当前中文
  CTC/AED；主基线应限于视觉 LoRA/prompt。
- HILTTA 无明确代码许可证；CPATTA 暂无可访问代码。若要复用实现，应先解决许可或独立重实现。
- 目前只有 speaker shift 证据。任何“可发表新方法”结论都必须在第二类真实 shift 上重现；
  合成亮度/压缩只能补充，不能替代真实设备、姿态、遮挡或 session shift。
- 本次没有找到直接把“字符编辑局部证据 + 正确区 KL + 稀疏反馈在线校准”用于 prequential
  VSR 的一手工作，但这不是首创证明；投稿前仍需做一次更新到截稿日的系统检索。
