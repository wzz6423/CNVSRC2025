# RSP-VSR 强基线与第二真实偏移审计（2026-07-23）

## 1. 审计范围与结论

本文只回答两件事：哪些公开方法能在当前冻结 Mandarin VSR、逐条
predict-then-update、稀疏反馈协议下形成诚实的匹配基线；CMLR 能否作为独立的第二类
真实偏移。本文未读取任何 holdout2 结果，也不把其他数据集上的论文数字当作本项目结果。

结论如下。

1. 当前可以直接实现的最小矩阵是：`static`、当前 incumbent、真正更新视觉前端
   `BatchNorm2d` 仿射参数的 `BN-TENT-VSR`/`ETA-VSR` 无反馈基线，以及只在相同固定
   纠正位置更新的 `online LoRA-F10` 反馈基线。现有 `tent_adapter` 只更新残差 adapter，
   必须继续称为 adapter-only diagnostic，不能改名为 TENT。
2. 完整 EATA 需要源域样本计算 Fisher；没有经过哈希锁定的源域校准集时，只能实现并
   声称 ETA。CoTTA 可移植，但分类 posterior、图像增强和全参数更新都要改造成序列 VSR
   版本；在验证这种改写之前，应称 `CoTTA-VSR` 或 `CoTTA-inspired VSR`，不能声称逐项
   复现原论文。
3. Personalized Lip Reading 是按目标说话人离线监督适应，不是在线 prequential 方法。
   当前协议中的 LoRA 只能称 `online LoRA sparse-feedback baseline`，不能声称复现该论文。
4. CMLR 官方页面与下载入口在 2026-07-23 仍可访问，但下载经过带提取码的百度网盘；
   使用权仅限高校和科研机构的研究用途。它是 11 位新闻主播、同一类国家新闻节目、
   2009--2018 年的真实视频，适合作为与 Chinese-LiPS 演讲视频不同的真实来源偏移，
   但不是有明确相机/场景标签的多条件数据集。
5. 当前只能证明公开的 checkpoint 训练清单没有列出 CMLR，不能证明底层视频、主播或
   节目片段绝对不重叠。CN-CVS 同样含新闻分支，因此必须在下载后做逐样本、主播、日期、
   文本和感知哈希审计，审计通过前不得把 CMLR 称为严格未见域。

## 2. 冻结协议与本地证据边界

- 当前确认实验使用的基座为
  `model_avg_cncvs_2_3_cnvsrc.pth`，SHA-256
  `577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c`。
- CNVSRC2025 官方 model zoo 只把该权重的训练数据列为
  `CN-CVS + CN-CVS2/3 + CNVSRC`；仓库没有提供对应训练集逐样本 UID、源 URL、主播身份或
  时间戳清单。因而“训练清单未列出 CMLR”是可核查事实，“与 CMLR 绝对无重叠”不是。
- 当前模型是 3D CNN/ResNet 视觉前端、12 层 Conformer 和 CTC/双向 attention 解码器。
  ResNet 前端含 `BatchNorm2d`，Conformer 主要使用 LayerNorm。当前在线引擎冻结整个基座，
  只对 CTC 前的残差 adapter 求梯度；这与 TENT、EATA 和 CoTTA 的原参数范围不同。
- 所有候选都必须保留相同的逐条顺序：先用样本 `t` 的更新前参数预测并记分，再允许用
  样本 `t` 的无标签信号或预先固定反馈更新。不得用更新后结果回填当前样本。

本地依据：[CNVSRC2025 README](https://github.com/asip-cslt/CNVSRC2025/blob/e5c4454016ba4eef9e586e77dd58e8981bb5c3e1/VSR/README.md)、
[`continual_adapt.py`](../continual_adapt.py)、
[`plasticity/engine.py`](../plasticity/engine.py) 和
[`paper_rsp_vsr_ieee/sections/experiments.tex`](../../paper_rsp_vsr_ieee/sections/experiments.tex)。

## 3. 基线逐项审计

### 3.1 CoTTA

**原方法。** CoTTA 面向无源数据的 continual test-time adaptation。官方论文描述三项核心：
weight-averaged teacher、必要时的 augmentation-averaged teacher prediction，以及每步把少量
参数随机恢复到源权重。官方实现复制 student、EMA teacher 和 source anchor，原 ImageNet
配置 batch size 为 64；当 anchor 置信度低于阈值时做 32 次增强推理，student 更新后 EMA
系数为 0.999，随机恢复概率为 0.001。论文明确主张长期更新全网络参数。

**当前场景所需输入与参数。** 不需要反馈标签或源数据，但需要：源权重快照、student、EMA
teacher、anchor、优化器状态，以及对整段视频时间一致的多视图增强。若保持原参数范围，
应更新当前 VSR 主干所有可训练 weight/bias，而不是只更新 75k adapter；显存、吞吐和
checkpoint 开销必须单独报告。

**最小移植范围。** 需要新增独立基线路径，而不是复用 adapter optimizer：

1. 在 CTC logits 上预注册序列 posterior 与熵的定义，包括 blank 是否进入平均、padding
   mask、帧或 utterance 聚合；
2. 把图像增强改为同一 utterance 内时间一致的 crop/flip/color/blur/noise，避免逐帧随机
   变换制造伪运动；
3. 保存 student/teacher/anchor、随机恢复 RNG 和优化器，以支持严格恢复和哈希验收；
4. 保留无反馈轨，不得消费稀疏纠正；同一 stream 上报告前向次数和峰值显存。

分类 softmax 到 CTC 序列 posterior 的改变不是原代码中的等价替换。因此实现若保留全部
机制可称 `CoTTA-VSR adaptation`；若删除 32-view teacher、anchor 或 stochastic restore，
只能称对应消融或 `CoTTA-inspired`。

**许可。** 官方仓库根目录是 MIT；其中分割扩展另有 non-commercial 说明。本项目只应参考
根目录分类实现并保留版权/许可，不应复制分割扩展后声称同一许可。

一手来源：[CVPR 2022 论文页](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html)、
[官方代码](https://github.com/qinenergy/cotta/tree/c212a204b32be4005092e4323105a24a29ad2952)、
[核心实现](https://github.com/qinenergy/cotta/blob/c212a204b32be4005092e4323105a24a29ad2952/imagenet/cotta.py)、
[MIT 许可](https://github.com/qinenergy/cotta/blob/c212a204b32be4005092e4323105a24a29ad2952/LICENSE)。

### 3.2 EATA 与 ETA

**原方法。** TENT 官方实现在测试时最小化预测熵，只收集 `BatchNorm2d` 的 scale/bias，
并通过关闭 running statistics 使用当前 batch 统计。EATA 在此类熵最小化上先排除高熵的
不可靠样本，再按与历史平均预测的
余弦相似度排除冗余样本；完整 EATA 还用 Fisher 加权的参数保持项缓解遗忘。官方代码只
收集 `BatchNorm2d` 的 scale/bias，并关闭 running statistics。官方复现脚本默认 batch
size 64、Fisher size 2000，并用 source/original 数据计算 Fisher；不带 Fisher 的官方分支
明确命名为 ETA。

**当前场景所需输入与参数。** ETA 不需要标签或源数据；完整 EATA 需要在看目标流结果前
冻结的源域校准样本和其哈希清单。官方 Fisher 代码实际用模型自身 argmax 作为目标，不要求
人工标签，但仍要求源图像。当前 ResNet 前端有可更新的 `BatchNorm2d`，所以参数入口存在；
单 utterance 与 CTC 序列输出则不满足原 ImageNet batch/classification 设定。

**最小移植范围。** 先实现 `BN-TENT-VSR`，再实现 ETA 的可靠/冗余筛选：

1. 仅打开视觉 ResNet 中全部 `BatchNorm2d.weight/bias`，关闭其 running stats；不要使用
   adapter 参数，并记录真实可更新参数数；
2. 对 padding 后有效的 CTC posterior 定义 blank-aware utterance entropy，把筛选单位固定为
   utterance；历史平均 posterior 必须对齐固定字符词表；
3. batch size 1 时，BN 统计实际上聚合该 utterance 的时空位置，必须加数值稳定性测试；
4. 只有提供哈希锁定的源域校准集并计算 Fisher 后，方法名才升级为 `EATA-VSR`。

阈值不能直接照搬 `log(1000)`，因为当前字符类数、CTC blank 占比和聚合单位不同；必须只在
development/calibration 上冻结。若没有源域校准集，论文表格必须写 ETA，不能写 EATA。

**许可。** TENT 与 EATA 官方仓库均为 MIT；保留版权和许可即可研究移植。

一手来源：[TENT ICLR 2021 论文页](https://openreview.net/forum?id=uXl3bZLkr3c)、
[TENT 官方代码](https://github.com/DequanWang/tent/tree/e9e926a668d85244c66a6d5c006efbd2b82e83e8)、
[ICML 2022/PMLR 论文页](https://proceedings.mlr.press/v162/niu22a.html)、
[官方代码](https://github.com/mr-eggplant/EATA/tree/f739b3668cc7617e9b9f1979c1a358497a3472c3)、
[EATA/ETA 实现](https://github.com/mr-eggplant/EATA/blob/f739b3668cc7617e9b9f1979c1a358497a3472c3/eata.py)、
[Fisher 构造](https://github.com/mr-eggplant/EATA/blob/f739b3668cc7617e9b9f1979c1a358497a3472c3/main.py)、
[MIT 许可](https://github.com/mr-eggplant/EATA/blob/f739b3668cc7617e9b9f1979c1a358497a3472c3/LICENSE)。

### 3.3 Personalized Lip Reading 与 online LoRA

**原方法。** AAAI 2025 工作在英文 VoxLRS-SA 上先训练目标说话人的视觉适应，再训练语言
prompt。视觉阶段在 Conformer 的 convolution、Q/K/V 上用 LoRA（rank 8、alpha 16），还
训练 user-dependent padding；论文报告每位说话人 300 updates。语言阶段冻结视觉端并训练
LLaMA3-8B 侧适应，论文报告 70 updates。实验使用每位说话人 1--45 分钟的带标签适应数据，
不是“预测一条、偶尔获得该条纠正”的在线协议。官方代码配置 batch size 1、gradient
accumulation 8，并默认 4 GPU；论文的完整系统还依赖 LRS3 初始化、VoxLRS-SA 和 LLaMA3-8B。

**当前场景可比范围。** 可以把“LoRA 作为参数高效视觉说话人适应”作为设计依据，但匹配
基线必须重新定义为 `online LoRA-F10`：从同一冻结 Mandarin checkpoint 初始化，在当前
Conformer 的 Q/K/V（可选 convolution module 必须预注册）插入 LoRA；每条样本先预测，只在
与 incumbent 完全相同的 75 个固定反馈位置做一次监督 CTC 更新；非查询样本不更新。参数量
应尽量接近 75,265，若 rank 离散化不能精确匹配则报告差值，不修改 hidden size 伪造相等。

该 online LoRA 是本项目构造的匹配基线，不是 Personalized Lip Reading 的复现。若要复现
原论文，还需要 VoxLRS-SA、LRS3 checkpoint、LLaMA3-8B、每说话人分钟级训练集，以及两阶段
训练；它也不能与仅 75 次纠正的 prequential 方法在同一监督预算列直接排名。

**许可。** 官方代码根目录是 CC BY-NC 4.0，限非商业使用且要求署名、标注修改；仓库还内嵌
AV-HuBERT、ESPnet、fairseq 等第三方代码，复用前必须逐项保留其许可。本项目自身的 CNVSRC
基座也只允许非商业比较/benchmark，故当前整体只能用于研究原型。

一手来源：[论文 arXiv 记录](https://arxiv.org/abs/2409.00986)、
[官方代码](https://github.com/JeongHun0716/Personalized-Lip-Reading/tree/4fa4b788da52d1382d3bc6a090742004153e3244)、
[LoRA/UDP 实现](https://github.com/JeongHun0716/Personalized-Lip-Reading/blob/4fa4b788da52d1382d3bc6a090742004153e3244/src/model_adaptation.py)、
[训练配置](https://github.com/JeongHun0716/Personalized-Lip-Reading/blob/4fa4b788da52d1382d3bc6a090742004153e3244/src/conf/adaptation/vision.yaml)、
[CC BY-NC 4.0](https://github.com/JeongHun0716/Personalized-Lip-Reading/blob/4fa4b788da52d1382d3bc6a090742004153e3244/LICENSE.md)。

## 4. 按可执行性分类

### 可直接实现

- `static`：相同 checkpoint、stream、decoder，无更新。
- `BN-TENT-VSR`：全视觉前端 BN affine，而不是 adapter-only；序列熵、blank/padding 规则先锁定。
- `ETA-VSR`：在上一项上增加 utterance 级可靠/冗余筛选；明确没有 Fisher。
- `online LoRA-F10`：只用相同固定查询位置和相同纠正，一次 predict-then-update；不使用
  目标 speaker 标签路由，不做离线重放。
- 当前 incumbent：与以上方法保持 checkpoint、manifest、查询位置、seed、decode 和统计脚本
  一致。无反馈方法与 F10 方法分轨比较，不跨监督预算宣称胜负。

### 需要额外资源或先决审计

- `EATA-VSR`：需要源域 calibration clips、逐样本清单与许可、Fisher artifact/hash，以及
  开跑前冻结的阈值。
- `CoTTA-VSR`：需要约三份模型状态、视频增强实现、全参数优化与额外前向预算；先做小型
  development 资源/数值 smoke，再进入匹配矩阵。
- Personalized Lip Reading 原论文复现：需要 VoxLRS-SA、LRS3 权重、LLaMA3-8B、目标说话人
  分钟级标签和多 GPU，不是当前最小矩阵的必要项。
- CMLR：需要人工从百度网盘下载、确认机构研究用途、校验压缩包、建立逐样本 provenance，
  并完成与 CN-CVS/CNVSRC 的重叠审计。

### 不应声称

- 不把现有 `tent_adapter` 称为 TENT、ETA 或 EATA。
- 不把没有 Fisher 的实现称为 EATA。
- 不把 CTC 序列化改写后的 CoTTA 称为官方代码“原样复现”。
- 不把 75 次在线纠正的 LoRA 称为 Personalized Lip Reading 复现。
- 不把英文 ImageNet-C/VoxLRS-SA 的公开结果抄作 Mandarin VSR 基线结果。
- 不把“checkpoint README 未列出 CMLR”提升为逐样本绝对无重叠证明。
- 不把 CMLR 的 2009--2018 时间跨度解释为已有明确相机、光照或场景标签。
- 不读取 holdout2 来选择算法、LoRA rank、熵定义、阈值或增强强度。

## 5. 推荐的最小 baseline 组合

按投入和信息量排序，推荐冻结以下矩阵：

| 轨道 | 必跑方法 | 作用 | 方法名边界 |
| --- | --- | --- | --- |
| 无反馈 | static | 绝对参照 | static |
| 无反馈 | BN-TENT-VSR | 真正的 norm-only TTA | 说明 CTC 序列熵改写 |
| 无反馈 | ETA-VSR | 强于朴素熵最小化且不需源数据 | 无 Fisher，绝不写 EATA |
| F10 | current combined-periodic | incumbent | 当前已冻结参数 |
| F10 | online LoRA-F10 | 同监督预算的 PEFT 基线 | PLR-inspired，不是 PLR 复现 |

若源域 calibration 数据可审计，再用 EATA-VSR 替换 ETA-VSR；若资源允许，再增加一条完整
机制的 CoTTA-VSR。不要为了凑方法数先实现残缺 CoTTA。所有方法报告 prequential CER、
paired bootstrap、A1/A2 static-corrected forgetting、实际更新次数、可更新参数、总前向/反向
次数、吞吐、峰值显存和 checkpoint 大小。无反馈轨与 F10 轨分别确定最强基线。

## 6. CMLR 作为第二真实 shift 的审计

### 6.1 一手事实与可获取性

CMLR 官方页由浙江大学 VIPA 发布，列出 102,072 句、11 位说话人；论文表格则列出
102,076 句。差异在训练集：官网为 71,448/10,206/20,418，论文为
71,452/10,206/20,418。来源为 2009 年 6 月至 2018 年 6 月的国家新闻节目
“新闻联播”，每句最长 29 个汉字，并提供词边界。官方论文说明
视频取自 China Network Television，11 位说话人是新闻主播；数据经过 ASR 初标，随后删除
英文字母、数字和少见标点，但未报告逐句人工转写或全量人工复核。论文把全数据按 7:1:2
**随机**划分为 train/validation/test，
不是 speaker-disjoint 划分。下载后必须用实际 manifest 和排除日志解释这 4 句差异，不能
先选其中一个数字当作验收总数。

2026-07-23 的现场检查结果：

- 官方页 `https://www.vipazoo.cn/CMLR.html` 可访问；
- 短链 `http://t.cn/A6waiog1` 返回 302 到
  `https://pan.baidu.com/s/16h7L_hagpumz-1JjktLFtw`；
- 百度页可访问并要求提取码，官方页给出的提取码为 `emqx`；
- 网盘目录列出 11 个 `s*.zip` 视频包，显示总大小约 41.4 GB；同时列出
  35,545,546 B 的 `text.zip` 以及 train/validation/test CSV；
- 尚未下载并解压这些文件，因此不能确认归档完整性、内部许可文本、实际 manifest、
  解压后结构或校验值。

官方页写明数据只向 universities and research institutes 开放、仅限 research purpose，
但页面没有提供正式 dataset license 文本。这不是开放商用许可；下载与论文实验只能在
符合机构身份和用途时进行，不能把网页可访问等同于允许商用、再分发、发布衍生数据或
模型权重。下载后还必须检查归档内是否有附加条款。

一手来源：[CMLR 官方页](https://www.vipazoo.cn/CMLR.html)、
[MMAsia 2019 论文](https://arxiv.org/abs/1908.04917)。

### 6.2 为什么它是候选，又为什么暂时不能直接开跑

它相对 Chinese-LiPS 的演讲/幻灯片视频形成真实来源、内容和采集流程偏移，并且有真实的
说话人身份和长期时间跨度。因此可作为第二真实 shift 候选。限制是：它全部来自同一类新闻
节目，官方没有提供相机、演播室、光照或年份条件标签；只能主张“跨数据集新闻广播偏移”，
不能主张已验证两个明确采集条件。

仓库内可核查的 standalone `VSR/data/cncvs/train.csv` 有 175,058 行，路径全部以
`speech/` 开头；但仓库未提供当前基座对应的 `cncvs_2_3_cnvsrc/train.csv`，只有
`valid.csv`。因此这份 standalone 清单不能代表缺失的 full combined 训练清单，也不能用来
排除新闻素材。另一方面，官方 CN-CVS 原始数据结构含 `news/n001...n028`，collector 也明确有
News 分支并处理新闻节目。CNVSRC README 没列出 CMLR，但两类数据都可能涉及公开中文新闻
视频，存在主播、日期、文本或底层视频重合的先验风险。数据下载后必须先完成：

1. 锁定 CMLR 压缩包和解压文件 SHA-256、官方 split 文件及文本；
2. 从 CN-CVS/CN-CVS2/3/CNVSRC 训练清单恢复源 URL、speaker、日期和文本；若官方无法提供，
   明确记录证据缺口；
3. 做 speaker/identity、规范化 transcript、日期/节目 ID、文件哈希和视频感知哈希交集；
4. 不沿用官方随机 split，按 speaker 划出本项目 development 与冻结 holdout；若 11 位主播
   数量不足以兼顾调参与确认，预先写明分配并禁止反复换人；
5. 按冻结 source character vocabulary 重编码，报告 OOV/删除率，生成 manifest/sidecar/hash；
6. 在任何结果分析前冻结 stream 顺序、反馈位置、唯一候选和最强基线。

只有上述审计没有发现重叠，或能把重叠样本/说话人完整剔除并重新锁定清单，才能写
“CMLR 是独立真实 shift”。若源训练逐样本清单始终拿不到，论文只能写“公开训练 inventory
未列出 CMLR，无法排除底层公开广播素材重叠”，并把它作为有证据限制的 external dataset，
不能作为最强无泄漏确认。

## 7. CMLR 不可用时的第二真实 shift

最可审计的替代不是另找一个来源不明的互联网压缩包，而是建立小型、同意授权的新采集
Mandarin VSR stream：

- 至少 6 位未参与任何源/目标训练的说话人；speaker-disjoint development/holdout；
- 每位说话人在至少两种预先定义的真实采集条件录制，例如固定室内正面与不同设备/距离/
  光照的自然条件；不得用同一视频的压缩或亮度变换冒充第二真实 shift；
- 在录制前固定句表来源、重复句比例、设备、帧率、距离和环境元数据，并取得可发表/共享
  统计结果的书面同意；
- 原始文件、转写、speaker/condition split、排除项和处理脚本全部哈希锁定；holdout 身份与
  结果在唯一候选冻结前不可见；
- 先做 source/Chinese-LiPS/CMLR（若有）之间的文本与人脸身份交集，再生成 A--B--C--A 或
  condition-switch stream。

这是唯一能从采集源头证明 speaker 与视频不重叠、同时拥有真实条件标签的方案。成本是伦理/
同意流程、录制和人工转写，不应低估。

次优替代是改用 `model_avg_cncvs_4s_30s.pth`，在 CNVSRC-Multi Dev 上建立独立实验，因为
该 checkpoint 的官方 inventory 只列 CN-CVS；但这更换了冻结基座，而且当前全量 checkpoint
已经见过 CNVSRC。因此它只能作为另一项 source-to-target study，不能冒充当前 checkpoint 的
第二确认 shift。合成压缩、亮度、遮挡只可作 robustness 辅助，不可替代真实 shift。

## 8. 开跑前风险清单

- **算法等价性：** CTC blank 和可变长度使分类 TTA 的熵/筛选不再直接等价，必须先锁定
  定义和单元测试。
- **batch size 1：** BN 统计、EATA 冗余筛选和梯度方差可能不稳定，必须报告空选择和 NaN。
- **预算失配：** CoTTA 最坏 32 次增强前向，PLR 原方法使用分钟级标签；不能只比较 CER 而
  隐去计算或监督差异。
- **参数失配：** CoTTA 全参数、EATA/BN-TENT 只更新 BN、LoRA/adapter 更新低秩或残差参数；
  真实参数和状态存储应报告，不要求通过改变架构强行做成同一数字。
- **许可：** CNVSRC 基座与 PLR/CMLR 均有限制性非商业/研究用途；发布代码、权重或数据前
  逐项复核，本文不构成法律意见。
- **数据泄漏：** CMLR 与 CN-CVS 都含新闻视频；没有 source-level 清单就没有绝对非重叠证据。
- **统计边界：** development 只用于冻结实现和 gate；holdout2 继续未读。任何未过预注册
  gate 的方法不得靠增加 seed、扫阈值或查看 holdout 复活。

## 9. 推荐执行顺序

1. 先实现并 smoke `BN-TENT-VSR`、ETA-VSR、online LoRA-F10；在新 development stream 上
   完成名称、参数范围、predict-then-update 和 artifact 验收。
2. 同步人工获取 CMLR，只做 provenance/许可/重叠/OOV 审计，不看模型结果。
3. 若源 calibration 可审计，再加入 EATA-VSR；资源仍充足才实现完整 CoTTA-VSR。
4. development 上冻结唯一候选和各监督轨最强基线后，才申请 CMLR speaker-disjoint holdout
   或新采集 holdout；CMLR 审计失败则直接走新采集方案。
5. 只有第二真实 shift 和当前冻结 holdout 都按预注册规则通过，才升级为方法有效性投稿；
   否则把论文定位为严格协议、负结果和工程可复现性研究，不继续叠加模块。
