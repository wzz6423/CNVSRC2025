# RSP-VSR

本仓库基于 [CNVSRC2025](https://github.com/asip-cslt/CNVSRC2025) 的官方
VSR 基线，研究面向中文连续唇读的可靠性门控结构可塑性。上游代码固定在
`e5c4454016ba4eef9e586e77dd58e8981bb5c3e1`。

当前工作区只检出了 `VSR/` 代码；大型训练清单与噪声文件仍保留在 Git 历史中，
可在服务器执行 `git sparse-checkout disable` 恢复完整工作树。

- 旧项目完整保存在本地 `backup/`，该目录不会提交。
- 新方法与服务器实验协议见 [VSR/docs/RSP_VSR.md](VSR/docs/RSP_VSR.md)。
- 最近邻工作与投稿门槛见 [VSR/docs/NOVELTY_REVIEW.md](VSR/docs/NOVELTY_REVIEW.md)。
- 官方基线说明见 [VSR/README.md](VSR/README.md)。
