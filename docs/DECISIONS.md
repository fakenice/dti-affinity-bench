# Decision log

本仓库关键口径与协议决定的记录。每条注明**仓内依据**（文件/代码位置）与可核对的日期
（= git 提交日期，`git log` 可逐条对上）；只收当时已留下书面或代码痕迹的决定。

## 2026-10-14（`c3601d6` 建仓）

1. **标题即结论：headline 是"切分协议比模型更能移动数字"，不是模型排行**。
   跨协议的排名不得读作模型质量（`README.md` §Headline 声明）。
2. **同一批特征/代码/seeds 只换切分协议**：协议效应与模型效应因此可分离
   （方差分解 `11_variance_decomposition.py`，91.3% vs 6.8%）。
3. **两套 concordance 口径并列报告**（Harrell 与 all-pairs），不选单一定义
   （`README.md` §Metrics）。
4. **数据取镜像并哈希钉死，且做一致性交叉核验**：DAVIS/KIBA 取
   `zihandun/GraphDTA-ColdStart` 与 `swmeyer1979/dti-contamination` 两个已规范化镜像
   （理由：原始站不可靠获取，镜像消除静默解析差异；代价是继承其 curate——故
   `scripts/00_fetch_data.py` 校验五文件行数=30,056、双源逐行一致 <0.02 pKd）；
   取数通道只有 `api.github.com` blobs 可达（`README.md`/`docs/data_sources.md`）。
5. **null 基线留在表里、不删**："predict the training mean" 打败学习模型是发现
   不是事故；坍缩有 paired bootstrap（Holm p ≤ 0.046）支撑，不是抽样噪声
   （`README.md` 结论 4）。
6. **泄漏是量出来的，不是声称的**：ECFP4 分桶报告 `random` 测试行 34.98% 泄漏，
   并据此更正 `cold_drug` 实非 scaffold split（48.79%）（`README.md` 结论 5）。

## 2026-11-03/04（`f456664`、`72d9f32`、`ccef2252…` `4d237c7`）

7. **点估计不许当结论**：docking/protocol 相关主张全部改为区间与显著性检验
   （`f456664`）；随后 grid 里**每个主张都带区间、显著性按方向拆分**（`72d9f32`）。
   并明确"显著 ≠ 幅度"：过 Holm 的模型间差通常比协议效应低一个量级
   （`README.md` 结论 2）。
8. **外部表（ChEMBL、BindingDB）复现 headline 必须同代码同协议**，且写在独立
   results 树（`results/external_chembl/`、`results/external_bindingdb/`），
   "不允许触碰主 `results/metrics.csv`"（`ccef2252…`、`e330859`；`README.md` §9）。
9. **smoke run 不进已提交的外部表与 grid**（`4d237c7`）。

## 2026-11-18（`20b9e81`）

10. **leaderboard 生成、永不手写**：`scripts/16_leaderboard.py` 在每个
    dataset × protocol 内按 seed 平均 RMSE 排名，重跑逐字节一致
    （`README.md` §Leaderboard；`results/leaderboard.md`）。
