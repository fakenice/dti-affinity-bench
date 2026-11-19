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

## 2026-11-19（Tobit 基线，把 §10 的一句"应该这么做"变成实测）

11. **删失似然必须跑成基线，不许只写在局限里**：`scripts/17_tobit_baseline.py` 与
    `03_models.py` 同特征、同 `standardize()` 守卫、同切分文件、同 `evaluate()`，
    λ 只用 03 自己的 alpha 网格在 valid 段的删失 NLL 上选；结论写进 `README.md` §4b，
    §10 原来的"this repo uses ordinary squared error and *reports* the consequence"
    改为记录测得的量级（最多 0.0226 pKd，`cold_drug` 反而 +0.0220）。
12. **基线不进 leaderboard**：`results/leaderboard.md` 仍只由 `results/metrics.csv`
    生成，Tobit 行单独放 `results/metrics_tobit.csv`——它是单次确定性 MLE 拟合，与
    3 seed 平均的 grid 行不是同一个重复口径，并表排名会造出一个假的可比性
    （`README.md` §7 布局说明；决定 10 的"生成、永不手写"不变）。
13. **超参影响写"量出来的"版本**：`tobit_baseline.json.lambda_sensitivity` 逐个 λ 记录
    测试集 RMSE，因此"λ 选得对不对"是可数的（10/10 格 NLL 选出的 λ 恰为测试最优 λ，
    后悔 ≤0.005 %），而不是用 valid NLL 的变化幅度做比率——该比率在最优 NLL 接近 0 时
    （kiba `cold_drug` 最优 −0.036）会给出虚假的"λ 很敏感"读数，已弃用。
