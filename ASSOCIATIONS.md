# 患者级 RNA、免疫与生存关联

在发布包根目录安装 CPU 分析依赖。这个模块可以在完成 H&E 导出后单独运行。

也可在构建 Docker 镜像时加 `--build-arg WITH_ANALYSIS=1`，执行分析时以 `--entrypoint python` 调用 `-m henets.associate`。模型资产包中的 `references/` 提供下文使用的两个参考文件。

```bash
python -m pip install -r requirements-analysis.txt
python -m henets.associate --help
```

## 从 STAR TPM 计算 RNA 评分

通用 GDC 工具生成的 `manifest.csv` 可以直接使用。自有数据需要 CSV 列 `sample_id,patient_id,cancer_type,rna_path`。每行对应一例独立患者及其一份 bulk RNA；相对路径以清单所在目录为基准。GDC 清单只有 `project_id` 时会将它原样用作癌种代码，临床表也应使用相同的代码，例如 `TCGA-BRCA`。同一个 GDC 项目内存在不同组织学类型时，先按研究定义筛选或赋予明确的 `cancer_type`。如提供 `rna_md5` 或 `rna_sha256` 列，程序逐文件验证。

```bash
python -m henets.associate rna --manifest cohort/manifest.csv \
  --gene-sets henets-assets/references/gene_sets_snapshot.json --markers henets-assets/references/mcpcounter_genes.tsv \
  --out output/rna
```

两个外供参考文件必须与版本指纹一致。

| 文件 | 内容 | SHA-256 |
| --- | --- | --- |
| `gene_sets_snapshot.json` | KEGG hsa04613 在项目固定 Ensembl 映射中的 191 个基因 | `9eae4b586d28a4b5e257ee82c8acd537314cb9b861611e9f5c211ee056e3352e` |
| `mcpcounter_genes.tsv` | MCP-counter 官方十类细胞标记，版本 b6eac73e91c246fcff0bb1a5c68a816cd588fc48 | `408f6c5d02c8f9bd2f1599c598367881853520934f8ce398a4886a8b296922bb` |

输入应使用基因层面的 GDC STAR 表，含 `gene_id,gene_name,tpm_unstranded`。程序保留有效 Ensembl 项、去除版本号并保留 PAR_Y 后缀。各癌种内须有相同的基因集合和顺序，包含固定 NET 基因映射的全部靶基因。每种癌症每批最多 8 人，GSEApy 1.3.2 以样本内基因排名计算 `rna_nets_score`，保存原始 ssGSEA ES，权重 0.25。实际比较需要一致的测序、注释与处理方式。MCP-counter 计算每类可用标记表达的 log2(TPM+1) 均值，要求每类覆盖率至少 50%。这些值表示相对细胞丰度，不能解读为细胞比例。参考方法见 [KEGG hsa04613](https://www.kegg.jp/entry/hsa04613) 与 [MCP-counter 官方标记和实现](https://github.com/ebecht/MCPcounter/tree/b6eac73e91c246fcff0bb1a5c68a816cd588fc48)。

结果目录中 `rna_scores.csv` 包含患者键、NET 通路分数和十个 `mcp_*` 分数。`rna_input_audit.csv` 记录各例输入 SHA-256 与基因数，`immune_marker_coverage.csv` 记录各癌种的标记覆盖率。RNA 的空间分辨率为送测组织块，患者匹配只支持患者级分析。

## 准备病例表并运行关联分析

`--features` 接受 H&E 导出生成的 `report/patient_image_features.csv`。`--patients` 至少包含 `sample_id,patient_id,cancer_type`，且恰好覆盖每个 H&E 样本。所有 ID 均使用去标识化代码，只允许 ASCII 字母、数字、下划线与连字符。每名患者只能纳入一张切片，需在看结局前依研究方案选择切片。一人多行、病例身份不吻合或临床表与 H&E 范围不一致时，命令会报错。分析不输出原始文件路径或自由文本列。

```csv
sample_id,patient_id,cancer_type,OS_days,OS_event,PFI_days,PFI_event
slide_001,case_001,TCGA-BRCA,1480,0,670,1
slide_002,case_002,TCGA-BRCA,520,1,,
```

OS/PFI 使用研究者提供的标准化同源时间和状态对，时间单位为天，0 表示删失，1 表示事件。需要自行记录时间起点、随访来源、删失与事件定义。TCGA 分析可按患者从 TCGA-CDR 提取 `OS.time/OS` 与 `PFI.time/PFI`；临床补充 XML 本身不能提供通用、已核定的 PFI。每个端点可整对省略；单例缺失、半对数据或非正时间保留为未纳入及原因。非法数值、无穷或 0/1 之外的状态要求纠正。参考 [TCGA 临床端点定义与按癌种使用建议](https://doi.org/10.1016/j.cell.2018.02.052)。

```bash
python -m henets.associate run \
  --features output/report/patient_image_features.csv --patients cohort/patients.csv \
  --rna-scores output/rna/rna_scores.csv --out output/associations
```

已有 RNA 或免疫分数也可直接作为 `patients.csv` 的数值列。默认选择 `rna_nets_score` 和所有 `mcp_*` 列，也可以指定 `--scores rna_nets_score my_immune_score`。同名分数只能从一个文件提供。独立 RNA 表可只覆盖部分患者，未匹配的患者仍可用于有合格结局的生存分析。没有分数列时仅进行生存分析。结果目录必须使用新路径。

- 每个癌种单独输出五项影像特征（全组织、肿瘤、间质、其他组织中位数，及肿瘤减间质）与各分子分数的双侧 Spearman 相关。每对只取完整记录，至少 5 人且双方均有变异时报告 rho、近似 P 值及 2,000 次患者重采样的 95% 百分位 CI。有效重采样少于 90% 时 CI 留空。每个癌种内对全部计划的特征与分数组合做 BH 校正，无法估计的组合保留在家族大小中。
- OS/PFI 各自使用全组织中位数拟合单变量 Cox 模型，Efron 处理并列事件。HR 的单位为该癌种、该端点完整病例中的 1 个样本标准差。至少需要 10 名有效患者、5 个事件及有变异的影像值。输出模型状态、HR、95% CI、未校正 P 值及描述性的 Schoenfeld 残差与 log 时间相关。小样本和稀疏事件可能导致不稳定估计。
- `patients.csv` 保存进入分析的去标识化数据与端点排除原因，`missingness.csv` 记录各癌种各变量的有效数。`correlations.csv` 和 `survival_cox.csv` 包含每个检验的有效人数与状态，`analysis_metadata.json` 记录输入、输出指纹及参数。患者表与临床分析结果应保存在获准存放这些数据的位置。

## 和固定 BRCA 分析的关系

本模块面向新队列的探索性分析。冻结的 TCGA-BRCA 工作流预先限定了 1,055 例的组织学、文件选择与数据完整性，设定了单一主要 RNA 相关、43 个次要相关的 BH 家族、排除已查看首批 30 例的敏感性分析，以及年龄、分期、组织学、ER/PR/HER2 的分层调整方案。通用命令只执行上文列出的癌种内分析。更换癌种时需另外验证组织区室与蛋白代理分数，bulk RNA 相关或生存相关都不能单独证明空间 NET 测量准确。
