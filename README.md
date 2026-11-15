# HENETs · 跨癌种 H&E 空间蛋白评分

输入一张有物理分辨率的 H&E 全切片，即可得到 MPO、CitH3 的区域预测值、HENETs 空间评分以及肿瘤、间质、其他组织的区室蒙版。自有队列只需提供切片清单。TCGA、RNA 和临床信息均为可选输入。

## 在云端运行第一张切片

准备一台能运行 NVIDIA GPU 容器的 Linux x86_64 主机，将仓库克隆或解压到本地。从[公开模型发布页](https://github.com/shenxiaotianCNS/HENETs-models/releases/tag/assets-v0.1.1)下载对应权重，按页面说明校验并解压到仓库外，得到 `weights/`、`encoder/` 和可选的 `references/`。切片目录和模型目录以只读方式挂载。运行目录使用支持文件锁和原子重命名的持久化磁盘。

```bash
docker build -t henets:cloud-v0.1.1 .
export HENETS_ASSETS=/workspace/henets-assets
export HENETS_DATA=/workspace/data
export HENETS_RUNS=/workspace/runs
mkdir -p "$HENETS_RUNS"

docker run --rm --gpus all \
  -v "$HENETS_ASSETS:/assets:ro" -v "$HENETS_DATA:/data:ro" -v "$HENETS_RUNS:/runs" \
  henets:cloud-v0.1.1 run --slide /data/slide01.svs --sample-id slide01 \
  --weights /assets/weights --encoder /assets/encoder --out /runs/study
```

先打开 `$HENETS_RUNS/study/slide01/export/index.html`。队列入口是 `$HENETS_RUNS/study/report/index.html`。模型清单含 20 份权重的 SHA-256；推理前还会校验编码器源码。推理依赖、支持的切片格式及 CPU 预览方法见 [RUNTIME.md](RUNTIME.md)。

## 分析一个或多个癌种

在数据目录放置 UTF-8 CSV。必填字段为 `sample_id` 和 `he_path`，可附 `cancer_type`。每行对应一张切片，使用唯一的匿名 ID。相对切片路径以 CSV 所在目录为基准。例如：

```csv
sample_id,he_path,cancer_type
liver_01,slides/liver_01.svs,liver
breast_01,slides/breast_01.svs,breast
```

沿用上述容器挂载，将 `run` 后的单张切片参数换成 `--manifest /data/slides.csv`，保留 `--weights`、`--encoder` 和 `--out`。指定 `--sample liver_01` 可只跑一个病例。图像推理不读取癌种或结局作为模型输入。

**多 GPU 或多节点**可按同一份清单分片。每个任务分配一块 GPU，使用相同的容器内挂载路径，写入同一持久化输出目录。在两个任务中分别指定 `--shard 0/2` 和 `--shard 1/2`。只有每张切片的导出都通过校验后，才运行一次汇总：

```bash
docker run --rm -v "$HENETS_DATA:/data:ro" -v "$HENETS_RUNS:/runs" \
  henets:cloud-v0.1.1 report --manifest /data/slides.csv --out /runs/study
```

中断后使用原命令与挂载点重试。每个病例独立加锁，源片 SHA-256、运行配置和代码指纹绑定到该病例。已验证导出的病例可以直接复用。默认在导出文件与完整区室蒙版的校验通过后清理这张切片的推理缓存；调试期间可以加 `--keep-cache`。各队列使用独立的 `--out`。

## 如何读结果

| 结果文件（每张切片） | 内容 |
|---|---|
| `compartments.tif` | 0.325 µm/px 分析坐标下的分块 BigTIFF。0 背景、1 肿瘤、2 间质、3 其他、255 未处理。 |
| `export/regional_scores_qc.csv` | 每个采样窗口的分析坐标、两种蛋白预测值、HENETs 分数、区室与质控。 |
| `export/score_grid.npz` | 带坐标变换的原始分数、质控后分数及空间网格。 |
| `export/compartment_summary.csv` | 分区的有效窗口数、均值、中位数及四分位数。 |
| `report/patient_image_features.csv`（队列根目录） | 每张切片一行的汇总；一位患者有多张切片时需先确定患者级汇总规则。 |

每个 512×512 采样窗以中心 224×224 像素预测蛋白。HENETs 为 `100 × sqrt(clip(MPO,0,1) × clip(CitH3,0,1))`，不同切片共用 0–100 色标，不做逐片归一化。混合区可计入整体分数；低组织量与有指定近黑线伪影的窗口从主汇总排除。纯区室要求组织占比至少 0.5、纯度至少 0.6，病例级区室中位数至少需要 20 个合格中心。

蛋白模型在肝内胆管癌配对标本上训练。用于其他癌种时，分数描述模型预测的 NETs 相关蛋白信号，尚不能解释为胞外 NETs 结构的实测数量。跨癌种比较需要分别复核区室标注、取样、染色与扫描差异。

## 可选的 TCGA 和关联分析

- 使用 [GDC 数据准备](TCGA.md) 按指定的 TCGA 项目选择、下载并校验公开诊断切片、原发肿瘤 STAR 表达谱和临床 XML。
- 使用 [患者级关联分析](ASSOCIATIONS.md) 计算实测 RNA 的 NET 通路与免疫评分，并按癌种检验与 HENETs 的相关和生存关联。临床结局由使用者按预设定义准备。

## 安装验收

`python examples/make_demo.py /path/to/data/demo.svs` 生成一张 1024×1024 的人工 Aperio 格式测试切片。也可在上面的镜像中通过 `--entrypoint python` 运行 `examples/make_demo.py`，把输出目录以可写方式挂载。然后运行单片命令，将 `slide01.svs` 和 ID 换为 `demo.svs` 和 `demo`。应得到 4 个窗口、同尺寸的区室蒙版和病例报告。该测试可检查安装、几何和计算流程。生物学验证需要独立标本与相应的病理或实验参照。

源包只列入 `SOURCE_MANIFEST.json` 中的文件。模型权重由独立的 HENETs-models 仓库分发。保留分发包内的第三方通知；项目自己的代码和模型的公开再分发许可由权利人决定。
