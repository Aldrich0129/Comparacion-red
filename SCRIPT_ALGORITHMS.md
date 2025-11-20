# 网络脚本算法说明

本文档总结了仓库中每个脚本的输入、主要算法步骤、生成的网络/图表以及与论文方案的对应关系，便于对照代码与分析方法。

## build_networks.py（基础全网构建）
- **输入**：`Mercados_company_means_FIXED.xlsx`、`RISK.xlsx`、`RETURN.xlsx` 中的逐年工作表；每个表至少包含 `empresa`、`Code` 以及若干财务指标。
- **特征处理**：数值列按列标准化（均值 0、方差 1），缺失用列均值填补，避免距离计算出现 NaN。
- **距离与相关性**：
  - 计算公司特征向量的欧氏距离，并转换为相似度 \(S=1/(1+d)\)。
  - 计算公司间相关性矩阵并应用阈值 `CORR_THRESHOLD=0.3`，仅保留 |corr| 大于阈值的配对。
- **边集过滤**：仅当“相关性通过阈值且相似度>0”时添加边，权重为相似度；若全部为 0 则跳过该图。
- **节点属性**：附加公司名称 `empresa`、所属数据集 `dataset`、工作表名/变量名 `sheet` 与 `type`。
- **输出**：
  - 静态 PNG（`output/png`）和按年份/变量的动态 HTML。
  - 节点与整体网络指标（中心性、聚类系数、平均最短路、直径等）写入 `output/network_metrics.xlsx`。

## 01_core_networks.py（方案 1：Top-k 核心网络）
- **输入**：使用 `build_networks.py` 的同源公司网络。
- **核心抽取**：
  - 在完整图上计算 eigenvector centrality，按得分取 `CORE_TOP_K=30` 个节点。
  - 在这些节点诱导子图中，取边权分位数 Top `EDGE_TOP_PCT=0.10`，仅保留最强边生成核心网络。
  - 若边被完全滤除则保留节点集合以便可视化。
- **可视化**：
  - Matplotlib 静态图：节点大小与 eigenvector 成比例；标题标注“Top-k 节点 + Top-p 边”。
  - Plotly 动态 HTML：提供“年份滑块”和“变量滑块”两类交互视图。
- **指标**：核心子图的节点/图指标写入 `output_core/network_core_metrics.xlsx`。

## 02_community_representatives.py（方案 2：社区代表网络）
- **输入**：同样基于公司层面网络。
- **社区检测**：优先使用 `python-louvain` 的 Louvain；若缺失则退回 NetworkX 内置实现。分辨率可由 `LOUVAIN_RESOLUTION` 调节。
- **代表节点选择**：
  - 在完整图上计算 eigenvector centrality。
  - 每个社区内按 eigenvector 排序，取前 `COMMUNITY_TOP_N=3` 作为代表。
- **边过滤**：代表节点诱导子图后，取边权 Top `EDGE_TOP_PCT=0.10`；若过滤后无边则保留原代表节点集合。
- **节点属性**：写入 `community` ID 以便着色。
- **可视化**：
  - Matplotlib 静态图以社区着色、eigenvector 控制节点大小。
  - Plotly 动态 HTML 分为“按年份”和“按变量”两套滑块。
- **指标**：输出至 `output_community/network_community_metrics.xlsx`。

### 为什么同色节点之间可能没有连边？
- 代表节点只取社区内中心性最高的少数公司，这些公司之间的边仍需通过“最强边”过滤（Top 10%）。
- 如果代表点之间的相似度低于分位阈值，或原始社区内部边本就稀疏，过滤后会出现同一社区（相同颜色）节点间无连接的情况；代码在这种情况下保留节点但不强制添加边，以真实反映强关系结构。

## 03_industry_networks.py（方案 3：行业聚合 + Drill-down）
- **输入**：年度 Excel 工作表与行业映射表 `industrial.xlsx`（`company_code` ↔ `new_industry`）。
- **行业聚合**：
  - 每个年份的 sheet 先按 `Code` 关联行业，缺失行业归为 `Other`；对行业内数值列求均值并保留公司数量 `n_companies`。
  - 使用与 `build_networks.py` 相同的相似度与相关性阈值（相似度=\(1/(1+d)\)，`CORR_THRESHOLD=0.3`）构建“行业网络”。
- **行业内部网络**：
  - 对每个“年份 × 行业”子集再次调用 `build_company_graph`，生成行业内部的公司网络。
- **可视化**：
  - 静态 PNG：行业网络与对应的内部公司网络都会输出到 `output_industrial/png`，命名包含数据集、行业与年份。
  - 动态 HTML：在 `output_industrial/html` 生成带年份滑块的行业网络动图，并支持点击行业节点查看右侧的内部公司网络（drill-down）。
- **指标与汇总**：
  - 全部行业网络与内部网络的节点/整体指标写入 `output_industrial/industry_network_metrics.xlsx`，与基础脚本的指标格式保持一致，便于统一对比。

## 04_variable_snapshots.py（方案 4：关键变量 × 关键年份）
- **输入**：整体面板数据（`FULL_EXCEL` 聚合），关注变量列表与年份列表。
- **过滤与建图**：
  - 仅保留目标变量与指定年份的公司记录，缺失变量或公司不足则跳过。
  - 调用 `build_company_graph` 构建单变量相似度网络（使用默认相关性阈值）。
  - 在该网络上执行 Top-k 抽取：eigenvector 排序取 `TOP_K=20` 节点，并在诱导子图中保留边权 Top `EDGE_TOP_PCT=0.10`。
- **输出**：
  - 静态 PNG 放在主 `output/png`。
  - 按变量的年份滑动 HTML。
  - 指标写入 `output/network_keyview_metrics.xlsx`。

## 05_matrices_and_timeseries.py（方案 5：矩阵与时间序列）
- **输入**：三套数据集的所有年度公司网络。
- **邻接矩阵热力图**：
  - 对每个年份图按 eigenvector 排序节点。
  - 以权重矩阵绘制热力图，展示强连边的块结构。
- **中心性时间序列**：
  - 汇总所有年份的 eigenvector 值，按公司求平均排名，取 Top 10。
  - 绘制这些公司的年度 eigenvector 折线，展示结构演化。
- **输出**：热力图与时间序列 PNG 存放在 `output_matrix_ts` 目录。

## build_company_list.py（辅助：生成公司列表）
- **输入**：多工作表的公司数据 Excel（默认 `Mercados_company_means_FIXED.xlsx`）。
- **处理**：遍历所有包含 `Code` 与 `empresa` 的 sheet，清理字符串后合并并按公司名去重，排序。
- **输出**：导出去重后的公司名单到 `company_list.xlsx`，包含 `company_code` 与 `company_name`。

## 关键阈值与通用假设
- 相关性阈值：公司网络普遍使用 `CORR_THRESHOLD=0.3` 过滤弱相关边。
- 强边保留：核心网络与社区代表网络在诱导子图中保留边权 Top 10% 以突出结构主干。
- 相似度定义：除非特别说明，均采用 \(1/(1+\text{欧氏距离})\) 作为相似度或边权。
- 布局与配色：静态图多用 spring layout，节点大小与 eigenvector 或公司数相关，颜色用于区分社区或度数。
