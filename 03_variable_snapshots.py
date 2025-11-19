"""
build_keyview_networks.py

方案 4：只展示“关键变量 × 关键年份”的小型网络视图。

- 变量示例：ROE, OperatingMargin, DebtToEquity
- 年份示例：2008, 2015, 2024
- 每张图：只保留 eigenvector centrality 排名前 TOP_K 的公司节点，
          并在这些节点诱导子图中保留边权 Top EDGE_TOP_PCT 的边。

输出（复用原 app 的 output 结构）：
- 静态 PNG：output/png/global_keyview_<variable>_<year>.png
- 动态 HTML：output/html/global_keyview_<variable>_dynamic_years.html
- 指标 Excel：output/network_keyview_metrics.xlsx
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

import build_networks as bn  # 复用你现有的构网和可视化函数

# =========================
# 配置区（根据需要修改）
# =========================

# 关键变量（必须与 Excel 中的列名一致）
KEY_VARIABLES = [
    "ROE",
    "OperatingMargin",
    "DebtToEquity",
]

# 关键年份
KEY_YEARS = [2008, 2015, 2024]

# 核心子图参数
TOP_K = 20          # 每个图保留的公司数量（按 eigenvector 排名前 TOP_K）
EDGE_TOP_PCT = 0.10 # 在核心子图中保留边权 top 10% 的边

# 复用原 app 的输出目录
OUTPUT_DIR = bn.OUTPUT_DIR       # 一般为 "output"
PNG_DIR    = bn.PNG_DIR          # "output/png"
HTML_DIR   = bn.HTML_DIR         # "output/html"
METRICS_FILE = os.path.join(OUTPUT_DIR, "network_keyview_metrics.xlsx")


# =========================
# 帮助函数：核心子图构建
# =========================

def build_keyview_subgraph(G: nx.Graph,
                           top_k: int = TOP_K,
                           edge_top_pct: float = EDGE_TOP_PCT) -> nx.Graph | None:
    """
    从完整公司网络 G 中抽取：
      1) eigenvector centrality 前 top_k 的公司节点；
      2) 在这些节点诱导子图中，仅保留边权处于 Top edge_top_pct 百分位的边。
    """
    if G.number_of_nodes() == 0:
        return None

    # 1) eigenvector centrality
    try:
        ev = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    except nx.PowerIterationFailedConvergence:
        # 若不收敛，退化为按度排序
        deg = dict(G.degree(weight="weight"))
        max_deg = max(deg.values()) if deg else 1.0
        ev = {n: (deg[n] / max_deg if max_deg > 0 else 0.0) for n in G.nodes()}

    # Top-k 节点
    sorted_nodes = sorted(ev.items(), key=lambda x: x[1], reverse=True)
    k = min(top_k, len(sorted_nodes))
    core_nodes = [n for n, _ in sorted_nodes[:k]]

    H = G.subgraph(core_nodes).copy()
    if H.number_of_edges() == 0:
        # 没有边，就直接返回包含 top-k 节点的子图
        return H

    # 2) 在核心子图中按边权过滤 Top 百分位
    weights = [d.get("weight", 0.0) for _, _, d in H.edges(data=True)]
    weights = [w for w in weights if w is not None]
    if not weights:
        return H

    edge_top_pct = float(edge_top_pct)
    edge_top_pct = max(0.0, min(edge_top_pct, 1.0))
    if edge_top_pct <= 0.0:
        return H

    # 令 q = 1 - p，保留权重大于等于分位数 q 的边
    if len(weights) > 1:
        thr = float(np.quantile(weights, 1.0 - edge_top_pct))
    else:
        thr = weights[0]

    G_core = nx.Graph()
    G_core.add_nodes_from(H.nodes(data=True))
    for u, v, d in H.edges(data=True):
        if d.get("weight", 0.0) >= thr:
            G_core.add_edge(u, v, **d)

    # 如果全部被过滤掉，则退回到 H
    if G_core.number_of_edges() == 0:
        return H
    return G_core


# =========================
# 主流程
# =========================

def main():
    # 确保原始输出目录存在（output/png, output/html）
    bn.ensure_dirs()

    # 载入“整体面板数据”（empresa, Code, Year + 所有财务比率）
    panel = bn.load_panel_from_full(bn.FULL_EXCEL)
    if panel.empty:
        print("[WARN] Panel vacío: no se ha cargado ningún dato desde FULL_EXCEL.")
        return

    # 过滤：只保留我们关心的变量中实际存在的列
    available_vars = [v for v in KEY_VARIABLES if v in panel.columns]
    missing_vars = [v for v in KEY_VARIABLES if v not in panel.columns]

    if missing_vars:
        print("[WARN] Las siguientes variables no se encuentran en el panel y se omiten:")
        for v in missing_vars:
            print("   -", v)

    if not available_vars:
        print("[ERROR] Ninguna de las variables clave está disponible en el Excel.")
        return

    all_node_metrics = []
    all_graph_metrics = []

    # 为每个变量构建“年份滑动”的 keyview 动态 HTML
    graphs_by_var_and_year: dict[str, dict[str, tuple[nx.Graph, dict]]] = {}

    for var in available_vars:
        graphs_by_var_and_year[var] = {}
        for year in KEY_YEARS:
            # 选取该年份的截面数据
            sub = panel[panel["Year"] == year].copy()
            if sub.empty:
                print(f"[WARN] No hay datos para Year={year} en variable {var}.")
                continue

            # 只保留 empresa, Code, 以及当前变量列
            cols = [c for c in bn.BASE_COLS + [var] if c in sub.columns]
            df_y = sub[cols].dropna(subset=[var])
            if df_y.shape[0] < 2:
                print(f"[WARN] Very few companies for {var} - {year}, se omite.")
                continue

            # 利用原 build_company_graph，但 df 只含 1 个数值变量 → 单变量距离网络
            sheet_label = str(year)
            G_full, meta_full = bn.build_company_graph(
                df_y,
                dataset_name="GLOBAL",
                sheet_name=sheet_label,
                corr_threshold=bn.CORR_THRESHOLD,
            )
            if G_full is None or G_full.number_of_nodes() == 0:
                print(f"[WARN] No se ha podido construir grafo completo para {var} - {year}.")
                continue

            # 抽取：Top-k 节点 + Top p 边
            G_key = build_keyview_subgraph(G_full)
            if G_key is None or G_key.number_of_nodes() == 0:
                print(f"[WARN] Grafo keyview vacío para {var} - {year}.")
                continue

            # 给节点写入 Year 属性，便于指标分析
            nx.set_node_attributes(G_key, {n: int(year) for n in G_key.nodes()}, "Year")
            # 额外写入当前变量名称
            nx.set_node_attributes(G_key, {n: var for n in G_key.nodes()}, "variable")

            # 构造 meta（注意：必须包含 dataset / sheet / type / variables）
            meta_key = {
                "dataset": "GLOBAL",
                "sheet": f"{var}_{year}",
                "type": "keyview",
                "variables": [var],
            }

            # 静态 PNG（复用原来的 plot_static_graph 风格）
            fname = f"global_keyview_{bn.sanitize_filename(var)}_{year}.png"
            out_png = os.path.join(PNG_DIR, fname)
            bn.plot_static_graph(G_key, meta_key, out_png)

            # 指标：节点 + 图整体
            node_df = bn.compute_node_metrics(G_key, meta_key)
            graph_dict = bn.compute_graph_metrics(G_key, meta_key)
            all_node_metrics.append(node_df)
            all_graph_metrics.append(graph_dict)

            # 动态 HTML 的缓存
            graphs_by_var_and_year[var][str(year)] = (G_key, meta_key)

            print(f"[OK] Keyview generado para {var} - {year}: {out_png}")

    # --------- 保存指标到 Excel --------- #
    if all_node_metrics and all_graph_metrics:
        nodes_df = pd.concat(all_node_metrics, ignore_index=True)
        graphs_df = pd.DataFrame(all_graph_metrics)
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(METRICS_FILE, engine="openpyxl") as writer:
            graphs_df.to_excel(writer, sheet_name="graphs_keyview", index=False)
            nodes_df.to_excel(writer, sheet_name="nodes_keyview", index=False)
        print(f"[OK] Métricas de keyview guardadas en {METRICS_FILE}")

    # --------- 为每个变量生成“按年份滑动”的动态 HTML --------- #
    for var, graphs_year in graphs_by_var_and_year.items():
        if not graphs_year:
            continue
        out_html = os.path.join(
            HTML_DIR,
            f"global_keyview_{bn.sanitize_filename(var)}_dynamic_years.html"
        )
        bn.build_animated_years_html(graphs_year, "GLOBAL", out_html)
        print(f"[OK] HTML dinámico (keyview, años) para {var}: {out_html}")


if __name__ == "__main__":
    main()
