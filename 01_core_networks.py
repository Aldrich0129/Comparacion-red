# build_core_networks.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import build_networks as bn

# =========================
# 配置区：方案 1 - Top-k 核心网络
# =========================

# 复用原始脚本中的输入文件配置
FULL_EXCEL   = bn.FULL_EXCEL   # "Mercados_company_means_FIXED.xlsx"
RISK_EXCEL   = bn.RISK_EXCEL   # "RISK.xlsx"
RETURN_EXCEL = bn.RETURN_EXCEL # "RETURN.xlsx"

# 为核心网络单独开一个输出目录，避免覆盖原来的结果
OUTPUT_DIR   = "output_core"
PNG_DIR      = os.path.join(OUTPUT_DIR, "png")
HTML_DIR     = os.path.join(OUTPUT_DIR, "html")
METRICS_FILE = os.path.join(OUTPUT_DIR, "network_core_metrics.xlsx")

# Top-k 节点和 Top-p 边的参数
CORE_TOP_K    = 30     # 例如取 eigenvector centrality 前 30 个公司
EDGE_TOP_PCT  = 0.10   # 保留边权 top 10% 的边（在核心子图内部）


def ensure_dirs():
    os.makedirs(PNG_DIR, exist_ok=True)
    os.makedirs(HTML_DIR, exist_ok=True)


# =========================
# 核心子图构建
# =========================

def build_core_subgraph(G: nx.Graph,
                        top_k: int = CORE_TOP_K,
                        edge_top_pct: float = EDGE_TOP_PCT):
    """
    给定一个完整的公司网络 G:
      1) 计算 eigenvector centrality；
      2) 选取 top-k 节点；
      3) 在这些节点诱导的子图中，仅保留权重属于 top edge_top_pct 百分位的边。

    返回: (G_core, ev_dict)
        - G_core: 过滤后的核心子图
        - ev_dict: 原图上所有节点的 eigenvector centrality（可用于打分/导出）
    """
    if G.number_of_nodes() == 0:
        return None, {}

    # 1) eigenvector centrality（在完整图上计算）
    try:
        ev = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    except nx.PowerIterationFailedConvergence:
        # 若不收敛，则退化为度中心性排序
        deg = dict(G.degree(weight="weight"))
        max_deg = max(deg.values()) if deg else 1.0
        ev = {n: deg[n] / max_deg for n in G.nodes()}

    # 2) 选取 top-k 节点
    sorted_nodes = sorted(ev.items(), key=lambda x: x[1], reverse=True)
    k = min(top_k, len(sorted_nodes))
    core_nodes = [n for n, _ in sorted_nodes[:k]]

    # 3) 在 core_nodes 上诱导子图 & 过滤边
    H = G.subgraph(core_nodes).copy()
    if H.number_of_edges() == 0:
        # 没有边时直接返回诱导子图（可能全部为孤立点）
        return H, ev

    # 收集当前子图中的所有边权
    weights = [d.get("weight", 0.0) for _, _, d in H.edges(data=True)]
    weights = [w for w in weights if w is not None]
    if len(weights) == 0:
        return H, ev

    # 边权阈值（top edge_top_pct 百分位）
    q = 1.0 - float(edge_top_pct)
    q = min(max(q, 0.0), 1.0)
    thr = float(np.quantile(weights, q)) if len(weights) > 1 else weights[0]

    # 构建仅保留强边的核心子图
    G_core = nx.Graph()
    G_core.add_nodes_from(H.nodes(data=True))
    for u, v, d in H.edges(data=True):
        w = d.get("weight", 0.0)
        if w >= thr:
            G_core.add_edge(u, v, **d)

    return G_core, ev


# =========================
# 静态图绘制（PNG）
# =========================

def plot_core_graph(G: nx.Graph,
                    meta: dict,
                    ev: dict,
                    out_path: str,
                    top_k: int = CORE_TOP_K,
                    edge_top_pct: float = EDGE_TOP_PCT):
    """
    使用 matplotlib 绘制“核心网络图”（Top-k + Top-p 边）。
    节点大小 ∝ eigenvector centrality（在完整图上算的 ev）。
    """
    if G.number_of_nodes() == 0:
        return

    # 仅取核心子图中的节点对应的 ev
    ev_vals = np.array([ev.get(n, 0.0) for n in G.nodes()])
    if ev_vals.max() > ev_vals.min():
        ev_norm = (ev_vals - ev_vals.min()) / (ev_vals.max() - ev_vals.min())
    else:
        ev_norm = np.ones_like(ev_vals) * 0.5
    sizes = 800 * (0.3 + 0.7 * ev_norm)

    pos = nx.spring_layout(G, weight="weight", seed=42)

    plt.figure(figsize=(8, 8))
    # 边
    nx.draw_networkx_edges(
        G, pos,
        width=0.8,
        edge_color="#999999",
        alpha=0.8
    )
    # 节点
    nx.draw_networkx_nodes(
        G, pos,
        node_size=sizes,
        node_color="#F4B41A",
        edgecolors="black",
        linewidths=0.8
    )
    # 标签：用 empresa
    labels = {n: G.nodes[n].get("empresa", str(n)) for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=7,
        font_color="#111827"
    )

    ds = meta.get("dataset", "")
    sheet = meta.get("sheet", "")

    if ds == "GLOBAL":
        base_title = "Núcleo de la red global"
    elif ds == "RISK":
        base_title = "Núcleo de la red de riesgo"
    elif ds == "RETURN":
        base_title = "Núcleo de la red de rentabilidad"
    else:
        base_title = f"Núcleo de la red ({ds})"

    if str(sheet).isdigit():
        subtitle = f"Año {sheet}"
    else:
        subtitle = str(sheet)

    plt.title(
        f"{base_title} - {subtitle}\n"
        f"Top-{top_k} nodos (centralidad) + Top {int(edge_top_pct*100)}% aristas",
        fontsize=12
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================
# 动态 HTML：按年份（years slider）
# =========================

def build_animated_years_html_core(graphs, dataset_name: str, out_html: str):
    """
    graphs: dict[year_str -> (G_core, meta, ev)]
    生成一个 Plotly HTML，滑块控制年份，只展示“核心网络”（Top-k）。
    """
    years = sorted(int(y) for y in graphs.keys() if str(y).isdigit())
    if not years:
        return
    years_str = [str(y) for y in years]

    frames = []
    first_data = None

    for y in years_str:
        G, meta, ev = graphs[y]
        if G.number_of_nodes() == 0:
            continue

        # 节点尺寸基于完整图的 eigenvector（ev）
        ev_vals = np.array([ev.get(n, 0.0) for n in G.nodes()])
        if ev_vals.max() > ev_vals.min():
            ev_norm = (ev_vals - ev_vals.min()) / (ev_vals.max() - ev_vals.min())
        else:
            ev_norm = np.ones_like(ev_vals) * 0.5
        sizes = 22 * (0.3 + 0.7 * ev_norm)

        pos = nx.spring_layout(G, weight="weight", seed=42)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        labels = [G.nodes[n].get("empresa", str(n)) for n in G.nodes()]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.8, color="#bbbbbb"),
            hoverinfo="none"
        )
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            hoverinfo="text",
            marker=dict(
                size=sizes,
                color="#F4B41A",
                line=dict(width=1, color="black")
            )
        )
        frame = go.Frame(
            data=[edge_trace, node_trace],
            name=str(y)
        )
        frames.append(frame)
        if first_data is None:
            first_data = [edge_trace, node_trace]

    if first_data is None:
        return

    if dataset_name == "GLOBAL":
        title_prefix = "Núcleo de la red global (Top-k) - Años"
    elif dataset_name == "RISK":
        title_prefix = "Núcleo de la red de riesgo (Top-k) - Años"
    elif dataset_name == "RETURN":
        title_prefix = "Núcleo de la red de rentabilidad (Top-k) - Años"
    else:
        title_prefix = f"Núcleo de la red ({dataset_name}) - Años"

    sliders = [{
        "steps": [{
            "args": [[str(y)], {"frame": {"duration": 500, "redraw": True},
                                "mode": "immediate"}],
            "label": str(y),
            "method": "animate"
        } for y in years_str],
        "currentvalue": {"prefix": "Año: "}
    }]

    fig = go.Figure(
        data=first_data,
        frames=frames,
        layout=go.Layout(
            title=title_prefix,
            showlegend=False,
            updatemenus=[{
                "type": "buttons",
                "buttons": [{
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True}]
                }]
            }],
            sliders=sliders
        )
    )
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")


# =========================
# 动态 HTML：按变量（ratio slider）
# =========================

def build_animated_variables_html_core(var_graphs, out_html: str):
    """
    var_graphs: dict[var_name -> (G_core, meta, ev)]
    生成“按变量”的核心网络动态 HTML。
    """
    variables = list(var_graphs.keys())
    if not variables:
        return

    frames = []
    first_data = None

    for var in variables:
        G, meta, ev = var_graphs[var]
        if G.number_of_nodes() == 0:
            continue

        ev_vals = np.array([ev.get(n, 0.0) for n in G.nodes()])
        if ev_vals.max() > ev_vals.min():
            ev_norm = (ev_vals - ev_vals.min()) / (ev_vals.max() - ev_vals.min())
        else:
            ev_norm = np.ones_like(ev_vals) * 0.5
        sizes = 22 * (0.3 + 0.7 * ev_norm)

        pos = nx.spring_layout(G, weight="weight", seed=42)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        labels = [G.nodes[n].get("empresa", str(n)) for n in G.nodes()]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.8, color="#bbbbbb"),
            hoverinfo="none"
        )
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            hoverinfo="text",
            marker=dict(
                size=sizes,
                color="#F4B41A",
                line=dict(width=1, color="black")
            )
        )
        frame = go.Frame(
            data=[edge_trace, node_trace],
            name=var
        )
        frames.append(frame)
        if first_data is None:
            first_data = [edge_trace, node_trace]

    if first_data is None:
        return

    sliders = [{
        "steps": [{
            "args": [[var], {"frame": {"duration": 500, "redraw": True},
                             "mode": "immediate"}],
            "label": var,
            "method": "animate"
        } for var in variables],
        "currentvalue": {"prefix": "Variable: "}
    }]

    fig = go.Figure(
        data=first_data,
        frames=frames,
        layout=go.Layout(
            title="Núcleo de la red global por variables (Top-k)",
            showlegend=False,
            updatemenus=[{
                "type": "buttons",
                "buttons": [{
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True}]
                }]
            }],
            sliders=sliders
        )
    )
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")


# =========================
# 主流程
# =========================

def main():
    ensure_dirs()

    datasets = [
        ("GLOBAL", FULL_EXCEL),
        ("RISK",   RISK_EXCEL),
        ("RETURN", RETURN_EXCEL),
    ]

    all_node_metrics = []
    all_graph_metrics = []

    # 为动态 HTML 保存的核心网络
    core_graphs_by_dataset_and_year = {
        "GLOBAL": {},
        "RISK": {},
        "RETURN": {}
    }
    var_core_graphs = {}   # variable -> (G_core, meta, ev)

    # ---------- 1. 各数据集 / 各年份：公司网络的核心子图 ---------- #
    for ds_name, path in datasets:
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            continue

        xls = pd.ExcelFile(path)
        for sh in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sh)
            G, meta = bn.build_company_graph(df, ds_name, sh, bn.CORR_THRESHOLD)
            if G is None:
                continue

            G_core, ev = build_core_subgraph(G)
            if G_core is None or G_core.number_of_nodes() == 0:
                continue

            # 为核心网络构建 meta_core（type 标记为 core）
            meta_core = dict(meta)
            meta_core["type"] = meta.get("type", "") + "_core"
            meta_core["sheet"] = sh

            # 静态 PNG
            fname = f"{ds_name.lower()}_{bn.sanitize_filename(sh)}_core_top{CORE_TOP_K}_p{int(EDGE_TOP_PCT*100)}.png"
            out_png = os.path.join(PNG_DIR, fname)
            plot_core_graph(G_core, meta_core, ev, out_png)

            # 指标（针对核心子图再计算一遍网络指标）
            node_df = bn.compute_node_metrics(G_core, meta_core)
            graph_dict = bn.compute_graph_metrics(G_core, meta_core)
            all_node_metrics.append(node_df)
            all_graph_metrics.append(graph_dict)

            # 动态按年份 HTML 用
            if str(sh).isdigit():
                core_graphs_by_dataset_and_year[ds_name][str(sh)] = (G_core, meta_core, ev)

    # ---------- 2. 按变量构建核心网络 ---------- #
    panel = bn.load_panel_from_full(FULL_EXCEL)
    if not panel.empty:
        num_cols = [
            c for c in panel.columns
            if c not in bn.BASE_COLS + ["Year", "SourceFile"]
            and pd.api.types.is_numeric_dtype(panel[c])
        ]
        for var in num_cols:
            Gv, meta_v = bn.build_variable_graph(panel, var, dataset_name="GLOBAL")
            if Gv is None:
                continue

            Gv_core, ev_v = build_core_subgraph(Gv)
            if Gv_core is None or Gv_core.number_of_nodes() == 0:
                continue

            meta_core_v = dict(meta_v)
            meta_core_v["type"] = meta_v.get("type", "") + "_core"

            fname = f"global_ratio_{bn.sanitize_filename(var)}_core_top{CORE_TOP_K}_p{int(EDGE_TOP_PCT*100)}.png"
            out_png = os.path.join(PNG_DIR, fname)
            plot_core_graph(Gv_core, meta_core_v, ev_v, out_png)

            node_df_v = bn.compute_node_metrics(Gv_core, meta_core_v)
            graph_dict_v = bn.compute_graph_metrics(Gv_core, meta_core_v)
            all_node_metrics.append(node_df_v)
            all_graph_metrics.append(graph_dict_v)

            var_core_graphs[var] = (Gv_core, meta_core_v, ev_v)

    # ---------- 3. 保存核心网络的指标 ---------- #
    if all_node_metrics and all_graph_metrics:
        nodes_df = pd.concat(all_node_metrics, ignore_index=True)
        graphs_df = pd.DataFrame(all_graph_metrics)
        with pd.ExcelWriter(METRICS_FILE, engine="openpyxl") as writer:
            graphs_df.to_excel(writer, sheet_name="graphs_summary", index=False)
            nodes_df.to_excel(writer, sheet_name="nodes_metrics", index=False)
        print(f"[OK] Métricas de núcleos guardadas en {METRICS_FILE}")

    # ---------- 4. 动态 HTML：按年份（核心网络） ---------- #
    for ds_name in ["GLOBAL", "RISK", "RETURN"]:
        graphs_year = core_graphs_by_dataset_and_year.get(ds_name, {})
        if graphs_year:
            out_html = os.path.join(
                HTML_DIR,
                f"{ds_name.lower()}_core_dynamic_years.html"
            )
            build_animated_years_html_core(graphs_year, ds_name, out_html)
            print(f"[OK] HTML dinámico (núcleo, años) para {ds_name}: {out_html}")

    # ---------- 5. 动态 HTML：按变量（核心网络） ---------- #
    if var_core_graphs:
        out_html_vars = os.path.join(HTML_DIR, "global_core_dynamic_variables.html")
        build_animated_variables_html_core(var_core_graphs, out_html_vars)
        print(f"[OK] HTML dinámico (núcleo, variables): {out_html_vars}")


if __name__ == "__main__":
    main()
