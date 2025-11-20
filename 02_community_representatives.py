# build_community_networks.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import build_networks as bn

# 尝试导入 Louvain 社区检测库
_louvain_partition_fn = None
try:
    import community as community_louvain  # python-louvain 典型用法

    def _louvain_partition_fn(G, resolution, random_state):
        return community_louvain.best_partition(
            G,
            weight="weight",
            resolution=resolution,
            random_state=random_state,
        )

except ImportError:
    try:
        from community import community_louvain  # 有些版本在子模块

        def _louvain_partition_fn(G, resolution, random_state):
            return community_louvain.best_partition(
                G,
                weight="weight",
                resolution=resolution,
                random_state=random_state,
            )

    except ImportError:
        # NetworkX 从 2.8 起内置了 louvain_communities，可作为后备方案
        try:
            from networkx.algorithms.community import louvain_communities

            def _louvain_partition_fn(G, resolution, random_state):
                comms = louvain_communities(
                    G,
                    weight="weight",
                    resolution=resolution,
                    seed=random_state,
                )
                partition = {}
                for cid, nodes in enumerate(comms):
                    for n in nodes:
                        partition[n] = cid
                return partition

        except Exception as e:  # pragma: no cover - 极端环境
            raise ImportError(
                "需要安装 python-louvain 或使用 NetworkX>=2.8 才能运行本脚本。"
            ) from e

# =========================
# 配置区：方案 2 - 社区代表网络
# =========================

FULL_EXCEL   = bn.FULL_EXCEL
RISK_EXCEL   = bn.RISK_EXCEL
RETURN_EXCEL = bn.RETURN_EXCEL

OUTPUT_DIR   = "output_community"
PNG_DIR      = os.path.join(OUTPUT_DIR, "png")
HTML_DIR     = os.path.join(OUTPUT_DIR, "html")
METRICS_FILE = os.path.join(OUTPUT_DIR, "network_community_metrics.xlsx")

COMMUNITY_TOP_N   = 3      # 每个社区展示的代表公司数量（1 = centroid，公司数更多就用 3）
EDGE_TOP_PCT      = 0.10   # 在代表公司之间保留边权 top 10% 的边
LOUVAIN_RESOLUTION = 1.0   # Louvain 分辨率参数，可调节社区数量大小


def ensure_dirs():
    os.makedirs(PNG_DIR, exist_ok=True)
    os.makedirs(HTML_DIR, exist_ok=True)


# =========================
# 社区代表子图构建
# =========================

def detect_louvain_communities(G: nx.Graph,
                               resolution: float = LOUVAIN_RESOLUTION,
                               random_state: int = 42):
    """
    使用 Louvain 算法对图 G 进行社区划分。
    返回: dict[node -> community_id]
    """
    if G.number_of_nodes() == 0:
        return {}
    return _louvain_partition_fn(G, resolution, random_state)


def build_community_representative_graph(
    G: nx.Graph,
    meta: dict,
    top_n: int = COMMUNITY_TOP_N,
    edge_top_pct: float = EDGE_TOP_PCT
):
    """
    从原始图 G 构建“社区代表网络”:

    1) Louvain 社区划分，得到 node -> community_id；
    2) 在每个社区内部，根据 eigenvector centrality 选出 top_n 代表公司；
    3) 仅保留这些代表公司作为节点；
    4) 在代表公司诱导的子图中，保留边权 top edge_top_pct 百分位的边；
    5) 将 community_id 作为节点属性 'community'。

    返回: (G_comm, meta_comm, ev_full, partition)
    """
    if G.number_of_nodes() == 0:
        return None, meta, {}, {}

    # 1) 社区划分
    partition = detect_louvain_communities(G)

    # 2) eigenvector centrality（在完整网络上计算）
    try:
        ev_full = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    except nx.PowerIterationFailedConvergence:
        deg = dict(G.degree(weight="weight"))
        max_deg = max(deg.values()) if deg else 1.0
        ev_full = {n: deg[n] / max_deg for n in G.nodes()}

    # 将节点按社区分组
    comm_to_nodes = {}
    for n, cid in partition.items():
        comm_to_nodes.setdefault(cid, []).append(n)

    # 3) 为每个社区选出 top_n 代表公司
    rep_nodes = []
    for cid, nodes in comm_to_nodes.items():
        # 按 eigenvector centrality 排序
        nodes_sorted = sorted(
            nodes,
            key=lambda x: ev_full.get(x, 0.0),
            reverse=True
        )
        rep_nodes.extend(nodes_sorted[:min(top_n, len(nodes_sorted))])

    if not rep_nodes:
        return None, meta, ev_full, partition

    # 4) 在代表公司诱导子图中保留强边
    H = G.subgraph(rep_nodes).copy()
    if H.number_of_edges() == 0:
        # 没有边时也保留节点，便于可视化社区结构
        nx.set_node_attributes(H, {n: partition.get(n, -1) for n in H.nodes()}, "community")
        return H, meta, ev_full, partition

    weights = [d.get("weight", 0.0) for _, _, d in H.edges(data=True)]
    weights = [w for w in weights if w is not None]
    if len(weights) == 0:
        nx.set_node_attributes(H, {n: partition.get(n, -1) for n in H.nodes()}, "community")
        return H, meta, ev_full, partition

    q = 1.0 - float(edge_top_pct)
    q = min(max(q, 0.0), 1.0)
    thr = float(np.quantile(weights, q)) if len(weights) > 1 else weights[0]

    G_comm = nx.Graph()
    G_comm.add_nodes_from(H.nodes(data=True))
    for u, v, d in H.edges(data=True):
        if d.get("weight", 0.0) >= thr:
            G_comm.add_edge(u, v, **d)

    # 如果过滤完后完全无边，至少保留所有代表点
    if G_comm.number_of_edges() == 0:
        G_comm = H

    # 5) 写入社区标签
    nx.set_node_attributes(G_comm, {n: partition.get(n, -1) for n in G_comm.nodes()}, "community")

    # meta 拷贝并标记类型
    meta_comm = dict(meta)
    meta_comm["type"] = meta.get("type", "") + "_community"

    return G_comm, meta_comm, ev_full, partition


# =========================
# 静态图绘制（PNG）：按社区着色
# =========================

def plot_community_graph(G: nx.Graph,
                         meta: dict,
                         ev_full: dict,
                         out_path: str,
                         top_n: int = COMMUNITY_TOP_N,
                         edge_top_pct: float = EDGE_TOP_PCT):
    """
    绘制“社区代表网络”：
      - 节点颜色：社区 ID
      - 节点大小：在完整图上的 eigenvector centrality
    """
    if G.number_of_nodes() == 0:
        return

    communities = [G.nodes[n].get("community", -1) for n in G.nodes()]
    unique_comms = sorted(set(communities))

    # 调色板
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]
    color_map = {
        cid: palette[i % len(palette)]
        for i, cid in enumerate(unique_comms)
    }
    node_colors = [color_map.get(c, "#cccccc") for c in communities]

    # 节点大小基于完整图的 eigenvector centrality
    ev_vals = np.array([ev_full.get(n, 0.0) for n in G.nodes()])
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
        edge_color="#aaaaaa",
        alpha=0.8
    )
    # 节点
    nx.draw_networkx_nodes(
        G, pos,
        node_size=sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.8
    )
    # 标签
    labels = {
        n: f"{G.nodes[n].get('empresa', str(n))}"
        for n in G.nodes()
    }
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=7,
        font_color="#111827"
    )

    ds = meta.get("dataset", "")
    sheet = meta.get("sheet", "")

    if ds == "GLOBAL":
        base_title = "Red global - representantes de comunidades"
    elif ds == "RISK":
        base_title = "Red de riesgo - representantes de comunidades"
    elif ds == "RETURN":
        base_title = "Red de rentabilidad - representantes de comunidades"
    else:
        base_title = f"Red ({ds}) - representantes de comunidades"

    if str(sheet).isdigit():
        subtitle = f"Año {sheet}"
    else:
        subtitle = str(sheet)

    plt.title(
        f"{base_title} - {subtitle}\n"
        f"Louvain (Top-{top_n} empresas por comunidad, Top {int(edge_top_pct*100)}% aristas)",
        fontsize=12
    )
    plt.axis("off")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  -> Saved community representatives PNG: {out_path}")


# =========================
# 动态 HTML：按年份（社区代表）
# =========================

def build_animated_years_html_communities(comm_graphs, dataset_name: str, out_html: str):
    """
    comm_graphs: dict[year_str -> (G_comm, meta, ev_full)]
    生成“社区代表网络”的年度动态 HTML。
    """
    years = sorted(int(y) for y in comm_graphs.keys() if str(y).isdigit())
    if not years:
        return
    years_str = [str(y) for y in years]

    frames = []
    first_data = None

    # 调色板（最多 10 个社区，如更多则循环使用）
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]

    for y in years_str:
        G, meta, ev_full = comm_graphs[y]
        if G.number_of_nodes() == 0:
            continue

        # 社区颜色
        communities = [G.nodes[n].get("community", -1) for n in G.nodes()]
        unique_comms = sorted(set(communities))
        color_map = {
            cid: palette[i % len(palette)]
            for i, cid in enumerate(unique_comms)
        }
        node_colors = [color_map.get(c, "#cccccc") for c in communities]

        # 节点大小
        ev_vals = np.array([ev_full.get(n, 0.0) for n in G.nodes()])
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
        labels = [
            f"{G.nodes[n].get('empresa', str(n))} (C{G.nodes[n].get('community', -1)})"
            for n in G.nodes()
        ]

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
                color=node_colors,
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
        title_prefix = "Red global - representantes de comunidades (Louvain) - Años"
    elif dataset_name == "RISK":
        title_prefix = "Red de riesgo - representantes de comunidades (Louvain) - Años"
    elif dataset_name == "RETURN":
        title_prefix = "Red de rentabilidad - representantes de comunidades (Louvain) - Años"
    else:
        title_prefix = f"Red ({dataset_name}) - comunidades (Louvain) - Años"

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
    print(f"  -> Saved animated community HTML (years): {out_html}")


# =========================
# 动态 HTML：按变量（ratio slider，社区代表）
# =========================

def build_animated_variables_html_communities(var_comm_graphs, out_html: str):
    """
    var_comm_graphs: dict[var_name -> (G_comm, meta, ev_full)]
    生成“按变量”的社区代表网络动态 HTML。
    """
    variables = list(var_comm_graphs.keys())
    if not variables:
        return

    frames = []
    first_data = None

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]

    for var in variables:
        G, meta, ev_full = var_comm_graphs[var]
        if G.number_of_nodes() == 0:
            continue

        communities = [G.nodes[n].get("community", -1) for n in G.nodes()]
        unique_comms = sorted(set(communities))
        color_map = {
            cid: palette[i % len(palette)]
            for i, cid in enumerate(unique_comms)
        }
        node_colors = [color_map.get(c, "#cccccc") for c in communities]

        ev_vals = np.array([ev_full.get(n, 0.0) for n in G.nodes()])
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
        labels = [
            f"{G.nodes[n].get('empresa', str(n))} (C{G.nodes[n].get('community', -1)})"
            for n in G.nodes()
        ]

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
                color=node_colors,
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
            title="Red global - representantes de comunidades por variables (Louvain)",
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
    print(f"  -> Saved animated community HTML (variables): {out_html}")


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

    comm_graphs_by_dataset_and_year = {
        "GLOBAL": {},
        "RISK": {},
        "RETURN": {}
    }
    var_comm_graphs = {}

    # ---------- 1. 各数据集 / 各年份：社区代表网络 ---------- #
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

            G_comm, meta_comm, ev_full, partition = build_community_representative_graph(G, meta)
            if G_comm is None or G_comm.number_of_nodes() == 0:
                continue

            meta_comm["sheet"] = sh

            # 静态 PNG
            fname = f"{ds_name.lower()}_{bn.sanitize_filename(sh)}_community_top{COMMUNITY_TOP_N}_p{int(EDGE_TOP_PCT*100)}.png"
            out_png = os.path.join(PNG_DIR, fname)
            plot_community_graph(G_comm, meta_comm, ev_full, out_png)

            # 指标（针对社区代表网络）
            node_df = bn.compute_node_metrics(G_comm, meta_comm)
            graph_dict = bn.compute_graph_metrics(G_comm, meta_comm)
            all_node_metrics.append(node_df)
            all_graph_metrics.append(graph_dict)

            if str(sh).isdigit():
                comm_graphs_by_dataset_and_year[ds_name][str(sh)] = (G_comm, meta_comm, ev_full)

    # ---------- 2. 按变量的社区代表网络（仅 GLOBAL） ---------- #
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

            Gv_comm, meta_comm_v, ev_full_v, partition_v = build_community_representative_graph(Gv, meta_v)
            if Gv_comm is None or Gv_comm.number_of_nodes() == 0:
                continue

            fname = f"global_ratio_{bn.sanitize_filename(var)}_community_top{COMMUNITY_TOP_N}_p{int(EDGE_TOP_PCT*100)}.png"
            out_png = os.path.join(PNG_DIR, fname)
            plot_community_graph(Gv_comm, meta_comm_v, ev_full_v, out_png)

            node_df_v = bn.compute_node_metrics(Gv_comm, meta_comm_v)
            graph_dict_v = bn.compute_graph_metrics(Gv_comm, meta_comm_v)
            all_node_metrics.append(node_df_v)
            all_graph_metrics.append(graph_dict_v)

            var_comm_graphs[var] = (Gv_comm, meta_comm_v, ev_full_v)

    # ---------- 3. 指标输出 ---------- #
    if all_node_metrics and all_graph_metrics:
        nodes_df = pd.concat(all_node_metrics, ignore_index=True)
        graphs_df = pd.DataFrame(all_graph_metrics)
        with pd.ExcelWriter(METRICS_FILE, engine="openpyxl") as writer:
            graphs_df.to_excel(writer, sheet_name="graphs_summary", index=False)
            nodes_df.to_excel(writer, sheet_name="nodes_metrics", index=False)
        print(f"[OK] Métricas de redes de comunidades guardadas en {METRICS_FILE}")

    # ---------- 4. 年度动态 HTML（社区代表） ---------- #
    for ds_name in ["GLOBAL", "RISK", "RETURN"]:
        comm_graphs_year = comm_graphs_by_dataset_and_year.get(ds_name, {})
        if comm_graphs_year:
            out_html = os.path.join(
                HTML_DIR,
                f"{ds_name.lower()}_community_dynamic_years.html"
            )
            build_animated_years_html_communities(comm_graphs_year, ds_name, out_html)
            print(f"[OK] HTML dinámico (comunidades, años) para {ds_name}: {out_html}")

    # ---------- 5. 变量动态 HTML（社区代表） ---------- #
    if var_comm_graphs:
        out_html_vars = os.path.join(HTML_DIR, "global_community_dynamic_variables.html")
        build_animated_variables_html_communities(var_comm_graphs, out_html_vars)
        print(f"[OK] HTML dinámico (comunidades, variables): {out_html_vars}")


if __name__ == "__main__":
    main()
