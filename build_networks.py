"""
build_networks.py

构建公司网络（整体 / 风险 / 收益），基于欧氏距离 + 相似度 + 相关性阈值；
并导出：
- 每个年份 + 均值的静态网络图 (PNG)
- 对整体数据每个变量的“公司-年份”网络图
- 所有网络的图指标 (Excel)
- 4 个动态 HTML 网络（按年份 / 按变量）

根据需要修改下面 CONFIG 区块中的文件名和阈值。
"""

import os
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---------------- CONFIG ---------------- #

FULL_EXCEL   = "Mercados_company_means_FIXED.xlsx"
RISK_EXCEL   = "RISK.xlsx"
RETURN_EXCEL = "RETURN.xlsx"

# 变量网络中，相似度的阈值（单变量，公司-年份网络）
SIM_THRESHOLD_VAR = 0.8   # 可调：0.7~0.9 之间试

# 相关性阈值（|corr| < CORR_THRESHOLD 的边会被删除）
CORR_THRESHOLD = 0.3

OUTPUT_DIR   = "output"
PNG_DIR      = os.path.join(OUTPUT_DIR, "png")
HTML_DIR     = os.path.join(OUTPUT_DIR, "html")
METRICS_FILE = os.path.join(OUTPUT_DIR, "network_metrics.xlsx")

BASE_COLS = ["empresa", "Code"]   # 固定前两列

# ---------------------------------------- #


def ensure_dirs():
    os.makedirs(PNG_DIR, exist_ok=True)
    os.makedirs(HTML_DIR, exist_ok=True)


def standardize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    对数值变量做列方向标准化：(x - mean)/std
    缺失值先用列均值填补，避免距离计算出 NaN。
    """
    X = df.copy().astype(float)
    X = X.fillna(X.mean())
    std = X.std(ddof=0)
    std_replaced = std.replace(0, 1.0)  # 避免除以 0
    X_std = (X - X.mean()) / std_replaced
    return X_std.fillna(0.0)


def compute_distance_similarity(X_std: np.ndarray):
    """
    X_std: (n_nodes, n_features)
    返回:
      dist  : 欧氏距离矩阵
      sim   : 相似度矩阵 S_ij = 1 / (1 + d_ij)
    """
    diff = X_std[:, None, :] - X_std[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = 1.0 / (1.0 + dist)
    np.fill_diagonal(sim, 0.0)
    return dist, sim


def compute_corr_mask(X_std: np.ndarray, threshold: float):
    """
    对公司向量（行）做相关性矩阵，返回布尔掩码：
    mask_ij = |corr_ij| >= threshold
    """
    if X_std.shape[1] < 2 or threshold is None:
        return np.ones((X_std.shape[0], X_std.shape[0]), dtype=bool)

    corr = np.corrcoef(X_std, rowvar=True)  # 行=公司
    np.fill_diagonal(corr, 1.0)
    mask = np.abs(corr) >= threshold
    return mask


def build_company_graph(df: pd.DataFrame,
                        dataset_name: str,
                        sheet_name: str,
                        corr_threshold: float):
    """
    针对单个 sheet 构建公司网络：
      - 节点：公司（Code）
      - 边：基于所有数值变量的距离 + 相关性过滤
    返回 (G, meta) 或 (None, None) 如果变量不足。
    """
    if not all(c in df.columns for c in BASE_COLS):
        return None, None

    # 挑出数值变量列
    num_cols = [
        c for c in df.columns
        if c not in BASE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(num_cols) == 0 or df.shape[0] < 2:
        return None, None

    X_std = standardize_numeric(df[num_cols])
    X_arr = X_std.values
    dist, sim = compute_distance_similarity(X_arr)
    corr_mask = compute_corr_mask(X_arr, corr_threshold)

    # 同时满足“相关性阈值”和“相似度>0”的边
    adj = np.where(corr_mask, sim, 0.0)
    np.fill_diagonal(adj, 0.0)

    # 若全部为 0，跳过
    if np.all(adj == 0):
        return None, None

    G = nx.from_numpy_array(adj)
    codes = df["Code"].astype(str).tolist()
    empresas = df["empresa"].astype(str).tolist()

    mapping = {i: codes[i] for i in range(len(codes))}
    G = nx.relabel_nodes(G, mapping)

    # 属性
    empresa_map = {codes[i]: empresas[i] for i in range(len(codes))}
    nx.set_node_attributes(G, empresa_map, "empresa")
    nx.set_node_attributes(G, dataset_name, "dataset")
    nx.set_node_attributes(G, sheet_name, "sheet")

    meta = {
        "dataset": dataset_name,
        "sheet": sheet_name,
        "variables": num_cols,
        "type": "company_sheet"
    }
    return G, meta


def build_variable_graph(panel_df: pd.DataFrame,
                         var_name: str,
                         dataset_name: str = "GLOBAL"):
    """
    针对单个变量构建“公司网络”：
      - 节点：公司（Code）
      - 特征：该变量在不同年份的时间序列
      - 步骤：
        1) 取 empresa, Code, Year, var_name
        2) pivot: index=Code, columns=Year, values=var_name
        3) 对列（年份）做标准化，并用列均值填补缺失
        4) 用时间序列向量算距离 + 相似度 + 相关性阈值
    """
    if var_name not in panel_df.columns:
        return None, None

    df = panel_df[["empresa", "Code", "Year", var_name]].copy()
    # 至少要有公司代码和年份
    df = df.dropna(subset=["Code", "Year"])
    if df.shape[0] < 2:
        return None, None

    df["Year"] = df["Year"].astype(int)

    # 为保证 empresa 和 Code 的映射，用第一个非空名称
    empresa_map = (
        df.sort_values("Year")
          .groupby("Code")["empresa"]
          .first()
          .to_dict()
    )

    # 宽表：每行 = 公司，列 = 年份
    wide = df.pivot_table(index="Code", columns="Year",
                          values=var_name, aggfunc="mean")

    # 如果公司少于 2 家，无图可画
    if wide.shape[0] < 2:
        return None, None

    # 对每个年份列做标准化（公司维度）
    X = wide.astype(float)
    X = X.fillna(X.mean())              # 缺失用该年均值填补
    std = X.std(ddof=0).replace(0, 1.0) # 防止除 0
    X_std = (X - X.mean()) / std
    X_std = X_std.fillna(0.0)

    X_arr = X_std.values  # (n_companies, n_years)

    # 距离 & 相似度
    dist, sim = compute_distance_similarity(X_arr)

    # 用同一个相关性阈值过滤（在“时间序列空间”上的相关性）
    corr_mask = compute_corr_mask(X_arr, CORR_THRESHOLD)
    adj = np.where(corr_mask, sim, 0.0)
    np.fill_diagonal(adj, 0.0)

    if np.all(adj == 0):
        return None, None

    # 构建图：节点先是 0..n-1，再映射到公司 Code
    G = nx.from_numpy_array(adj)
    codes = list(wide.index.astype(str))
    mapping = {i: codes[i] for i in range(len(codes))}
    G = nx.relabel_nodes(G, mapping)

    # 设置节点属性
    empresa_attr = {code: empresa_map.get(code, code) for code in codes}
    nx.set_node_attributes(G, empresa_attr, "empresa")
    nx.set_node_attributes(G, dataset_name, "dataset")
    nx.set_node_attributes(G, var_name, "variable")

    meta = {
        "dataset": dataset_name,
        "sheet": var_name,          # 用变量名当作“sheet 标识”
        "variables": [var_name],
        "type": "variable_panel"    # 标记：这是按变量构建的公司网络
    }
    return G, meta



def compute_node_metrics(G: nx.Graph, meta: dict):
    """
    返回每个节点的指标 DataFrame。
    """
    # eigenvector centrality
    try:
        ev = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    except nx.PowerIterationFailedConvergence:
        ev = {n: 0.0 for n in G.nodes()}

    degree = dict(G.degree())
    strength = dict(G.degree(weight="weight"))
    closeness = nx.closeness_centrality(G, distance=lambda u, v, d: 1.0 / d["weight"])
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)

    rows = []
    for n in G.nodes():
        rows.append({
            "dataset": meta["dataset"],
            "sheet": meta["sheet"],
            "type": meta["type"],
            "node": n,
            "empresa": G.nodes[n].get("empresa"),
            "Code": G.nodes[n].get("Code", n),
            "Year": G.nodes[n].get("Year"),
            "eigenvector": ev.get(n, 0.0),
            "degree": degree.get(n, 0),
            "strength": strength.get(n, 0.0),
            "closeness": closeness.get(n, 0.0),
            "betweenness": betweenness.get(n, 0.0)
        })
    return pd.DataFrame(rows)


def compute_graph_metrics(G: nx.Graph, meta: dict):
    """
    返回单个网络整体指标（dict）。
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)
    avg_degree = sum(dict(G.degree()).values()) / n if n > 0 else 0.0
    avg_strength = (sum(dict(G.degree(weight="weight")).values()) / n
                    if n > 0 else 0.0)
    avg_clustering = nx.average_clustering(G, weight="weight")

    # 只在最大连通子图上计算平均最短路径 & 直径
    try:
        if nx.is_connected(G):
            H = G
        else:
            comps = list(nx.connected_components(G))
            H = G.subgraph(max(comps, key=len)).copy()

        avg_path_length = nx.average_shortest_path_length(
            H, weight=lambda u, v, d: 1.0 / d["weight"]
        )
        diameter = nx.diameter(H)
    except Exception:
        avg_path_length = math.nan
        diameter = math.nan

    return {
        "dataset": meta["dataset"],
        "sheet": meta["sheet"],
        "type": meta["type"],
        "num_nodes": n,
        "num_edges": m,
        "density": density,
        "avg_degree": avg_degree,
        "avg_strength": avg_strength,
        "avg_clustering": avg_clustering,
        "avg_path_length": avg_path_length,
        "diameter": diameter
    }


def plot_static_graph(G: nx.Graph, meta: dict, out_path: str):
    """
    使用 matplotlib 画静态 PNG 图，风格尽量接近你之前 igraph 图。
    节点大小 ∝ eigenvector centrality。
    """
    if G.number_of_nodes() == 0:
        return

    # eigenvector centrality（之前已算过，可再算一次）
    try:
        ev = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    except nx.PowerIterationFailedConvergence:
        ev = {n: 0.0 for n in G.nodes()}

    # 归一化到 [0.3, 1.0]
    ev_vals = np.array(list(ev.values()))
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
        width=0.5,
        edge_color="#bbbbbb",
        alpha=0.7
    )
    # 节点
    nx.draw_networkx_nodes(
        G, pos,
        node_size=sizes,
        node_color="#DAA520",
        edgecolors="black",
        linewidths=0.8
    )
    # 标签：用 empresa
    labels = {n: G.nodes[n].get("empresa", str(n)) for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=7,
        font_color="#000080"
    )

    # 标题
    ds = meta["dataset"]
    sheet = meta["sheet"]
    if ds == "GLOBAL":
        title_prefix = "Red Global"
    elif ds == "RISK":
        title_prefix = "Red de Riesgo"
    else:
        title_prefix = "Red de Rentabilidad"

    if sheet.isdigit():
        title = f"{title_prefix} - Año {sheet}"
    else:
        title = f"{title_prefix} - {sheet}"

    plt.title(title, fontsize=14, pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def sanitize_filename(text: str) -> str:
    return "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in str(text)
    )


def load_panel_from_full(full_excel: str) -> pd.DataFrame:
    """
    从整体 Excel 中堆叠所有年份的 sheet（sheet 名为数字），
    用于按变量构建“公司-年份”网络。
    """
    xls = pd.ExcelFile(full_excel)
    frames = []
    for sh in xls.sheet_names:
        if not sh.isdigit():
            continue
        df = pd.read_excel(full_excel, sheet_name=sh)
        if not all(c in df.columns for c in BASE_COLS):
            continue
        if "Year" not in df.columns:
            df["Year"] = int(sh)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    # 只保留数值列 + empresa/Code/Year
    return panel


def build_animated_years_html(graphs, dataset_name: str, out_html: str):
    """
    graphs: dict[year_str -> (G, meta)]
    用 plotly 生成“按年份滑动”的动态 HTML。
    """
    years = sorted(int(y) for y in graphs.keys() if y.isdigit())
    if not years:
        return
    years_str = [str(y) for y in years]

    frames = []
    first_data = None

    for y in years_str:
        G, meta = graphs[y]
        if G.number_of_nodes() == 0:
            continue

        try:
            ev = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
        except nx.PowerIterationFailedConvergence:
            ev = {n: 0.0 for n in G.nodes()}

        ev_vals = np.array(list(ev.values()))
        if ev_vals.max() > ev_vals.min():
            ev_norm = (ev_vals - ev_vals.min()) / (ev_vals.max() - ev_vals.min())
        else:
            ev_norm = np.ones_like(ev_vals) * 0.5
        sizes = 20 * (0.3 + 0.7 * ev_norm)

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
            line=dict(width=0.5, color="#bbbbbb"),
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
                color="#DAA520",
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
        title_prefix = "Red Global - Años"
    elif dataset_name == "RISK":
        title_prefix = "Red de Riesgo - Años"
    else:
        title_prefix = "Red de Rentabilidad - Años"

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
    fig.write_html(out_html, include_plotlyjs="cdn")


def build_animated_variables_html(var_graphs, out_html: str):
    """
    var_graphs: dict[var_name -> (G, meta)]
    动态 HTML：滑动选择变量。
    """
    variables = list(var_graphs.keys())
    if not variables:
        return

    frames = []
    first_data = None

    for var in variables:
        G, meta = var_graphs[var]
        if G.number_of_nodes() == 0:
            continue

        try:
            ev = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
        except nx.PowerIterationFailedConvergence:
            ev = {n: 0.0 for n in G.nodes()}

        ev_vals = np.array(list(ev.values()))
        if ev_vals.max() > ev_vals.min():
            ev_norm = (ev_vals - ev_vals.min()) / (ev_vals.max() - ev_vals.min())
        else:
            ev_norm = np.ones_like(ev_vals) * 0.5
        sizes = 20 * (0.3 + 0.7 * ev_norm)

        pos = nx.spring_layout(G, weight="weight", seed=42)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        labels = [f"{G.nodes[n].get('empresa','')} ({G.nodes[n].get('Year','')})"
                  for n in G.nodes()]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="#bbbbbb"),
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
                color="#DAA520",
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
            title="Red Global - Variables (selecciona ratio)",
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
    fig.write_html(out_html, include_plotlyjs="cdn")


def main():
    ensure_dirs()

    datasets = [
        ("GLOBAL", FULL_EXCEL),
        ("RISK",   RISK_EXCEL),
        ("RETURN", RETURN_EXCEL),
    ]

    all_node_metrics = []
    all_graph_metrics = []

    # 保存用于动态 HTML 的结构
    graphs_by_dataset_and_year = {
        "GLOBAL": {},
        "RISK": {},
        "RETURN": {}
    }
    var_graphs = {}   # variable -> (G, meta)

    # --------- 1. company networks for each sheet --------- #
    for ds_name, path in datasets:
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            continue
        xls = pd.ExcelFile(path)
        for sh in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sh)
            G, meta = build_company_graph(df, ds_name, sh, CORR_THRESHOLD)
            if G is None:
                continue

            # 静态 PNG
            fname = f"{ds_name.lower()}_{sanitize_filename(sh)}.png"
            out_png = os.path.join(PNG_DIR, fname)
            plot_static_graph(G, meta, out_png)

            # 指标
            node_df = compute_node_metrics(G, meta)
            graph_dict = compute_graph_metrics(G, meta)
            all_node_metrics.append(node_df)
            all_graph_metrics.append(graph_dict)

            # 给动态 HTML 用（只记录年份 sheet）
            if sh.isdigit():
                graphs_by_dataset_and_year[ds_name][sh] = (G, meta)

    # --------- 2. variable-level networks (from FULL_EXCEL) --------- #
    panel = load_panel_from_full(FULL_EXCEL)
    if not panel.empty:
        # 所有数值变量，排除 base + Year + SourceFile 等
        num_cols = [
            c for c in panel.columns
            if c not in BASE_COLS + ["Year", "SourceFile"]
            and pd.api.types.is_numeric_dtype(panel[c])
        ]
        for var in num_cols:
            Gv, meta_v = build_variable_graph(panel, var, dataset_name="GLOBAL")
            if Gv is None:
                continue

            # 静态 PNG
            fname = f"global_ratio_{sanitize_filename(var)}.png"
            out_png = os.path.join(PNG_DIR, fname)
            plot_static_graph(Gv, meta_v, out_png)

            # 指标
            node_df_v = compute_node_metrics(Gv, meta_v)
            graph_dict_v = compute_graph_metrics(Gv, meta_v)
            all_node_metrics.append(node_df_v)
            all_graph_metrics.append(graph_dict_v)

            var_graphs[var] = (Gv, meta_v)

    # --------- 3. 保存指标到 Excel --------- #
    if all_node_metrics and all_graph_metrics:
        nodes_df = pd.concat(all_node_metrics, ignore_index=True)
        graphs_df = pd.DataFrame(all_graph_metrics)
        with pd.ExcelWriter(METRICS_FILE, engine="openpyxl") as writer:
            graphs_df.to_excel(writer, sheet_name="graphs_summary", index=False)
            nodes_df.to_excel(writer, sheet_name="nodes_metrics", index=False)
        print(f"[OK] Métricas guardadas en {METRICS_FILE}")

    # --------- 4. 动态 HTML：按年份 --------- #
    for ds_name in ["GLOBAL", "RISK", "RETURN"]:
        graphs_year = graphs_by_dataset_and_year.get(ds_name, {})
        if graphs_year:
            out_html = os.path.join(
                HTML_DIR,
                f"{ds_name.lower()}_dynamic_years.html"
            )
            build_animated_years_html(graphs_year, ds_name, out_html)
            print(f"[OK] HTML dinámico (años) para {ds_name}: {out_html}")

    # --------- 5. 动态 HTML：按变量 --------- #
    if var_graphs:
        out_html_vars = os.path.join(HTML_DIR, "global_dynamic_variables.html")
        build_animated_variables_html(var_graphs, out_html_vars)
        print(f"[OK] HTML dinámico (variables): {out_html_vars}")


if __name__ == "__main__":
    main()
