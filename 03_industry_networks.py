"""
build_industry_drilldown_networks.py

功能：
- 在保持原始构网算法（build_networks.py）的前提下，
  把公司按行业 new_industry 聚合，生成“行业网络”（按年份动态滑动）。
- 同时，对每个「年份 × 行业」构建该行业内部的“公司网络”。
- 输出一个可 drill-down 的 HTML：
  左侧：行业之间的动态网络（按年份 slider / 播放）
  右侧：点击某个行业节点后，显示该年份下该行业内部的公司网络。

依赖：
    pip install pandas numpy networkx plotly openpyxl

要求存在的文件：
    - build_networks.py
    - Mercados_company_means_FIXED.xlsx   （GLOBAL）
    - RISK.xlsx
    - RETURN.xlsx
    - industrial.xlsx   （包含 company_code, company_name, new_industry）
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly

# ----------------- 导入你原来的 build_networks ----------------- #

import importlib.util
import sys

BASE_DIR = Path(".").resolve()
BUILD_NETWORKS_PATH = BASE_DIR / "build_networks.py"

spec = importlib.util.spec_from_file_location("build_networks", BUILD_NETWORKS_PATH)
bn = importlib.util.module_from_spec(spec)
sys.modules["build_networks"] = bn
spec.loader.exec_module(bn)

# 复用原脚本中的配置与函数
CORR_THRESHOLD = bn.CORR_THRESHOLD          # 相关性阈值
BUILD_COMPANY_GRAPH = bn.build_company_graph

# ----------------- 本脚本配置 ----------------- #

INDUSTRY_FILE = BASE_DIR / "industrial.xlsx"
GLOBAL_FILE   = BASE_DIR / "Mercados_company_means_FIXED.xlsx"
RISK_FILE     = BASE_DIR / "RISK.xlsx"
RETURN_FILE   = BASE_DIR / "RETURN.xlsx"

OUTPUT_DIR = BASE_DIR                       # 直接输出到当前目录
# 也可以改成 OUTPUT_DIR = BASE_DIR / "output_html"


# ============================================================
# 1. 工具函数：公司 -> 行业聚合
# ============================================================

def aggregate_to_industry(
    df_sheet: pd.DataFrame,
    industry_map: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    把某一年份的公司数据 df_sheet 按 industrial.xlsx 中的 new_industry 聚合。
    返回：
      - grouped: 行业层面 DataFrame（每行一个行业，用于构建行业网络）
      - merged: 带有 new_industry 的原始公司层面数据（用于后续行业内部网络）
    """
    merged = df_sheet.merge(
        industry_map[["company_code", "new_industry"]],
        left_on="Code",
        right_on="company_code",
        how="left"
    )

    # 未匹配到行业的公司归为 "Other"
    merged["new_industry"] = merged["new_industry"].fillna("Other")

    # 数值列（去掉 Year 避免扰动）
    num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    if "Year" in num_cols:
        num_cols.remove("Year")

    # 按行业取均值
    grouped = (
        merged
        .groupby("new_industry", as_index=False)[num_cols]
        .mean()
        .rename(columns={"new_industry": "empresa"})
    )

    # Code 用行业名作伪代码，保证每个行业一个节点
    grouped["Code"] = grouped["empresa"]

    # Year 列（如果有）
    if "Year" in merged.columns:
        year_vals = merged["Year"].dropna().unique()
        if len(year_vals) >= 1:
            grouped["Year"] = int(pd.Series(year_vals).mode().iloc[0])

    # 行业内公司数
    counts = merged.groupby("new_industry")["Code"].nunique().rename("n_companies")
    grouped = grouped.merge(
        counts.reset_index().rename(columns={"new_industry": "empresa"}),
        on="empresa",
        how="left"
    )

    return grouped, merged


# ============================================================
# 2. 将 NetworkX 图转成 Plotly traces（通用）
# ============================================================

def graph_to_traces(G: nx.Graph, label_attr: str = "empresa"):
    """
    将 NetworkX 图转为两条 Plotly trace：
      - edge_trace：线段
      - node_trace：节点（带标签）
    label_attr: 节点属性中用于展示的文本字段（公司名或行业名）
    """
    if G is None or G.number_of_nodes() == 0:
        return None, None

    # 基于 spring_layout 的布局（固定 seed，保证不同年份视觉连贯）
    k = 1 / np.sqrt(max(1, G.number_of_nodes()))
    pos = nx.spring_layout(G, seed=42, k=k)

    # 边
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#999"),
        hoverinfo="none",
        showlegend=False,
    )

    # 节点：大小 ∝ eigenvector 中心性
    try:
        ev = nx.eigenvector_centrality(G, max_iter=500, weight="weight")
    except Exception:
        ev = {n: 0.0 for n in G.nodes()}
    ev_vals = np.array(list(ev.values()))
    if ev_vals.max() > ev_vals.min():
        ev_norm = (ev_vals - ev_vals.min()) / (ev_vals.max() - ev_vals.min())
    else:
        ev_norm = np.ones_like(ev_vals) * 0.5
    node_sizes = 20 + 40 * ev_norm

    node_x, node_y, texts, hovers = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        label = str(G.nodes[node].get(label_attr, node))
        texts.append(label)
        deg = G.degree(node)
        hovers.append(f"{label}<br>Degree: {deg}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=texts,
        textposition="top center",
        hovertext=hovers,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            line=dict(width=0.5, color="#333"),
        ),
        showlegend=False,
    )

    return edge_trace, node_trace


# ============================================================
# 3. 构建：每个数据集（GLOBAL / RISK / RETURN）的
#    - 按年份的行业网络
#    - 对应的行业内部公司网络
# ============================================================

def build_industry_frames_for_dataset(
    excel_path: Path,
    dataset_name: str,
    industry_map: pd.DataFrame
) -> Tuple[Dict[str, nx.Graph], Dict[str, Dict[str, nx.Graph]]]:
    """
    对一个数据集（GLOBAL / RISK / RETURN）：
      - 读取每个年份的 sheet
      - 聚合为行业层面数据，并构建行业网络图
      - 同时为每个「年份 × 行业」构建行业内部的公司网络图

    返回：
      graphs_years:  {year_str -> G_industry}
      inner_graphs:  {year_str -> {industry_name -> G_inner_companies}}
    """
    xls = pd.ExcelFile(excel_path)
    graphs_years: Dict[str, nx.Graph] = {}
    inner_graphs: Dict[str, Dict[str, nx.Graph]] = {}

    for sheet in xls.sheet_names:
        # 只用纯数字年份的 sheet（2006, 2007, ...）
        if not sheet.isdigit():
            continue

        df = pd.read_excel(xls, sheet_name=sheet)
        grouped, merged = aggregate_to_industry(df, industry_map)

        # 行业网络
        G_ind, meta_ind = BUILD_COMPANY_GRAPH(
            grouped,
            dataset_name=f"{dataset_name}_SECTOR",
            sheet_name=sheet,
            corr_threshold=CORR_THRESHOLD,
        )
        if G_ind is None or G_ind.number_of_nodes() == 0:
            continue

        graphs_years[sheet] = G_ind

        # 行业内部公司网络（一个行业一个子图）
        inner_dict: Dict[str, nx.Graph] = {}
        for ind_name, sub in merged.groupby("new_industry"):
            sub2 = sub.copy()
            # 删除辅助列
            for col in ["company_code", "new_industry"]:
                if col in sub2.columns:
                    sub2 = sub2.drop(columns=[col])
            G_in, meta_in = BUILD_COMPANY_GRAPH(
                sub2,
                dataset_name=f"{dataset_name}_INNER",
                sheet_name=f"{sheet}_{ind_name}",
                corr_threshold=CORR_THRESHOLD,
            )
            if G_in is not None and G_in.number_of_nodes() > 0:
                inner_dict[ind_name] = G_in

        inner_graphs[sheet] = inner_dict

    return graphs_years, inner_graphs


# ============================================================
# 4. 行业网络：构建按年份动画的 Plotly Figure
# ============================================================

def build_industry_animated_figure(
    graphs_years: Dict[str, nx.Graph],
    dataset_name: str
) -> go.Figure:
    """
    为一个数据集构建“行业网络按年份动画”的 Plotly Figure，
    带 slider + play/pause。
    """
    years = sorted(graphs_years.keys())
    if not years:
        raise ValueError(f"{dataset_name}: graphs_years 为空。")

    init_year = years[0]

    edge0, node0 = graph_to_traces(graphs_years[init_year], label_attr="empresa")

    # 每个年份一帧
    frames = []
    for y in years:
        edge_t, node_t = graph_to_traces(graphs_years[y], label_attr="empresa")
        frames.append(
            go.Frame(
                data=[edge_t, node_t],
                name=str(y),
                layout=go.Layout(
                    title_text=f"{dataset_name} - red por sectores, año {y}"
                ),
            )
        )

    # slider
    slider_steps = [
        {
            "args": [
                [str(y)],
                {
                    "frame": {"duration": 600, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                },
            ],
            "label": str(y),
            "method": "animate",
        }
        for y in years
    ]

    sliders = [
        {
            "active": 0,
            "y": -0.05,
            "x": 0.1,
            "len": 0.8,
            "xanchor": "left",
            "yanchor": "top",
            "pad": {"b": 10, "t": 30},
            "currentvalue": {
                "visible": True,
                "prefix": "Año: ",
                "xanchor": "right",
                "font": {"size": 14},
            },
            "steps": slider_steps,
        }
    ]

    updatemenus = [
        {
            "type": "buttons",
            "direction": "left",
            "x": 0.1,
            "y": 1.1,
            "showactive": False,
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 800, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 300},
                        },
                    ],
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }
    ]

    layout = go.Layout(
        title=f"{dataset_name} - red por sectores (2006–2024)",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=40, r=40, b=40, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        updatemenus=updatemenus,
        sliders=sliders,
    )

    fig = go.Figure(data=[edge0, node0], layout=layout, frames=frames)
    return fig


# ============================================================
# 5. 行业内部公司网络：为每个「年份 × 行业」构建单独 Figure
# ============================================================

def build_company_figures(
    inner_graphs: Dict[str, Dict[str, nx.Graph]]
) -> Dict[str, Dict[str, dict]]:
    """
    把 inner_graphs 转成可直接序列化到 JSON 的 Plotly dict：
      company_figs[year][industry] = {data:..., layout:...}
    """
    company_figs: Dict[str, Dict[str, dict]] = {}

    for year, ind_dict in inner_graphs.items():
        year_dict: Dict[str, dict] = {}
        for ind_name, G in ind_dict.items():
            edge_t, node_t = graph_to_traces(G, label_attr="empresa")
            if edge_t is None or node_t is None:
                continue

            layout = go.Layout(
                title=f"{ind_name} - red interna ({year})",
                showlegend=False,
                hovermode="closest",
                margin=dict(l=40, r=40, b=40, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            fig = go.Figure(data=[edge_t, node_t], layout=layout)
            year_dict[ind_name] = fig.to_plotly_json()
        company_figs[year] = year_dict

    return company_figs


# ============================================================
# 6. 生成带 drill-down 功能的 HTML
# ============================================================

def build_drilldown_html(
    fig_industry: go.Figure,
    company_figs: Dict[str, Dict[str, dict]],
    dataset_name: str,
    out_html: Path,
) -> Path:
    """
    将“行业网络动画 Figure + 行业内部网络 Figures”封装成双面板 HTML。
    左：行业动态网络（按年份 slider）
    右：点击行业节点后显示行业内部公司网络。
    """
    industry_json = json.dumps(
        fig_industry.to_plotly_json(),
        cls=plotly.utils.PlotlyJSONEncoder,
    )
    company_json = json.dumps(
        company_figs,
        cls=plotly.utils.PlotlyJSONEncoder,
    )

    years = sorted(company_figs.keys())
    init_year = years[0] if years else ""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>{dataset_name} - Red por sectores con drill-down</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
        background-color: #f6f7fb;
    }}
    .container {{
        max-width: 1400px;
        margin: 0 auto;
        padding: 16px 24px 32px 24px;
    }}
    h1 {{
        font-size: 1.6rem;
        margin: 0 0 6px 0;
        font-weight: 600;
    }}
    p.subtitle {{
        margin: 0 0 14px 0;
        font-size: 0.9rem;
        color: #4b5563;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.7rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        background: rgba(37,99,235,0.06);
        color: #1d4ed8;
        border: 1px solid rgba(37,99,235,0.25);
        margin-bottom: 4px;
    }}
    .grid {{
        display: grid;
        grid-template-columns: 1.1fr 1fr;
        grid-gap: 18px;
        align-items: stretch;
    }}
    .card {{
        background-color: #ffffff;
        border-radius: 18px;
        padding: 14px 16px 18px 16px;
        box-shadow: 0 16px 35px rgba(15,23,42,0.08);
        border: 1px solid rgba(148,163,184,0.18);
        min-height: 420px;
    }}
    .card h2 {{
        font-size: 1.1rem;
        margin: 0 0 4px 0;
    }}
    .card p.helper {{
        margin: 0 0 10px 0;
        font-size: 0.8rem;
        color: #6b7280;
    }}
    #industry_network, #company_network {{
        width: 100%;
        height: 600px;
    }}
    .hint {{
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 6px;
    }}
  </style>
</head>
<body>
<div class="container">
  <div class="badge">Network Drill-down</div>
  <h1>{dataset_name} – red dinámica por sectores con vista interna</h1>
  <p class="subtitle">
    Use el control temporal para cambiar de año. Haga clic en un nodo de sector para desplegar,
    en el panel derecho, la red interna de empresas de ese sector en el año seleccionado.
  </p>
  <div class="grid">
    <div class="card">
      <h2>Red entre sectores (2006–2024)</h2>
      <p class="helper">Slider temporal y animación por años. Cada nodo representa un sector agregado.</p>
      <div id="industry_network"></div>
      <p class="hint">Sugerencia: seleccione un año en el slider y luego haga clic en un sector.</p>
    </div>
    <div class="card">
      <h2>Red interna de empresas por sector</h2>
      <p class="helper">Se actualizará automáticamente al hacer clic en un nodo de sector.</p>
      <div id="company_network"></div>
      <p class="hint" id="company_hint"></p>
    </div>
  </div>
</div>

<script>
  // Datos generados en Python
  const industryFig = {industry_json};
  const companyFigures = {company_json};
  let currentYear = "{init_year}";
  let currentIndustry = null;

  const industryDiv = document.getElementById("industry_network");
  const companyDiv = document.getElementById("company_network");
  const companyHint = document.getElementById("company_hint");

  function renderIndustryFigure() {{
    Plotly.newPlot(industryDiv, industryFig.data, industryFig.layout, {{
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"]
    }}).then(function() {{
      if (industryFig.frames) {{
        Plotly.addFrames(industryDiv, industryFig.frames);
      }}
    }});
  }}

  function updateCompanyFigure(year, industryName) {{
    if (!year || !industryName) {{
      companyDiv.innerHTML = "";
      companyHint.textContent = "Seleccione un sector en la red de la izquierda para ver su estructura interna.";
      return;
    }}
    const yearMap = companyFigures[year];
    if (!yearMap || !yearMap[industryName]) {{
      companyDiv.innerHTML = "";
      companyHint.textContent = "No hay datos de red interna para '" + industryName + "' en " + year + ".";
      return;
    }}
    const fig = yearMap[industryName];
    companyHint.textContent = "Sector: " + industryName + " | Año: " + year;
    Plotly.newPlot(companyDiv, fig.data, fig.layout, {{
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"]
    }});
  }}

  function initInteractions() {{
    industryDiv.on("plotly_click", function(eventData) {{
      if (!eventData || !eventData.points || eventData.points.length === 0) return;
      const pt = eventData.points[0];
      const indName = pt.text || (pt.customdata && pt.customdata[0]) || null;
      if (!indName) return;
      currentIndustry = indName;
      updateCompanyFigure(currentYear, currentIndustry);
    }});

    industryDiv.on("plotly_sliderchange", function(eventData) {{
      if (!eventData || !eventData.step || !eventData.step.label) return;
      currentYear = String(eventData.step.label);
      if (currentIndustry) {{
        updateCompanyFigure(currentYear, currentIndustry);
      }} else {{
        updateCompanyFigure(currentYear, null);
      }}
    }});
  }}

  // Render inicial
  renderIndustryFigure();
  initInteractions();
  // Mensaje inicial
  updateCompanyFigure(currentYear, null);
</script>
</body>
</html>"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html


# ============================================================
# 7. 主入口：对 GLOBAL / RISK / RETURN 全部生成
# ============================================================

def main():
    industry_map = pd.read_excel(INDUSTRY_FILE)

    datasets = [
        ("GLOBAL", GLOBAL_FILE, "global_industry_drilldown_years.html"),
        ("RISK",   RISK_FILE,   "risk_industry_drilldown_years.html"),
        ("RETURN", RETURN_FILE, "return_industry_drilldown_years.html"),
    ]

    for ds_name, path, html_name in datasets:
        print(f"\n=== Procesando dataset: {ds_name} ===")
        graphs_years, inner_graphs = build_industry_frames_for_dataset(
            excel_path=path,
            dataset_name=ds_name,
            industry_map=industry_map,
        )
        print(f"{ds_name}: años con red de sectores -> {sorted(graphs_years.keys())}")

        fig_ind = build_industry_animated_figure(graphs_years, ds_name)
        company_figs = build_company_figures(inner_graphs)

        out_html = OUTPUT_DIR / html_name
        build_drilldown_html(fig_ind, company_figs, ds_name, out_html)

        print(f"{ds_name}: HTML con drill-down generado -> {out_html}")

    print("\nTodo listo. Puedes在 Streamlit 中用 components.html 嵌入上述 HTML 文件。")


if __name__ == "__main__":
    main()
