"""方案 5：以热力图 / 矩阵与中心性时间序列替代大型网络图。

该脚本复用 build_networks.py 的数据读入逻辑：
- 从 GLOBAL / RISK / RETURN 三个 Excel 中读取每个 sheet（年份）。
- 针对每个网络生成一个邻接矩阵热力图（按 eigenvector centrality 排序）。
- 针对每个数据集，汇总所有年份的 eigenvector centrality，绘制 Top-N 公司的
  时间序列图，展示结构性变化。

输出目录保持独立，避免覆盖原有 output：
- output_matrix_ts/matrices/*.png  （邻接矩阵热力图）
- output_matrix_ts/timeseries/*.png（中心性时间序列）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_networks as bn

# =====================
# 配置
# =====================

DATASETS: List[Tuple[str, str]] = [
    ("GLOBAL", bn.FULL_EXCEL),
    ("RISK", bn.RISK_EXCEL),
    ("RETURN", bn.RETURN_EXCEL),
]

OUTPUT_DIR = "output_matrix_ts"
HEATMAP_DIR = os.path.join(OUTPUT_DIR, "matrices")
TIMESERIES_DIR = os.path.join(OUTPUT_DIR, "timeseries")

TOP_N_CENTRAL = 10  # 时间序列图中展示的公司数量


# =====================
# 公共工具
# =====================

def ensure_dirs() -> None:
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    os.makedirs(TIMESERIES_DIR, exist_ok=True)


def compute_eigenvector(G: nx.Graph) -> Dict[str, float]:
    if G.number_of_nodes() == 0:
        return {}
    try:
        return nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    except nx.PowerIterationFailedConvergence:
        # 若不收敛，退化为度中心性
        deg = dict(G.degree(weight="weight"))
        max_deg = max(deg.values()) if deg else 1.0
        return {n: (deg.get(n, 0.0) / max_deg if max_deg > 0 else 0.0) for n in G.nodes()}


# =====================
# 邻接矩阵热力图
# =====================

def generate_adjacency_heatmap(
    G: nx.Graph,
    meta: dict,
    eigenvector: Dict[str, float],
    out_path: str,
) -> None:
    if G.number_of_nodes() == 0:
        return

    # 节点排序：按 eigenvector 由高到低
    node_order = sorted(
        G.nodes(),
        key=lambda n: eigenvector.get(n, 0.0),
        reverse=True,
    )
    A = nx.to_numpy_array(G, nodelist=node_order, weight="weight")

    plt.figure(figsize=(6, 6))
    plt.imshow(A, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Peso / similitud")
    plt.xticks([], [])
    plt.yticks([], [])

    ds = meta.get("dataset", "")
    sheet = meta.get("sheet", "")
    if str(sheet).isdigit():
        subtitle = f"Año {sheet}"
    else:
        subtitle = str(sheet)

    if ds == "GLOBAL":
        prefix = "Matriz de adyacencia - Red global"
    elif ds == "RISK":
        prefix = "Matriz de adyacencia - Red de riesgo"
    elif ds == "RETURN":
        prefix = "Matriz de adyacencia - Red de rentabilidad"
    else:
        prefix = f"Matriz de adyacencia - {ds}"

    plt.title(f"{prefix}\n{subtitle}")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


# =====================
# 中心性时间序列
# =====================

def generate_centrality_timeseries(
    centrality_records: pd.DataFrame,
    dataset: str,
    out_path: str,
    top_n: int = TOP_N_CENTRAL,
) -> None:
    df = centrality_records[centrality_records["dataset"] == dataset].copy()
    if df.empty:
        return

    df["year"] = df["year"].astype(int)

    mean_ev = (
        df.groupby("node")["eigenvector"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    top_nodes = mean_ev.index.tolist()
    sub = df[df["node"].isin(top_nodes)].copy()
    if sub.empty:
        return

    plt.figure(figsize=(7, 4))
    for node in top_nodes:
        tmp = sub[sub["node"] == node]
        plt.plot(
            tmp["year"],
            tmp["eigenvector"],
            marker="o",
            linewidth=1,
            label=tmp["empresa"].iloc[0] if not tmp["empresa"].isna().all() else node,
        )

    plt.xlabel("Año")
    plt.ylabel("Centralidad de vector propio")
    plt.title(f"Top {len(top_nodes)} empresas por centralidad - {dataset}")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


# =====================
# 主流程
# =====================

def main() -> None:
    ensure_dirs()

    centrality_rows: List[Dict[str, object]] = []

    for dataset, path in DATASETS:
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            continue

        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            G, meta = bn.build_company_graph(df, dataset, sheet, bn.CORR_THRESHOLD)
            if G is None:
                continue

            ev = compute_eigenvector(G)
            meta = dict(meta)
            meta["dataset"] = dataset
            meta["sheet"] = sheet

            fname = f"{dataset.lower()}_{bn.sanitize_filename(sheet)}_adjacency.png"
            out_png = os.path.join(HEATMAP_DIR, fname)
            generate_adjacency_heatmap(G, meta, ev, out_png)
            print(f"[OK] Heatmap guardado: {out_png}")

            if str(sheet).isdigit():
                year_val = int(sheet)
                for node in G.nodes():
                    centrality_rows.append(
                        {
                            "dataset": dataset,
                            "year": year_val,
                            "node": node,
                            "empresa": G.nodes[node].get("empresa", node),
                            "eigenvector": ev.get(node, 0.0),
                        }
                    )

    if centrality_rows:
        centrality_df = pd.DataFrame(centrality_rows)
        for dataset, _ in DATASETS:
            out_png = os.path.join(
                TIMESERIES_DIR,
                f"centrality_timeseries_{dataset.lower()}.png",
            )
            generate_centrality_timeseries(centrality_df, dataset, out_png)
            print(f"[OK] Serie temporal guardada: {out_png}")
    else:
        print("[WARN] No se han acumulado datos de centralidad para generar las series.")


if __name__ == "__main__":
    main()
