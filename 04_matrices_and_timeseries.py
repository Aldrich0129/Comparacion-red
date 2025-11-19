# 04_matrices_and_timeseries.py
"""
方案 5：矩阵 / 热力图 / 中心性时间序列

功能：
- 从 network_macro_merged.xlsx 的 "nodes_with_stress" 读取节点指标
- 从 edges/{dataset}_{year}_edges.csv 读取边表
- 对每个 dataset-year 生成加权邻接矩阵热力图：
    outputs/matrices/adjacency_{dataset}_{year}.png
- 对每个 dataset 生成 Top N 公司 eigenvector_adj 随时间变化折线图：
    outputs/timeseries/centrality_timeseries_{dataset}.png
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================
# 配置
# =====================

BASE_DIR = Path(__file__).resolve().parent

NODES_FILE = BASE_DIR / "network_macro_merged.xlsx"
EDGES_DIR = BASE_DIR / "edges"
OUTPUT_DIR = BASE_DIR / "outputs"

DATASETS = ["GLOBAL", "RISK", "RETURN"]
YEARS = list(range(2006, 2025))

TOP_N_CENTRAL = 10   # 时间序列图中展示的公司数量


# =====================
# 公共工具
# =====================

def ensure_output_dir(subdir: str) -> Path:
    out = OUTPUT_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_nodes_all() -> pd.DataFrame:
    """载入全部 nodes_with_stress，用于时间序列。"""
    return pd.read_excel(NODES_FILE, sheet_name="nodes_with_stress")


def load_nodes(dataset: str, year: int) -> pd.DataFrame:
    df = pd.read_excel(NODES_FILE, sheet_name="nodes_with_stress")
    sub = df[(df["dataset"] == dataset) & (df["year"] == year)].copy()
    if sub.empty:
        raise ValueError(f"nodes_with_stress 中找不到 dataset={dataset}, year={year} 的记录")
    return sub


def load_edges(dataset: str, year: int) -> pd.DataFrame:
    edges_file = EDGES_DIR / f"{dataset}_{year}_edges.csv"
    if not edges_file.exists():
        raise FileNotFoundError(f"未找到边表文件: {edges_file}")
    edges = pd.read_csv(edges_file)
    required = {"source", "target", "weight"}
    if not required.issubset(edges.columns):
        raise ValueError(f"{edges_file} 缺少必要列: {required}")
    return edges


def build_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    for _, r in nodes_df.iterrows():
        code = r["Code"]
        G.add_node(
            code,
            empresa=r.get("empresa", code),
            eigenvector=r.get("eigenvector", np.nan),
            eigenvector_adj=r.get("eigenvector_adj", r.get("eigenvector", np.nan)),
            degree=r.get("degree", np.nan),
            strength=r.get("strength", np.nan),
            dataset=r.get("dataset"),
            year=int(r.get("year")),
        )

    for _, r in edges_df.iterrows():
        u, v = r["source"], r["target"]
        if u in G.nodes and v in G.nodes:
            G.add_edge(u, v, weight=float(r.get("weight", 1.0)))

    return G


# =====================
# 邻接矩阵热力图
# =====================

def generate_adjacency_heatmap(dataset: str, year: int) -> Path:
    nodes_df = load_nodes(dataset, year)
    edges_df = load_edges(dataset, year)
    G = build_graph(nodes_df, edges_df)

    # 节点排序：按 eigenvector_adj 由高到低
    node_order = sorted(
        G.nodes(),
        key=lambda n: G.nodes[n].get("eigenvector_adj", G.nodes[n].get("eigenvector", 0.0)),
        reverse=True,
    )

    A = nx.to_numpy_array(G, nodelist=node_order, weight="weight")

    out_dir = ensure_output_dir("matrices")
    png_path = out_dir / f"adjacency_{dataset}_{year}.png"

    plt.figure(figsize=(6, 6))
    plt.imshow(A, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Peso / similitud")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title(f"Matriz de adyacencia - {dataset} {year}")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    return png_path


# =====================
# 中心性时间序列
# =====================

def generate_centrality_timeseries(nodes_with_stress: pd.DataFrame, dataset: str) -> Path:
    df = nodes_with_stress[nodes_with_stress["dataset"] == dataset].copy()
    # 有些 Year 列为空，使用 'year' 替代
    if "year" in df.columns and df["Year"].isna().all():
        df["Year"] = df["year"]
    df["Year"] = df["Year"].astype(int)

    # 按公司计算 eigenvector_adj 的平均值，选 Top N
    mean_ec = (
        df.groupby("Code")["eigenvector_adj"]
        .mean()
        .sort_values(ascending=False)
        .head(TOP_N_CENTRAL)
    )
    top_codes = mean_ec.index.tolist()

    sub = df[df["Code"].isin(top_codes)].copy()
    sub = sub.sort_values(["Code", "Year"])

    out_dir = ensure_output_dir("timeseries")
    png_path = out_dir / f"centrality_timeseries_{dataset}.png"

    plt.figure(figsize=(7, 4))
    for code in top_codes:
        tmp = sub[sub["Code"] == code]
        plt.plot(tmp["Year"], tmp["eigenvector_adj"], marker="o", linewidth=1, label=code)

    plt.xlabel("Año")
    plt.ylabel("Centralidad ajustada (eigenvector_adj)")
    plt.title(f"Top {TOP_N_CENTRAL} empresas por centralidad - {dataset}")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    return png_path


def main():
    print("=== 方案5：矩阵 & 中心性时间序列 - 批量生成开始 ===")

    # 1) 邻接矩阵热力图
    for dataset in DATASETS:
        for year in YEARS:
            try:
                print(f"[{dataset} {year}] 生成邻接矩阵热力图...")
                png_path = generate_adjacency_heatmap(dataset, year)
                print(f"  -> PNG: {png_path}")
            except FileNotFoundError as e:
                print(f"  !! 跳过（缺少文件）: {e}")
            except Exception as e:
                print(f"  !! 跳过（其他错误）: {e}")

    # 2) 中心性时间序列
    nodes_all = load_nodes_all()
    for dataset in DATASETS:
        try:
            print(f"[{dataset}] 生成中心性时间序列...")
            png_path = generate_centrality_timeseries(nodes_all, dataset)
            print(f"  -> PNG: {png_path}")
        except Exception as e:
            print(f"  !! 跳过（错误）: {e}")

    print("=== 完成 ===")


if __name__ == "__main__":
    main()
