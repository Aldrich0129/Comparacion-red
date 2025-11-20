"""
build_industry_networks.py

在原有 build_networks.py 的基础上，把公司网络按“行业（new_industry）”聚合：
- 节点：行业
- 特征：该行业内所有公司的财务/风险/收益指标的截面平均值

输出：
1) 行业网络静态 PNG（每年 + AVG_BY_COMPANY + WAVG_BY_COMPANY，GLOBAL / RISK / RETURN）
2) 行业网络按年份的动态 HTML（*_industry_dynamic_years.html）
3) 基于 GLOBAL 的“行业-年份-变量”网络 + PNG + 动态 HTML
   (global_industry_dynamic_variables.html)
4) 所有行业网络的节点 & 整体指标：network_metrics_industry.xlsx

关键点：
- 完全复用 build_networks.py 中的标准化、距离、相似度、相关性阈值逻辑，
  只是在建图前把公司按 new_industry 聚合。
- 行业映射来自 industrial.xlsx，要求至少包含：
    - company_code  （公司代码，如 0001.HK）
    - new_industry  （行业名，已是你整理好的聚合口径）
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import build_networks as bn  # 直接复用原脚本中的函数


# ---------------- 路径 & 基本配置 ---------------- #

BASE_DIR = Path(__file__).resolve().parent

# 原数据文件（保持和 build_networks.py 一致）
FULL_EXCEL = BASE_DIR / "Mercados_company_means_FIXED.xlsx"
RISK_EXCEL = BASE_DIR / "RISK.xlsx"
RETURN_EXCEL = BASE_DIR / "RETURN.xlsx"

# 行业映射文件：company_code -> new_industry
INDUSTRY_EXCEL = BASE_DIR / "industrial.xlsx"
INDUSTRY_SHEET_NAME = None  # 若为 None，则使用第一个 sheet

# 输出目录（和原 output 区分开）
OUTPUT_DIR = BASE_DIR / "output_industry"
PNG_DIR = OUTPUT_DIR / "png"
HTML_DIR = OUTPUT_DIR / "html"
METRICS_FILE = OUTPUT_DIR / "network_metrics_industry.xlsx"


# ---------------- 工具函数 ---------------- #

def ensure_dirs():
    """创建输出目录。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    PNG_DIR.mkdir(exist_ok=True, parents=True)
    HTML_DIR.mkdir(exist_ok=True, parents=True)


def load_industry_mapping(
    path: Path = INDUSTRY_EXCEL,
    sheet_name: str | None = INDUSTRY_SHEET_NAME
) -> pd.DataFrame:
    """
    读取 industrial.xlsx，返回 Code -> new_industry 映射表。

    预期列：
        company_code   公司代码（与各 Excel 中 Code 对应）
        new_industry   行业名称（已是你希望使用的聚合行业）
    """
    xls = pd.ExcelFile(path)
    sh = sheet_name or xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sh)

    if "company_code" not in df.columns or "new_industry" not in df.columns:
        raise ValueError(
            "industrial.xlsx 必须至少包含列 'company_code' 和 'new_industry'"
        )

    map_df = (
        df[["company_code", "new_industry"]]
        .dropna()
        .drop_duplicates()
    )
    return map_df


def aggregate_sheet_to_industry(
    df_sheet: pd.DataFrame,
    ind_map: pd.DataFrame
) -> pd.DataFrame | None:
    """
    将单个 sheet（公司层面）按行业聚合成“行业表”。

    步骤：
      1) 按 Code 连接 industry 映射 -> 得到 new_industry
      2) 丢弃无行业的公司
      3) 对每个 new_industry，在所有数值变量上取均值
      4) 生成行业层面的 DataFrame：
           - empresa = new_industry  (用于标签)
           - Code   = new_industry  (作为网络节点 ID)
           - 其余列为各数值变量的行业均值
           - n_companies = 该行业内公司数
    """
    # 要求至少有 empresa, Code（和原脚本一致）
    if not all(c in df_sheet.columns for c in bn.BASE_COLS):
        return None

    df = df_sheet.merge(
        ind_map,
        left_on="Code",
        right_on="company_code",
        how="left"
    )

    before = df.shape[0]
    df = df.dropna(subset=["new_industry"])
    after = df.shape[0]

    print(f"  - rows with industry: {after}/{before}")
    if after < 2:
        return None

    # 选数值列做聚合
    num_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    # Year 是常数列，保留与否问题不大；留着也 OK，会被标准化成 0 向量
    grouped = df.groupby("new_industry")

    agg = grouped[num_cols].mean().reset_index()
    # 行业内公司数量
    agg["n_companies"] = grouped.size().values

    # 对齐 build_company_graph 的格式：BASE_COLS = ["empresa", "Code"]
    agg = agg.rename(columns={"new_industry": "empresa"})
    agg["Code"] = agg["empresa"]

    # 确保 empresa / Code 在前
    cols = ["empresa", "Code"] + [
        c for c in agg.columns if c not in ["empresa", "Code"]
    ]
    agg = agg[cols]

    return agg


def make_industry_panel_for_var(
    panel_ind: pd.DataFrame,
    var_name: str
) -> pd.DataFrame:
    """
    从公司层面的 panel（已接上 new_industry）生成
    该变量的“行业-年份”面板数据，用于按变量构建行业网络。

    返回列：
        empresa  (行业名)
        Code     (行业名，同 empresa)
        Year
        var_name
    """
    df = panel_ind[["new_industry", "Year", var_name]].copy()
    df = df.dropna(subset=["new_industry", "Year"])
    if df.empty:
        return pd.DataFrame(columns=["empresa", "Code", "Year", var_name])

    df["Year"] = df["Year"].astype(int)

    grp = df.groupby(["new_industry", "Year"])[var_name].mean().reset_index()
    grp = grp.rename(columns={"new_industry": "empresa"})
    grp["Code"] = grp["empresa"]

    cols = ["empresa", "Code", "Year", var_name]
    return grp[cols]


# ---------------- 主函数：构建行业网络 ---------------- #

def main():
    ensure_dirs()

    ind_map = load_industry_mapping()
    print(
        f"[INFO] Loaded industry map: "
        f"{ind_map['new_industry'].nunique()} industries, "
        f"{len(ind_map)} rows."
    )

    datasets = [
        ("GLOBAL", FULL_EXCEL),
        ("RISK",   RISK_EXCEL),
        ("RETURN", RETURN_EXCEL),
    ]

    all_node_metrics: list[pd.DataFrame] = []
    all_graph_metrics: list[dict] = []

    # 保存用于动态 HTML（按年份）的结构
    graphs_by_dataset_and_year: dict[str, dict[str, tuple]] = {
        "GLOBAL": {},
        "RISK": {},
        "RETURN": {},
    }

    var_graphs: dict[str, tuple] = {}  # 行业-变量网络（只对 GLOBAL）

    # ---------- 1. 各数据集 / 各年份的“行业网络” ---------- #

    for ds_name, path in datasets:
        path = Path(path)  # 保险起见转成 Path
        print(f"\n[INFO] Building industry networks for {ds_name} ({path.name})")

        xls = pd.ExcelFile(path)
        for sh in xls.sheet_names:
            print(f"  -> sheet: {sh}")
            df_sheet = pd.read_excel(path, sheet_name=sh)

            df_ind = aggregate_sheet_to_industry(df_sheet, ind_map)
            if df_ind is None or df_ind.shape[0] < 2:
                print("     (skip: not enough industries after aggregation)")
                continue

            # 这里 dataset 名字用 f"{ds_name}_IND" 方便在指标里区分
            G, meta = bn.build_company_graph(
                df_ind,
                dataset_name=f"{ds_name}_IND",
                sheet_name=sh,
                corr_threshold=bn.CORR_THRESHOLD,
            )
            if G is None or G.number_of_nodes() == 0:
                print("     (skip: empty graph)")
                continue

            # 静态 PNG
            fname = f"{ds_name.lower()}_IND_{bn.sanitize_filename(sh)}.png"
            out_png = PNG_DIR / fname
            bn.plot_static_graph(G, meta, str(out_png))

            # 网络指标
            node_df = bn.compute_node_metrics(G, meta)
            graph_dict = bn.compute_graph_metrics(G, meta)
            all_node_metrics.append(node_df)
            all_graph_metrics.append(graph_dict)

            # 动态 HTML（按年份）只记录数字年份的 sheet
            if sh.isdigit():
                graphs_by_dataset_and_year[ds_name][sh] = (G, meta)

    # ---------- 2. 基于 GLOBAL 的“按变量的行业网络” ---------- #

    print("\n[INFO] Building variable-level industry networks from GLOBAL panel")

    panel = bn.load_panel_from_full(str(FULL_EXCEL))
    if not panel.empty:
        # 接上行业
        panel_ind = panel.merge(
            ind_map,
            left_on="Code",
            right_on="company_code",
            how="left"
        )
        panel_ind = panel_ind.dropna(subset=["new_industry"])

        # 数值变量：排除基础列和映射列
        num_cols = [
            c for c in panel_ind.columns
            if c not in bn.BASE_COLS
                      + ["Year", "SourceFile",
                         "company_code", "company_name", "new_industry"]
            and pd.api.types.is_numeric_dtype(panel_ind[c])
        ]
        print(f"  - #variables for industry-variable networks: {len(num_cols)}")

        for var in num_cols:
            print(f"  -> variable: {var}")
            panel_var = make_industry_panel_for_var(panel_ind, var)
            if panel_var["empresa"].nunique() < 2:
                print("     (skip: <2 industries)")
                continue

            Gv, meta_v = bn.build_variable_graph(
                panel_var,
                var_name=var,
                dataset_name="GLOBAL_IND"
            )
            if Gv is None or Gv.number_of_nodes() == 0:
                print("     (skip: empty graph)")
                continue

            # 静态 PNG
            fname = f"GLOBAL_IND_var_{bn.sanitize_filename(var)}.png"
            out_png = PNG_DIR / fname
            bn.plot_static_graph(Gv, meta_v, str(out_png))

            # 指标
            node_df_v = bn.compute_node_metrics(Gv, meta_v)
            graph_dict_v = bn.compute_graph_metrics(Gv, meta_v)
            all_node_metrics.append(node_df_v)
            all_graph_metrics.append(graph_dict_v)

            var_graphs[var] = (Gv, meta_v)
    else:
        print("  (panel empty, skip variable-level industry networks)")

    # ---------- 3. 指标写入 Excel ---------- #

    if all_node_metrics and all_graph_metrics:
        nodes_df_all = pd.concat(all_node_metrics, ignore_index=True)
        graphs_df_all = pd.DataFrame(all_graph_metrics)

        with pd.ExcelWriter(METRICS_FILE, engine="openpyxl") as writer:
            nodes_df_all.to_excel(writer, sheet_name="nodes_metrics", index=False)
            graphs_df_all.to_excel(writer, sheet_name="graphs_summary", index=False)

        print(f"\n[OK] Saved industry network metrics to: {METRICS_FILE}")
    else:
        print("\n[WARN] No industry networks built, metrics file not written.")

    # ---------- 4. 行业网络：按年份的动态 HTML ---------- #

    for ds_name, _ in datasets:
        graphs_year = graphs_by_dataset_and_year.get(ds_name, {})
        if graphs_year:
            out_html = HTML_DIR / f"{ds_name.lower()}_industry_dynamic_years.html"
            # 这里 dataset_name 仍传 'GLOBAL' / 'RISK' / 'RETURN'，
            # 只是 title 上显示为 "Red Global - Años"，但节点实际是行业。
            bn.build_animated_years_html(graphs_year, ds_name, str(out_html))
            print(f"[OK] Industry dynamic HTML (years) for {ds_name}: {out_html}")
        else:
            print(f"[INFO] No yearly industry graphs for {ds_name}, skip HTML.")

    # ---------- 5. 行业-变量网络：动态 HTML ---------- #

    if var_graphs:
        out_html_vars = HTML_DIR / "global_industry_dynamic_variables.html"
        bn.build_animated_variables_html(var_graphs, str(out_html_vars))
        print(f"[OK] Industry dynamic HTML (variables): {out_html_vars}")
    else:
        print("[INFO] No industry variable-level graphs, skip HTML.")

    print("\n[DONE] Industry-aggregated networks construction finished.")


if __name__ == "__main__":
    main()
