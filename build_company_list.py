import pandas as pd
from pathlib import Path


def build_company_list(
    input_file: str = "Mercados_company_means_FIXED.xlsx",
    output_file: str = "company_list.xlsx",
):
    """
    从包含多工作表的 Excel 中读取所有公司，
    生成不重复的公司列表（代号 + 名称）。
    假设列名为: 'Code'（公司代号）, 'empresa'（公司名称）
    """

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_path}")

    xls = pd.ExcelFile(input_path)

    frames = []

    for sheet in xls.sheet_names:
        # 逐个工作表读取
        df = pd.read_excel(input_path, sheet_name=sheet)

        # 只处理包含公司信息的 sheet（有 Code 和 empresa 两列）
        if "Code" in df.columns and "empresa" in df.columns:
            tmp = df[["Code", "empresa"]].copy()

            # 清理一下字符串，防止前后空格导致重复判断出错
            tmp["Code"] = tmp["Code"].astype(str).str.strip()
            tmp["empresa"] = tmp["empresa"].astype(str).str.strip()

            # 去掉公司名为空的行
            tmp = tmp.dropna(subset=["empresa"])

            frames.append(tmp)

    if not frames:
        raise ValueError("未在任何工作表中找到同时包含 'Code' 和 'empresa' 的数据。")

    # 合并所有工作表中的公司
    all_companies = pd.concat(frames, ignore_index=True)

    # 按公司名去重（避免同一公司在不同年份重复）
    unique_companies = (
        all_companies
        .drop_duplicates(subset=["empresa"], keep="first")
        .reset_index(drop=True)
    )

    # 可选：按 Code 排序一下，方便查看
    unique_companies = unique_companies.sort_values("Code").reset_index(drop=True)

    # 为了更直观，可以改个列名（也可以保持原来的 Code / empresa）
    unique_companies = unique_companies.rename(
        columns={"Code": "company_code", "empresa": "company_name"}
    )

    # 导出到新的 Excel
    unique_companies.to_excel(output_file, index=False)

    print(f"✅ 已生成公司列表: {output_file}")
    print(f"📊 共 {len(unique_companies)} 家不重复的公司")


if __name__ == "__main__":
    build_company_list()
