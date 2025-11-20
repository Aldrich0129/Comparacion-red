"""
build_industry_networks.py

功能：
基于“超级节点（Super-nodes）”理论构建产业网络。
1. 读取公司层面的财务数据。
2. 读取行业分类数据（支持多种编码格式，防止乱码）。
3. 按行业（Industry）聚合数据：计算每个行业所有公司的指标平均值（质心）。
4. 构建网络：
   - 节点：行业（Industry）
   - 连边：行业平均向量之间的相似度（基于欧氏距离）
   - 节点大小：反映该行业包含的公司数量（权重）

输出：
- 静态 PNG 图片（按年份）
- 动态 HTML 交互图（按年份聚合动画）
- Excel 网络指标分析报告
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# ---------------- CONFIGURATION ---------------- #

# 输入文件配置
DATA_FILES = {
    "GLOBAL": "Mercados_company_means_FIXED.xlsx",
    "RISK":   "RISK.xlsx",
    "RETURN": "RETURN.xlsx"
}

INDUSTRY_FILE = "industrial.xlsx"

# 相似度阈值配置
# 由于聚合后的行业数据比单家公司数据更平滑，相关性通常更高
SIM_THRESHOLD = 0.85  

# 输出目录
OUTPUT_DIR = "output_industry_networks"
PNG_DIR = os.path.join(OUTPUT_DIR, "png")
HTML_DIR = os.path.join(OUTPUT_DIR, "html")
METRICS_FILE = os.path.join(OUTPUT_DIR, "Industry_Network_Metrics.xlsx")

# 创建目录
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

# ---------------- DATA LOADING & PROCESSING ---------------- #

def read_csv_robust(filepath):
    """尝试多种编码读取 CSV 文件"""
    encodings = ['utf-8', 'gbk', 'gb18030', 'latin1', 'utf-16']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            # 简单的检查：如果读出来只有一列，且包含逗号，可能分隔符不对，但这通常是编码问题的前兆
            return df
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    print(f"[ERROR] Could not read {filepath} with any standard encoding.")
    return None

def load_industry_mapping(filepath):
    """读取行业对照表，返回 {company_code: industry_name} 字典"""
    df = read_csv_robust(filepath)
    
    if df is None:
        print("[ERROR] Failed to load industry mapping file.")
        return {}

    try:
        # 清理列名
        df.columns = [c.strip() for c in df.columns]
        
        # 确保有关键列
        if 'company_code' not in df.columns or 'new_industry' not in df.columns:
            print("Warning: Industry file columns mismatch. Trying index 0 and 2.")
            # 尝试按位置读取：第1列是代码，第3列是行业
            mapping = dict(zip(df.iloc[:, 0].astype(str).str.strip(), df.iloc[:, 2].astype(str).str.strip()))
        else:
            mapping = dict(zip(df['company_code'].astype(str).str.strip(), df['new_industry'].astype(str).str.strip()))
            
        print(f"[INFO] Loaded mapping for {len(mapping)} companies.")
        return mapping
    except Exception as e:
        print(f"[ERROR] Error parsing industry mapping: {e}")
        return {}

def get_file_pattern(prefix):
    """根据前缀匹配当文件夹下的年份CSV文件"""
    return f"{prefix} - 20*.csv"

def process_year_file(filepath, industry_map):
    """
    处理单一年份的文件：
    1. 读取数据
    2. 映射行业
    3. 按行业聚合（计算均值）
    """
    df = read_csv_robust(filepath)
    if df is None:
        return None, None

    try:
        # 清理 Code 列，用于匹配
        if 'Code' in df.columns:
            df['Code_Clean'] = df['Code'].astype(str).str.strip()
        elif 'company_code' in df.columns:
            df['Code_Clean'] = df['company_code'].astype(str).str.strip()
        else:
            # 尝试找包含 .SZ/.SS/.HK 的列
            found = False
            for col in df.columns:
                if df[col].astype(str).str.contains(r'\.(SZ|HK|SS|SH)', regex=True).any():
                    df['Code_Clean'] = df[col].astype(str).str.strip()
                    found = True
                    break
            if not found:
                return None, None

        # 映射行业
        df['Industry'] = df['Code_Clean'].map(industry_map)
        
        # 过滤掉没有行业归属的公司
        df_clean = df.dropna(subset=['Industry'])
        
        if df_clean.empty:
            return None, None

        # 提取数值列用于计算
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        # 排除无意义的列
        numeric_cols = [c for c in numeric_cols if 'Year' not in c and c != 'Unnamed: 0']
        
        # Super-node Aggregation 核心逻辑：按行业分组
        # 1. 计算均值 (每个行业的特征向量)
        df_industry_mean = df_clean.groupby('Industry')[numeric_cols].mean()
        
        # 2. 计算公司数量 (作为节点大小权重)
        df_industry_count = df_clean.groupby('Industry')['Code_Clean'].count()
        
        # 将数量合并进去
        df_industry_mean['company_count'] = df_industry_count
        
        # 数据归一化处理 (MinMax)，防止量纲影响欧氏距离
        feature_cols = [c for c in df_industry_mean.columns if c != 'company_count']
        if not feature_cols:
            return None, None
            
        scaler = MinMaxScaler()
        df_industry_mean[feature_cols] = scaler.fit_transform(df_industry_mean[feature_cols].fillna(0))
        
        return df_industry_mean, feature_cols
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None, None

# ---------------- NETWORK CONSTRUCTION ---------------- #

def compute_similarity(vec_a, vec_b):
    """基于欧氏距离计算相似度: 1 / (1 + distance)"""
    dist = np.linalg.norm(vec_a - vec_b)
    return 1.0 / (1.0 + dist)

def build_network_from_aggregated(df_agg, feature_cols, threshold=0.8):
    """
    构建网络图
    df_agg: 索引为 Industry 的 DataFrame
    feature_cols:用于计算距离的列名列表
    """
    G = nx.Graph()
    
    industries = df_agg.index.tolist()
    
    # 添加节点
    for ind in industries:
        count = df_agg.loc[ind, 'company_count']
        G.add_node(ind, size=count, label=ind)
    
    # 添加边
    n = len(industries)
    for i in range(n):
        for j in range(i + 1, n):
            ind_a = industries[i]
            ind_b = industries[j]
            
            vec_a = df_agg.loc[ind_a, feature_cols].values
            vec_b = df_agg.loc[ind_b, feature_cols].values
            
            sim = compute_similarity(vec_a, vec_b)
            
            if sim >= threshold:
                G.add_edge(ind_a, ind_b, weight=sim)
                
    return G

# ---------------- VISUALIZATION ---------------- #

def save_static_plot(G, year, dataset_name):
    """保存静态 PNG"""
    plt.figure(figsize=(12, 10))
    
    # 布局
    pos = nx.spring_layout(G, k=0.6, seed=42) # k值大一些，节点分得更开
    
    # 节点大小 (根据公司数量缩放)
    # 避免太小，基础大小200 + 数量*10
    node_sizes = [200 + nx.get_node_attributes(G, 'size')[node] * 10 for node in G.nodes()]
    
    # 节点颜色 (使用度中心性)
    d = dict(G.degree(weight='weight'))
    node_colors = list(d.values())
    
    # 绘图
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray')
    
    # 标签 (行业名称)
    # 自动调整标签位置防止重叠是个难题，这里简单向上偏移
    label_pos = {k: (v[0], v[1]+0.05) for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=9, font_weight='bold', font_family='sans-serif')
    
    plt.title(f"Industry Super-Node Network: {dataset_name} ({year})", fontsize=15)
    plt.axis('off')
    
    filename = os.path.join(PNG_DIR, f"{dataset_name}_{year}.png")
    plt.savefig(filename, format="PNG", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved PNG: {filename}")

def calculate_graph_metrics(G, year, dataset_name):
    """计算网络指标"""
    if len(G.nodes) == 0:
        return {}
        
    metrics = {
        "Dataset": dataset_name,
        "Year": year,
        "Nodes (Industries)": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Density": nx.density(G),
        "Avg Degree": np.mean([d for n, d in G.degree()]),
        "Avg Weighted Degree": np.mean([d for n, d in G.degree(weight='weight')]),
        "Avg Clustering": nx.average_clustering(G, weight='weight'),
        "Transitivity": nx.transitivity(G)
    }
    
    # 尝试计算连通分量相关指标
    if nx.is_connected(G):
        metrics["Avg Path Length"] = nx.average_shortest_path_length(G, weight='weight')
        metrics["Diameter"] = nx.diameter(G)
    elif len(G.nodes) > 1:
        # 如果不连通，计算最大连通子图
        largest_cc = max(nx.connected_components(G), key=len)
        if len(largest_cc) > 1:
            subG = G.subgraph(largest_cc)
            metrics["Avg Path Length"] = nx.average_shortest_path_length(subG, weight='weight')
            metrics["Diameter"] = nx.diameter(subG)
        else:
            metrics["Avg Path Length"] = 0
            metrics["Diameter"] = 0
    else:
        metrics["Avg Path Length"] = 0
        metrics["Diameter"] = 0
        
    return metrics

# ---------------- DYNAMIC HTML GENERATION ---------------- #

def build_dynamic_html_timeline(graphs_dict, dataset_name, output_path):
    """
    生成带时间滑块的 Plotly HTML
    graphs_dict: { year: NetworkX Graph }
    """
    years = sorted(graphs_dict.keys())
    if not years:
        return

    # 1. 准备基础 Figure
    fig = go.Figure()
    
    frames = []
    steps = []

    # 先画第一年的（作为初始状态）
    first_year = years[0]
    G_first = graphs_dict[first_year]
    pos_first = nx.spring_layout(G_first, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in G_first.edges():
        x0, y0 = pos_first[edge[0]]
        x1, y1 = pos_first[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in G_first.nodes():
        x, y = pos_first[node]
        node_x.append(x)
        node_y.append(y)
        # Tooltip 显示信息
        size_val = G_first.nodes[node].get('size', 10)
        deg = G_first.degree(node)
        info = f"Industry: {node}<br>Companies: {size_val}<br>Degree: {deg}"
        node_text.append(info)
        node_size.append(math.sqrt(size_val) * 5 + 5) # 调整大小比例
        node_color.append(deg)

    # 添加初始 Trace (Edges)
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Relations'
    ))

    # 添加初始 Trace (Nodes)
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n for n in G_first.nodes()], # 节点上直接显示名字
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='Degree Centrality',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        ),
        name='Industries'
    ))

    # 3. 构建 Frames (每一年的数据)
    for year in years:
        G = graphs_dict[year]
        pos = nx.spring_layout(G, seed=42) # 重新计算布局
        
        e_x, e_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            e_x.extend([x0, x1, None])
            e_y.extend([y0, y1, None])
            
        n_x, n_y = [], []
        n_text_hover = []
        n_labels = []
        n_sizes = []
        n_colors = []
        
        for node in G.nodes():
            x, y = pos[node]
            n_x.append(x)
            n_y.append(y)
            size_val = G.nodes[node].get('size', 10)
            deg = G.degree(node)
            n_text_hover.append(f"Year: {year}<br>Industry: {node}<br>Companies: {size_val}<br>Degree: {deg}")
            n_labels.append(node)
            n_sizes.append(math.sqrt(size_val) * 5 + 5)
            n_colors.append(deg)
            
        frame = go.Frame(
            data=[
                go.Scatter(x=e_x, y=e_y), # Update edges
                go.Scatter(
                    x=n_x, y=n_y, 
                    text=n_labels,
                    hovertext=n_text_hover,
                    marker=dict(color=n_colors, size=n_sizes)
                ) # Update nodes
            ],
            name=str(year)
        )
        frames.append(frame)
        
        # Slider step
        step = dict(
            method="animate",
            args=[
                [str(year)],
                {"frame": {"duration": 1000, "redraw": True},
                 "mode": "immediate",
                 "transition": {"duration": 500}}
            ],
            label=str(year)
        )
        steps.append(step)

    fig.frames = frames

    # 4. 布局设置 (Slider & Buttons)
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Year: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        title=f"Dynamic Industry Network - {dataset_name}",
        showlegend=False,
        width=1000,
        height=800,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.1, y=0, xanchor="right", yanchor="top",
            pad={"t": 50, "r": 10},
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 1000, "redraw": True}, 
                                       "fromcurrent": True, "transition": {"duration": 500}}]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False}, 
                                         "mode": "immediate", "transition": {"duration": 0}}])]
        )],
        sliders=sliders
    )

    fig.write_html(output_path)
    print(f"  -> Saved Dynamic HTML: {output_path}")

# ---------------- MAIN EXECUTION ---------------- #

def main():
    print("--- Starting Industry Network Generation ---")
    
    # 1. 加载行业映射
    print(f"Loading industry map from {INDUSTRY_FILE}...")
    industry_map = load_industry_mapping(INDUSTRY_FILE)
    if not industry_map:
        print("Error: Could not load industry map (Empty). Exiting.")
        return

    all_metrics = []

    # 2. 遍历三个数据集 (GLOBAL, RISK, RETURN)
    for ds_name, file_prefix in DATA_FILES.items():
        print(f"\nProcessing Dataset: {ds_name}")
        
        # 查找该类别下的所有年份文件
        pattern = get_file_pattern(file_prefix)
        files = glob.glob(pattern)
        
        # 排序年份
        try:
            files.sort(key=lambda x: int(x.split(' - ')[-1].replace('.csv', '')))
        except:
            files.sort() # fallback
            
        if not files:
            print(f"  No files found for pattern: {pattern}")
            continue

        graphs_by_year = {}

        for fpath in files:
            # 提取年份
            try:
                year_str = fpath.split(' - ')[-1].replace('.csv', '')
            except:
                year_str = "Unknown"
                
            print(f"  > Processing Year: {year_str}")
            
            # 核心：聚合处理
            df_agg, features = process_year_file(fpath, industry_map)
            
            if df_agg is None or df_agg.empty:
                print(f"    Skipping {year_str} (No valid data after merging)")
                continue
                
            # 核心：建图
            G = build_network_from_aggregated(df_agg, features, threshold=SIM_THRESHOLD)
            
            # 保存
            graphs_by_year[year_str] = G
            
            # 1. 静态图
            save_static_plot(G, year_str, ds_name)
            
            # 2. 计算指标
            m = calculate_graph_metrics(G, year_str, ds_name)
            all_metrics.append(m)

        # 3. 生成该数据集的动态时间轴 HTML
        if graphs_by_year:
            html_path = os.path.join(HTML_DIR, f"{ds_name}_dynamic_industry.html")
            build_dynamic_html_timeline(graphs_by_year, ds_name, html_path)

    # 4. 保存所有指标到 Excel
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        # 调整列顺序
        cols = ["Dataset", "Year", "Nodes (Industries)", "Edges", "Density", "Avg Degree"] + \
               [c for c in metrics_df.columns if c not in ["Dataset", "Year", "Nodes (Industries)", "Edges", "Density", "Avg Degree"]]
        metrics_df = metrics_df[cols]
        
        metrics_df.to_excel(METRICS_FILE, index=False)
        print(f"\n[SUCCESS] Metrics saved to {METRICS_FILE}")
    
    print("\n--- Processing Complete ---")
    print(f"Check the '{OUTPUT_DIR}' folder for results.")

if __name__ == "__main__":
    main()