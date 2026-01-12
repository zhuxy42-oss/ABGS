import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
# 保持您的引用不变
from testing_gcn import create_example2, load_para
from options.test_options import TestOptions
from models.networks import EdgeRankingGNN, EdgeRankingGNN2
from data.simplification_data import gather_graph_normal1

def compute_grouped_feature_importance(model, loader, device, metric_func, node_feature_names, edge_feature_groups):
    """
    计算【点特征】和【成组的边特征】的置换重要性
    
    Args:
        edge_feature_groups: 一个字典或列表，定义了哪些列属于同一组。
                             格式示例: { "Inner Angles": [1, 2], "Dihedral": [0] }
    """
    model.eval()
    
    # 1. 计算基准分数 (Baseline Score)
    print("Computing baseline performance...")
    baseline_scores = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            score = metric_func(model, data) 
            baseline_scores.append(score.item())
    
    baseline_mean = np.mean(baseline_scores)
    print(f"Baseline Score (MSE Loss): {baseline_mean:.6f}")
    
    importances = {}
    importances_std = {}
    n_repeats = 5  # 重复次数
    
    print(f"{'Feature Group Name':<30} | {'Importance':<12} | {'Std Dev'}")
    print("-" * 60)

    # ==========================================
    # Part A: 分析点特征 (Node Features) - 保持不变(或是也改成Group模式，这里暂时按单列处理)
    # ==========================================
    for i, feature_name in enumerate(node_feature_names):
        diffs = []
        for _ in range(n_repeats):
            perm_scores = []
            for data in loader:
                data = data.to(device)
                original_x = data.x.clone()
                
                # 打乱第 i 列
                perm = torch.randperm(data.x.size(0))
                data.x[:, i] = data.x[perm, i]
                
                with torch.no_grad():
                    score = metric_func(model, data)
                    perm_scores.append(score.item())
                
                data.x = original_x # 恢复
            
            perm_mean = np.mean(perm_scores)
            diffs.append(perm_mean - baseline_mean)
        
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        importances[feature_name] = mean_diff
        importances_std[feature_name] = std_diff
        print(f"[Node] {feature_name:<23} | {mean_diff:.8f}     | {std_diff:.8f}")

    # ==========================================
    # Part B: 分析成组的边特征 (Grouped Edge Features)
    # ==========================================
    # 遍历定义的每一个组
    for group_name, col_indices in edge_feature_groups.items():
        diffs = []
        for _ in range(n_repeats):
            perm_scores = []
            for data in loader:
                data = data.to(device)
                original_edge_attr = data.edge_attr.clone() # 备份
                
                # === 核心修改：生成一个随机排列，应用到组内的所有列 ===
                # 这样既打破了与y的关系，又保持了组内特征(如Angle1和Angle2)的联合分布结构
                perm = torch.randperm(data.edge_attr.size(0))
                
                for col_idx in col_indices:
                    data.edge_attr[:, col_idx] = data.edge_attr[perm, col_idx]
                
                with torch.no_grad():
                    score = metric_func(model, data)
                    perm_scores.append(score.item())
                
                data.edge_attr = original_edge_attr # 恢复
            
            perm_mean = np.mean(perm_scores)
            diffs.append(perm_mean - baseline_mean)
        
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        importances[group_name] = mean_diff
        importances_std[group_name] = std_diff
        print(f"[Edge] {group_name:<23} | {mean_diff:.8f}     | {std_diff:.8f}")
        
    return importances, importances_std

# ==========================================
# 使用示例
# ==========================================

def my_metric_func(model, data):
    criterion = nn.MSELoss()
    out = model(data)
    label_regression = data.y
    loss = criterion(out, torch.tensor(label_regression[0]).to(device))
    return loss

# 1. 定义特征名称
node_feature_names = [
    "Node Size",    # 0
    "Node LBO"      # 1
]

# 2. 【核心修改】定义边特征的“组” (Feature Groups)
# 格式： "显示名称": [列索引列表]
edge_feature_groups = {
    "Dihedral Angle":        [0],       # 单个特征
    "Inner Angles (Comb)":   [1, 2],    # 【合并扰动】Inner Angle 1 & 2
    "Len/Height (Comb)":     [3, 4],    # 【合并扰动】Len/Height 1 & 2
    "Global Edge Ratio":     [5],       # 单个特征
    "Normal Angle":          [6],       # 单个特征
    "Edge LBO":              [7]        # 单个特征
}

# 3. 加载环境
vtk_path = "/home/zhuxunyang/coding/simply/datasets/sf0613/test/data/110_0.vtk"
opt = TestOptions().parse()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
loader, _ = gather_graph_normal1("/home/zhuxunyang/coding/simply/datasets/sf0613/val", opt, device, 1, load_para())

# 加载模型
model = EdgeRankingGNN2(node_in_dim=2, edge_in_dim=8, hidden_dim=64, num_layers=2).to(device=device)
checkpoint = torch.load(
    "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize3_1225.pth",
    weights_only=False
)
model.load_state_dict(checkpoint)

# 4. 运行分析 (传入新的 edge_feature_groups)
imps, imps_std = compute_grouped_feature_importance(
    model, 
    loader, 
    device, 
    my_metric_func, 
    node_feature_names, 
    edge_feature_groups 
)

# 5. 绘图
def plot_combined_importance(importances, importances_std):
    names = list(importances.keys())
    values = list(importances.values())
    errors = list(importances_std.values())
    
    # 区分颜色：点特征用橙色，边特征用蓝色
    colors = []
    for name in names:
        if name in node_feature_names:
            colors.append('#ff7f0e') # Orange for Nodes
        else:
            colors.append('#1f77b4') # Blue for Edges

    # 排序
    indices = np.argsort(values)
    sorted_names = [names[i] for i in indices]
    sorted_values = [values[i] for i in indices]
    sorted_errors = [errors[i] for i in indices]
    sorted_colors = [colors[i] for i in indices]
    
    plt.figure(figsize=(12, 7))
    plt.barh(sorted_names, sorted_values, xerr=sorted_errors, capsize=5, color=sorted_colors, edgecolor='black', alpha=0.8)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Node Features'),
        Patch(facecolor='#1f77b4', edgecolor='black', label='Edge Features (Grouped)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.xlabel('Importance (Increase in MSE Loss)')
    plt.title('Grouped Feature Importance')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

plot_combined_importance(imps, imps_std)