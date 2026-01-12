import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root_path = "/home/zhuxunyang/coding/simply/csv_data0105"
# 注意：请确保你的本地路径下有这些文件，否则运行会报错
# 为了演示，我假设数据已经读取
# ... (读取数据的代码保持不变) ...
# 这里保留你的原始读取逻辑
trad_5 = pd.read_csv(f'{root_path}/110_trad_convergence_data_5.csv')
trad_10 = pd.read_csv(f'{root_path}/110_trad_convergence_data_10.csv')
trad_12_5 = pd.read_csv(f'{root_path}/110_trad_convergence_data_12.5.csv')

gcn_5 = pd.read_csv(f'{root_path}/110_GCN_convergence_data_5.csv')
gcn_10 = pd.read_csv(f'{root_path}/110_GCN_convergence_data_10.csv')
gcn_12_5 = pd.read_csv(f'{root_path}/110_GCN_convergence_data_12.5.csv')

# 处理GCN数据
def process_gcn_element_num(df):
    df = df.copy()
    df['element_num'] = df['element_num'].astype(str).str.replace('[', '').str.replace(']', '').astype(float)
    return df

gcn_5 = process_gcn_element_num(gcn_5)
gcn_10 = process_gcn_element_num(gcn_10)
gcn_12_5 = process_gcn_element_num(gcn_12_5)

x_ticks = np.linspace(1700, 200, 10).astype(int)

# ================= 修改点 1：加大字号数值 =================
font_params = {
    'title_size': 32,      # 标题字体大小
    'label_size': 31,      # 坐标轴标签字体大小
    'x_tick_size': 30,     # x轴刻度标签字体大小
    'y_tick_size': 30,     # 【修改】y轴刻度标签字体大小 (原24 -> 改为34，或者更大)
    'legend_size': 28,     # 图例字体大小
}
# ========================================================

# 2. 创建子图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(32, 10)) 

trad_color = 'blue'
gcn_color = 'red'
trad_marker = 'o'
gcn_marker = 's'
line_width = 4  
marker_size = 10 

# --- 第一个图：5% ---
ax1.plot(trad_5['query_cell_num'], trad_5['element_num'], 
         color=trad_color, linewidth=line_width, label='LBO-ABMS', marker=trad_marker, markersize=marker_size)
ax1.plot(gcn_5['query_cell_num'], gcn_5['element_num'], 
         color=gcn_color, linewidth=line_width, label='GCN-ABMS', marker=gcn_marker, markersize=marker_size)
ax1.set_xticks(x_ticks)
ax1.set_xticklabels([f'{x}' for x in x_ticks], rotation=45, fontsize=font_params['x_tick_size'])

# ================= 修改点 2：应用到所有刻度 (which='both') =================
# 对于对数坐标(log scale)，加上 which='both' 可以确保主刻度和次刻度字体都生效
ax1.tick_params(axis='y', which='both', labelsize=font_params['y_tick_size'])
# ======================================================================

ax1.set_xlim(1700, 200)
ax1.set_xlabel('Query Cell Number', fontweight='bold', fontsize=font_params['label_size'])
ax1.set_ylabel('Element Number', fontweight='bold', fontsize=font_params['label_size'])
ax1.set_title('5% Data:Query Cell vs Element Number', fontweight='bold', pad=25, fontsize=font_params['title_size'])
ax1.legend(loc='best', frameon=True, fontsize=font_params['legend_size'])
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# --- 第二个图：10% ---
ax2.plot(trad_10['query_cell_num'], trad_10['element_num'], 
         color=trad_color, linewidth=line_width, label='LBO-ABMS', marker=trad_marker, markersize=marker_size)
ax2.plot(gcn_10['query_cell_num'], gcn_10['element_num'], 
         color=gcn_color, linewidth=line_width, label='GCN-ABMS', marker=gcn_marker, markersize=marker_size)
ax2.set_xticks(x_ticks)
ax2.set_xticklabels([f'{x}' for x in x_ticks], rotation=45, fontsize=font_params['x_tick_size'])

# 【修改】应用 Y 轴大字号
ax2.tick_params(axis='y', which='both', labelsize=font_params['y_tick_size'])

ax2.set_xlim(1700, 200)
ax2.set_xlabel('Query Cell Number', fontweight='bold', fontsize=font_params['label_size'])
ax2.set_ylabel('Element Number', fontweight='bold', fontsize=font_params['label_size'])
ax2.set_title('10% Data:Query Cell vs Element Number', fontweight='bold', pad=25, fontsize=font_params['title_size'])
ax2.legend(loc='best', frameon=True, fontsize=font_params['legend_size'])
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# --- 第三个图：12.5% ---
ax3.plot(trad_12_5['query_cell_num'], trad_12_5['element_num'], 
         color=trad_color, linewidth=line_width, label='LBO-ABMS', marker=trad_marker, markersize=marker_size)
ax3.plot(gcn_12_5['query_cell_num'], gcn_12_5['element_num'], 
         color=gcn_color, linewidth=line_width, label='GCN-ABMS', marker=gcn_marker, markersize=marker_size)
ax3.set_xticks(x_ticks)
ax3.set_xticklabels([f'{x}' for x in x_ticks], rotation=45, fontsize=font_params['x_tick_size'])

# 【修改】应用 Y 轴大字号
ax3.tick_params(axis='y', which='both', labelsize=font_params['y_tick_size'])

ax3.set_xlim(1700, 200)
ax3.set_xlabel('Query Cell Number', fontweight='bold', fontsize=font_params['label_size'])
ax3.set_ylabel('Element Number', fontweight='bold', fontsize=font_params['label_size'])
ax3.set_title('12.5% Data:Query Cell vs Element Number', fontweight='bold', pad=25, fontsize=font_params['title_size'])
ax3.legend(loc='best', frameon=True, fontsize=font_params['legend_size'])
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

plt.tight_layout()

# 3. 保存
save_path = "adaptive_by_ratio_HD_0105.pdf"
plt.savefig(save_path, dpi=600, bbox_inches='tight')
print(f"高清收敛曲线图已保存至: {save_path}")
plt.show()