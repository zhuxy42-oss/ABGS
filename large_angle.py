import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 设置与数据准备
# ==========================================

# 设置全局字体大小（加大字号）
plt.rcParams.update({
    'font.size': 14,          # 全局字体大小
    'axes.titlesize': 16,     # 标题字体大小
    'axes.labelsize': 14,     # 轴标签字体大小
    'xtick.labelsize': 12,    # X轴刻度字体大小
    'ytick.labelsize': 12,    # Y轴刻度字体大小
    'font.family': 'sans-serif', # 字体类型
    # 如果中文显示乱码，请解开下面这行的注释并设置你系统中有的支持中文的字体（如 SimHei, Microsoft YaHei 等）
    # 'font.family': ['SimHei'], 
})

# 生成网格数据 (背景场)
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

# 定义类似图中的圆锥场函数 (中心低，四周高)
# 中心大概在 (0.5, 0.45)
center_x, center_y = 0.5, 0.45
# Z = 距离 * 比例 + 偏移
Z = np.sqrt((X - center_x)**2 + (Y - center_y)**2) * 0.3 + 0.1

# 定义三角形的顶点坐标
# 绿色点 (Left)
p_green = np.array([0.3, 0.4])
# 蓝色点 (Right)
p_blue = np.array([0.7, 0.4])
# 红色点 (Bottom/Max Angle)
p_red = np.array([0.5, 0.27])
# 橙色点 (投影点/垂足)，位于绿蓝连线的中点
p_orange = np.array([0.5, 0.4])

# 计算这些点对应的 Z 值 (为了在 3D 图中贴合曲面或作为参考)
# 这里假设三角形是悬浮在空间中的，或者我们直接取函数表面的值
z_green = np.sqrt((p_green[0] - center_x)**2 + (p_green[1] - center_y)**2) * 0.3 + 0.1
z_blue = np.sqrt((p_blue[0] - center_x)**2 + (p_blue[1] - center_y)**2) * 0.3 + 0.1
z_red = np.sqrt((p_red[0] - center_x)**2 + (p_red[1] - center_y)**2) * 0.3 + 0.1
z_orange = np.sqrt((p_orange[0] - center_x)**2 + (p_orange[1] - center_y)**2) * 0.3 + 0.1

# ==========================================
# 2. 绘图开始
# ==========================================

fig = plt.figure(figsize=(16, 8))

# --- 左图：2D 等高线图 ---
ax1 = fig.add_subplot(1, 2, 1)

# 绘制等高线填充 (Contourf)
# levels=30 让颜色过渡更平滑，cmap='viridis_r' (反转) 使得中间深(紫)外围浅(黄)
# 原图中中间是紫色，外围黄色，这对应标准的 'viridis' (低值紫，高值黄)
contour = ax1.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.9)

# 绘制三角形连线
triangle_x = [p_green[0], p_blue[0], p_red[0], p_green[0]]
triangle_y = [p_green[1], p_blue[1], p_red[1], p_green[1]]
ax1.plot(triangle_x, triangle_y, color='red', linewidth=3, zorder=2)

# 绘制垂线 (红色点到橙色点)
ax1.plot([p_red[0], p_orange[0]], [p_red[1], p_orange[1]], color='red', linewidth=2, linestyle='-')

# 绘制顶点 (Scatter)
# zorder设高一点以保证点在在最上层
ax1.scatter(p_green[0], p_green[1], color='green', s=150, zorder=3, edgecolors='black')
ax1.scatter(p_blue[0], p_blue[1], color='blue', s=150, zorder=3, edgecolors='black')
ax1.scatter(p_orange[0], p_orange[1], color='orange', s=100, zorder=3, edgecolors='black')
ax1.scatter(p_red[0], p_red[1], color='red', s=200, zorder=3, edgecolors='black') # 红色点最大

# 设置轴标签和标题
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title("Triangle with Perpendicular from Max Angle Vertex\n(Max Angle: 118.1°)")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_aspect('equal')

# --- 右图：3D 曲面图 ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# 绘制曲面 (透明度 alpha=0.6)
surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, linewidth=0, antialiased=True)

# 绘制上方的一个平面 (模拟图中的淡黄色顶盖)
# Z_plane = np.full_like(X, np.max(Z))
# ax2.plot_surface(X, Y, Z_plane, color='yellow', alpha=0.1)

# 在 3D 中绘制三角形
# 我们将 2D 坐标映射到 3D 空间。原图中三角形似乎是在空间中浮动的
# 这里我们直接用计算出的 z_red, z_green 等坐标
ax2.plot([p_green[0], p_blue[0]], [p_green[1], p_blue[1]], [z_green, z_blue], color='brown', linewidth=2)
ax2.plot([p_blue[0], p_red[0]], [p_blue[1], p_red[1]], [z_blue, z_red], color='brown', linewidth=2)
ax2.plot([p_red[0], p_green[0]], [p_red[1], p_green[1]], [z_red, z_green], color='brown', linewidth=2)

# 绘制点
ax2.scatter(p_green[0], p_green[1], z_green, color='green', s=100, edgecolors='white')
ax2.scatter(p_blue[0], p_blue[1], z_blue, color='blue', s=100, edgecolors='white')
ax2.scatter(p_orange[0], p_orange[1], z_orange, color='orange', s=80, edgecolors='white')
ax2.scatter(p_red[0], p_red[1], z_red, color='red', s=120, edgecolors='white')

# 绘制箭头 (Vector)
# 从红色点指向橙色点 (或者反过来，参考原图看起来是从红点指出去的蓝色箭头)
# 原图是 3D 视角下的垂线向量
ax2.quiver(p_red[0], p_red[1], z_red, 
           p_orange[0]-p_red[0], p_orange[1]-p_red[1], z_orange-z_red,
           color='blue', arrow_length_ratio=0.1, linewidth=2.5)

# 设置 3D 轴标签
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Size Value')

# 调整视角以匹配原图
ax2.view_init(elev=35, azim=-100) # elev=仰角, azim=方位角

# 设置坐标轴范围
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_zlim(0.1, 0.3) # 根据 Z 的计算值调整

plt.tight_layout()
plt.savefig("large_angle.png", dpi=600)
plt.show()