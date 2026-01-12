import meshio
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.spatial import cKDTree
import torch

def gradinet(mesh):
    grad = np.zeros(mesh)

def map_triangle_to_2d(points_3d):
    """
    修改点：
    1. 移除所有.item()操作
    2. 全部使用张量运算保持梯度
    3. 统一数据类型为float64
    """
    dtype = torch.float64
    points_3d = points_3d.to(dtype)

    # 计算边长 (保持梯度)
    a = torch.norm(points_3d[1] - points_3d[0])
    b = torch.norm(points_3d[2] - points_3d[1])
    c = torch.norm(points_3d[0] - points_3d[2])

    # 添加微小量防止除零
    eps = torch.tensor(1e-10, dtype=dtype, device=points_3d.device)
    a_safe = torch.max(a, eps)
    c_safe = torch.max(c, eps)

    # 计算余弦值 (保持梯度)
    cos_theta = (a**2 + c**2 - b**2) / (2 * a_safe * c_safe)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # 计算正弦值 (保持梯度)
    sin_theta = torch.sqrt(1 - cos_theta**2 + eps)  # 添加数值稳定性

    # 构造2D坐标 (全部使用张量运算)
    p0 = torch.zeros(2, dtype=dtype, device=points_3d.device)
    p1 = torch.stack([a, torch.tensor(0.0, dtype=dtype, device=points_3d.device)])
    p2 = torch.stack([c * cos_theta, c * sin_theta])

    return torch.stack([p0, p1, p2])


def calculate_gradinet(points_2d, h_values):
    """
    修改点：
    1. 移除所有detach操作
    2. 保持所有操作用张量运算
    3. 统一数据类型
    """
    dtype = torch.float64
    device = points_2d.device
    
    x0, y0 = points_2d[0][0], points_2d[0][1]
    x1, y1 = points_2d[1][0], points_2d[1][1]
    x2, y2 = points_2d[2][0], points_2d[2][1]

    # 计算面积 (保持梯度)
    A = 0.5 * torch.abs(x0*(y1 - y2) + x1*(y2 - y0) + x2*(y0 - y1))
    A = A + 1e-10  # 防止除零

    # 计算系数 (保持梯度)
    b0 = y1 - y2
    b1 = y2 - y0
    b2 = y0 - y1
    
    c0 = x2 - x1
    c1 = x0 - x2
    c2 = x1 - x0

    # 构建K矩阵 (保持梯度)
    denominator = 4 * A**2
    k00 = (b0**2 + c0**2) / denominator
    k01 = (b0*b1 + c0*c1) / denominator
    k02 = (b0*b2 + c0*c2) / denominator
    
    k11 = (b1**2 + c1**2) / denominator
    k12 = (b1*b2 + c1*c2) / denominator
    
    k22 = (b2**2 + c2**2) / denominator

    K = torch.tensor([[k00, k01, k02],
                     [k01, k11, k12],
                     [k02, k12, k22]], 
                    dtype=dtype, device=device)

    # 保持H的梯度
    H = h_values.to(dtype).view(-1, 1)  # 关键修改！

    # 正确矩阵乘法顺序
    gradient_squared = (H.t() @ K @ H).squeeze()  # (1,1) -> 标量

    return gradient_squared


def hausdorff_distance(mesh1, mesh2):
    """
    计算两个网格之间的豪斯多夫距离
    :param mesh1: 第一个网格对象，包含 vertices (N1, 3)
    :param mesh2: 第二个网格对象，包含 vertices (N2, 3)
    :return: 豪斯多夫距离（标量张量，保留梯度）
    """
    points1 = mesh1.vertices  # 确保 requires_grad=True
    points2 = mesh2.vertices  # 确保 requires_grad=True

    # 计算所有点对之间的距离矩阵
    points1_expanded = points1.unsqueeze(1)  # 形状变为 (N1, 1, 3)
    points2_expanded = points2.unsqueeze(0)  # 形状变为 (1, N2, 3)
    distances = torch.sqrt(torch.sum((points1_expanded - points2_expanded) ** 2, dim=-1))  # 形状为 (N1, N2)

    # 计算 h(A, B)：从 points2 到 points1 的最大最小距离
    min_distances_AB, _ = torch.min(distances, dim=0)  # 形状为 (N2,)
    h_AB = torch.max(min_distances_AB)

    # 计算 h(B, A)：从 points1 到 points2 的最大最小距离
    min_distances_BA, _ = torch.min(distances, dim=1)  # 形状为 (N1,)
    h_BA = torch.max(min_distances_BA)

    # 豪斯多夫距离是两者的最大值
    hausdorff_dist = torch.max(h_AB, h_BA)

    return hausdorff_dist


