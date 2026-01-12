import torch
import numpy as np
import sys
import os
import meshio
# import pyvista as pv
import ctypes
import subprocess
from scipy.spatial import cKDTree
from collections import defaultdict
from scipy.spatial import KDTree
from typing import Tuple
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from util.calculate_gradient import map_triangle_to_2d, calculate_gradinet
from util.bending import integrate_over_triangle
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import pyvista as pv
from trame_vtk.tools.vtksz2html import write_html
from scipy.sparse import coo_matrix, csr_matrix, diags
from models.layers.mesh import Mesh
import trimesh
from util.nlp_smooth import smooth_sizing_function
from util.qem import BatchQEMSimplifier
from util.xiao_nlp1 import smooth_mesh_sizing

def simplify_mesh_with_kdtree_projection(custom_mesh, target_face_count):
    """
    使用 PyVista 进行 QEM 坍缩，并使用 KDTree 将新面片的区域 ID (surface_id) 
    投影回原网格进行最近邻匹配。
    
    参数:
        custom_mesh: CustomMesh 对象
        target_face_count: 目标面片数量
        
    返回:
        new_mesh: 简化后的 CustomMesh 对象
    """
    
    # =========================================================
    # 1. 准备数据与 PyVista 对象
    # =========================================================
    # 转换为 Numpy (CPU)
    vertices_np = custom_mesh.vertices.cpu().numpy()
    faces_np = custom_mesh.faces.cpu().numpy()
    
    # 构造 PyVista 需要的 faces 格式 [3, v1, v2, v3, 3, ...]
    faces_pv = np.hstack([np.full((faces_np.shape[0], 1), 3), faces_np]).flatten()
    mesh_pv = pv.PolyData(vertices_np, faces_pv)
    
    # 绑定 Point Data (VTK 会自动插值这些属性)
    # A. 尺寸场
    if custom_mesh.sizing_values is not None:
        mesh_pv.point_data['sizing_values'] = custom_mesh.sizing_values.cpu().numpy().flatten()
    
    # B. 特征点 (转为 float 以便 VTK 插值，后续再阈值化)
    if custom_mesh.feature_point is not None:
        mesh_pv.point_data['feature_point'] = custom_mesh.feature_point.cpu().numpy().astype(np.float32)

    # =========================================================
    # 2. 准备原网格的 KDTree (用于 region ID 投影)
    # =========================================================
    # 计算原网格的所有面中心
    # faces_np shape: (F, 3), vertices_np shape: (V, 3)
    # 这种写法利用了 numpy 的高级索引，比循环快得多
    orig_face_centers = vertices_np[faces_np].mean(axis=1) # (F, 3)
    
    # 获取原网格的 regions
    if custom_mesh.surface_id is not None:
        orig_regions = custom_mesh.surface_id
        if torch.is_tensor(orig_regions):
            orig_regions = orig_regions.cpu().numpy()
        # 建立 KDTree
        tree = cKDTree(orig_face_centers)
    else:
        orig_regions = None
        tree = None

    # =========================================================
    # 3. 执行坍缩 (Decimation)
    # =========================================================
    current_faces = mesh_pv.n_faces_original if hasattr(mesh_pv, 'n_faces_original') else mesh_pv.n_cells
    if current_faces <= target_face_count:
        print("Target face count is higher than current face count. No simplification needed.")
        return custom_mesh
    
    reduction = 1.0 - (target_face_count / current_faces)
    
    # 使用 decimate_pro (QEM算法)
    # preserve_topology=True: 保持流形结构
    # feature_angle=45.0: 尝试保留特征边
    # splitting=False: 禁止网格分裂
    simplified_pv = mesh_pv.decimate_pro(reduction, preserve_topology=True, feature_angle=10.0, splitting=False)
    
    # =========================================================
    # 4. 提取数据并重建属性
    # =========================================================
    
    # A. 提取顶点
    new_vertices = torch.tensor(simplified_pv.points, dtype=torch.float32)
    
    # B. 提取面
    # simplified_pv.faces 格式也是 [3, v1, v2, v3, ...], 需要 reshape 解析
    if simplified_pv.n_faces > 0:
        new_faces_np = simplified_pv.faces.reshape(-1, 4)[:, 1:]
        new_faces = torch.tensor(new_faces_np, dtype=torch.long)
    else:
        # 极少数情况可能坍缩没了
        return None

    # C. 提取自动插值的尺寸场
    new_sizing_values = None
    if 'sizing_values' in simplified_pv.point_data:
        data = simplified_pv.point_data['sizing_values']
        new_sizing_values = torch.tensor(data, dtype=torch.float32)
        # 保持维度一致 (N, 1)
        if len(custom_mesh.sizing_values.shape) > 1:
            new_sizing_values = new_sizing_values.unsqueeze(1)

    # D. 提取并恢复特征点
    new_feature_points = None
    if 'feature_point' in simplified_pv.point_data:
        data = simplified_pv.point_data['feature_point']
        # 插值后可能会变成 0.3, 0.7 等，取 0.5 作为阈值恢复布尔值
        new_feature_points = torch.tensor(data > 0.5, dtype=torch.bool)

    # =========================================================
    # 5. 核心：使用 KDTree 投影传递 Region ID
    # =========================================================
    new_regions = None
    if tree is not None and orig_regions is not None:
        # 计算新网格的面中心
        new_verts_np = new_vertices.numpy()
        new_faces_indices = new_faces.numpy()
        new_face_centers = new_verts_np[new_faces_indices].mean(axis=1)
        
        # 查询最近邻 (k=1)
        # distances: 到最近点的距离, indices: 原网格中对应面的索引
        distances, indices = tree.query(new_face_centers, k=1)
        
        # 映射 ID
        new_regions_data = orig_regions[indices]
        
        # 转回 Tensor
        new_regions = torch.tensor(new_regions_data)
        
        # 可选：打印投影误差统计
        # print(f"Region projection max distance: {distances.max():.6f}")

    # =========================================================
    # 6. 构建新 CustomMesh 对象
    # =========================================================
    # 获取类引用 (假设在类外部使用，或者你可以直接写 CustomMesh)
    MeshClass = custom_mesh.__class__
    
    new_mesh = MeshClass(
        vertices=new_vertices,
        faces=new_faces,
        sizing_values=new_sizing_values,
        regions=new_regions,
        feature_points=new_feature_points,
        version=1 # 假设 version 1 跳过某些初始化以加速
    )
    
    return new_mesh

def generate_unique_surface_points():
    """
    在中心为(0,0,0)、边长为200的立方体表面以间距0.1进行无重复采样
    """
    # 生成坐标范围（从-100到100，步长0.1）
    coord = np.round(np.arange(-100, 100 + 0.05, 0.1), 1)
    n = len(coord)
    
    # 初始化点列表
    all_points = []
    
    # 1. 生成两个相对的面（x=±100），包括所有边界点
    Y, Z = np.meshgrid(coord, coord, indexing='ij')
    total_points = n * n
    
    # x=100 面
    points_x100 = np.zeros((total_points, 3))
    points_x100[:, 0] = 100
    points_x100[:, 1] = Y.flatten()
    points_x100[:, 2] = Z.flatten()
    all_points.append(points_x100)
    
    # x=-100 面
    points_x_minus100 = np.zeros((total_points, 3))
    points_x_minus100[:, 0] = -100
    points_x_minus100[:, 1] = Y.flatten()
    points_x_minus100[:, 2] = Z.flatten()
    all_points.append(points_x_minus100)
    
    # 2. 生成y=±100的面，但排除已经包含在x面中的边缘
    # y=100 面：排除x=±100和z=±100的边界（因为这些点已经在x面或z面）
    mask_y100 = ((Y != -100) & (Y != 100) & (Z != -100) & (Z != 100))
    points_y100 = np.zeros((np.sum(mask_y100), 3))
    points_y100[:, 0] = Y[mask_y100]
    points_y100[:, 1] = 100
    points_y100[:, 2] = Z[mask_y100]
    all_points.append(points_y100)
    
    # y=-100 面：同样排除边界
    points_y_minus100 = np.zeros((np.sum(mask_y100), 3))
    points_y_minus100[:, 0] = Y[mask_y100]
    points_y_minus100[:, 1] = -100
    points_y_minus100[:, 2] = Z[mask_y100]
    all_points.append(points_y_minus100)
    
    # 3. 生成z=±100的面，排除所有边界（这些点已经在前面的面中出现）
    # 创建内部点的掩码（排除所有边界）
    mask_internal = ((Y != -100) & (Y != 100) & (Z != -100) & (Z != 100))
    
    # z=100 面
    points_z100 = np.zeros((np.sum(mask_internal), 3))
    points_z100[:, 0] = Y[mask_internal]
    points_z100[:, 1] = Z[mask_internal]
    points_z100[:, 2] = 100
    all_points.append(points_z100)
    
    # z=-100 面
    points_z_minus100 = np.zeros((np.sum(mask_internal), 3))
    points_z_minus100[:, 0] = Y[mask_internal]
    points_z_minus100[:, 1] = Z[mask_internal]
    points_z_minus100[:, 2] = -100
    all_points.append(points_z_minus100)
    
    # 合并所有点
    all_points = np.vstack(all_points)
    
    return all_points

def calculate_uvw(point, tri_verts):
    """
    计算点相对于三角形的重心坐标
    """
    v0, v1, v2 = tri_verts
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = point - v0
    
    # 使用更稳定的方法计算重心坐标
    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        # 退化三角形，返回均匀坐标
        return np.array([1/3, 1/3, 1/3])
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    return np.array([u, v, w])

def barycentric_interpolation(point, face, mesh):
        """重心坐标插值"""
        v0 = mesh.vertices[face[0]].numpy()
        v1 = mesh.vertices[face[1]].numpy()
        v2 = mesh.vertices[face[2]].numpy()
        
        # 计算重心坐标
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = point - v0
        
        d00 = np.dot(v0v1, v0v1)
        d01 = np.dot(v0v1, v0v2)
        d11 = np.dot(v0v2, v0v2)
        d20 = np.dot(v0p, v0v1)
        d21 = np.dot(v0p, v0v2)
        
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / (denom + 1e-8)
        w = (d00 * d21 - d01 * d20) / (denom + 1e-8)
        u = 1 - v - w
        
        # 插值尺寸值
        sizes = [mesh.sizing_values[i].item() for i in face]
        if (u * sizes[0] + v * sizes[1] + w * sizes[2]) < 0.0001:
            print("warning")
        return u * sizes[0] + v * sizes[1] + w * sizes[2]

def barycentric_interpolation_batch(points, faces, mesh):
    """
    批量重心坐标插值
    
    Args:
        points: (N, 3) 点坐标数组
        faces: (N, 3) 面对应的顶点索引数组
        mesh: 网格对象，包含vertices和sizing_values
    
    Returns:
        interpolated_values: (N,) 插值结果数组
    """
    # 转换为numpy数组（如果输入是tensor）
    if torch.is_tensor(points):
        points = points.detach().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().numpy()
    
    # 获取所有需要的顶点
    all_vertex_indices = faces.reshape(-1)
    all_vertices = mesh.vertices[all_vertex_indices].numpy()
    
    # 重塑为 (N, 3, 3) - 每个面三个顶点的坐标
    v0 = all_vertices[0::3].reshape(-1, 3)  # 第一个顶点
    v1 = all_vertices[1::3].reshape(-1, 3)  # 第二个顶点
    v2 = all_vertices[2::3].reshape(-1, 3)  # 第三个顶点
    
    # 计算向量
    v0v1 = v1 - v0  # (N, 3)
    v0v2 = v2 - v0  # (N, 3)
    v0p = points - v0  # (N, 3)
    
    # 计算点积
    d00 = np.sum(v0v1 * v0v1, axis=1)  # (N,)
    d01 = np.sum(v0v1 * v0v2, axis=1)  # (N,)
    d11 = np.sum(v0v2 * v0v2, axis=1)  # (N,)
    d20 = np.sum(v0p * v0v1, axis=1)   # (N,)
    d21 = np.sum(v0p * v0v2, axis=1)   # (N,)
    
    # 计算分母
    denom = d00 * d11 - d01 * d01  # (N,)
    
    # 计算重心坐标 v, w
    v = (d11 * d20 - d01 * d21) / (denom + 1e-8)  # (N,)
    w = (d00 * d21 - d01 * d20) / (denom + 1e-8)  # (N,)
    u = 1 - v - w  # (N,)
    
    # 获取对应的尺寸值
    all_sizing_values = mesh.sizing_values
    
    # 重塑为每个面的三个尺寸值
    s0 = all_sizing_values[faces[:, 0]]  # (N,)
    s1 = all_sizing_values[faces[:, 1]]  # (N,)
    s2 = all_sizing_values[faces[:, 2]]  # (N,)
    
    # 插值计算
    interpolated_values = torch.from_numpy(u) * s0.squeeze(1) + torch.from_numpy(v) * s1.squeeze(1) + torch.from_numpy(w) * s2.squeeze(1)  # (N,)
    
    # # 检查异常值（可选）
    # small_values_mask = interpolated_values < 0.0001
    # if np.any(small_values_mask):
    #     print(f"Warning: {np.sum(small_values_mask)} points have very small interpolation values")
    
    return interpolated_values

def uvw(point, mesh, cell):
        index0, index1, index2 = cell

        v0 = (mesh.vertices[index1] - mesh.vertices[index0]).numpy()
        v1 = (mesh.vertices[index2] - mesh.vertices[index0]).numpy()
        v2 = (torch.from_numpy(point) - mesh.vertices[index0]).numpy()

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d02 = np.dot(v0, v2)
        d11 = np.dot(v1, v1)
        d12 = np.dot(v1, v2)
        denom = d00 * d11 - d01 * d01

        v = (d11 * d02 - d01 * d12) / (denom + 1e-8)
        w = (d00 * d12 - d01 * d02) / (denom + 1e-8)
        u = 1.0 - v - w
        return u, v, w

def uvw_vectorized(points, mesh, cells):
    """
    向量化版本的uvw函数，同时计算多个点相对于对应三角形的重心坐标
    
    参数:
        points: (N, 3) numpy数组，N个点的坐标
        mesh: 网格对象，包含vertices属性
        cells: (N, 3) numpy数组，每个点对应的三角形顶点索引
    
    返回:
        uvw_coords: (N, 3) numpy数组，每个点的重心坐标(u, v, w)
    """
    # 将网格顶点转换为numpy数组
    vertices = mesh.vertices.numpy()
    
    # 提取三角形的三个顶点索引
    indices_0 = cells[:, 0]  # (N,)
    indices_1 = cells[:, 1]  # (N,)
    indices_2 = cells[:, 2]  # (N,)
    
    # 获取顶点坐标
    v0_points = vertices[indices_0]  # (N, 3) 三角形第一个顶点
    v1_points = vertices[indices_1]  # (N, 3) 三角形第二个顶点  
    v2_points = vertices[indices_2]  # (N, 3) 三角形第三个顶点
    
    # 计算向量
    v0 = v1_points - v0_points  # (N, 3) 边向量 v0 = v1 - v0
    v1 = v2_points - v0_points  # (N, 3) 边向量 v1 = v2 - v0
    v2 = points - v0_points     # (N, 3) 点相对于v0的向量
    
    # 计算点积
    d00 = np.einsum('ij,ij->i', v0, v0)  # (N,) v0·v0
    d01 = np.einsum('ij,ij->i', v0, v1)  # (N,) v0·v1
    d02 = np.einsum('ij,ij->i', v0, v2)  # (N,) v0·v2
    d11 = np.einsum('ij,ij->i', v1, v1)  # (N,) v1·v1
    d12 = np.einsum('ij,ij->i', v1, v2)  # (N,) v1·v2
    
    # 计算分母
    denom = d00 * d11 - d01 * d01  # (N,)
    
    # 计算v和w坐标
    v_coord = (d11 * d02 - d01 * d12) / (denom + 1e-8)  # (N,)
    w_coord = (d00 * d12 - d01 * d02) / (denom + 1e-8)  # (N,)
    u_coord = 1.0 - v_coord - w_coord  # (N,)
    
    # 组合成(N, 3)数组
    uvw_coords = np.column_stack((u_coord, v_coord, w_coord))
    
    return uvw_coords

def angle_between_normals(normal1: torch.Tensor, normal2: torch.Tensor, degrees: bool = False) -> float:
    """
    计算两个法向量的夹角

    参数:
        normal1 (torch.Tensor): 第一个法向量 (3,)
        normal2 (torch.Tensor): 第二个法向量 (3,)
        degrees (bool): 是否返回角度制（默认弧度制）

    返回:
        float: 夹角（弧度或角度）
    """

    # 将numpy数组转换为torch.Tensor
    if isinstance(normal1, np.ndarray):
        normal1 = torch.from_numpy(normal1)
    if isinstance(normal2, np.ndarray):
        normal2 = torch.from_numpy(normal2)    

    # 确保输入是单位向量
    normal1 = normal1 / (torch.norm(normal1, p=2) + 1e-8)
    normal2 = normal2 / (torch.norm(normal2, p=2) + 1e-8)
    
    # 计算点积并夹紧到[-1, 1]（避免浮点误差）
    dot = torch.clamp(torch.dot(normal1, normal2), -1.0, 1.0)
    
    # 计算夹角
    angle_rad = torch.acos(dot)
    
    if degrees:
        return math.degrees(angle_rad.item())
    else:
        return angle_rad.item()

def angle_between_normals_batch(normals1: torch.Tensor, normals2: torch.Tensor, degrees: bool = False) -> torch.Tensor:
    """
    批量计算法向夹角的向量化版本
    
    参数:
        normals1 (torch.Tensor): (n, 3) 法向量组
        normals2 (torch.Tensor): (n, 3) 法向量组
        degrees (bool): 是否返回角度制
        
    返回:
        torch.Tensor: (n,) 夹角张量
    """
    # 批量归一化
    normals1 = normals1 / (torch.norm(normals1, p=2, dim=1, keepdim=True) + 1e-8)
    normals2 = normals2 / (torch.norm(normals2, p=2, dim=1, keepdim=True) + 1e-8)
    
    # 批量点积
    dots = (normals1 * normals2).sum(dim=1)
    dots = torch.clamp(dots, -1.0, 1.0)
    
    # 计算夹角
    angles = torch.acos(dots)
    if degrees:
        angles = torch.rad2deg(angles)
    
    return angles

def _overlap_on_axis(tri1: np.ndarray, tri2: np.ndarray, axis: np.ndarray) -> bool:
    """
    检查两个三角形在给定轴上的投影是否重叠
    
    参数:
        tri1 (np.ndarray): 第一个三角形
        tri2 (np.ndarray): 第二个三角形
        axis (np.ndarray): 投影轴
        
    返回:
        bool: 如果投影重叠返回True，否则返回False
    """
    # 归一化轴
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    
    # 计算三角形1在轴上的投影
    proj1 = [np.dot(v, axis) for v in tri1]
    min1, max1 = min(proj1), max(proj1)
    
    # 计算三角形2在轴上的投影
    proj2 = [np.dot(v, axis) for v in tri2]
    min2, max2 = min(proj2), max(proj2)
    
    # 检查投影是否重叠
    return not (max1 < min2 or max2 < min1)

def _coplanar_triangles_overlap(tri1: np.ndarray, tri2: np.ndarray) -> bool:
    """
    检查共面三角形是否重叠
    
    参数:
        tri1 (np.ndarray): 第一个三角形
        tri2 (np.ndarray): 第二个三角形
        
    返回:
        bool: 如果共面三角形重叠返回True，否则返回False
    """
    # 将3D问题转化为2D问题
    normal = np.cross(tri1[1] - tri1[0], tri1[2] - tri1[0])
    axis = np.eye(3)[np.argmax(np.abs(normal))]  # 选择最大分量对应的轴
    
    # 创建旋转矩阵
    if axis[0] == 1:  # x轴最大
        u = (tri1[1] - tri1[0])[:2]
        v = (tri1[2] - tri1[0])[:2]
    elif axis[1] == 1:  # y轴最大
        u = (tri1[1] - tri1[0])[::2]
        v = (tri1[2] - tri1[0])[::2]
    else:  # z轴最大
        u = (tri1[1] - tri1[0])[:2]
        v = (tri1[2] - tri1[0])[:2]
    
    # 检查三角形2的点是否在三角形1内
    for point in tri2:
        if _point_in_triangle(point, tri1):
            return True
    
    # 检查三角形1的点是否在三角形2内
    for point in tri1:
        if _point_in_triangle(point, tri2):
            return True
    
    # 检查边相交
    edges1 = [(tri1[0], tri1[1]), (tri1[1], tri1[2]), (tri1[2], tri1[0])]
    edges2 = [(tri2[0], tri2[1]), (tri2[1], tri2[2]), (tri2[2], tri2[0])]
    
    for e1 in edges1:
        for e2 in edges2:
            if _edges_intersect(e1, e2):
                return True
    
    return False

def _point_in_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    """
    检查点是否在三角形内(2D或3D)
    
    参数:
        point (np.ndarray): 要检查的点
        triangle (np.ndarray): 三角形顶点
        
    返回:
        bool: 如果点在三角形内返回True，否则返回False
    """
    # 使用重心坐标法
    v0 = triangle[2] - triangle[0]
    v1 = triangle[1] - triangle[0]
    v2 = point - triangle[0]
    
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    return (u >= 1e-6) and (v >= 1e-6) and (u + v <= 1)

def _edges_intersect(edge1: Tuple[np.ndarray, np.ndarray], 
                    edge2: Tuple[np.ndarray, np.ndarray]) -> bool:
    """
    检查两条线段是否相交(2D)
    
    参数:
        edge1: 第一条线段的两个端点
        edge2: 第二条线段的两个端点
        
    返回:
        bool: 如果线段相交返回True，否则返回False
    """
    p1, p2 = edge1
    p3, p4 = edge2
    
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)


def is_vertex_appearing_more_than_twice(feature_points, edge_index, v):
    # mask = (edge_index[0] == v) | (edge_index[1] == v)
    # connected_edges = edge_index[:, mask]
    mask = (edge_index[:, 0] == v) | (edge_index[:, 1] == v)
    connected_edges = edge_index[mask]
    neighbors = connected_edges[connected_edges != v].unique()
    feature_count = feature_points[neighbors].sum().item()
    return feature_count

def is_vertex_appearing_more_than_twice_vec(feature_points, edge_index, vs):
    """
    向量化版本，支持输入多个顶点进行批量处理
    
    参数:
        feature_points: 特征点数组
        edge_index: 边索引数组，形状为(N, 2)
        vs: 多个顶点组成的数组/列表
        
    返回:
        每个顶点对应的特征计数数组
    """
    # 确保输入是numpy数组
    vs = np.asarray(vs)
    edge_index = np.asarray(edge_index)
    feature_points = np.asarray(feature_points)
    
    # 获取所有顶点和边的数量
    num_vs = len(vs)
    num_edges = edge_index.shape[0]
    
    # 创建掩码矩阵: [顶点数量, 边数量]，表示每条边是否包含对应顶点
    # 维度为 (num_vs, num_edges)
    mask = (edge_index[:, 0] == vs[:, np.newaxis]) | (edge_index[:, 1] == vs[:, np.newaxis])
    
    # 计算每个顶点的邻居特征总和
    feature_counts = np.zeros(num_vs, dtype=int)
    
    for i in range(num_vs):
        # 获取与当前顶点相连的边
        connected_edges = edge_index[mask[i]]
        # 提取邻居顶点并去重
        neighbors = connected_edges[connected_edges != vs[i]]
        if len(neighbors) > 0:
            # 计算邻居中特征点的总数
            feature_counts[i] = feature_points[neighbors].sum()
    
    return feature_counts

class EmptyCustomMesh:
    """表示无效网格的空类（非流形边情况）"""
    def __init__(self):
        self.is_empty = True  # 附加标识属性，方便检测

    def __bool__(self):
        """自定义布尔值，便于条件判断"""
        return False

class BaseMesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

class CustomMesh(BaseMesh):
    def __init__(self, vertices, faces, edge_topology=None, sizing_values=None, regions=None, feature_points=None, version=0):
        """
        初始化 CustomMesh 类。

        参数:
        vertices (torch.Tensor): 顶点坐标张量。
        faces (torch.Tensor): 面拓扑张量。
        edge_topology (torch.Tensor, 可选): 边拓扑张量。默认为 None。
        sizing_values (torch.Tensor, 可选): 尺寸值张量。默认为 None。
        """
        super().__init__(vertices=vertices, faces=faces)
        self.is_empty = False
        if edge_topology is None:
            self.create_edge_index1()
        else:
            self.edges = edge_topology
        self.sizing_values = sizing_values if sizing_values is not None else None

        if regions is None:
            self.surface_id = np.ones(shape=(len(self.faces), 1))
        else:
            self.surface_id = regions
        self._edge_face_map = self._build_edge_face_map()  # 边到面片的映射缓存
        self._vertex_face_map = self._build_vertex_face_map1()
        self.face_normal, self.vertex_normal = self.compute_normals_with_pyvista()

        if version == 0:
            self._vertex_face_vectors = self._build_vertex_face_vetcor_map1()
            self.convex = self.classify_vertex_convexity1()
            if feature_points is None:
                self.feature_point, _ = self.calculate_feature_points1()
            else:
                self.feature_point = feature_points
        
        # 预计算一些用于微分算子的属性
        self._cotangent_weights = None
        self._voronoi_areas = None
        self._laplace_matrix = None

    def __getattr__(self, name):
        if name not in self.__dict__:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return self.__dict__[name]

# class CustomMesh(BaseMesh):
#     def __init__(self, vertices, faces, edge_topology=None, sizing_values=None, regions=None, feature_points=None, version=0):
#         """
#         初始化 CustomMesh 类，并记录各阶段耗时。
#         """
#         self.timings = {}  # 存储各阶段耗时
#         total_start = time.time()
        
#         # 1. 调用父类初始化
#         super_start = time.time()
#         super().__init__(vertices=vertices, faces=faces)
#         self.timings['super_init'] = time.time() - super_start
        
#         self.is_empty = False
        
#         # 2. 边拓扑处理
#         edge_start = time.time()
#         if edge_topology is None:
#             self.create_edge_index1()
#         else:
#             self.edges = edge_topology
#         self.timings['edge_topology'] = time.time() - edge_start
        
#         # 3. 尺寸值处理
#         sizing_start = time.time()
#         self.sizing_values = sizing_values if sizing_values is not None else None
#         self.timings['sizing_values'] = time.time() - sizing_start
        
#         # 4. 表面ID处理
#         region_start = time.time()
#         self.surface_id = regions
#         self.timings['regions'] = time.time() - region_start
        
#         # 5. 边面映射构建
#         edge_face_start = time.time()
#         self._edge_face_map = self._build_edge_face_map()  # 边到面片的映射缓存
#         self.timings['edge_face_map'] = time.time() - edge_face_start
        
#         # 6. 顶点面映射构建
#         vertex_face_start = time.time()
#         self._vertex_face_map = self._build_vertex_face_map()
#         self.timings['vertex_face_map'] = time.time() - vertex_face_start
        
#         # 7. 法向量计算
#         normal_start = time.time()
#         self.face_normal, self.vertex_normal = self.compute_normals_with_pyvista()
#         self.timings['normals'] = time.time() - normal_start
        
#         if version == 0:
#             # 8. 顶点面向量构建
#             vertex_vector_start = time.time()
#             self._vertex_face_vectors = self._build_vertex_face_vetcor_map1()
#             self.timings['vertex_face_vectors'] = time.time() - vertex_vector_start

#             # 9. 顶点凸性分类
#             convex_start = time.time()
#             # self.convex = self.classify_vertex_convexity_fully_vectorized()
#             self.convex = self.classify_vertex_convexity1()
#             self.timings['convexity'] = time.time() - convex_start
        
            
#             # 10. 特征点计算
#             feature_start = time.time()
#             if feature_points is None:
#                 self.feature_point, _ = self.calculate_feature_points1()
#             else:
#                 self.feature_point = feature_points
#             self.timings['feature_points'] = time.time() - feature_start
        
#         # 总时间
#         self.timings['total'] = time.time() - total_start
        
#         # 打印性能报告
#         self._print_timing_report()

#     def _print_timing_report(self):
#         """打印详细的性能分析报告"""
#         print("\n" + "="*60)
#         print("CustomMesh 初始化性能分析报告")
#         print("="*60)
        
#         # 获取网格基本信息
#         n_vertices = len(self.vertices) if hasattr(self, 'vertices') else 0
#         n_faces = len(self.faces) if hasattr(self, 'faces') else 0
#         n_edges = len(self.edges) if hasattr(self, 'edges') else 0
        
#         print(f"网格规模: {n_vertices} 顶点, {n_faces} 面, {n_edges} 边")
#         print(f"总耗时: {self.timings['total']:.6f} 秒")
#         print("\n各阶段详细耗时:")
#         print("-" * 50)
        
#         # 按耗时排序
#         sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
#         for stage, duration in sorted_timings:
#             if stage != 'total':
#                 percentage = (duration / self.timings['total']) * 100
#                 print(f"{stage:20s}: {duration:8.6f}s ({percentage:6.2f}%)")
        
#         # 识别性能瓶颈
#         bottlenecks = [stage for stage, duration in sorted_timings 
#                       if stage != 'total' and duration > 0.1 and duration / self.timings['total'] > 0.1]
        
#         if bottlenecks:
#             print(f"\n性能瓶颈: {', '.join(bottlenecks)}")
        
#         print("="*60)

#     def get_timing_summary(self):
#         """获取时间统计摘要"""
#         return {
#             'total_time': self.timings['total'],
#             'stage_times': {k: v for k, v in self.timings.items() if k != 'total'},
#             'vertex_count': len(self.vertices) if hasattr(self, 'vertices') else 0,
#             'face_count': len(self.faces) if hasattr(self, 'faces') else 0,
#             'edge_count': len(self.edges) if hasattr(self, 'edges') else 0
#         }

    @classmethod
    def from_vtk(cls, vtk_path: str):
        # 使用 meshio 读取 VTK 文件
        mesh = meshio.read(vtk_path)

        # 转换顶点数据的字节序
        vertices = mesh.points
        if vertices.dtype.byteorder != '=':
            # 交换字节序并设置为本地字节序
            vertices = vertices.byteswap().view(vertices.dtype.newbyteorder('='))
        vertices = torch.tensor(vertices, dtype=torch.float32)

        # 检查是否有单元数据
        if len(mesh.cells) == 0:
            raise ValueError("VTK 文件不包含任何单元数据")

        # 优化面数据提取
        faces = []
        for cell_block in mesh.cells:
            if cell_block.type == 'triangle':
                # 预分配数组大小
                faces_arr = np.ascontiguousarray(cell_block.data, dtype=np.int64)
                faces_tensor = torch.from_numpy(faces_arr)
                faces.append(faces_tensor)
        faces = faces[0]
        
        unique_counts = torch.tensor([torch.unique(tri).size(0) for tri in faces])
        non_degenerate_mask = unique_counts == 3
        faces = faces[non_degenerate_mask]

        try:
            surface_id = np.concatenate([d for d in mesh.cell_data.get('surface_id', [])])
            surface_id = surface_id[non_degenerate_mask]
        except:
            surface_id = np.concatenate([d for d in mesh.cell_data.get('regions', [])])
            surface_id = np.expand_dims(surface_id, axis=1)
            surface_id = surface_id[non_degenerate_mask]

        # 获取尺寸值数据
        sizing_values = None
        # for name in ['sizing_value', 'vertex_size']:
        for name in ['sizing_value']:
            if name in mesh.point_data:
                size = np.ascontiguousarray(mesh.point_data[name], dtype=np.float32)
                sizing_values = torch.from_numpy(size)
                break

        mesh = cls(vertices=vertices, faces=faces, sizing_values=sizing_values, regions=surface_id)
        return mesh

    def clone_mesh(self):
        new_mesh = CustomMesh(self.vertices, self.faces, self.edges, self.sizing_values, version=1)
        return new_mesh

    def get_all_info(self):
        """
        获取点、面拓扑、边拓扑和尺寸信息。

        返回:
        tuple: 包含顶点、面、边拓扑和尺寸值的元组。
        """
        if self.sizing_values is None:
            print("There has ", len(self.vertices), " vertices ", len(self.faces), " faces ", len(self.edges), " edges ", "There has no size value")
        else:
            print("There has ", len(self.vertices), " vertices ", len(self.faces), " faces ", len(self.edges), " edges ", "There has size value")

    def get_surface_area(self):
        areas = torch.zeros(size=(len(self.faces), ))
        i = 0
        for tri in self.faces:
            v0, v1, v2 = self.vertices[tri]
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross_product = torch.linalg.cross(edge1, edge2)
            area = 0.5 * torch.linalg.vector_norm(cross_product)
            areas[i] = area
            i = i + 1
        return areas

    def get_vertex_normal_batch(self, vertex_idxes: int) -> torch.Tensor:
        # 1. 找到包含该顶点的所有面片
        connected_faces = {}
        normals = []
        faces_np = np.array(self.faces)
        for id in vertex_idxes:
            contains_id = (faces_np == id).any(axis=1)
            # 找到所有包含当前顶点的面片索引
            face_indices = np.where(contains_id)[0]
            # 将结果存入字典
            connected_faces[id.item()] = face_indices.tolist()

        result_normal = {}

        mesh_center = torch.mean(self.vertices, dim=0)

        for id in vertex_idxes:
            normals = []
            related_face = connected_faces[id.item()]
            for face_idx in related_face:
                # 获取面片的三个顶点
                v0, v1, v2 = self.vertices[self.faces[face_idx]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                
                # 计算面法向（未归一化）
                face_normal = torch.linalg.cross(edge1, edge2)
                
                # 计算面中心位置
                face_center = (v0 + v1 + v2) / 3
                
                # 计算从顶点到面中心的向量
                to_face_center = face_center - mesh_center
                
                # 调整法向方向：确保法向与顶点到面中心向量的夹角小于90度
                if torch.dot(face_normal, to_face_center) < 0:
                    face_normal = -face_normal
                
                
                normals.append(face_normal)
            
            normals = torch.stack(normals)
            
            # 3. 计算加权或非加权平均

            vertex_normal = normals.mean(dim=0)
            
            # 4. 归一化结果
            vertex_normal = vertex_normal / (torch.norm(vertex_normal, p=2) + 1e-8)
            result_normal[id.item()] = vertex_normal
        
        return result_normal
    
    def compute_edge_features(self):
        """
        计算所有边特征并合并为 (n, 7) 的张量
        
        Returns:
            torch.Tensor: 形状为 (n_edges, 7) 的特征矩阵，列顺序为:
                0: 二面角
                1-2: 两个对角
                3-4: 两个高基比
                5: 全局边比率
                6: 顶点法向量夹角
        """
        edge_points = self._get_edge_points()
        
        # 计算各特征
        edges_lengths = np.linalg.norm(self.vertices[edge_points[:, 0]] - self.vertices[edge_points[:, 1]], 
                                     ord=2, axis=1) + 1e-8
        dihedral = self._compute_dihedral_angle(edge_points)          # (n,)
        opposite_angles = self._compute_opposite_angles(edge_points)  # (n, 2)
        face_ratios = self._compute_face_ratios(edge_points)         # (n, 2)
        global_ratio = self._compute_global_edge_ratio()             # (n,)
        normal_angle = angle_between_normals_batch(                  # (n,)
            self.vertex_normal[edge_points[:, 0]], 
            self.vertex_normal[edge_points[:, 1]], 
            True
        )
        # print(np.max(dihedral), np.min(dihedral))

        
        # 转换为PyTorch张量（如果尚未是张量）
        if not isinstance(dihedral, torch.Tensor):
            dihedral = torch.from_numpy(dihedral).float()
        if not isinstance(opposite_angles, torch.Tensor):
            opposite_angles = torch.from_numpy(opposite_angles).float()
        if not isinstance(face_ratios, torch.Tensor):
            face_ratios = torch.from_numpy(face_ratios).float()
        if not isinstance(global_ratio, torch.Tensor):
            global_ratio = torch.from_numpy(global_ratio).float()
        if not isinstance(normal_angle, torch.Tensor):
            normal_angle = torch.from_numpy(normal_angle).float()
        
        # 合并所有特征 (n, 1+2+2+1+1 = 7)
        features = torch.cat([
            dihedral.unsqueeze(1),            # (n,1)
            opposite_angles,                   # (n,2)
            face_ratios,                       # (n,2)
            global_ratio.unsqueeze(1),        # (n,1)
            normal_angle.unsqueeze(1)         # (n,1)
        ], dim=1)
        
        return features.transpose(1, 0)
    
    def compute_edge_features1(self):
        """
        Compute all edge features including:
        - Dihedral angle between adjacent faces
        - Two opposite angles in adjacent faces
        - Two edge-length ratios (height/base) for adjacent faces
        - One global edge length ratio (normalized by average edge length)
        
        Returns:
            dict: Dictionary containing all edge features
        """
        edge_points = self._get_edge_points()
        
        features = {}
        
        # 1. Dihedral angles
        features['dihedral_angle'] = self._compute_dihedral_angle(edge_points)
        
        # 2. Two opposite angles in adjacent faces
        features['opposite_angles'] = self._compute_opposite_angles(edge_points)
        
        # 3. Two edge-length ratios (height/base) for adjacent faces
        features['face_ratios'] = self._compute_face_ratios(edge_points)
        
        # 4. Global edge length ratio (normalized by average edge length)
        features['global_edge_ratio'] = self._compute_global_edge_ratio()

        features['edge_vertex_normal_angle'] = angle_between_normals_batch(self.vertex_normal[edge_points[:, 0]], self.vertex_normal[edge_points[:, 1]], True)
        
        return features
    
    def _get_edge_points(self):
        """Returns edge_points (#E x 4) tensor, with four vertex ids per edge
           edge_points[edge_id, 0] and edge_points[edge_id, 1] are the two vertices defining the edge
           edge_points[edge_id, 2] and edge_points[edge_id, 3] are the opposite vertices in adjacent faces
        """
        edge_points = np.zeros([len(self.edges), 4], dtype=np.int32)
        
        for edge_id, edge in enumerate(self.edges):
            v0, v1 = edge
            # Find faces sharing this edge
            # faces = self._edge_face_map.get(tuple(sorted((v0, v1))), [])
            key = tuple(sorted((v0.item(), v1.item())))
            faces = self._edge_face_map[key]
            
            if len(faces) >= 1:
                face1 = self.faces[faces[0]]
                opp_vertex1 = [v for v in face1 if v not in edge][0]
                edge_points[edge_id, 2] = opp_vertex1
                
            if len(faces) >= 2:
                face2 = self.faces[faces[1]]
                opp_vertex2 = [v for v in face2 if v not in edge][0]
                edge_points[edge_id, 3] = opp_vertex2
                
            edge_points[edge_id, 0] = v0
            edge_points[edge_id, 1] = v1
            
        return edge_points
    
    def _compute_dihedral_angle(self, edge_points):
        """Compute dihedral angle between adjacent faces for each edge"""
        normals_a = self._get_normals(edge_points, 0)  # normal of first face
        normals_b = self._get_normals(edge_points, 3)  # normal of second face
        
        # Handle edges with only one adjacent face
        valid_edges = (edge_points[:, 3] != 0)  # edges with two adjacent faces
        
        dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
        angles = np.zeros(len(edge_points))
        angles[valid_edges] = np.pi - np.arccos(dot[valid_edges])
        
        return angles
    
    def _get_normals(self, edge_points, side):
        """Get normals for faces adjacent to edges"""
        vertices = self.vertices.detach().cpu().numpy() if torch.is_tensor(self.vertices) else self.vertices
        
        edge_a = vertices[edge_points[:, side // 2 + 2]] - vertices[edge_points[:, side // 2]]
        edge_b = vertices[edge_points[:, 1 - side // 2]] - vertices[edge_points[:, side // 2]]
        
        normals = np.cross(edge_a, edge_b)
        div = self._fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
        normals /= div[:, np.newaxis]
        return normals
    
    def _compute_opposite_angles(self, edge_points):
        """Compute two opposite angles in adjacent faces for each edge"""
        vertices = self.vertices.detach().cpu().numpy() if torch.is_tensor(self.vertices) else self.vertices
        
        angles = np.zeros((len(edge_points), 2))
        
        # First face angle
        edges_a = vertices[edge_points[:, 0]] - vertices[edge_points[:, 2]]
        edges_b = vertices[edge_points[:, 1]] - vertices[edge_points[:, 2]]
        
        edges_a /= self._fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
        edges_b /= self._fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
        dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
        angles[:, 0] = np.arccos(dot)
        
        # Second face angle (for edges with two adjacent faces)
        valid_edges = (edge_points[:, 3] != 0)
        if np.any(valid_edges):
            edges_a = vertices[edge_points[valid_edges, 0]] - vertices[edge_points[valid_edges, 3]]
            edges_b = vertices[edge_points[valid_edges, 1]] - vertices[edge_points[valid_edges, 3]]
            
            edges_a /= self._fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
            edges_b /= self._fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
            dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
            angles[valid_edges, 1] = np.arccos(dot)
        
        return angles
    
    def _compute_face_ratios(self, edge_points):
        """Compute two edge-length ratios (height/base) for adjacent faces"""
        vertices = self.vertices.detach().cpu().numpy() if torch.is_tensor(self.vertices) else self.vertices
        
        ratios = np.zeros((len(edge_points), 2))
        
        # Edge lengths (base)
        edges_lengths = np.linalg.norm(vertices[edge_points[:, 0]] - vertices[edge_points[:, 1]], 
                                     ord=2, axis=1) + 1e-8
        
        # First face ratio
        point_o = vertices[edge_points[:, 2]]
        point_a = vertices[edge_points[:, 0]]
        point_b = vertices[edge_points[:, 1]]
        ratios[:, 0] = self._compute_height_base_ratio(point_o, point_a, point_b, edges_lengths)
        
        # Second face ratio (for edges with two adjacent faces)
        valid_edges = (edge_points[:, 3] != 0)
        if np.any(valid_edges):
            point_o = vertices[edge_points[valid_edges, 3]]
            ratios[valid_edges, 1] = self._compute_height_base_ratio(
                point_o, 
                vertices[edge_points[valid_edges, 0]], 
                vertices[edge_points[valid_edges, 1]], 
                edges_lengths[valid_edges]
            )
        
        return ratios
    
    def _compute_height_base_ratio(self, point_o, point_a, point_b, edges_lengths):
        """Compute height/base ratio for a triangle"""
        line_ab = point_b - point_a
        projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / self._fixed_division(
            np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
        closest_point = point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
        height = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
        return height / edges_lengths
    
    def _compute_global_edge_ratio(self):
        """Compute global edge length ratio (normalized by average edge length)"""
        vertices = self.vertices.detach().cpu().numpy() if torch.is_tensor(self.vertices) else self.vertices
        
        edge_lengths = np.linalg.norm(vertices[self.edges[:, 0]] - vertices[self.edges[:, 1]], 
                                     ord=2, axis=1)
        avg_length = np.mean(edge_lengths)
        return edge_lengths / avg_length
    
    def _fixed_division(self, a, epsilon=1e-10):
        """安全除法处理（单参数版）"""
        a = np.copy(a)
        a[np.abs(a) < epsilon] = epsilon  # 防止除以0
        return a

    # def _build_edge_face_map(self):
    #     """构建边到面的映射字典"""
    #     for i, face in enumerate(self.faces):
    #         edges = [
    #             tuple(sorted((face[0], face[1]))),
    #             tuple(sorted((face[1], face[2]))),
    #             tuple(sorted((face[2], face[0])))
    #         ]
    #         for edge in edges:
    #             if edge not in self._edge_face_map:
    #                 self._edge_face_map[edge] = []
    #             self._edge_face_map[edge].append(i)

    def _get_face_edges(self, face_idx):
        """获取面的三条边"""
        face = self.faces[face_idx]
        return [
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[2], face[0])))
        ]
    
    

    def compute_normals_with_pyvista(self):
        """
        使用 PyVista 计算面法向和点法向
        参数:
            vertices: [V, 3] 的顶点数组
            faces: [F, 3] 的面片数组（三角形网格）
        返回:
            face_normals: [F, 3] 的面法向
            vertex_normals: [V, 3] 的点法向
        """
        # 创建 PyVista 网格
        faces_pv = np.hstack([np.full((self.faces.shape[0], 1), 3), self.faces]).ravel()  # PyVista 需要的面格式
        vertices_np = self.vertices.cpu().numpy() if torch.is_tensor(self.vertices) else np.asarray(self.vertices)
        mesh = pv.PolyData(vertices_np, faces_pv)
        
        # 计算面法向（PyVista 自动计算）
        face_normals = mesh.cell_normals  # [F, 3]
        
        # 计算点法向（需显式启用）
        mesh.compute_normals(point_normals=True, cell_normals=False)  # 只计算点法向
        vertex_normals = mesh.point_normals  # [V, 3]
        
        return torch.from_numpy(np.asarray(face_normals)), torch.from_numpy(np.asarray(vertex_normals))
        # return face_normals, vertex_normals

    def visualize_normals_pyvista(self, scale=0.1, sample_ratio=0.1, save_path="normals.html"):
        """使用 PyVista 可视化法向（修复箭头绘制问题）"""
        
        # pv.OFF_SCREEN = True
        
        # 确保数据是 NumPy 数组
        vertices_np = self.vertices.cpu().numpy() if torch.is_tensor(self.vertices) else np.asarray(self.vertices)
        faces_np = self.faces.cpu().numpy() if torch.is_tensor(self.faces) else np.asarray(self.faces)
        
        # 创建 PyVista 网格
        faces_pv = np.hstack([np.full((self.faces.shape[0], 1), 3), self.faces]).ravel()
        mesh = pv.PolyData(vertices_np, faces_pv)
        
        # 计算面中心点
        face_centers = mesh.cell_centers().points
        
        # 采样部分面法向（避免箭头过多）
        n_faces = face_centers.shape[0]
        sample_step = max(1, int(1 / sample_ratio))
        sampled_indices = range(0, n_faces, sample_step)
        
        # 创建法向箭头数据
        normals_pv = pv.PolyData(face_centers[sampled_indices])
        normals_pv["normals"] = self.face_normal[sampled_indices]
        
        # 绘制
        plotter = pv.Plotter(off_screen=True)  # 离屏渲染
        plotter.add_mesh(mesh, color="lightblue", opacity=0.8, label="Mesh")
        
        # 批量绘制箭头
        arrows = normals_pv.glyph(
            orient="normals",
            scale=False,  # 禁用自动缩放
            factor=scale  # 手动控制箭头长度
        )
        plotter.add_mesh(arrows, color="red", label="Face Normals")
        
        # 点法向（可选）
        if hasattr(self, 'vertex_normal'):
            vertex_normals_pv = pv.PolyData(vertices_np)
            vertex_normals_pv["normals"] = self.vertex_normal
            vertex_arrows = vertex_normals_pv.glyph(
                orient="normals",
                scale=False,
                factor=scale * 0.5
            )
            plotter.add_mesh(vertex_arrows, color="green", label="Vertex Normals")
        
        # plotter.add_legend()
        # plotter.show()
        plotter.export_html(save_path)

    def get_adjacent_faces(self, face):
        """
        获取指定面片的所有邻接面
        
        参数:
            face (list/tuple): 包含三个顶点ID的序列，如 [v0, v1, v2]
            
        返回:
            tuple: (当前面片ID, 邻接面片ID列表)
        """
        # 1. 验证输入格式
        if len(face) != 3:
            raise ValueError("面片必须包含3个顶点ID")
        
        # 2. 查找当前面片ID
        face_id = self._find_face_id(face)
        if face_id is None:
            raise ValueError("输入面片不存在于网格中")
        
        # 3. 构建边到面片的映射（惰性初始化）
        if self._edge_face_map is None:
            self._build_edge_face_map()
        
        # 4. 收集邻接面
        adjacent_ids = set()
        face_edges = [
            tuple(sorted((face[0], face[1]))),
            tuple(sorted((face[1], face[2]))),
            tuple(sorted((face[2], face[0])))
        ]
        
        for edge in face_edges:
            adjacent_ids.update(self._edge_face_map.get(edge, []))
        
        # 移除自身并返回结果
        adjacent_ids.discard(face_id)
        return (face_id, list(adjacent_ids))

    # def _build_vertex_face_map(self) -> dict:
    #     """构建顶点到邻接面的映射字典"""
    #     vertex_face_map = {}
    #     for face_idx, face in enumerate(self.faces.view(-1, 3)):
    #         for v_idx in face:
    #             v_idx = v_idx.item()
    #             if v_idx not in vertex_face_map:
    #                 vertex_face_map[v_idx] = []
    #             vertex_face_map[v_idx].append(face_idx)
    #     return vertex_face_map

    # def _build_vertex_face_vetcor_map(self) -> dict:
    #     """构建顶点到邻接面向量（顶点指向对面边中点）的映射字典
        
    #     返回:
    #         dict: {
    #             顶点索引: [向量1, 向量2, ...],
    #             ...
    #         }
    #         其中每个向量是从该顶点指向所在面另一侧边中点的向量
    #     """
    #     vertex_face_vectors = defaultdict(list)
        
    #     for v_idx, _ in enumerate(self.vertices):
    #         try:
    #             v2face = self._vertex_face_map[v_idx]
    #         except:
    #             continue
    #         v2tris = self.faces[v2face]
    #         for tris in v2tris:
    #             v1, v2 = tris[tris != v_idx]
    #             vector = (self.vertices[v1] + self.vertices[v2]) / 2 - self.vertices[v_idx]
    #             vertex_face_vectors[v_idx].append(vector)
    #     return vertex_face_vectors

    def _build_vertex_face_vetcor_map(self) -> dict:
        """构建顶点到邻接面向量的映射字典（向量化版本）"""
        vertex_face_vectors = defaultdict(list)
        
        # 预计算所有面的顶点
        all_faces = self.faces
        
        for v_idx in range(len(self.vertices)):
            # if v_idx not in self._vertex_face_map:
            #     continue
                
            # 获取包含该顶点的所有面
            face_indices = self._vertex_face_map[v_idx]
            if not face_indices:
                continue
                
            # 批量处理所有面
            faces = all_faces[face_indices]
            
            # 找到每个面中不等于v_idx的两个顶点
            # 创建掩码来找到另外两个顶点
            mask = faces != v_idx
            other_vertices = faces[mask].reshape(-1, 2)
            
            # 批量计算中点
            mid_points = (self.vertices[other_vertices[:, 0]] + self.vertices[other_vertices[:, 1]]) * 0.5
            
            # 批量计算向量
            vectors = mid_points - self.vertices[v_idx]
            
            # 添加到结果
            vertex_face_vectors[v_idx] = vectors
        
        return vertex_face_vectors
    
    def _build_vertex_face_vetcor_map1(self) -> dict:
        """构建顶点到邻接面向量的映射字典（修复索引错误版本）"""
        vertex_face_vectors = defaultdict(list)
        
        # 确保vertices和faces是numpy数组以提高访问速度
        vertices = np.asarray(self.vertices)
        all_faces = np.asarray(self.faces)
        
        # 遍历_vertex_face_map中的所有键值对
        # 使用items()确保正确获取索引和对应的面列表
        for v_idx, face_indices in enumerate(self._vertex_face_map):
            # 检查面索引是否有效且不为空
            if not face_indices or not isinstance(face_indices, (list, np.ndarray)):
                continue
                
            # 确保面索引是整数列表
            try:
                face_indices = [int(idx) for idx in face_indices]
            except (TypeError, ValueError):
                continue  # 跳过无效的面索引
            
            # 获取包含该顶点的所有面
            try:
                faces = all_faces[face_indices]
            except IndexError:
                continue  # 跳过索引越界的情况
            
            # 找到每个面中不等于v_idx的两个顶点
            mask = faces != v_idx
            other_vertices = faces[mask].reshape(-1, 2)
            
            # 批量计算中点和向量
            mid_points = (vertices[other_vertices[:, 0]] + vertices[other_vertices[:, 1]]) * 0.5
            vectors = mid_points - vertices[v_idx]
            
            # 添加到结果
            vertex_face_vectors[v_idx] = vectors
            # vertex_face_vectors[v_idx] = torch.from_numpy(vectors)
        
        return vertex_face_vectors


    def _build_vertex_face_map(self):
        """
        构建顶点到邻接面的映射：每个顶点对应一个列表，存储包含该顶点的所有面的索引。
        
        返回:
            vertex_face_map (list of list): 长度为顶点数N，vertex_face_map[v]是顶点v的邻接面索引列表。
        """
        # 获取顶点数量和面数量
        num_vertices = self.vertices.shape[0]  # 假设vertices形状为[N, 3]
        num_faces = self.faces.shape[0]        # 假设faces形状为[F, 3]（三角形网格）

        vertex_face_map = [[] for _ in range(num_vertices)]
        for face_idx in range(num_faces):
            vertices_in_face = self.faces[face_idx]  # 形状为[3]的张量
            
            for v_idx in vertices_in_face:
                v_idx = v_idx.item()  # 转换为Python整数（若为张量）
                vertex_face_map[v_idx].append(face_idx)
        
        return vertex_face_map
    
    def _build_vertex_face_map1(self):
        """
        构建顶点到邻接面的映射：每个顶点对应一个列表，存储包含该顶点的所有面的索引。
        
        返回:
            vertex_face_map (list of list): 长度为顶点数N，vertex_face_map[v]是顶点v的邻接面索引列表。
        """
        # 获取顶点数量和面数量
        num_vertices = self.vertices.shape[0]  # 顶点形状为[N, 3]
        num_faces = self.faces.shape[0]        # 面形状为[F, 3]（三角形网格）
        
        # 每个面索引重复3次（每个面有3个顶点），形状为[F*3]
        face_indices = torch.arange(num_faces, device=self.faces.device).repeat_interleave(3)
        
        # 将所有面的顶点索引展平，形状为[F*3]
        vertex_indices = self.faces.flatten()
        
        # 找到每个顶点对应的面索引在展平数组中的位置
        # 首先对顶点索引排序，获取排序后的索引和原始位置
        sorted_indices = torch.argsort(vertex_indices)
        sorted_vertices = vertex_indices[sorted_indices]
        sorted_faces = face_indices[sorted_indices]
        
        # 找到每个顶点的起始和结束位置
        # 计算每个顶点在排序后数组中出现的次数
        counts = torch.bincount(sorted_vertices, minlength=num_vertices)
        
        # 计算累积和，得到每个顶点对应的面索引在结果中的起始位置
        offsets = torch.zeros(num_vertices + 1, dtype=torch.long, device=self.faces.device)
        offsets[1:] = counts.cumsum(dim=0)
        
        # 创建结果列表
        vertex_face_map = []
        for v in range(num_vertices):
            start = offsets[v]
            end = offsets[v + 1]
            vertex_face_map.append(sorted_faces[start:end].tolist())
        
        return vertex_face_map

    def classify_vertex_convexity(self, angle_tolerance: float = 5.0) -> torch.Tensor:
        """
        通过顶点法向与邻接面向量的夹角判断凹凸性
        
        参数:
            angle_tolerance: 角度容差（处理数值误差）
        
        返回:
            torch.Tensor: 1（凸点），-1（凹点），0（边界点/无法判定）
        """
        convexity = torch.zeros(len(self.vertices), dtype=torch.int32)
        
        for v_idx, vectors in self._vertex_face_vectors.items():
            if len(vectors) == 0 or (vectors.shape[0] <= 3):
                convexity[v_idx] = 0  # 孤立点或边界点
                continue
            
            # 顶点法向（单位向量）
            v_normal = self.vertex_normal[v_idx]
            
            # 计算与所有邻接面向量的点积
            # face_vectors = torch.stack(vectors)  # [n_faces, 3]
            dots = torch.sum(v_normal * vectors, dim=1)  # [n_faces]
            
            # 判断凹凸性（考虑容差）
            if torch.all(dots > -angle_tolerance / 90.0):  # 近似 dot > 0
                convexity[v_idx] = 1  # 凸点
            elif torch.any(dots <= angle_tolerance / 90.0):  # 近似 dot <= 0
                convexity[v_idx] = -1  # 凹点
        
        return convexity

    def classify_vertex_convexity1(self, angle_tolerance: float = 5.0) -> torch.Tensor:
        """
        通过顶点法向与邻接面向量的夹角判断凹凸性（加速版本）
        
        参数:
            angle_tolerance: 角度容差（处理数值误差）
        
        返回:
            torch.Tensor: 1（凸点），-1（凹点），0（边界点/无法判定）
        """
        # 初始化结果张量
        convexity = torch.zeros(len(self.vertices), dtype=torch.int32)
        
        # 提前计算容差阈值（避免重复计算）
        tolerance = angle_tolerance / 90.0
        
        # 收集所有需要处理的顶点数据（过滤掉无需处理的顶点）
        valid_indices = []
        all_vectors = []
        
        for v_idx, vectors in self._vertex_face_vectors.items():
            # 只处理有足够邻接面的顶点
            if len(vectors) > 3:
                valid_indices.append(v_idx)
                all_vectors.append(vectors)
        
        if not valid_indices:
            return convexity  # 没有需要处理的顶点
        
        # 批量处理所有有效顶点
        # 获取所有有效顶点的法向量 [n_valid, 3]
        normals = self.vertex_normal[valid_indices]
        
        # 计算每个顶点的向量数量，用于后续拆分
        vector_counts = [v.shape[0] for v in all_vectors]
        
        # 将所有向量合并为一个大张量 [total_vectors, 3]
        # stacked_vectors = torch.cat(all_vectors, dim=0)
        stacked_vectors = torch.tensor(np.concatenate(all_vectors, axis=0))
        
        # 为每个向量复制对应的法向量 [total_vectors, 3]
        expanded_normals = normals.repeat_interleave(torch.tensor(vector_counts, device=normals.device), dim=0)
        
        # 批量计算点积 [total_vectors]
        dots = torch.sum(expanded_normals * stacked_vectors, dim=1)
        
        # 拆分点积结果，恢复每个顶点的点积集合
        split_dots = torch.split(dots, vector_counts)
        
        # 批量判断凹凸性
        for i, v_idx in enumerate(valid_indices):
            vertex_dots = split_dots[i]
            # 所有点积都大于负容差 -> 凸点
            if torch.all(vertex_dots > -tolerance):
                convexity[v_idx] = 1
            # 存在点积小于等于容差 -> 凹点
            elif torch.any(vertex_dots <= tolerance):
                convexity[v_idx] = -1
            # 其他情况保持0（边界点）
        
        return convexity
    
    def classify_vertex_convexity_fully_vectorized(self, angle_tolerance: float = 5.0) -> torch.Tensor:
        """
        完全向量化的版本
        """
        convexity = torch.zeros(len(self.vertices), dtype=torch.int32)
        tolerance = angle_tolerance / 90.0
        
        valid_indices = []
        all_vectors = []
        
        for v_idx, vectors in self._vertex_face_vectors.items():
            if len(vectors) > 3:
                valid_indices.append(v_idx)
                all_vectors.append(vectors)
        
        if not valid_indices:
            return convexity
        
        # 转换为张量
        valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long)
        vector_counts = torch.tensor([v.shape[0] for v in all_vectors], dtype=torch.long)
        total_vectors = vector_counts.sum().item()
        
        # 创建索引映射
        cum_counts = torch.cumsum(vector_counts, dim=0)
        start_indices = torch.cat([torch.tensor([0]), cum_counts[:-1]])
        
        # 批量计算所有条件
        all_convex = torch.ones(len(valid_indices), dtype=torch.bool)
        all_concave = torch.zeros(len(valid_indices), dtype=torch.bool)
        
        for i, vectors in enumerate(all_vectors):
            dots = torch.sum(self.vertex_normal[valid_indices[i]] * vectors, dim=1)
            all_convex[i] = torch.all(dots > -tolerance)
            all_concave[i] = torch.any(dots <= tolerance)
        
        # 批量赋值
        convex_indices = valid_indices_tensor[all_convex]
        concave_indices = valid_indices_tensor[all_concave & ~all_convex]
        
        convexity[convex_indices] = 1
        convexity[concave_indices] = -1
        
        return convexity

    
    def is_convex_vertex(mesh, vertex_id: int, angle_tol: float = 5.0) -> bool:
        """
        判断顶点是否为凸点
        
        参数:
            mesh: 网格对象（需包含 vertex_normal 和 _vertex_face_vectors）
            vertex_id: 顶点ID
            angle_tol: 角度容差（度）
        
        返回:
            bool: True=凸点，False=凹点
        """
        # 获取顶点法向和邻接面向量
        v_normal = mesh.vertex_normal[vertex_id]
        face_vectors = mesh.face_normal[mesh._vertex_face_map[vertex_id]]
        
        # 计算点积（带容差）
        dots = [np.dot(v_normal, v) for v in face_vectors]
        convex = all(d > -np.sin(np.radians(angle_tol)) for d in dots)
        
        return convex

    def _build_edge_face_map(self):
        """构建边到面片的映射关系"""
        edge_face_map = defaultdict(list)
        faces = self.faces.cpu().numpy() if torch.is_tensor(self.faces) else self.faces
        
        for face_id, face in enumerate(faces):
            # 生成三条边（确保边表示为有序元组）
            edges = [
                tuple(sorted((face[0], face[1]))),
                tuple(sorted((face[1], face[2]))),
                tuple(sorted((face[2], face[0])))
            ]
            for edge in edges:
                edge_face_map[edge].append(face_id)
        return edge_face_map
    
    def _build_edge_face_map1(self):
        """构建边到面片的映射关系 - 向量化版本"""
        edge_face_map = defaultdict(list)
        faces = self.faces.cpu().numpy() if torch.is_tensor(self.faces) else self.faces
        
        # 一次性生成所有边
        edges = np.sort(np.column_stack([
            faces[:, [0, 1]],  # 边1
            faces[:, [1, 2]],  # 边2
            faces[:, [2, 0]]   # 边3
        ]), axis=1)
        
        # 重塑为 (3*num_faces, 2) 的形状
        all_edges = edges.reshape(-1, 2)
        
        # 为每条边创建对应的面ID
        face_ids = np.repeat(np.arange(len(faces)), 3)
        
        # 使用字典存储映射关系
        for edge, face_id in zip(map(tuple, all_edges), face_ids):
            edge_face_map[edge].append(face_id)
        
        return edge_face_map

    def _find_face_id(self, vertex_ids):
        """
        通过顶点ID查找面片ID
        返回:
            int/None: 找到返回面片ID，否则返回None
        """
        faces = self.faces.cpu().numpy() if torch.is_tensor(self.faces) else self.faces
        vertex_set = set(vertex_ids)
        
        for face_id, face in enumerate(faces):
            if set(face) == vertex_set:
                return face_id
        return None

    def create_edge_index(self):
        edges = []
        edges_count = 0
        edge2key = dict()
        Face = self.faces.detach().numpy()
        for face in Face:
            faces_edges = []
            for i in range(3):
                cur_edge = (face[i], face[(i + 1) % 3])
                faces_edges.append(cur_edge)
            for edge in faces_edges:
                edge = tuple(sorted(list(edge)))
                if edge not in edge2key:
                    edge2key[edge] = edges_count
                    edges.append(list(edge))
                    edges_count += 1
        self.edges = torch.tensor(edges, dtype=torch.int32)

    def create_edge_index1(self):
        Face = self.faces.numpy() if self.faces.is_cuda else self.faces.detach().numpy()
        
        # 创建所有边
        edges1 = Face[:, [0, 1]]
        edges2 = Face[:, [1, 2]]
        edges3 = Face[:, [2, 0]]
        
        # 合并并排序边
        all_edges = np.vstack([edges1, edges2, edges3])
        sorted_edges = np.sort(all_edges, axis=1)
        
        # 去重
        unique_edges = np.unique(sorted_edges, axis=0)
        
        self.edges = torch.tensor(unique_edges, dtype=torch.int32)
    
    # 辅助函数：计算顶点度数（连接边数）
    def get_vertex_degree(self, vertex):
        degree = 0
        for edge in self.edges:
            if edge[0] == vertex or edge[1] == vertex:
                degree += 1
        return degree

    # 辅助函数：判断顶点是否在特征边中出现超过两次
    def is_vertex_appearing_more_than_twice(self, vertex):
        if not self.feature_point[vertex]:
            return 0
            
        count = 0
        for edge in self.edges:
            if (edge[0] == vertex or edge[1] == vertex) and self.feature_point[edge[0]] and self.feature_point[edge[1]]:
                count += 1
        # 返回出现次数
        return count

    def detect_non_manifold_vertices(self):
        """
        检测非流形顶点（邻域非闭合环）
        返回:
            non_manifold_vertices (list): 非流形顶点的索引列表
        """
        num_vertices = self.vertices.size(0)
        non_manifold_vertices = []

        Flag = True

        # 1. 构建顶点到面的映射：vertex_faces[v] = [包含v的面索引]
        vertex_faces = [[] for _ in range(num_vertices)]
        for face_idx, face in enumerate(self.faces):
            for vertex_idx in face:
                vertex_faces[vertex_idx].append(face_idx)

        # 2. 检查每个顶点的邻域拓扑
        for v in range(num_vertices):
            faces_v = vertex_faces[v]
            # 孤立点判定
            if not faces_v:
                non_manifold_vertices.append(v)
                continue

            # 构建邻接图 (graph[a] = [b, c] 表示a与b,c相邻)
            graph = {}
            for face_idx in faces_v:
                face = self.faces[face_idx].tolist()  # 转为列表
                # 定位顶点v在面中的位置
                try:
                    idx = face.index(v)
                except ValueError:
                    continue
                # 获取v在面中的两个相邻顶点
                a = face[(idx + 1) % 3]
                b = face[(idx + 2) % 3]
                
                # 添加无向边 (a, b)
                if a not in graph:
                    graph[a] = []
                if b not in graph:
                    graph[b] = []
                graph[a].append(b)
                graph[b].append(a)
            
            nodes = list(graph.keys())
            # 无相邻顶点（理论不应出现）
            if not nodes:
                non_manifold_vertices.append(v)
                continue
            
            # 3. 检查图的连通性
            visited = set()
            stack = [nodes[0]]
            while stack:
                current = stack.pop()
                visited.add(current)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            # 存在多个连通分量 → 非流形
            if len(visited) != len(nodes):
                # non_manifold_vertices.append(v)
                # continue
                Flag = False
                break
            
            # 4. 检查度数：所有节点度数必须为2（严格闭合环）
            for node in nodes:
                if len(graph[node]) != 2:  # 非闭合环
                    # non_manifold_vertices.append(v)
                    # break
                    Flag = False
                    break
        
        return Flag

    def num_mesh(self):
        total_elements = 0.0
        elements = []
        for face in self.faces:
            coord_3d = torch.stack([self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]])
            coord_2d = map_triangle_to_2d(coord_3d)
            size = np.array([self.sizing_values[face[0]], self.sizing_values[face[1]], self.sizing_values[face[2]]])
            num = integrate_over_triangle(coord_2d.numpy(), size)
            elements.append(num)
            # total_elements += integrate_over_triangle(coord_2d.numpy(), size)
        # return total_elements
        return np.concatenate(elements)


    def collapsing_edge_id(self, edge_id):
        if edge_id >= len(self.edges):
            raise ValueError(f"边ID {edge_id} 超出范围 (总边数: {len(self.edges)})")
        
        v1, v2 = self.edges[edge_id]
        if v1 == v2:
            raise ValueError("边的两个顶点不能相同")

        # 计算新顶点位置
        new_vertex_pos = (self.vertices[v1] + self.vertices[v2]) / 2

        new_vertices = torch.cat([self.vertices[:v2], self.vertices[v2+1:]])
        new_vertices[v1] = new_vertex_pos

        # 修改顶点索引
        new_faces = self.faces.clone()
        new_faces[new_faces == v2] = v1
        new_faces[new_faces > v2] -= 1

        # 检查退化面
        # unique_counts = torch.tensor([torch.unique(tri).size(0) for tri in new_faces])
        # non_degenerate_mask = unique_counts == 3
        # newfaces = new_faces[non_degenerate_mask]
        # related_newfaces = []
        # for i, face in enumerate(newfaces):
        #     if v1 in face:
        #         related_newfaces.append(face)

        face_tensor = new_faces
        # 检查每个面是否有重复顶点
        non_degenerate_mask = (face_tensor[:, 0] != face_tensor[:, 1]) & \
            (face_tensor[:, 1] != face_tensor[:, 2]) & \
            (face_tensor[:, 2] != face_tensor[:, 0])
        newfaces = new_faces[non_degenerate_mask]

        new_regions = self.surface_id
        new_regions = new_regions[non_degenerate_mask]

        new_size_value = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
        new_sizing_values = torch.cat([self.sizing_values[:v2], self.sizing_values[v2+1:]])
        new_sizing_values[v1] = new_size_value

        # if self.feature_point[v1] == self.feature_point[v2]:
        #     new_feature_point = self.feature_point[v1]
        # else:
        #     new_feature_point = True
        # new_feature_points = torch.cat([self.feature_point[:v2], self.feature_point[v2+1:]])
        # new_feature_points[v1] = new_feature_point

        # mesh = CustomMesh(vertices=new_vertices, faces=newfaces, sizing_values=new_sizing_values, regions=new_regions, feature_points=new_feature_points)
        mesh = CustomMesh(vertices=new_vertices, faces=newfaces, sizing_values=new_sizing_values, regions=new_regions, version=1)
        # post_faces = []
        # for i, face in enumerate(newfaces):
        #     if v1 in face:
        #         post_faces.append(face)
        # post_faces = [face for face in newfaces if v1 in face]
        mask = torch.any(newfaces == v1, dim=1)
        post_faces = newfaces[mask]
        return mesh, v1, post_faces

    def collapsing_edge_id1(self, edge_id):
        total_start = time.time()
        
        # 1. 输入验证
        start_time = time.time()
        if edge_id >= len(self.edges):
            raise ValueError(f"边ID {edge_id} 超出范围 (总边数: {len(self.edges)})")
        
        v1, v2 = self.edges[edge_id]
        if v1 == v2:
            raise ValueError("边的两个顶点不能相同")
        validation_time = time.time() - start_time

        # 2. 计算新顶点位置
        start_time = time.time()
        new_vertex_pos = (self.vertices[v1] + self.vertices[v2]) / 2
        vertex_calc_time = time.time() - start_time

        # 3. 创建新顶点数组
        start_time = time.time()
        new_vertices = torch.cat([self.vertices[:v2], self.vertices[v2+1:]])
        new_vertices[v1] = new_vertex_pos
        vertices_creation_time = time.time() - start_time

        # 4. 修改面索引
        start_time = time.time()
        new_faces = self.faces.clone()
        new_faces[new_faces == v2] = v1
        new_faces[new_faces > v2] -= 1
        faces_modification_time = time.time() - start_time

        # 5. 检查退化面
        start_time = time.time()
        # unique_counts = torch.tensor([torch.unique(tri).size(0) for tri in new_faces])
        # non_degenerate_mask = unique_counts == 3
        face_tensor = new_faces
        # 检查每个面是否有重复顶点
        non_degenerate_mask = (face_tensor[:, 0] != face_tensor[:, 1]) & \
            (face_tensor[:, 1] != face_tensor[:, 2]) & \
            (face_tensor[:, 2] != face_tensor[:, 0])
        newfaces = new_faces[non_degenerate_mask]
        degenerate_check_time = time.time() - start_time

        # 6. 查找相关面（需要优化的部分）
        start_time = time.time()
        # related_newfaces = []
        # for i, face in enumerate(newfaces):
        #     if v1 in face:
        #         related_newfaces.append(face)
        mask = torch.any(newfaces == v1, dim=1)
        post_faces = newfaces[mask]
        related_faces_time = time.time() - start_time

        # 7. 处理区域信息
        start_time = time.time()
        new_regions = self.surface_id
        new_regions = new_regions[non_degenerate_mask]
        regions_time = time.time() - start_time

        # 8. 处理尺寸值
        start_time = time.time()
        new_size_value = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
        new_sizing_values = torch.cat([self.sizing_values[:v2], self.sizing_values[v2+1:]])
        new_sizing_values[v1] = new_size_value
        sizing_time = time.time() - start_time

        # 9. 创建新网格对象
        start_time = time.time()
        mesh = CustomMesh(vertices=new_vertices, faces=newfaces, sizing_values=new_sizing_values, regions=new_regions, version=1)
        mesh_creation_time = time.time() - start_time

        # # 10. 再次查找相关面（重复操作，可以优化）
        # start_time = time.time()
        # post_faces = [face for face in newfaces if v1 in face]
        # post_faces_time = time.time() - start_time

        total_time = time.time() - total_start

        # 打印时间分析
        print(f"总时间: {total_time:.6f}s")
        print(f"输入验证: {validation_time:.6f}s ({validation_time/total_time*100:.1f}%)")
        print(f"顶点计算: {vertex_calc_time:.6f}s ({vertex_calc_time/total_time*100:.1f}%)")
        print(f"顶点创建: {vertices_creation_time:.6f}s ({vertices_creation_time/total_time*100:.1f}%)")
        print(f"面修改: {faces_modification_time:.6f}s ({faces_modification_time/total_time*100:.1f}%)")
        print(f"退化检查: {degenerate_check_time:.6f}s ({degenerate_check_time/total_time*100:.1f}%)")
        print(f"相关面查找1: {related_faces_time:.6f}s ({related_faces_time/total_time*100:.1f}%)")
        print(f"区域处理: {regions_time:.6f}s ({regions_time/total_time*100:.1f}%)")
        print(f"尺寸值处理: {sizing_time:.6f}s ({sizing_time/total_time*100:.1f}%)")
        print(f"网格创建: {mesh_creation_time:.6f}s ({mesh_creation_time/total_time*100:.1f}%)")
        # print(f"相关面查找2: {post_faces_time:.6f}s ({post_faces_time/total_time*100:.1f}%)")

        return mesh, v1, post_faces
    
    def collapsing_edge(self, v1, v2):
        if v1 == v2:
            raise ValueError("边的两个顶点不能相同")

        # 计算新顶点位置
        new_vertex_pos = (self.vertices[v1] + self.vertices[v2]) / 2
        
        # related_faces = []
        # for face in self.faces:
        #     if v1 in face or v2 in face:
        #         related_faces.append(face)
        
        distance_sum = 0.0
        # for face in related_faces:
        #     a, b, c = face
        #     va = self.vertices[a]
        #     vb = self.vertices[b]
        #     vc = self.vertices[c]
            
        #     # 计算面法线
        #     ab = vb - va
        #     ac = vc - va
        #     normal = torch.linalg.cross(ab, ac)
            
        #     # 跳过退化面
        #     if torch.norm(normal) < 1e-8:
        #         continue
            
        #     # 计算平面方程参数
        #     d = -torch.dot(va, normal)
            
        #     # 计算点到面的距离
        #     distance = torch.abs(torch.dot(new_vertex_pos, normal) + d) / torch.norm(normal)
        #     distance_sum += distance

        new_vertices = torch.cat([self.vertices[:v2], self.vertices[v2+1:]])
        new_vertices[v1] = new_vertex_pos

        # 修改顶点索引
        new_faces = self.faces.clone()
        new_faces[new_faces == v2] = v1
        new_faces[new_faces > v2] -= 1

        # 检查退化面
        unique_counts = torch.tensor([torch.unique(tri).size(0) for tri in new_faces])
        non_degenerate_mask = unique_counts == 3
        newfaces = new_faces[non_degenerate_mask]

        new_regions = self.surface_id
        new_regions = new_regions[non_degenerate_mask]

        new_size_value = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
        new_sizing_values = torch.cat([self.sizing_values[:v2], self.sizing_values[v2+1:]])
        new_sizing_values[v1] = new_size_value

        mesh = CustomMesh(vertices=new_vertices, faces=newfaces, sizing_values=new_sizing_values, regions=new_regions)
        return mesh, v1
    
    # def collapsing_edge_id(self, edge_id):
    #     if edge_id >= len(self.edges):
    #         raise ValueError(f"边ID {edge_id} 超出范围 (总边数: {len(self.edges)})")
        
    #     v1, v2 = self.edges[edge_id]
    #     if v1 == v2:
    #         raise ValueError("边的两个顶点不能相同")

    #     # 计算新顶点位置
    #     new_vertex_pos = (self.vertices[v1] + self.vertices[v2]) / 2
    #     new_vertices = torch.cat([self.vertices[:v2], self.vertices[v2+1:]])
    #     new_vertices[v1] = new_vertex_pos

    #     # 修改顶点索引
    #     new_faces = self.faces.clone()
    #     new_faces[new_faces == v2] = v1
    #     new_faces[new_faces > v2] -= 1

    #     # 检查退化面
    #     unique_counts = torch.tensor([torch.unique(tri).size(0) for tri in new_faces])
    #     non_degenerate_mask = unique_counts == 3
    #     newfaces = new_faces[non_degenerate_mask]

    #     related_newfaces = []
    #     for i, face in enumerate(newfaces):
    #         if v1 in face:
    #             related_newfaces.append(face)

    #     new_regions = self.surface_id
    #     new_regions = new_regions[non_degenerate_mask]

    #     new_size_value = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
    #     new_sizing_values = torch.cat([self.sizing_values[:v2], self.sizing_values[v2+1:]])
    #     new_sizing_values[v1] = new_size_value

    #     mesh = CustomMesh(vertices=new_vertices, faces=newfaces, sizing_values=new_sizing_values, regions=new_regions)
    #     post_element = 0
    #     post_faces = []
    #     for i, face in enumerate(newfaces):
    #         if v1 in face:
    #             post_faces.append(face)
    #     return mesh, post_faces
    
    # def collapsing_edge_id_optimized(self, edge_id):
    #     if edge_id >= len(self.edges):
    #         raise ValueError(f"边ID {edge_id} 超出范围")
        
    #     v1, v2 = self.edges[edge_id]
    #     if v1 == v2:
    #         raise ValueError("边的两个顶点不能相同")

    #     # 使用原地操作和预分配内存
    #     device = self.vertices.device
    #     n_vertices = len(self.vertices)
        
    #     # 1. 创建新顶点数组
    #     new_vertices = torch.empty(n_vertices - 1, 3, device=device)
    #     new_vertices[:v2] = self.vertices[:v2]
    #     new_vertices[v2:] = self.vertices[v2+1:]
    #     new_vertices[v1 if v1 < v2 else v1 - 1] = (self.vertices[v1] + self.vertices[v2]) * 0.5

    #     # 2. 快速处理面索引
    #     new_faces = self.faces.clone()
        
    #     # 使用张量操作替换循环
    #     # 替换v2为v1
    #     v2_positions = (new_faces == v2)
    #     new_faces[v2_positions] = v1
        
    #     # 调整索引
    #     adjustment_mask = (new_faces > v2)
    #     new_faces[adjustment_mask] -= 1

    #     # 3. 快速退化面检测
    #     # 检查每个面的三个顶点是否都不相同
    #     degenerate = (new_faces[:, 0] == new_faces[:, 1]) | \
    #                 (new_faces[:, 1] == new_faces[:, 2]) | \
    #                 (new_faces[:, 2] == new_faces[:, 0])
        
    #     non_degenerate = ~degenerate
    #     newfaces = new_faces[non_degenerate]

    #     # 4. 找到相关面
    #     v1_mask = (newfaces == v1).any(dim=1)
    #     related_newfaces = newfaces[v1_mask]

    #     # 5. 处理尺寸值
    #     new_sizing_values = torch.empty(n_vertices - 1, device=device)
    #     new_sizing_values[:v2] = self.sizing_values[:v2].squeeze(1)
    #     new_sizing_values[v2:] = self.sizing_values[v2+1:].squeeze(1)
    #     new_sizing_values[v1 if v1 < v2 else v1 - 1] = (self.sizing_values[v1] + self.sizing_values[v2]) * 0.5

    #     # 6. 处理区域（如果有）
    #     if hasattr(self, 'surface_id'):
    #         new_regions = self.surface_id[non_degenerate]
    #     else:
    #         new_regions = None

    #     # 创建网格
    #     mesh_args = {
    #         'vertices': new_vertices,
    #         'faces': newfaces,
    #         'sizing_values': new_sizing_values
    #     }
    #     if new_regions is not None:
    #         mesh_args['regions'] = new_regions
            
    #     mesh = CustomMesh(**mesh_args)
        
    #     return mesh, related_newfaces

    def collapsing_edge_id_optimized(self, edge_id):
        start_time = time.time()
        total_start = start_time
        
        if edge_id >= len(self.edges):
            raise ValueError(f"边ID {edge_id} 超出范围")
        
        v1, v2 = self.edges[edge_id]
        if v1 == v2:
            raise ValueError("边的两个顶点不能相同")

        # 使用原地操作和预分配内存
        device = self.vertices.device
        n_vertices = len(self.vertices)
        
        # 1. 创建新顶点数组
        vert_start = time.time()
        new_vertices = torch.empty(n_vertices - 1, 3, device=device)
        new_vertices[:v2] = self.vertices[:v2]
        new_vertices[v2:] = self.vertices[v2+1:]
        new_vertices[v1 if v1 < v2 else v1 - 1] = (self.vertices[v1] + self.vertices[v2]) * 0.5
        vert_time = time.time() - vert_start

        # 2. 快速处理面索引
        face_start = time.time()
        new_faces = self.faces.clone()
        
        # 使用张量操作替换循环
        v2_positions = (new_faces == v2)
        new_faces[v2_positions] = v1
        
        # 调整索引
        adjustment_mask = (new_faces > v2)
        new_faces[adjustment_mask] -= 1
        face_time = time.time() - face_start

        # 3. 快速退化面检测
        degenerate_start = time.time()
        degenerate = (new_faces[:, 0] == new_faces[:, 1]) | \
                    (new_faces[:, 1] == new_faces[:, 2]) | \
                    (new_faces[:, 2] == new_faces[:, 0])
        
        non_degenerate = ~degenerate
        newfaces = new_faces[non_degenerate]
        degenerate_time = time.time() - degenerate_start

        # 4. 找到相关面
        related_start = time.time()
        v1_mask = (newfaces == v1).any(dim=1)
        related_newfaces = newfaces[v1_mask]
        related_time = time.time() - related_start

        # 5. 处理尺寸值
        sizing_start = time.time()
        new_sizing_values = torch.empty(n_vertices - 1, device=device)
        # 添加维度检查以确保正确索引
        if self.sizing_values.ndim > 1:
            new_sizing_values[:v2] = self.sizing_values[:v2].squeeze(1)
            new_sizing_values[v2:] = self.sizing_values[v2+1:].squeeze(1)
            sizing_val = (self.sizing_values[v1] + self.sizing_values[v2]) * 0.5
        else:
            new_sizing_values[:v2] = self.sizing_values[:v2]
            new_sizing_values[v2:] = self.sizing_values[v2+1:]
            sizing_val = (self.sizing_values[v1] + self.sizing_values[v2]) * 0.5
        
        new_sizing_values[v1 if v1 < v2 else v1 - 1] = sizing_val
        sizing_time = time.time() - sizing_start

        # 6. 处理区域（如果有）
        region_start = time.time()
        if hasattr(self, 'surface_id'):
            new_regions = self.surface_id[non_degenerate]
        else:
            new_regions = None
        region_time = time.time() - region_start

        # 创建网格
        mesh_start = time.time()
        mesh_args = {
            'vertices': new_vertices,
            'faces': newfaces,
            'sizing_values': new_sizing_values
        }
        if new_regions is not None:
            mesh_args['regions'] = new_regions
            
        mesh = CustomMesh(**mesh_args)
        mesh_time = time.time() - mesh_start

        total_time = time.time() - total_start
        
        # 输出时间统计
        print(f"=== 边折叠性能分析 ===")
        print(f"总时间: {total_time:.6f}s")
        print(f"顶点处理: {vert_time:.6f}s ({vert_time/total_time*100:.1f}%)")
        print(f"面索引处理: {face_time:.6f}s ({face_time/total_time*100:.1f}%)")
        print(f"退化面检测: {degenerate_time:.6f}s ({degenerate_time/total_time*100:.1f}%)")
        print(f"相关面查找: {related_time:.6f}s ({related_time/total_time*100:.1f}%)")
        print(f"尺寸值处理: {sizing_time:.6f}s ({sizing_time/total_time*100:.1f}%)")
        print(f"区域处理: {region_time:.6f}s ({region_time/total_time*100:.1f}%)")
        print(f"网格创建: {mesh_time:.6f}s ({mesh_time/total_time*100:.1f}%)")
        print(f"====================")
        
        return mesh, related_newfaces
    
    # def collapsing_edge_new(self, v1, v2):
    #     if v1 == v2:
    #         raise ValueError("边的两个顶点不能相同")
    #     delete_point_id = -1
    #     remain_point_id = -1
    #     new_size_value = -1
    #     is_center = True
    #     if self.feature_point[v1] == self.feature_point[v2]:
    #         if not self.feature_point[v1]:
    #             new_vertex_pos = (self.vertices[v1] + self.vertices[v2]) / 2
    #             delete_point_id = v2
    #             remain_point_id = v1
    #             new_size_value = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
    #             is_center = False
    #         else:
    #             v1_condition = is_vertex_appearing_more_than_twice(self.feature_edge, v1)
    #             v2_condition = is_vertex_appearing_more_than_twice(self.feature_edge, v2)
    #             if (not v1_condition and not v2_condition) or (v1_condition and v2_condition):
    #                 new_vertex_pos = (self.vertices[v1] + self.vertices[v2]) / 2
    #                 delete_point_id = v2
    #                 remain_point_id = v1
    #                 new_size_value = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
    #             else:
    #                 if v1_condition:
    #                     new_vertex_pos = self.vertices[v1]
    #                     delete_point_id = v2
    #                     remain_point_id = v1
    #                     new_size_value = self.sizing_values[v1]
    #                     is_center = False
    #                 elif v2_condition:
    #                     new_vertex_pos = self.vertices[v2]
    #                     delete_point_id = v1
    #                     remain_point_id = v2
    #                     new_size_value = self.sizing_values[v2]
    #                     is_center = False

    #     elif self.feature_point[v1] and not self.feature_point[v2]:
    #         new_vertex_pos = self.vertices[v1]
    #         delete_point_id = v2
    #         remain_point_id = v1
    #         new_size_value = self.sizing_values[v1]
    #         is_center = False
    #     elif self.feature_point[v2] and not self.feature_point[v1]:
    #         new_vertex_pos = self.vertices[v2]
    #         delete_point_id = v1
    #         remain_point_id = v2
    #         new_size_value = self.sizing_values[v2]
    #         is_center = False

    #     new_vertices = torch.cat([self.vertices[:delete_point_id], self.vertices[delete_point_id+1:]])
    #     new_vertices[remain_point_id] = new_vertex_pos

    #     new_feature_points = torch.cat([self.feature_point[:delete_point_id], self.feature_point[delete_point_id+1:]])


    #     # 修改顶点索引
    #     new_faces = self.faces.clone()
    #     new_faces[new_faces == delete_point_id] = remain_point_id
    #     new_faces[new_faces > delete_point_id] -= 1

    #     # 检查退化面
    #     unique_counts = torch.tensor([torch.unique(tri).size(0) for tri in new_faces])
    #     non_degenerate_mask = unique_counts == 3
    #     newfaces = new_faces[non_degenerate_mask]

    #     post_face = []
    #     if is_center:
    #         # mask = (newfaces.faces == remain_point_id).any(dim=1)
    #         for n, face in enumerate(newfaces):
    #             if remain_point_id in face:
    #                 post_face.append(face)
    #         # post_face = [face for face in newfaces.faces if remain_point_id in face]
    #     else:
    #         # for n, face in enumerate(self.faces):
    #         #     if delete_point_id in face and remain_point_id not in face:
    #         #         post_face.append(face == delete_point_id = remain_point_id)
    #         for face in self.faces:
    #             if delete_point_id in face and remain_point_id not in face:
    #                 # 替换 delete_point_id 为 remain_point_id
    #                 modified_face = [remain_point_id if v == delete_point_id else v for v in face]
    #                 modified_face = torch.tensor(modified_face)
    #                 modified_face[modified_face > delete_point_id] -= 1
    #                 post_face.append(modified_face)


    #     new_regions = self.surface_id
    #     new_regions = new_regions[non_degenerate_mask]

    #     new_sizing_values = torch.cat([self.sizing_values[:delete_point_id], self.sizing_values[delete_point_id+1:]])
    #     new_sizing_values[remain_point_id] = new_size_value

    #     mesh = CustomMesh(vertices=new_vertices, faces=newfaces, sizing_values=new_sizing_values, regions=new_regions, feature_points=new_feature_points)
    #     return mesh, is_center, remain_point_id, post_face

    def collapsing_edge_new(self, v1, v2):
        timers = {
            'input_check': 0,
            'condition_check': 0,
            'vertex_processing': 0,
            'face_processing': 0,
            'post_face_processing': 0,
            'mesh_construction': 0
        }
        
        # 1. 输入检查
        start = time.time()
        if v1.item() == v2.item():
            return None, True, -1, -1, []
        sorted_edge1 = torch.tensor((v1.item(), v2.item()))
        sorted_edge2 = torch.tensor((v2.item(), v1.item()))
        if not (any(torch.equal(sorted_edge1, edge) for edge in self.edges) or any(torch.equal(sorted_edge2, edge) for edge in self.edges)):
            return None, True, -1, -1, []
        timers['input_check'] = time.time() - start
        
        # 初始化变量
        delete_point_id = -1
        remain_point_id = -1
        new_size_value = -1
        is_center = True
        
        # 2. 条件判断区块
        start = time.time()
        if self.feature_point[v1] == self.feature_point[v2]:
            if not self.feature_point[v1]:
                new_vertex_pos = (self.vertices[v1] + self.vertices[v2]) / 2
                delete_point_id = v2
                remain_point_id = v1
                new_size_value = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
                is_center = True
            else:
                v1_condition = is_vertex_appearing_more_than_twice(self.feature_point, self.edges, v1)
                v2_condition = is_vertex_appearing_more_than_twice(self.feature_point,self.edges, v2)
                if v1_condition == v2_condition:
                    new_vertex_pos = (self.vertices[v1] + self.vertices[v2]) / 2
                    delete_point_id = v2
                    remain_point_id = v1
                    new_size_value = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
                else:
                    if v1_condition > v2_condition:
                        new_vertex_pos = self.vertices[v1]
                        delete_point_id = v2
                        remain_point_id = v1
                        new_size_value = self.sizing_values[v1]
                        is_center = False
                    elif v1_condition < v2_condition:
                        new_vertex_pos = self.vertices[v2]
                        delete_point_id = v1
                        remain_point_id = v2
                        new_size_value = self.sizing_values[v2]
                        is_center = False

        elif self.feature_point[v1] and not self.feature_point[v2]:
            new_vertex_pos = self.vertices[v1]
            delete_point_id = v2
            remain_point_id = v1
            new_size_value = self.sizing_values[v1]
            is_center = False
        elif self.feature_point[v2] and not self.feature_point[v1]:
            new_vertex_pos = self.vertices[v2]
            delete_point_id = v1
            remain_point_id = v2
            new_size_value = self.sizing_values[v2]
            is_center = False

        timers['condition_check'] = time.time() - start
        
        original_remain_point_id = remain_point_id.item()

        # 3. 顶点处理
        start = time.time()
        # print(self.vertices[1512], self.vertices[1511], self.vertices[1510], self.vertices[1452])
        new_vertices = torch.cat([self.vertices[:delete_point_id], self.vertices[delete_point_id+1:]])
        new_feature_points = torch.cat([self.feature_point[:delete_point_id], self.feature_point[delete_point_id+1:]])
        if delete_point_id < remain_point_id:
            new_vertices[remain_point_id - 1] = new_vertex_pos
            new_feature_points[remain_point_id - 1] = self.feature_point[remain_point_id]
        else:
            new_vertices[remain_point_id] = new_vertex_pos
            new_feature_points[remain_point_id] = self.feature_point[remain_point_id]
        # print(new_vertices[1511], new_vertices[1510], new_vertices[1509], new_vertices[1452])
        
        timers['vertex_processing'] = time.time() - start
        
        # 4. 面处理
        start = time.time()
        new_faces = self.faces.clone()

        
        # if delete_point_id < remain_point_id:
        #     new_faces[new_faces > delete_point_id] -= 1
        #     new_faces[new_faces == delete_point_id] = remain_point_id
        # elif delete_point_id > remain_point_id:
        #     new_faces[new_faces == delete_point_id] = remain_point_id
        #     new_faces[new_faces > delete_point_id] -= 1

        new_faces[new_faces == delete_point_id] = remain_point_id
        new_faces[new_faces > delete_point_id] -= 1


        flag = True
        
        if delete_point_id < remain_point_id:
            remain_point_id -= 1
            flag = False

        # 检查退化面
        unique_counts = torch.tensor([torch.unique(tri).size(0) for tri in new_faces])
        non_degenerate_mask = unique_counts == 3
        newfaces = new_faces[non_degenerate_mask]
        timers['face_processing'] = time.time() - start

        # 5. post_face处理
        start = time.time()
        post_face = []
        if is_center:
            for n, face in enumerate(newfaces):
                if remain_point_id in face:
                    post_face.append(face)
        else:
            if not flag:
                operate_id = torch.tensor(original_remain_point_id)
            else:
                operate_id = remain_point_id
            for face in self.faces:
                if delete_point_id in face and operate_id not in face:
                    modified_face = [operate_id if v == delete_point_id else v for v in face]
                    modified_face = torch.tensor(modified_face)
                    modified_face[modified_face > delete_point_id] -= 1
                    post_face.append(modified_face)

        timers['post_face_processing'] = time.time() - start
        
        # 6. 新网格构建
        start = time.time()
        new_regions = self.surface_id[non_degenerate_mask]
        new_sizing_values = torch.cat([self.sizing_values[:delete_point_id], 
                                    self.sizing_values[delete_point_id+1:]])
        new_sizing_values[remain_point_id] = new_size_value
        
        mesh = CustomMesh(vertices=new_vertices, faces=newfaces, 
                        sizing_values=new_sizing_values, regions=new_regions, feature_points=new_feature_points)
        timers['mesh_construction'] = time.time() - start
        
        return mesh, is_center, remain_point_id, delete_point_id, post_face
    

    def collapsing_edge_new1(self, v1, v2):
        timers = {
            'input_check': 0,
            'condition_check': 0,
            'vertex_processing': 0,
            'face_processing': 0,
            'post_face_processing': 0,
            'mesh_construction': 0
        }
        
        # 1. 输入检查
        start = time.time()
        if v1 == v2:
            raise ValueError("边的两个顶点不能相同")
        timers['input_check'] = time.time() - start
        
        # 初始化变量
        delete_point_id = -1
        remain_point_id = -1
        new_size_value = -1
        is_center = True
        
        # 2. 条件判断区块
        start = time.time()
        if self.feature_point[v1] == self.feature_point[v2]:
            if not self.feature_point[v1]:
                new_vertex_pos = (self.vertices[v1] + self.vertices[v2]) / 2
                delete_point_id = v2
                remain_point_id = v1
                new_size_value = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
                is_center = True
            else:
                v1_condition = is_vertex_appearing_more_than_twice(self.feature_point, self.edges, v1)
                v2_condition = is_vertex_appearing_more_than_twice(self.feature_point,self.edges, v2)
                if v1_condition == v2_condition:
                    new_vertex_pos = (self.vertices[v1] + self.vertices[v2]) / 2
                    delete_point_id = v2
                    remain_point_id = v1
                    new_size_value = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
                else:
                    if v1_condition > v2_condition:
                        new_vertex_pos = self.vertices[v1]
                        delete_point_id = v2
                        remain_point_id = v1
                        new_size_value = self.sizing_values[v1]
                        is_center = False
                    elif v1_condition < v2_condition:
                        new_vertex_pos = self.vertices[v2]
                        delete_point_id = v1
                        remain_point_id = v2
                        new_size_value = self.sizing_values[v2]
                        is_center = False

        elif self.feature_point[v1] and not self.feature_point[v2]:
            new_vertex_pos = self.vertices[v1]
            delete_point_id = v2
            remain_point_id = v1
            new_size_value = self.sizing_values[v1]
            is_center = False
        elif self.feature_point[v2] and not self.feature_point[v1]:
            new_vertex_pos = self.vertices[v2]
            delete_point_id = v1
            remain_point_id = v2
            new_size_value = self.sizing_values[v2]
            is_center = False

        timers['condition_check'] = time.time() - start
        
        # 3. 顶点处理
        start = time.time()
        new_vertices = torch.cat([self.vertices[:delete_point_id], self.vertices[delete_point_id+1:]])
        new_vertices[remain_point_id] = new_vertex_pos
        new_feature_points = torch.cat([self.feature_point[:delete_point_id], self.feature_point[delete_point_id+1:]])
        timers['vertex_processing'] = time.time() - start
        
        # 4. 面处理
        start = time.time()
        new_faces = self.faces.clone()
        new_faces[new_faces == delete_point_id] = remain_point_id
        new_faces[new_faces > delete_point_id] -= 1

        deleted_edge_indices = []
        # 识别与坍缩边相关的面
        edge_faces = set()
        delete_edges = []
        for i, face in enumerate(self.faces):
            if (v1 in face) and (v2 in face):
                # 获取第三个顶点（既不是v1也不是v2的顶点）
                # third_v = next(v for v in face if v not in {v1, v2})
                third_v = -1
                for v in face:
                    if v.item() != v1 and v.item() != v2:
                        third_v = v
                delete_edge = torch.tensor(sorted((third_v.item(), delete_point_id)))
                delete_edges.append(delete_edge)
        delete_edges.append(torch.tensor(sorted((v1, v2))))
        for i, edge in enumerate(self.edges):
            for e in delete_edges:
                if e[0] == edge[0] and e[1] == edge[1]:
                    deleted_edge_indices.append(i)
                
        
        # 检查退化面
        unique_counts = torch.tensor([torch.unique(tri).size(0) for tri in new_faces])
        non_degenerate_mask = unique_counts == 3
        newfaces = new_faces[non_degenerate_mask]
        timers['face_processing'] = time.time() - start
        
        # 5. post_face处理
        start = time.time()
        post_face = []
        if is_center:
            for n, face in enumerate(newfaces):
                if remain_point_id in face:
                    post_face.append(face)
        else:
            for face in self.faces:
                if delete_point_id in face and remain_point_id not in face:
                    modified_face = [remain_point_id if v == delete_point_id else v for v in face]
                    modified_face = torch.tensor(modified_face)
                    modified_face[modified_face > delete_point_id] -= 1
                    post_face.append(modified_face)
        modified_edges = set()
        for face in post_face:
            third_v = remain_point_id
            modified_edges.add(tuple(sorted((face[0].item(), face[1].item()))))
            modified_edges.add(tuple(sorted((face[1].item(), face[2].item()))))
            modified_edges.add(tuple(sorted((face[0].item(), face[2].item()))))

        timers['post_face_processing'] = time.time() - start
        
        # 6. 新网格构建
        start = time.time()
        new_regions = self.surface_id[non_degenerate_mask]
        new_sizing_values = torch.cat([self.sizing_values[:delete_point_id], 
                                    self.sizing_values[delete_point_id+1:]])
        new_sizing_values[remain_point_id] = new_size_value
        
        mesh = CustomMesh(vertices=new_vertices, faces=newfaces, 
                        sizing_values=new_sizing_values, regions=new_regions, feature_points=new_feature_points)
        timers['mesh_construction'] = time.time() - start

        return mesh, is_center, remain_point_id, modified_edges, deleted_edge_indices

    def collapsing_edge_new2(self, v1, v2):
        # 1. 输入检查优化 - 使用向量化操作替代循环
        if v1.item() == v2.item():
            return None, True, -1, -1, []
        
        # 使用张量操作替代循环检查边是否存在
        edge_exists = torch.any((self.edges == torch.tensor([v1.item(), v2.item()])).all(dim=1) | 
                            (self.edges == torch.tensor([v2.item(), v1.item()])).all(dim=1))
        if not edge_exists:
            return None, True, -1, -1, []
        
        # 初始化变量
        delete_point_id = -1
        remain_point_id = -1
        new_size_value = -1
        is_center = True
        
        # 2. 条件判断优化 - 预先计算特征点条件
        v1_feature = self.feature_point[v1]
        v2_feature = self.feature_point[v2]
        
        if v1_feature == v2_feature:
            if not v1_feature:
                # 两个都不是特征点
                new_vertex_pos = (self.vertices[v1] + self.vertices[v2]) / 2
                delete_point_id = v2
                remain_point_id = v1
                new_size_value = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
                is_center = True
            else:
                # 两个都是特征点 - 预先缓存特征点出现次数
                if not hasattr(self, '_feature_count_cache'):
                    # 缓存特征点出现次数
                    self._feature_count_cache = self._precompute_feature_counts()
                
                v1_count = self._feature_count_cache[v1.item()]
                v2_count = self._feature_count_cache[v2.item()]
                
                if v1_count == v2_count:
                    new_vertex_pos = (self.vertices[v1] + self.vertices[v2]) / 2
                    delete_point_id = v2
                    remain_point_id = v1
                    new_size_value = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
                else:
                    if v1_count > v2_count:
                        new_vertex_pos = self.vertices[v1]
                        delete_point_id = v2
                        remain_point_id = v1
                        new_size_value = self.sizing_values[v1]
                        is_center = False
                    else:
                        new_vertex_pos = self.vertices[v2]
                        delete_point_id = v1
                        remain_point_id = v2
                        new_size_value = self.sizing_values[v2]
                        is_center = False
        elif v1_feature and not v2_feature:
            new_vertex_pos = self.vertices[v1]
            delete_point_id = v2
            remain_point_id = v1
            new_size_value = self.sizing_values[v1]
            is_center = False
        else:  # v2_feature and not v1_feature
            new_vertex_pos = self.vertices[v2]
            delete_point_id = v1
            remain_point_id = v2
            new_size_value = self.sizing_values[v2]
            is_center = False

        original_remain_point_id = remain_point_id.item()
        delete_idx = delete_point_id.item()
        remain_idx = remain_point_id.item()

        # 3. 顶点处理优化 - 使用索引操作避免不必要的拼接
        # 创建新顶点数组，直接赋值而不是拼接
        new_vertices = self.vertices.clone()
        new_feature_points = self.feature_point.clone()
        new_sizing_values = self.sizing_values.clone()
        
        # 更新保留点的位置和属性
        new_vertices[remain_point_id] = new_vertex_pos
        new_sizing_values[remain_point_id] = new_size_value
        
        # 创建删除掩码
        keep_mask = torch.ones(len(self.vertices), dtype=torch.bool)
        keep_mask[delete_point_id] = False
        
        # 应用删除掩码
        new_vertices = new_vertices[keep_mask]
        new_feature_points = new_feature_points[keep_mask]
        new_sizing_values = new_sizing_values[keep_mask]
        
        # 4. 面处理优化 - 使用向量化操作
        new_faces = self.faces.clone()
        
        # 一次性更新面索引
        # 先将删除点替换为保留点
        new_faces[new_faces == delete_idx] = remain_idx
        
        # 调整大于删除点的索引
        mask = new_faces > delete_idx
        new_faces[mask] -= 1
        
        # 检查退化面 - 使用向量化操作
        # 检查每个面是否有重复顶点
        face_unique_counts = (new_faces[:, 0] != new_faces[:, 1]) & \
                            (new_faces[:, 0] != new_faces[:, 2]) & \
                            (new_faces[:, 1] != new_faces[:, 2])
        
        newfaces = new_faces[face_unique_counts]
        
        # 更新保留点索引
        if delete_idx < remain_idx:
            remain_idx -= 1
            flag = False
        else:
            flag = True

        # 5. post_face处理优化
        post_face = []
        if is_center:
            # 使用向量化查找包含保留点的面
            contain_remain = torch.any(newfaces == remain_idx, dim=1)
            post_face_tensor = newfaces[contain_remain]
            post_face = [face for face in post_face_tensor]
        else:
            operate_id = original_remain_point_id if not flag else remain_idx
            
            # 预计算需要处理的面
            original_delete_faces = self.faces[torch.any(self.faces == delete_idx, dim=1)]
            # 排除已经包含操作点的面
            faces_to_process = original_delete_faces[~torch.any(original_delete_faces == operate_id, dim=1)]
            
            for face in faces_to_process:
                modified_face = face.clone()
                modified_face[modified_face == delete_idx] = operate_id
                # 调整索引
                modified_face[modified_face > delete_idx] -= 1
                post_face.append(modified_face)

        # 6. 新网格构建优化
        new_regions = self.surface_id[face_unique_counts]
        
        mesh = CustomMesh(
            vertices=new_vertices, 
            faces=newfaces, 
            sizing_values=new_sizing_values, 
            regions=new_regions, 
            feature_points=new_feature_points
        )
        
        return mesh, is_center, remain_point_id, delete_point_id, post_face

    def collapsing_edges(self, edge_ids):
        timers = {
            'input_check': 0,
            'component_processing': 0,
            'vertex_processing': 0,
            'face_processing': 0,
            'post_face_processing': 0,
            'mesh_construction': 0
        }
        
        # 1. 输入检查
        start = time.time()
        if not edge_ids:
            return None, True, -1, -1
        
        # 获取所有要坍缩的边
        edges_to_collapse = []
        for edge_id in edge_ids:
            if edge_id < 0 or edge_id >= len(self.edges):
                continue
            edge = self.edges[edge_id]
            edges_to_collapse.append((edge[0].item(), edge[1].item()))
        
        if not edges_to_collapse:
            return None, True, -1, -1
        
        # 提取所有涉及的顶点
        all_vertices = set()
        for v1, v2 in edges_to_collapse:
            all_vertices.add(v1)
            all_vertices.add(v2)
        all_vertices = list(all_vertices)
        
        timers['input_check'] = time.time() - start
        
        # 2. 处理连通分量
        start = time.time()
        # 构建图并找到连通分量
        graph = {v: set() for v in all_vertices}
        for v1, v2 in edges_to_collapse:
            graph[v1].add(v2)
            graph[v2].add(v1)
        
        # 使用BFS找到连通分量
        components = []
        visited = set()
        for vertex in all_vertices:
            if vertex in visited:
                continue
            component = []
            queue = [vertex]
            visited.add(vertex)
            while queue:
                v = queue.pop(0)
                component.append(v)
                for neighbor in graph[v]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
        
        # 为每个连通分量选择保留顶点并计算新属性
        component_info = {}
        for comp in components:
            # 提取特征点
            feature_vertices = [v for v in comp if self.feature_point[v]]
            
            if not feature_vertices:
                # 没有特征点 - 使用所有顶点的平均值
                new_vertex_pos = torch.mean(torch.stack([self.vertices[v] for v in comp]), dim=0)
                new_size_value = torch.mean(torch.stack([self.sizing_values[v] for v in comp]))
                is_center = True
                # 选择度数最高的顶点作为保留点
                degrees = {v: self.get_vertex_degree(v) for v in comp}
                remain_vertex = max(degrees, key=degrees.get)
            
            else:
                # 计算特征点的"出现次数条件"
                conditions = {}
                for v in feature_vertices:
                    conditions[v] = self.is_vertex_appearing_more_than_twice(v)
                
                max_condition = max(conditions.values())
                max_cond_vertices = [v for v, cond in conditions.items() if cond == max_condition]
                
                if len(max_cond_vertices) == 1:
                    # 有唯一最高条件的特征点
                    remain_vertex = max_cond_vertices[0]
                    new_vertex_pos = self.vertices[remain_vertex]
                    new_size_value = self.sizing_values[remain_vertex]
                    is_center = False
                else:
                    # 多个特征点条件相同 - 使用这些特征点的平均值
                    new_vertex_pos = torch.mean(torch.stack([self.vertices[v] for v in max_cond_vertices]), dim=0)
                    new_size_value = torch.mean(torch.stack([self.sizing_values[v] for v in max_cond_vertices]))
                    is_center = True
                    # 选择度数最高的特征点作为保留点
                    degrees = {v: self.get_vertex_degree(v) for v in max_cond_vertices}
                    remain_vertex = max(degrees, key=degrees.get)
            
            component_info[remain_vertex] = {
                'vertices': comp,
                'new_pos': new_vertex_pos,
                'new_size': torch.tensor([new_size_value.item()]),
                'is_center': is_center,
                'feature': self.feature_point[remain_vertex] if not is_center else any(self.feature_point[v] for v in comp)
            }
        
        timers['component_processing'] = time.time() - start
        
        # 3. 顶点处理
        start = time.time()
        # 确定要删除的顶点（所有非保留顶点）
        delete_vertices = set()
        for comp in components:
            remain_vertex = next(iter(component_info.keys()))
            delete_vertices.update([v for v in comp if v != remain_vertex])
        delete_indices = list(delete_vertices)
        
        # 构建顶点索引映射表
        index_map = {}
        new_index = 0
        for orig_idx in range(len(self.vertices)):
            if orig_idx in delete_indices:
                continue
            index_map[orig_idx] = new_index
            new_index += 1
        
        # 更新顶点：删除顶点，并更新保留顶点的位置和尺寸
        updated_vertices = []
        updated_sizing = []
        updated_features = []
        for i in range(len(self.vertices)):
            if i in delete_indices:
                continue
            
            # 检查是否是某个连通分量的保留顶点
            if i in component_info:
                info = component_info[i]
                updated_vertices.append(info['new_pos'])
                updated_sizing.append(info['new_size'])
                updated_features.append(info['feature'])
            else:
                updated_vertices.append(self.vertices[i])
                updated_sizing.append(self.sizing_values[i])
                updated_features.append(self.feature_point[i])
        
        new_vertices = torch.stack(updated_vertices) if updated_vertices else torch.empty((0, 3))
        new_sizing_values = torch.stack(updated_sizing) if updated_sizing else torch.empty((0,))
        new_feature_points = torch.tensor(updated_features, dtype=torch.bool) if updated_features else torch.empty((0,), dtype=torch.bool)
        
        timers['vertex_processing'] = time.time() - start
        
        # 4. 面处理
        start = time.time()
        new_faces = []
        for face in self.faces:
            # 替换顶点：被删除的顶点映射到其连通分量的保留顶点
            new_face = []
            for vertex in face:
                v_item = vertex.item()
                # 如果顶点在删除列表中，找到它所属的连通分量保留顶点
                if v_item in delete_indices:
                    for remain_vertex, info in component_info.items():
                        if v_item in info['vertices']:
                            new_face.append(remain_vertex)
                            break
                else:
                    new_face.append(v_item)
            
            # 应用全局索引重映射
            remapped_face = [index_map[v] for v in new_face]
            
            # 检查退化面（顶点不重复）
            if len(set(remapped_face)) == 3:
                new_faces.append(remapped_face)
        
        new_faces = torch.tensor(new_faces, dtype=torch.long) if new_faces else torch.empty((0, 3), dtype=torch.long)
        

        
        # 6. 新网格构建
        start = time.time()
        valid_mask = [i for i in range(len(self.faces)) 
                 if i < len(new_faces)]
    
        if len(valid_mask) != len(new_faces):
            # 安全处理：如果数量不匹配，创建全1的区域ID
            new_regions = torch.ones(len(new_faces), dtype=torch.long)
        else:
            new_regions = self.surface_id[valid_mask]
        
        mesh = CustomMesh(vertices=new_vertices, faces=new_faces,
                        sizing_values=new_sizing_values, regions=new_regions,
                        feature_points=new_feature_points)
        timers['mesh_construction'] = time.time() - start
        
        # 返回保留顶点的新索引
        remain_vertices_new_idx = {v: index_map[v] for v in component_info.keys()}
        return mesh, True, remain_vertices_new_idx, delete_indices



    # def collapse_multiple_edges(self, edges_to_collapse):
    #     """
    #     向量化实现多个边的坍缩，保持未受影响顶点的拓扑结构并删除零面积面片
        
    #     参数:
    #     edges_to_collapse - 要坍缩的边列表，每个元素为(v1, v2)元组
    #     """
    #     # 输入验证
    #     if not edges_to_collapse:
    #         return self, [], [], []
        
    #     # # 转换为numpy数组以便处理
    #     # edges_array = np.array(edges_to_collapse)
    #     # v1_list = edges_array[:, 0]
    #     # v2_list = edges_array[:, 1]
        
    #     # 1. 确定每个边坍缩后的新顶点位置和删除策略
    #     new_vertex_positions = []
    #     delete_points = []
    #     remain_points = []
    #     new_size_values = []
        
    #     for edge_id in edges_to_collapse:
    #         v1, v2 = self.edges[edge_id.item()]
    #         if self.feature_point[v1] == self.feature_point[v2]:
    #             if not self.feature_point[v1]:
    #                 # 非特征点取中点
    #                 new_pos = (self.vertices[v1] + self.vertices[v2]) / 2
    #                 delete_points.append(v2)
    #                 remain_points.append(v1)
    #                 new_size = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
    #             else:
    #                 # 特征点处理
    #                 v1_condition = is_vertex_appearing_more_than_twice(self.feature_point, self.edges, v1)
    #                 v2_condition = is_vertex_appearing_more_than_twice(self.feature_point, self.edges, v2)
    #                 if v1_condition >= v2_condition:
    #                     new_pos = self.vertices[v1]
    #                     delete_points.append(v2)
    #                     remain_points.append(v1)
    #                     new_size = self.sizing_values[v1]
    #                 else:
    #                     new_pos = self.vertices[v2]
    #                     delete_points.append(v1)
    #                     remain_points.append(v2)
    #                     new_size = self.sizing_values[v2]
    #         elif self.feature_point[v1] and not self.feature_point[v2]:
    #             new_pos = self.vertices[v1]
    #             delete_points.append(v2)
    #             remain_points.append(v1)
    #             new_size = self.sizing_values[v1]
    #         else:
    #             new_pos = self.vertices[v2]
    #             delete_points.append(v1)
    #             remain_points.append(v2)
    #             new_size = self.sizing_values[v2]
                
    #         new_vertex_positions.append(new_pos)
    #         new_size_values.append(new_size)
        
    #     # 2. 排序并处理顶点重映射
    #     delete_points = np.array(delete_points)
    #     remain_points = np.array(remain_points)
        
    #     # 按降序排序删除点，确保删除操作不会影响后续索引
    #     sort_indices = np.argsort(-delete_points)
    #     delete_points = delete_points[sort_indices]
    #     remain_points = remain_points[sort_indices]
    #     new_vertex_positions = [new_vertex_positions[i] for i in sort_indices]
    #     new_size_values = [new_size_values[i] for i in sort_indices]
        
    #     # 构建顶点映射表
    #     vertex_remap = np.arange(len(self.vertices))
    #     deleted_mask = np.zeros(len(self.vertices), dtype=bool)
        
    #     for delete_id, remain_id in zip(delete_points, remain_points):
    #         vertex_remap[delete_id] = remain_id
    #         deleted_mask[delete_id] = True
        
    #     # 计算删除顶点后的新索引
    #     new_indices = np.cumsum(~deleted_mask) - 1
    #     vertex_remap = new_indices[vertex_remap]
        
    #     # 3. 更新顶点坐标
    #     new_vertices = self.vertices[~deleted_mask].clone()
        
    #     # 更新保留顶点的新位置
    #     for i, (delete_id, remain_id, new_pos) in enumerate(zip(delete_points, remain_points, new_vertex_positions)):
    #         new_idx = vertex_remap[remain_id]
    #         new_vertices[new_idx] = new_pos
        
    #     # 4. 更新面信息
    #     new_faces = self.faces.clone()
        
    #     # 替换所有要删除的顶点为保留顶点
    #     for delete_id, remain_id in zip(delete_points, remain_points):
    #         new_faces[new_faces == delete_id] = remain_id
        
    #     # 应用顶点重映射
    #     new_faces = torch.tensor(vertex_remap[new_faces.numpy()])
        
    #     # 5. 检测并移除退化面（零面积面片）
    #     def calculate_triangle_areas(vertices, faces):
    #         """计算三角形面片的面积"""
    #         v0 = vertices[faces[:, 0]]
    #         v1 = vertices[faces[:, 1]]
    #         v2 = vertices[faces[:, 2]]
            
    #         # 计算三角形两边向量
    #         edge1 = v1 - v0
    #         edge2 = v2 - v0
            
    #         # 计算叉积
    #         cross_product = torch.linalg.cross(edge1, edge2)
            
    #         # 计算面积（叉积模长的一半）
    #         areas = 0.5 * torch.norm(cross_product, dim=1)
    #         return areas
        
    #     # 设置面积阈值（可根据网格精度调整）
    #     area_threshold = 1e-10
    #     areas = calculate_triangle_areas(new_vertices, new_faces)
        
    #     # 保留面积大于阈值的面片
    #     non_degenerate_mask = areas > area_threshold
    #     new_faces = new_faces[non_degenerate_mask]
        
    #     # 7. 更新其他属性
    #     new_feature_points = self.feature_point[~deleted_mask]
    #     new_regions = self.surface_id[non_degenerate_mask]
        
    #     new_sizing_values = self.sizing_values[~deleted_mask]
    #     for i, (delete_id, remain_id, new_size) in enumerate(zip(delete_points, remain_points, new_size_values)):
    #         new_idx = vertex_remap[remain_id]
    #         new_sizing_values[new_idx] = new_size
        
    #     # 8. 构建新网格
    #     mesh = CustomMesh(
    #         vertices=new_vertices, 
    #         faces=new_faces,
    #         sizing_values=new_sizing_values, 
    #         regions=new_regions, 
    #         feature_points=new_feature_points
    #     )
        
    #     return mesh

    def collapse_multiple_edges(self, edges_to_collapse):
        """
        优化后的多个边坍缩实现，避免重叠面情况
        
        参数:
        edges_to_collapse - 要坍缩的边列表，每个元素为(v1, v2)元组
        
        返回:
        新网格对象，删除的顶点列表，保留的顶点列表，新顶点位置列表
        """
        if not edges_to_collapse:
            return self
        
        # 1. 准备数据结构
        new_vertex_positions = []
        delete_points = []
        remain_points = []
        new_size_values = []
        edge_set = set()  # 用于检测重复边
        
        # 2. 预处理边，确保没有重复或冲突的边
        unique_edges = []
        for edge_id in edges_to_collapse:
            v1, v2 = self.edges[edge_id]
            if v1 == v2:
                continue  # 跳过自环边
            
            # 确保边是唯一的且没有冲突
            if (v1, v2) not in edge_set and (v2, v1) not in edge_set:
                edge_set.add((v1, v2))
                unique_edges.append((v1, v2))
        
        if not unique_edges:
            return self
        # 3. 确定每个边的坍缩策略
        for v1, v2 in unique_edges:
            # 检查顶点是否已经被标记为删除
            if v1 in delete_points or v2 in delete_points:
                continue
                
            if self.feature_point[v1] == self.feature_point[v2]:
                if not self.feature_point[v1]:
                    # 非特征点取中点
                    new_pos = (self.vertices[v1] + self.vertices[v2]) / 2
                    delete_points.append(v2)
                    remain_points.append(v1)
                    new_size = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
                else:
                    # 特征点处理
                    v1_condition = is_vertex_appearing_more_than_twice(self.feature_point, self.edges, v1.item())
                    v2_condition = is_vertex_appearing_more_than_twice(self.feature_point, self.edges, v2.item())
                    if v1_condition >= v2_condition:
                        new_pos = self.vertices[v1]
                        delete_points.append(v2)
                        remain_points.append(v1)
                        new_size = self.sizing_values[v1]
                    else:
                        new_pos = self.vertices[v2]
                        delete_points.append(v1)
                        remain_points.append(v2)
                        new_size = self.sizing_values[v2]
            elif self.feature_point[v1] and not self.feature_point[v2]:
                new_pos = self.vertices[v1]
                delete_points.append(v2)
                remain_points.append(v1)
                new_size = self.sizing_values[v1]
            else:
                new_pos = self.vertices[v2]
                delete_points.append(v1)
                remain_points.append(v2)
                new_size = self.sizing_values[v2]
                
            new_vertex_positions.append(new_pos)
            new_size_values.append(new_size)
        
        if not delete_points:
            return self
        
        # 4. 处理顶点重映射
        delete_points = np.array(delete_points)
        remain_points = np.array(remain_points)
        
        # 按降序排序删除点，确保删除操作不会影响后续索引
        sort_indices = np.argsort(-delete_points)
        delete_points = delete_points[sort_indices]
        remain_points = remain_points[sort_indices]
        new_vertex_positions = [new_vertex_positions[i] for i in sort_indices]
        new_size_values = [new_size_values[i] for i in sort_indices]
        
        # 构建顶点映射表
        vertex_remap = np.arange(len(self.vertices))
        deleted_mask = np.zeros(len(self.vertices), dtype=bool)
        
        for delete_id, remain_id in zip(delete_points, remain_points):
            if deleted_mask[delete_id]:
                continue  # 跳过已经标记为删除的顶点
            vertex_remap[delete_id] = remain_id
            deleted_mask[delete_id] = True
        
        # 计算删除顶点后的新索引
        new_indices = np.cumsum(~deleted_mask) - 1
        vertex_remap = new_indices[vertex_remap]
        
        # 5. 更新顶点坐标
        new_vertices = self.vertices[~deleted_mask].clone()
        
        # 更新保留顶点的新位置
        for i, (delete_id, remain_id, new_pos) in enumerate(zip(delete_points, remain_points, new_vertex_positions)):
            new_idx = vertex_remap[remain_id]
            new_vertices[new_idx] = new_pos
        
        # 6. 更新面信息并检测退化面
        new_faces = self.faces.clone()
        
        # 替换所有要删除的顶点为保留顶点
        for delete_id, remain_id in zip(delete_points, remain_points):
            new_faces[new_faces == delete_id] = remain_id
        
        # 应用顶点重映射
        new_faces = torch.tensor(vertex_remap[new_faces.numpy()])
        
        # 检测并移除退化面（零面积面片）
        def calculate_triangle_areas(vertices, faces):
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross_product = torch.linalg.cross(edge1, edge2)
            return 0.5 * torch.norm(cross_product, dim=1)
        
        area_threshold = 1e-10
        areas = calculate_triangle_areas(new_vertices, new_faces)
        
        # 额外检查：移除包含重复顶点的面（防止重叠面）
        duplicate_vertex_mask = (new_faces[:, 0] == new_faces[:, 1]) | \
                            (new_faces[:, 1] == new_faces[:, 2]) | \
                            (new_faces[:, 0] == new_faces[:, 2])
        
        non_degenerate_mask = (areas > area_threshold) & ~duplicate_vertex_mask
        new_faces = new_faces[non_degenerate_mask]
        
        # 7. 更新其他属性
        new_feature_points = self.feature_point[~deleted_mask]
        new_regions = self.surface_id[non_degenerate_mask]
        
        new_sizing_values = self.sizing_values[~deleted_mask]
        for i, (delete_id, remain_id, new_size) in enumerate(zip(delete_points, remain_points, new_size_values)):
            new_idx = vertex_remap[remain_id]
            new_sizing_values[new_idx] = new_size
        
        # 8. 构建新网格
        mesh = CustomMesh(
            vertices=new_vertices, 
            faces=new_faces,
            sizing_values=new_sizing_values, 
            regions=new_regions,
            version=1
        )
        
        return mesh
    
    def collapse_multiple_edges1(self, edges_to_collapse):
        """
        优化后的多个边坍缩实现，避免重叠面情况
        
        参数:
        edges_to_collapse - 要坍缩的边列表，每个元素为(v1, v2)元组
        
        返回:
        新网格对象，拓扑改变的三角面片列表（在新网格中的顶点索引）
        """
        if not edges_to_collapse:
            return self, []
        
        # 1. 准备数据结构
        new_vertex_positions = []
        delete_points = []
        remain_points = []
        new_size_values = []
        edge_set = set()  # 用于检测重复边
        changed_triangles_indices = set()  # 用于记录拓扑发生改变的三角面片索引（原网格中的索引）

        # 2. 预处理边，确保没有重复或冲突的边
        unique_edges = []
        for edge_id in edges_to_collapse:
            v1, v2 = self.edges[edge_id]
            if v1 == v2:
                continue  # 跳过自环边
            # # 确保边是唯一的且没有冲突
            if (v1, v2) not in edge_set and (v2, v1) not in edge_set:
                edge_set.add((v1, v2))
                unique_edges.append((v1, v2))
        
        if not unique_edges:
            return self, []
        
        # 3. 确定每个边的坍缩策略
        for v1, v2 in unique_edges:
            # 检查顶点是否已经被标记为删除
            if v1 in delete_points or v2 in delete_points:
                continue
                
            if self.feature_point[v1] == self.feature_point[v2]:
                if not self.feature_point[v1]:
                    # 非特征点取中点
                    new_pos = (self.vertices[v1] + self.vertices[v2]) / 2
                    delete_points.append(v2)
                    remain_points.append(v1)
                    new_size = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
                else:
                    # 特征点处理
                    # v1_condition = is_vertex_appearing_more_than_twice(self.feature_point, self.edges, v1)
                    # v2_condition = is_vertex_appearing_more_than_twice(self.feature_point, self.edges, v2)
                    # if v1_condition >= v2_condition:
                    #     new_pos = self.vertices[v1]
                    #     delete_points.append(v2)
                    #     remain_points.append(v1)
                    #     new_size = self.sizing_values[v1]
                    # else:
                    #     new_pos = self.vertices[v2]
                    #     delete_points.append(v1)
                    #     remain_points.append(v2)
                    #     new_size = self.sizing_values[v2]
                    new_pos = (self.vertices[v1] + self.vertices[v2]) / 2
                    delete_points.append(v2)
                    remain_points.append(v1)
                    new_size = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
            elif self.feature_point[v1] and not self.feature_point[v2]:
                new_pos = self.vertices[v1]
                delete_points.append(v2)
                remain_points.append(v1)
                new_size = self.sizing_values[v1]
            elif self.feature_point[v2] and not self.feature_point[v1]:
                new_pos = self.vertices[v2]
                delete_points.append(v1)
                remain_points.append(v2)
                new_size = self.sizing_values[v2]
                
            new_vertex_positions.append(new_pos)
            new_size_values.append(new_size)
        
        if not delete_points:
            return self, []
        
        # 4. 处理顶点重映射
        delete_points = np.array(delete_points)
        remain_points = np.array(remain_points)
        
        # 按降序排序删除点，确保删除操作不会影响后续索引
        sort_indices = np.argsort(-delete_points)
        delete_points = delete_points[sort_indices]
        remain_points = remain_points[sort_indices]
        new_vertex_positions = [new_vertex_positions[i] for i in sort_indices]
        new_size_values = [new_size_values[i] for i in sort_indices]
        
        # 构建顶点映射表
        vertex_remap = np.arange(len(self.vertices))
        deleted_mask = np.zeros(len(self.vertices), dtype=bool)
        
        for delete_id, remain_id in zip(delete_points, remain_points):
            if deleted_mask[delete_id]:
                continue  # 跳过已经标记为删除的顶点
            vertex_remap[delete_id] = remain_id
            deleted_mask[delete_id] = True
        
        # 计算删除顶点后的新索引
        new_indices = np.cumsum(~deleted_mask) - 1
        vertex_remap = new_indices[vertex_remap]
        
        # 5. 更新顶点坐标
        new_vertices = self.vertices[~deleted_mask].clone()
        
        # 更新保留顶点的新位置
        for i, (delete_id, remain_id, new_pos) in enumerate(zip(delete_points, remain_points, new_vertex_positions)):
            new_idx = vertex_remap[remain_id]
            new_vertices[new_idx] = new_pos
        
        # 6. 更新面信息并检测退化面
        new_faces = self.faces.clone()
        
        # 首先收集所有包含要删除或保留顶点的面片（这些面片的拓扑可能改变）
        for delete_id, remain_id in zip(delete_points, remain_points):
            # 找到所有包含要删除顶点的面
            face_mask = ((new_faces == delete_id) | (new_faces == remain_id)).any(dim=1)
            changed_triangles_indices.update(torch.where(face_mask)[0].tolist())
            
            # 替换顶点
            new_faces[face_mask] = torch.where(
                new_faces[face_mask] == delete_id,
                torch.full_like(new_faces[face_mask], remain_id),
                new_faces[face_mask]
            )
        
        # 应用顶点重映射
        new_faces = torch.tensor(vertex_remap[new_faces.numpy()])
        
        # 检测并移除退化面（零面积面片）
        def calculate_triangle_areas(vertices, faces):
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross_product = torch.linalg.cross(edge1, edge2)
            return 0.5 * torch.norm(cross_product, dim=1)
        
        area_threshold = 1e-15
        areas = calculate_triangle_areas(new_vertices, new_faces)
        
        # 额外检查：移除包含重复顶点的面（防止重叠面）
        duplicate_vertex_mask = (new_faces[:, 0] == new_faces[:, 1]) | \
                            (new_faces[:, 1] == new_faces[:, 2]) | \
                            (new_faces[:, 0] == new_faces[:, 2])
        
        non_degenerate_mask = (areas > area_threshold) & ~duplicate_vertex_mask
        deleted_faces = ~non_degenerate_mask
        
        # 创建从旧面索引到新面索引的映射
        old_to_new_face_map = {}
        new_face_idx = 0
        for old_face_idx in range(len(non_degenerate_mask)):
            if non_degenerate_mask[old_face_idx]:
                old_to_new_face_map[old_face_idx] = new_face_idx
                new_face_idx += 1
        
        # 只保留最终存在的面片
        new_faces = new_faces[non_degenerate_mask]
        
        # 获取最终保留的且拓扑改变的面片（在新网格中的表示）
        changed_triangles_new_mesh = []
        for old_idx in changed_triangles_indices:
            if old_idx in old_to_new_face_map:
                new_idx = old_to_new_face_map[old_idx]
                changed_triangles_new_mesh.append(new_faces[new_idx].tolist())

        # 7. 更新其他属性
        new_feature_points = self.feature_point[~deleted_mask]
        new_regions = self.surface_id[non_degenerate_mask]
        
        new_sizing_values = self.sizing_values[~deleted_mask]
        for i, (delete_id, remain_id, new_size) in enumerate(zip(delete_points, remain_points, new_size_values)):
            new_idx = vertex_remap[remain_id]
            new_sizing_values[new_idx] = new_size
        
        # 8. 构建新网格
        mesh = CustomMesh(
            vertices=new_vertices, 
            faces=new_faces,
            sizing_values=new_sizing_values, 
            regions=new_regions,
            feature_points=new_feature_points
        )
        
        return mesh, changed_triangles_new_mesh
    
    def collapse_multiple_edges2(self, edges_to_collapse):
        """
        优化后的多个边坍缩实现，避免重叠面情况
        """
        if not edges_to_collapse:
            return self
        
        # 计时开始
        total_start = time.time()
        
        # 1. 准备数据结构 - 常数时间 O(1)
        start1 = time.time()
        new_vertex_positions = []
        delete_points = []
        remain_points = []
        new_size_values = []
        edge_set = set()  # 用于检测重复边
        time1 = time.time() - start1
        
        # 2. 预处理边，确保没有重复或冲突的边 - O(n)，n为edges_to_collapse长度
        start2 = time.time()
        unique_edges = []
        for edge_id in edges_to_collapse:
            v1, v2 = self.edges[edge_id]
            if v1 == v2:
                continue  # 跳过自环边
            
            # 确保边是唯一的且没有冲突
            if (v1, v2) not in edge_set and (v2, v1) not in edge_set:
                edge_set.add((v1, v2))
                unique_edges.append((v1, v2))
        
        if not unique_edges:
            return self
        time2 = time.time() - start2
        
        # 3. 确定每个边的坍缩策略 - O(m)，m为unique_edges长度
        start3 = time.time()
        # for v1, v2 in unique_edges:
        #     # 检查顶点是否已经被标记为删除
        #     if v1 in delete_points or v2 in delete_points:
        #         continue
                
        #     if self.feature_point[v1] == self.feature_point[v2]:
        #         if not self.feature_point[v1]:
        #             # 非特征点取中点
        #             new_pos = (self.vertices[v1] + self.vertices[v2]) / 2
        #             delete_points.append(v2)
        #             remain_points.append(v1)
        #             new_size = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
        #         else:
        #             # 特征点处理
        #             v1_condition = is_vertex_appearing_more_than_twice(self.feature_point, self.edges, v1.item())
        #             v2_condition = is_vertex_appearing_more_than_twice(self.feature_point, self.edges, v2.item())
        #             if v1_condition >= v2_condition:
        #                 new_pos = self.vertices[v1]
        #                 delete_points.append(v2)
        #                 remain_points.append(v1)
        #                 new_size = self.sizing_values[v1]
        #             else:
        #                 new_pos = self.vertices[v2]
        #                 delete_points.append(v1)
        #                 remain_points.append(v2)
        #                 new_size = self.sizing_values[v2]
        #     elif self.feature_point[v1] and not self.feature_point[v2]:
        #         new_pos = self.vertices[v1]
        #         delete_points.append(v2)
        #         remain_points.append(v1)
        #         new_size = self.sizing_values[v1]
        #     else:
        #         new_pos = self.vertices[v2]
        #         delete_points.append(v1)
        #         remain_points.append(v2)
        #         new_size = self.sizing_values[v2]
                
        #     new_vertex_positions.append(new_pos)
        #     new_size_values.append(new_size)

        # 假设unique_edges是一个形状为(N, 2)的numpy数组
        # 提取v1和v2
        unique_edges = np.array(unique_edges)
        # 预处理：过滤掉包含待删除点的边
        valid_mask = ~(np.isin(unique_edges[:, 0], delete_points) | np.isin(unique_edges[:, 1], delete_points))
        valid_edges = unique_edges[valid_mask]

        if len(valid_edges) == 0:
            return

        v1_arr = valid_edges[:, 0]
        v2_arr = valid_edges[:, 1]

        # 获取特征点信息
        v1_feature = self.feature_point[v1_arr]
        v2_feature = self.feature_point[v2_arr]

        # 情况1：两个顶点都是非特征点或都是特征点
        same_feature_mask = v1_feature == v2_feature

        # 情况1.1：都是非特征点
        non_feature_mask = same_feature_mask & ~v1_feature
        v1_non_feature = v1_arr[non_feature_mask]
        v2_non_feature = v2_arr[non_feature_mask]

        # 计算中点和新尺寸
        new_pos_non_feature = np.array((self.vertices[v1_non_feature] + self.vertices[v2_non_feature]) / 2)
        new_size_non_feature = (self.sizing_values[v1_non_feature] + self.sizing_values[v2_non_feature]) / 2

        # 标记要删除的点（v2）和保留的点（v1）
        delete_points.extend(v2_non_feature.tolist())
        remain_points.extend(v1_non_feature.tolist())

        # 情况1.2：都是特征点
        feature_mask = same_feature_mask & v1_feature
        v1_feature_verts = v1_arr[feature_mask]
        v2_feature_verts = v2_arr[feature_mask]

        # 检查顶点出现次数（需要单独处理，因为is_vertex_appearing_more_than_twice可能无法直接向量化）
        v1_conditions = np.array([is_vertex_appearing_more_than_twice(self.feature_point, self.edges, v1.item()) 
                                for v1 in v1_feature_verts])
        v2_conditions = np.array([is_vertex_appearing_more_than_twice(self.feature_point, self.edges, v2.item()) 
                                for v2 in v2_feature_verts])

        # 选择条件更好的顶点
        v1_better_mask = v1_conditions >= v2_conditions
        v2_better_mask = ~v1_better_mask

        # 处理v1条件更好的情况
        delete_points.extend(v2_feature_verts[v1_better_mask].tolist())
        remain_points.extend(v1_feature_verts[v1_better_mask].tolist())
        new_pos_feature_v1 = self.vertices[v1_feature_verts[v1_better_mask]]
        new_size_feature_v1 = self.sizing_values[v1_feature_verts[v1_better_mask]]

        # 处理v2条件更好的情况
        delete_points.extend(v1_feature_verts[v2_better_mask].tolist())
        remain_points.extend(v2_feature_verts[v2_better_mask].tolist())
        new_pos_feature_v2 = self.vertices[v2_feature_verts[v2_better_mask]]
        new_size_feature_v2 = self.sizing_values[v2_feature_verts[v2_better_mask]]

        # 合并特征点结果
        new_pos_feature = np.concatenate([new_pos_feature_v1, new_pos_feature_v2])
        new_size_feature = torch.from_numpy(np.concatenate([new_size_feature_v1, new_size_feature_v2]))

        # 情况2：v1是特征点，v2不是
        v1_feature_v2_not_mask = v1_feature & ~v2_feature
        v1_feature_only = v1_arr[v1_feature_v2_not_mask]
        v2_feature_only = v2_arr[v1_feature_v2_not_mask]

        delete_points.extend(v2_feature_only.tolist())
        remain_points.extend(v1_feature_only.tolist())
        new_pos_v1_feature = np.array(self.vertices[v1_feature_only])
        new_size_v1_feature = self.sizing_values[v1_feature_only]

        # 情况3：v2是特征点，v1不是
        v2_feature_v1_not_mask = v2_feature & ~v1_feature
        v1_not_feature = v1_arr[v2_feature_v1_not_mask]
        v2_not_feature = v2_arr[v2_feature_v1_not_mask]

        delete_points.extend(v1_not_feature.tolist())
        remain_points.extend(v2_not_feature.tolist())
        new_pos_v2_feature = np.array(self.vertices[v2_not_feature])
        new_size_v2_feature = self.sizing_values[v2_not_feature]

        # 合并所有结果
        new_vertex_positions.extend([
            *new_pos_non_feature,
            *new_pos_feature,
            *new_pos_v1_feature,
            *new_pos_v2_feature
        ])

        new_size_values.extend([
            *new_size_non_feature,
            *new_size_feature,
            *new_size_v1_feature,
            *new_size_v2_feature
        ])
        
        if not delete_points:
            return self
        time3 = time.time() - start3
        
        # 4. 处理顶点重映射 - O(v)，v为顶点数量
        start4 = time.time()
        delete_points = np.array(delete_points)
        remain_points = np.array(remain_points)
        
        # 按降序排序删除点，确保删除操作不会影响后续索引
        sort_indices = np.argsort(-delete_points)
        delete_points = delete_points[sort_indices]
        remain_points = remain_points[sort_indices]
        # new_vertex_positions = [
        #     torch.as_tensor(item).clone().detach() 
        #     if isinstance(item, np.ndarray) 
        #     else item.clone().detach() 
        #     for item in new_vertex_positions
        # ]
        # new_size_values = [
        #     torch.as_tensor(item).clone().detach() 
        #     if isinstance(item, np.ndarray) 
        #     else item.clone().detach() 
        #     for item in new_vertex_positions
        # ]
        new_vertex_positions = [new_vertex_positions[i] for i in sort_indices]
        new_size_values = [new_size_values[i] for i in sort_indices]
        
        # 构建顶点映射表
        vertex_remap = np.arange(len(self.vertices))
        deleted_mask = np.zeros(len(self.vertices), dtype=bool)
        
        for delete_id, remain_id in zip(delete_points, remain_points):
            if deleted_mask[delete_id]:
                continue  # 跳过已经标记为删除的顶点
            vertex_remap[delete_id] = remain_id
            deleted_mask[delete_id] = True
        
        # 计算删除顶点后的新索引
        new_indices = np.cumsum(~deleted_mask) - 1
        vertex_remap = new_indices[vertex_remap]
        time4 = time.time() - start4
        
        # 5. 更新顶点坐标 - O(v)
        start5 = time.time()
        new_vertices = self.vertices[~deleted_mask].clone()
        
        # 更新保留顶点的新位置
        for i, (delete_id, remain_id, new_pos) in enumerate(zip(delete_points, remain_points, new_vertex_positions)):
            new_idx = vertex_remap[remain_id]
            new_vertices[new_idx] = torch.from_numpy(new_pos)
        time5 = time.time() - start5
        
        # 6. 更新面信息并检测退化面 - O(f)，f为面数量
        start6 = time.time()
        new_faces = self.faces.clone()
        
        # 替换所有要删除的顶点为保留顶点
        for delete_id, remain_id in zip(delete_points, remain_points):
            new_faces[new_faces == delete_id] = remain_id
        
        # 应用顶点重映射
        new_faces = torch.tensor(vertex_remap[new_faces.numpy()])
        
        # 检测并移除退化面（零面积面片）
        def calculate_triangle_areas(vertices, faces):
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross_product = torch.linalg.cross(edge1, edge2)
            return 0.5 * torch.norm(cross_product, dim=1)
        
        area_threshold = 1e-10
        areas = calculate_triangle_areas(new_vertices, new_faces)
        
        # 额外检查：移除包含重复顶点的面（防止重叠面）
        duplicate_vertex_mask = (new_faces[:, 0] == new_faces[:, 1]) | \
                            (new_faces[:, 1] == new_faces[:, 2]) | \
                            (new_faces[:, 0] == new_faces[:, 2])
        
        non_degenerate_mask = (areas > area_threshold) & ~duplicate_vertex_mask
        new_faces = new_faces[non_degenerate_mask]
        time6 = time.time() - start6
        
        # 7. 更新其他属性 - O(v + f)
        start7 = time.time()
        new_feature_points = self.feature_point[~deleted_mask]
        new_regions = self.surface_id[non_degenerate_mask]
        
        new_sizing_values = self.sizing_values[~deleted_mask]
        for i, (delete_id, remain_id, new_size) in enumerate(zip(delete_points, remain_points, new_size_values)):
            new_idx = vertex_remap[remain_id]
            new_sizing_values[new_idx] = new_size
        time7 = time.time() - start7
        
        # 8. 构建新网格 - O(1)
        start8 = time.time()
        mesh = CustomMesh(
            vertices=new_vertices, 
            faces=new_faces,
            sizing_values=new_sizing_values, 
            regions=new_regions,
            # version=1
        )
        time8 = time.time() - start8
        
        total_time = time.time() - total_start
        
        # 打印各部分时间
        # print(f"各部分时间消耗:")
        # print(f"1. 准备数据结构: {time1:.6f}s")
        # print(f"2. 预处理边: {time2:.6f}s")
        # print(f"3. 确定坍缩策略: {time3:.6f}s")
        # print(f"4. 顶点重映射: {time4:.6f}s")
        # print(f"5. 更新顶点坐标: {time5:.6f}s")
        # print(f"6. 更新面信息: {time6:.6f}s")
        # print(f"7. 更新其他属性: {time7:.6f}s")
        # print(f"8. 构建新网格: {time8:.6f}s")
        # print(f"总时间: {total_time:.6f}s")
        
        return mesh


    def check_edge_face_intersections1(self):
        """检测边是否穿过面（向量化实现）
        返回:
            Flag (bool): 是否所有边都不穿过任何非邻接面
        """
        vertices = self.vertices  # shape: (V, 3)
        faces = self.faces        # shape: (F, 3)
        edges = self.edges        # shape: (E, 2)
        
        # 1. 预处理：排除邻接面（边和面共享顶点的情况）
        # 构建边-顶点关系矩阵 (E, V)
        edge_vertex_mask = torch.zeros((edges.size(0), vertices.size(0)), dtype=torch.float32)
        edge_vertex_mask[torch.arange(edges.size(0)).unsqueeze(1), edges] = 1.0
        
        # 构建面-顶点关系矩阵 (F, V)
        face_vertex_mask = torch.zeros((faces.size(0), vertices.size(0)), dtype=torch.float32)
        face_vertex_mask[torch.arange(faces.size(0)).unsqueeze(1), faces] = 1.0
        
        # 计算边和面是否共享顶点 (E, F)
        shared_vertex = (edge_vertex_mask @ face_vertex_mask.T) > 0.5  # 使用float矩阵乘法后比较
        
        # 2. 准备射线和三角形数据
        ray_origins = vertices[edges[:, 0]]  # (E, 3)
        ray_dirs = vertices[edges[:, 1]] - ray_origins  # (E, 3)
        
        # 三角形顶点 (F, 3, 3)
        triangles = vertices[faces]
        
        # 3. 批量计算射线-三角形相交（Möller-Trumbore算法）
        v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]  # 每个 (F, 3)
        e1 = v1 - v0  # (F, 3)
        e2 = v2 - v0  # (F, 3)
        
        # 计算行列式（使用广播）
        # h shape: (E, F, 3)
        h = torch.cross(
            ray_dirs.unsqueeze(1).expand(-1, faces.size(0), -1), 
            e2.unsqueeze(0).expand(edges.size(0), -1, -1), 
            dim=-1
        )
        
        # det shape: (E, F)
        det = torch.einsum('efi,efi->ef', h, e1.unsqueeze(0).expand_as(h))
        
        # 避免平行情况（行列式接近0）
        parallel = torch.abs(det) < 1e-8
        det = det + parallel.float()  # 平行时det=1避免除0
        
        # 计算u参数
        a = ray_origins.unsqueeze(1) - v0.unsqueeze(0)  # (E, F, 3)
        pvec = torch.cross(ray_dirs.unsqueeze(1), e1.unsqueeze(0), dim=-1)  # (E, F, 3)
        u = torch.einsum('efi,efi->ef', a, pvec) / det
        
        # 计算v参数
        qvec = torch.cross(a, e1.unsqueeze(0), dim=-1)  # (E, F, 3)
        v = torch.einsum('efi,efi->ef', ray_dirs.unsqueeze(1), qvec) / det
        
        # 计算t参数
        t = torch.einsum('efi,efi->ef', e2.unsqueeze(0), qvec) / det
        
        # 4. 判断相交条件
        valid = (~parallel) & (u >= -1e-6) & (v >= -1e-6) & ((u + v) <= 1 + 1e-6) & (t >= -1e-6) & (t <= 1 + 1e-6)
        
        # 5. 排除邻接面的相交情况
        valid = valid & (~shared_vertex)
        
        # 6. 检查是否有任何相交
        return not torch.any(valid)

    def get_max_size(self, para):
        xmax = -1000000000
        ymax = -1000000000
        zmax = -1000000000
        xmin = 1000000000
        ymin = 1000000000
        zmin = 1000000000
        for v in self.vertices:
            xmax = max(xmax, v[0])
            ymax = max(ymax, v[1])
            zmax = max(zmax, v[2])
            xmin = min(xmin, v[0])
            ymin = min(ymin, v[1])
            zmin = min(zmin, v[2])
        xLength = xmax - xmin
        yLength = ymax - ymin
        zLength = zmax - zmin

        L = max(xLength, max(yLength, zLength))

        return (L / para).item()

    def clean_edge(self, edge_ids):
        result = []
        used_vertices = set()
        for id in edge_ids:
            edge = self.edges[id]
            # 假设边用两个顶点表示，如 (vertex1, vertex2)
            vertex1, vertex2 = edge
            if vertex1 not in used_vertices and vertex2 not in used_vertices:
                result.append(id)
                used_vertices.add(vertex1)
                used_vertices.add(vertex2)
        return result

    def if_manifold(self):
        # 将faces和edges转换为NumPy数组
        faces = np.array(self.faces)
        edges = np.array(self.edges)
        
        # 对每个边，检查它的两个顶点是否都在某个面中
        # 使用向量化操作替代循环
        for edge in edges:
            v1, v2 = edge
            # 检查每个面是否同时包含v1和v2
            mask_v1 = (faces == v1).any(axis=1)
            mask_v2 = (faces == v2).any(axis=1)
            mask_both = mask_v1 & mask_v2
            tris_index = np.where(mask_both)[0]
            
            count = len(tris_index)
            
            if count != 2:
                # print(f"Error Edge {v1, v2}, have {count} adjuest face {tris_index} {self.faces[tris_index]}")
                return False
        
        return True

    def writeVTK(self, filename):
        try:
            with open(filename, 'w') as f:
                f.write("# vtk DataFile Version 2.0\n")
                f.write("TetWild Mesh\n")
                f.write("ASCII\n")
                f.write("DATASET UNSTRUCTURED_GRID\n")
                f.write(f"POINTS {len(self.vertices)} double\n")
                for coord in self.vertices:
                    f.write(f"{coord[0]:.17g} {coord[1]:.17g} {coord[2]:.17g}\n")

                # 假设 faces 中只包含三角形面
                total_cells = len(self.faces)
                total_cell_vertices = len(self.faces) * 4
                f.write(f"CELLS {total_cells} {total_cell_vertices}\n")
                for tri in self.faces:
                    f.write("3 ")
                    for vertex in tri:
                        f.write(f"{vertex} ")
                    f.write("\n")

                f.write(f"CELL_TYPES {total_cells}\n")
                for _ in self.faces:
                    f.write("5\n")

                f.write(f"CELL_DATA {len(self.faces)}\n")
                f.write(f"SCALARS surface_id int 1\n")
                f.write(f"LOOKUP_TABLE default\n")
                for region in self.surface_id:
                    f.write(f"{region[0]}\n")
                
                if self.sizing_values is not None:
                    f.write(f"POINT_DATA {len(self.vertices)}\n")
                    f.write("SCALARS sizing_value double 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for size in self.sizing_values:
                        f.write(f"{size.item()}\n")

                    # f.write(f"CELL_DATA {len(self.area_num)}\n")
                    # f.write(f"SCALARS element int 1\n")
                    # f.write(f"LOOKUP_TABLE default\n")
                    # for element in self.area_num:
                    #     f.write(f"{element}\n")
                
            return 0
        except IOError:
            print(f"Write VTK file failed. - {filename}")
            return -1

    def save_to_vtk(self, file_path):
        # Convert vertices and faces to numpy arrays
        vertices = self.vertices.detach().numpy()
        faces = self.faces.detach().numpy()
        surface_id = self.surface_id
        
        # Prepare cells data for meshio (VTK format expects explicit cell types)
        cells = [("triangle", faces)]
        
        # Prepare point data
        point_data = {}
        if self.sizing_values is not None:
            sizing_value = self.sizing_values.detach().numpy().flatten()
            point_data["sizing_value"] = list(sizing_value)
        cell_data = {}
        if self.surface_id is not None:
            surface_id = self.surface_id.flatten()
            cell_data["surface_id"] = [surface_id]
        
        # Create meshio mesh object
        mesh = meshio.Mesh(
            points=vertices,
            cells=cells,
            point_data=point_data,
            cell_data= cell_data
        )
        
        # Write to file
        mesh.write(file_path)
        print(f"Successfully saved mesh to {file_path}")

    def get_gradinet(self):
        # sum_grad = torch.tensor(0.0, dtype=torch.float64, device=self.vertices.device)
        grads = torch.zeros(size=(len(self.faces), ))
        i = 0
        for tri in self.faces:
            v0, v1, v2 = tri
            point_value = torch.stack([self.sizing_values[v0], 
                                    self.sizing_values[v1],
                                    self.sizing_values[v2]])
            point3d_coord = torch.stack([self.vertices[v0], 
                                    self.vertices[v1],
                                    self.vertices[v2]])
            
            point2d_coord = map_triangle_to_2d(point3d_coord)
            
            grad = calculate_gradinet(point2d_coord, point_value)
            grads[i] = grad
            i = i + 1
        
        return grads
    
    def num_mesh(self):
        total_elements = 0.0
        elements = []
        for face in self.faces:
            coord_3d = torch.stack([self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]])
            coord_2d = map_triangle_to_2d(coord_3d)
            size = np.array([self.sizing_values[face[0]], self.sizing_values[face[1]], self.sizing_values[face[2]]])
            num = integrate_over_triangle(coord_2d.numpy(), size)
            elements.append(num)
            # total_elements += integrate_over_triangle(coord_2d.numpy(), size)
        # return total_elements
        return elements
          
    def rotate_edge(self, edge_id):
        """
        旋转指定的边，并返回一个新的CustomMesh实例。

        参数:
        edge_id (tuple): 要旋转的边，如 (20, 30)
        
        返回:
        CustomMesh: 包含旋转后边的新网格实例
        """
        # 创建原始数据的深拷贝
        new_vertices = self.vertices.clone()
        new_faces = self.faces.clone()
        new_edges = self.edges.clone()
        new_sizing_values = self.sizing_values.clone() if self.sizing_values is not None else None
        
        v1, v2 = new_edges[edge_id]  # 边的两个顶点

        # 找到包含该边的两个三角形
        tri1_idx, tri2_idx = None, None
        for i, face in enumerate(new_faces):
            if v1 in face and v2 in face:
                if tri1_idx is None:
                    tri1_idx = i
                else:
                    tri2_idx = i
                    break

        if tri1_idx is None or tri2_idx is None:
            raise ValueError(f"Edge {edge_id} is not shared by exactly two triangles.")

        # 提取两个三角形
        tri1 = new_faces[tri1_idx]
        tri2 = new_faces[tri2_idx]

        # 找到两个三角形中不包含 edge_id 的顶点（新边的顶点）
        def find_opposite_vertex(triangle, edge):
            return [v for v in triangle if v not in edge][0]

        v_new1 = find_opposite_vertex(tri1, (v1, v2))  # 10
        v_new2 = find_opposite_vertex(tri2, (v1, v2))  # 40

        # 更新两个三角形
        new_tri1 = torch.tensor([v_new1, v1, v_new2])  # [10, 20, 40]
        new_tri2 = torch.tensor([v_new1, v2, v_new2])  # [10, 30, 40]

        # 替换原三角形
        new_faces[tri1_idx] = new_tri1
        new_faces[tri2_idx] = new_tri2

        # 更新边拓扑
        new_edges[edge_id] = torch.tensor([v_new1, v_new2])

        # 返回新的CustomMesh实例
        return CustomMesh(
            vertices=new_vertices,
            faces=new_faces,
            edge_topology=new_edges,
            sizing_values=new_sizing_values
        )
    
    def split_edge(self, edge_id):
        """
        在指定边中点插入一个点，并连接相关点，返回新的CustomMesh实例
        
        参数:
        edge_id (int): 要分割的边的索引
        
        返回:
        CustomMesh: 包含分割后边的新网格实例
        """
        # 创建原始数据的深拷贝
        new_vertices = self.vertices.clone()
        new_faces = self.faces.clone()
        new_edges = self.edges
        # new_sizing_values = self.sizing_values.clone() if self.sizing_values is not None else None
        new_faces_id = self.surface_id
        
        # 获取边的两个顶点
        v1, v2 = new_edges[edge_id]
        
        # 计算中点坐标
        midpoint = (new_vertices[v1] + new_vertices[v2]) / 2
        new_vertex_idx = len(new_vertices)
        new_vertices = torch.cat([new_vertices, midpoint.unsqueeze(0)])
        
        # 找到包含该边的两个三角形
        adjacent_faces = []
        adjacent_faces_id = []
        for i, face in enumerate(new_faces):
            if v1 in face and v2 in face:
                adjacent_faces.append(i)
                adjacent_faces_id.append(self.surface_id[i])
        
        if len(adjacent_faces) != 2:
            # raise ValueError(f"Edge {edge_id} is not shared by exactly two triangles.")
            print(f"Edge {edge_id} is not shared by exactly two triangles.")
            return EmptyCustomMesh()
        
        # 更新面拓扑
        new_faces_list = []
        new_faces_id_list = []
        new_sizing = 0
        for face_idx in adjacent_faces:
            face = new_faces[face_idx]
            # 找到对面的顶点
            opposite_vertex = [v for v in face if v not in (v1, v2)][0]

            new_sizing += self.sizing_values[opposite_vertex]
            
            # 创建两个新三角形
            new_face1 = torch.tensor([v1, new_vertex_idx, opposite_vertex])
            new_face2 = torch.tensor([new_vertex_idx, v2, opposite_vertex])
            
            new_faces_list.extend([new_face1, new_face2])
            new_faces_id_list.extend([self.surface_id[face_idx], self.surface_id[face_idx]])

        other_faces = []
        other_faces_id = []
        # 保留不相关的面
        for i, face in enumerate(new_faces):
            if i not in adjacent_faces:
                other_faces.append(face)
                other_faces_id.append(self.surface_id[i])
        new_faces = torch.stack(other_faces + new_faces_list)
        new_faces_id = np.stack(other_faces_id + new_faces_id_list)

        # other_faces = [f for i, f in enumerate(new_faces) if i not in adjacent_faces]
        # new_faces = torch.stack(other_faces + new_faces_list)
        # other_faces_id = [f for i, f in enumerate(new_faces_id) if i not in adjacent_faces_id]
        # new_faces_id = np.stack(other_faces_id + new_faces_id_list)
        
        # # 更新尺寸值（如果有）
        # if new_sizing_values is not None:
        #     new_size = new_sizing / 2
        #     new_sizing_values = torch.cat([new_sizing_values, new_size.unsqueeze(0)])
        
        # 返回新的CustomMesh实例
        return CustomMesh(
            vertices=new_vertices,
            faces=new_faces,

            regions=new_faces_id
        )

    def remove_non_manifold_faces(self):
        """
        去除非流形面并返回新的CustomMesh实例
        返回:
            CustomMesh: 只包含流形面的新网格
        """
        import numpy as np
        from collections import defaultdict

        # 转换数据为numpy格式（假设输入是PyTorch张量）
        vertices = self.vertices
        faces = self.faces

        # 初始化数据结构
        edge_face_count = defaultdict(int)
        face_mask = np.ones(len(faces), dtype=bool)
        

        # 第一遍：统计每条边被多少面引用
        for face in faces:
            for i in range(3):
                edge = tuple(sorted((face[i], face[(i+1)%3])))
                edge_face_count[edge] += 1

        # 第二遍：标记非流形面
        for i, face in enumerate(faces):
            # 检查是否为退化面（面积为0）
            if self.area[i] < 1e-10:
                face_mask[i] = False
                continue
                
            # 检查是否有边被超过2个面共享
            non_manifold = False
            for j in range(3):
                edge = tuple(sorted((face[j], face[(j+1)%3])))
                if edge_face_count[edge] > 2:
                    non_manifold = True
                    break
                    
            if non_manifold:
                face_mask[i] = False

        # 应用过滤
        valid_faces = faces[face_mask]
        
        # 返回新的CustomMesh实例
        return CustomMesh(
            vertices=self.vertices.clone(),
            faces=torch.tensor(valid_faces, dtype=torch.long),
            edge_topology=None,  # 需要重新计算
            sizing_values=self.sizing_values[face_mask] if self.sizing_values is not None else None
        )
    
    def calculate_feature_points(self, angle_threshold=0.9):
        feature_edges = set()
        feature_point = torch.zeros(len(self.vertices), dtype=torch.bool)
        edge_table = self._edge_face_map
        for edge, face_indices in edge_table.items():
            v1, v2 = edge
            if len(face_indices) != 2:  # 边界边
                feature_point[v1] = True
                feature_point[v2] = True
                feature_edges.add((v1, v2))
            else:
                f1, f2 = face_indices
                
                # 使用预计算的法向量
                normal1 = self.face_normal[f1]
                normal2 = self.face_normal[f2]
                
                cos_theta = torch.dot(normal1, normal2).item()
                
                if cos_theta <= angle_threshold:  # 尖锐边
                    feature_point[v1] = True
                    feature_point[v2] = True
                    feature_edges.add((v1, v2))
        return feature_point, feature_edges
    
    def calculate_feature_points1(self, angle_threshold=0.9):
        feature_edges = set()
        feature_point = torch.zeros(len(self.vertices), dtype=torch.bool)
        edge_table = self._edge_face_map
        
        # 分离边界边和需要计算角度的边
        boundary_edges = []
        edges_to_check = []
        face_pairs_to_check = []
        
        for edge, face_indices in edge_table.items():
            if len(face_indices) != 2:  # 边界边
                boundary_edges.append(edge)
            else:  # 需要计算角度的边
                edges_to_check.append(edge)
                face_pairs_to_check.append(face_indices)
        
        # 处理边界边
        for edge in boundary_edges:
            v1, v2 = edge
            feature_point[v1] = True
            feature_point[v2] = True
            feature_edges.add((v1, v2))
        
        # 向量化处理需要计算角度的边
        if edges_to_check:
            # 转换为张量
            edges_tensor = torch.tensor(edges_to_check, dtype=torch.long)  # (n_edges, 2)
            face_pairs_tensor = torch.tensor(face_pairs_to_check, dtype=torch.long)  # (n_edges, 2)
            
            # 批量获取法向量
            normals1 = self.face_normal[face_pairs_tensor[:, 0]]  # (n_edges, 3)
            normals2 = self.face_normal[face_pairs_tensor[:, 1]]  # (n_edges, 3)
            
            # 向量化点积计算
            cos_theta = torch.sum(normals1 * normals2, dim=1)  # (n_edges,)
            
            # 找出尖锐边
            sharp_mask = cos_theta <= angle_threshold
            sharp_edges = edges_tensor[sharp_mask]
            
            # 处理尖锐边
            for edge in sharp_edges:
                v1, v2 = edge.tolist()
                feature_point[v1] = True
                feature_point[v2] = True
                feature_edges.add((v1, v2))
        
        return feature_point, feature_edges

    def create_edge2face(self):
        edge_table = defaultdict(list)

        # 构建边表（记录每条边所属的面）
        for i, face in enumerate(self.faces):
            for j in range(3):
                v1 = face[j].item()
                v2 = face[(j+1)%3].item()
                if v1 > v2:
                    v1, v2 = v2, v1  # 确保边以小顶点到大顶点存储
                edge_table[(v1, v2)].append(i)

        return edge_table
    
    def visual_feature(self):
        colors = []
        for is_feature in self.feature_point:
            if is_feature:
                colors.append([1, 0, 0])
            else:
                colors.append([0, 0, 1])

        meshio.write(
                    os.path.join("visual.vtk"),
                    meshio.Mesh(
                        points=self.vertices.detach().numpy(),
                        cells=[("line", self.edges.detach().numpy())],
                        point_data={"Color": colors}
                    )
                )
        
    def visual_edge(self, edge_id, path):
        # colors = []
        # for i, edge in enumerate(self.edges):
        #     if i in edge_id:
        #         colors.append([1, 0, 0])
        #     else:
        #         colors.append([0, 0, 1])
        edge_id_set = set(edge_id)  # 转换为集合用于快速查找
        colors = np.zeros((len(self.edges), 3), dtype=np.float32)

        # 创建布尔索引数组
        is_highlighted = np.array([i in edge_id_set for i in range(len(self.edges))])

        # 使用布尔索引一次性设置颜色
        colors[is_highlighted] = [1, 0, 0]    # 红色
        colors[~is_highlighted] = [0, 0, 1]   # 蓝色
        meshio.write(
                    os.path.join(path),
                    meshio.Mesh(
                        points=self.vertices.detach().numpy(),
                        cells=[("line", self.edges.detach().numpy())],
                        cell_data={"Color": [colors]}
                    )
                )
        
    
    def check_triangle_overlap(self, face_idx1: int, face_idx2: int) -> bool:
        """
        检测两个三角面片是否重叠
        
        参数:
            face_idx1 (int): 第一个面片的索引
            face_idx2 (int): 第二个面片的索引
            
        返回:
            bool: 如果面片重叠返回True，否则返回False
        """
        # 获取两个三角形的顶点
        tri1 = self.get_face_vertices(face_idx1)
        tri2 = self.get_face_vertices(face_idx2)

        # tri1 = self.faces[face_idx1]
        # tri2 = self.faces[face_idx2]
        
        # 转换为numpy数组便于计算
        tri1 = tri1.cpu().numpy() if torch.is_tensor(tri1) else np.array(tri1)
        tri2 = tri2.cpu().numpy() if torch.is_tensor(tri2) else np.array(tri2)
        
        return self._tri_tri_overlap(tri1, tri2)
    
    def get_face_vertices(self, face_idx: int) -> torch.Tensor:
        """
        获取指定面片的顶点坐标
        
        参数:
            face_idx (int): 面片索引
            
        返回:
            torch.Tensor: 3x3张量，包含三个顶点的坐标
        """
        face = self.faces[face_idx]
        return self.vertices[face]
    
    @staticmethod
    def _tri_tri_overlap(tri1: np.ndarray, tri2: np.ndarray) -> bool:
        """
        使用分离轴定理检测两个三角形是否相交
        
        参数:
            tri1 (np.ndarray): 第一个三角形的顶点 (3x3)
            tri2 (np.ndarray): 第二个三角形的顶点 (3x3)
            
        返回:
            bool: 如果三角形相交返回True，否则返回False
        """
        # 计算三角形法向量
        normal1 = np.cross(tri1[1] - tri1[0], tri1[2] - tri1[0])
        normal2 = np.cross(tri2[1] - tri2[0], tri2[2] - tri2[0])
        
        # 1. 检查三角形1的法向量是否为分离轴
        if not _overlap_on_axis(tri1, tri2, normal1):
            return False
        
        # 2. 检查三角形2的法向量是否为分离轴
        if not _overlap_on_axis(tri1, tri2, normal2):
            return False
        
        # 3. 检查两个三角形边的叉积是否为分离轴
        edges1 = [tri1[1] - tri1[0], tri1[2] - tri1[1], tri1[0] - tri1[2]]
        edges2 = [tri2[1] - tri2[0], tri2[2] - tri2[1], tri2[0] - tri2[2]]
        
        for e1 in edges1:
            for e2 in edges2:
                axis = np.cross(e1, e2)
                if np.linalg.norm(axis) < 1e-8:  # 平行边，跳过
                    continue
                if not _overlap_on_axis(tri1, tri2, axis):
                    return False
        
        # 4. 检查三角形是否共面且重叠
        if np.dot(normal1, normal2) > 0.999:  # 共面情况
            return _coplanar_triangles_overlap(tri1, tri2)
        
        return True
    
    def detect_thin_walls(self, thickness_threshold=0.01):
        """检测薄壁区域，返回面片对的列表 [(face1, face2)]"""
        face_centers = torch.mean(self.vertices[self.faces], dim=1)  # [F, 3]
        thin_pairs = []
        
        # 遍历所有面片对，检测距离小于阈值的
        for i in range(len(self.faces)):
            for j in range(i + 1, len(self.faces)):
                dist = torch.norm(face_centers[i] - face_centers[j], p=2)
                if dist < thickness_threshold:
                    thin_pairs.append((i, j))
        return thin_pairs
    
    def collapse_aviliable(self, path):
        flag = []
        for edge in self.edges:
            v1, v2 = edge
            if self.feature_point[v1] != self.feature_point[v1]:
                flag.append([1, 0, 0])
            elif (self.feature_point[v1] == self.feature_point[v1]) and (not self.feature_point[v1]):
                flag.append([1, 0, 0])
            elif (self.feature_point[v1] == self.feature_point[v1]) and (self.feature_point[v1]):
                angle = angle_between_normals(self.get_vertex_normal(v1.item()), self.get_vertex_normal(v2.item()), degrees=True)
                if angle < 2:
                    flag.append([1, 0, 0])
                else:
                    flag.append([0, 0, 1])
        meshio.write(
                    os.path.join(path),
                    meshio.Mesh(
                        points=self.vertices.detach().numpy(),
                        cells=[("line", self.edges.detach().numpy())],
                        cell_data={"Color": [flag]}
                    )
                )
        
    def visualize_convexity(self, save_path):
        """
        可视化网格顶点凹凸性（凸点绿色，凹点红色）和边特征
        
        参数:
            save_path: 保存文件路径（如 "output.vtk"）
        """
        # 1. 计算顶点凹凸性（假设已有classify_vertex_convexity方法）
        convexity = self.classify_vertex_convexity()  # 返回Tensor: 1=凸点, -1=凹点, 0=边界
        
        # 2. 顶点颜色（RGB格式）
        vertex_colors = np.zeros((len(self.vertices), 3))
        for v_idx in range(len(self.vertices)):
            if convexity[v_idx] == 1:    # 凸点→绿色
                vertex_colors[v_idx] = [0, 1, 0]
            elif convexity[v_idx] == -1: # 凹点→红色
                vertex_colors[v_idx] = [1, 0, 0]
            else:                        # 边界点→灰色
                vertex_colors[v_idx] = [0.5, 0.5, 0.5]
        
        # 3. 边颜色（保留原始逻辑）
        edge_colors = []
        for edge in self.edges:
            v1, v2 = edge
            if self.feature_point[v1] != self.feature_point[v2]:
                edge_colors.append([1, 0, 0])  # 特征边→红色
            elif not self.feature_point[v1]:    # 非特征边→蓝色
                edge_colors.append([0, 0, 1])
            else:
                angle = angle_between_normals(
                    self.vertex_normal[v1.item()],
                    self.vertex_normal[v2.item()],
                    degrees=True
                )
                edge_colors.append([1, 0, 0] if angle < 2 else [0, 0, 1])
        
        # 4. 保存为VTK文件（兼容顶点和边颜色）
        mesh = meshio.Mesh(
            points=self.vertices.detach().numpy(),
            cells=[
                ("vertex", np.arange(len(self.vertices)).reshape(-1, 1)),  # 顶点
                ("line", self.edges.detach().numpy())                     # 边
            ],
            point_data={"VertexColor": vertex_colors},
            cell_data={"EdgeColor": [
                np.zeros((len(self.vertices), 3)),  # 顶点颜色占位（实际不使用）
                np.array(edge_colors)              # 边颜色
            ]}
        )
        meshio.write(save_path, mesh)

    def compute_LBO(self):
        sizing_values = self.sizing_values
        vertices_np = self.vertices.cpu().numpy() if torch.is_tensor(self.vertices) else self.vertices
        if not hasattr(self, '_cotangent_weights') or self._cotangent_weights is None:
            self._precompute_cotangent_weights(vertices_np)
        if not hasattr(self, '_vertex_neighbors') or self._vertex_neighbors is None:
            self._precompute_vertex_neighbors()
        LBO_v = []
        LBO_e = []
        for i, v in enumerate(vertices_np):
            neighbors = self._vertex_neighbors[i]
            total_weight = 0.0
            A = 0
            neighbor_face = self._vertex_face_map[i]
            for face_id in neighbor_face:
                v0, v1, v2 = self.vertices[self.faces[face_id]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross_product = torch.linalg.cross(edge1, edge2)
                area = 0.5 * torch.linalg.vector_norm(cross_product)
                A += area.item()
            for j in neighbors:
                # 获取边的余切权重
                edge_key = tuple(sorted((i, j)))
                weight = self._cotangent_weights.get(edge_key, 0.0)
                total_weight += (weight * (sizing_values[j][0].item() - sizing_values[i][0].item()))
            total_weight /= (A + 1e-8)
            LBO_v.append(total_weight)
        
        vertex_lbo = np.abs(LBO_v)

        # vertex_lbo_array = np.array(vertex_lbo)

        # # 获取排序索引（从大到小）
        # sorted_indices = np.argsort(vertex_lbo_array)[::-1]

        # # 根据索引获取排序后的列表
        # sorted_vertex_lbo = vertex_lbo_array[sorted_indices]

        for i, edge in enumerate(self.edges):
            v1, v2 = np.array(edge)
            LBO_e.append((vertex_lbo[v1] + vertex_lbo[v2]) / 2)
        edge_lbo = np.array(LBO_e)
        return vertex_lbo, edge_lbo

    def compute_LBO1(self):
        sizing_values = self.sizing_values
        vertices_np = self.vertices.cpu().numpy() if torch.is_tensor(self.vertices) else self.vertices
        
        if not hasattr(self, '_cotangent_weights') or self._cotangent_weights is None:
            self._precompute_cotangent_weights(vertices_np)
        if not hasattr(self, '_vertex_neighbors') or self._vertex_neighbors is None:
            self._precompute_vertex_neighbors()
            
        LBO_v = []
        LBO_e = []
        
        for i, v in enumerate(vertices_np):
            neighbors = self._vertex_neighbors[i]
            total_weight = 0.0
            A = 0
            neighbor_face = self._vertex_face_map[i]
            for face_id in neighbor_face:
                v0, v1, v2 = self.vertices[self.faces[face_id]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross_product = torch.linalg.cross(edge1, edge2)
                area = 0.5 * torch.linalg.vector_norm(cross_product)
                A += area.item()
            
            for j in neighbors:
                # 获取边的余切权重
                edge_key = tuple(sorted((i, j)))
                weight = self._cotangent_weights.get(edge_key, 0.0)
                # 注意：这里计算的是拉普拉斯算子，通常应为 sum(w * (val_i - val_j)) 或反之，请确认方向是否符合您的定义
                total_weight += (weight * (sizing_values[j][0].item() - sizing_values[i][0].item()))
            
            total_weight /= (A + 1e-8)
            LBO_v.append(total_weight)
        
        # 将列表转换为 numpy 数组并取绝对值
        vertex_lbo = np.array(np.abs(LBO_v))

        # =================【修改开始：平滑/截断过大的值】=================
        # 方法：使用 95% 分位数进行截断 (Winsorization)
        # 解释：找到数值排在第 95% 位置的数，将所有比它大的数都设为这个数。
        # 这样可以消除由退化网格引起的数值爆炸（Infinity/Huge values）。
        
        percentile_threshold = np.percentile(vertex_lbo, 95) # 您可以调整这个比例，例如 90 或 99
        
        # 将大于阈值的值“削平”
        vertex_lbo = np.clip(vertex_lbo, a_min=None, a_max=percentile_threshold)
        # =================【修改结束】===================================

        for i, edge in enumerate(self.edges):
            v1, v2 = np.array(edge)
            LBO_e.append((vertex_lbo[v1] + vertex_lbo[v2]) / 2)
            
        edge_lbo = np.array(LBO_e)
        return vertex_lbo, edge_lbo


    def compute_discrete_laplace_beltrami_with_sizing(self, weighting_scheme='cotangent'):
        """
        基于网格点上的尺寸值计算 Discrete Laplace-Beltrami Operator（加速版本）。
        
        参数:
        weighting_scheme (str): 权重计算方案，可选 'cotangent' (余切权重) 或 'uniform' (均匀权重)
        
        返回:
        L (scipy.sparse.csr_matrix): 离散 Laplace-Beltrami 算子矩阵 (稀疏矩阵)
        """
        n_vertices = self.vertices.shape[0]
        
        # 如果没有提供尺寸值，使用默认值1.0
        if self.sizing_values is None:
            sizing_values = torch.ones(n_vertices, dtype=self.vertices.dtype, device=self.vertices.device)
        else:
            sizing_values = self.sizing_values
        
        # 转换为numpy数组以便处理
        vertices_np = self.vertices.cpu().numpy() if torch.is_tensor(self.vertices) else self.vertices
        faces_np = self.faces.cpu().numpy() if torch.is_tensor(self.faces) else self.faces
        sizing_np = list(sizing_values.cpu().numpy()) if torch.is_tensor(sizing_values) else sizing_values
        
        # 预计算顶点邻接关系
        if not hasattr(self, '_vertex_neighbors') or self._vertex_neighbors is None:
            self._precompute_vertex_neighbors()
        
        # 预计算边到面的映射
        if not hasattr(self, '_edge_face_map') or self._edge_face_map is None:
            self._build_edge_face_map()
        
        # 构建邻接矩阵和权重矩阵
        rows, cols, weights = [], [], []
        
        if weighting_scheme == 'cotangent':
            # 预计算余切权重矩阵
            if not hasattr(self, '_cotangent_weights') or self._cotangent_weights is None:
                self._precompute_cotangent_weights(vertices_np)
            
            # 使用预计算的余切权重
            for i in range(n_vertices):
                neighbors = self._vertex_neighbors[i]
                total_weight = 0.0
                
                for j in neighbors:
                    # 获取边的余切权重
                    edge_key = tuple(sorted((i, j)))
                    weight = self._cotangent_weights.get(edge_key, 0.0)
                    
                    # 根据尺寸值调整权重
                    avg_sizing = (sizing_np[i] + sizing_np[j]) / 2.0
                    adjusted_weight = weight / (avg_sizing * avg_sizing)
                    
                    total_weight += adjusted_weight
                    rows.append(i)
                    cols.append(j)
                    weights.append(-adjusted_weight)
                
                # 对角元素
                rows.append(i)
                cols.append(i)
                weights.append(total_weight)
                
        elif weighting_scheme == 'uniform':
            # 均匀权重方案（向量化实现）
            for i in range(n_vertices):
                neighbors = self._vertex_neighbors[i]
                degree = len(neighbors)
                
                if degree > 0:
                    uniform_weight = 1.0 / degree
                    
                    for j in neighbors:
                        # 根据尺寸值调整权重
                        avg_sizing = (sizing_np[i] + sizing_np[j]) / 2.0
                        adjusted_weight = uniform_weight / (avg_sizing * avg_sizing)
                        
                        rows.append(i)
                        cols.append(j)
                        weights.append(-adjusted_weight)
                    
                    # 对角元素
                    rows.append(i)
                    cols.append(i)
                    weights.append(1.0)
                else:
                    # 孤立顶点
                    rows.append(i)
                    cols.append(i)
                    weights.append(0.0)
        
        rows = np.array(rows)
        cols = np.array(cols)
        w = np.abs(np.array(weights))
        weights = w.flatten()
        # 创建稀疏矩阵
        L = coo_matrix((weights, (rows, cols)), shape=(n_vertices, n_vertices))
        return L.tocsr()
    
    def compute_discrete_laplace_beltrami_with_sizing_vertex(self, weighting_scheme='cotangent'):
        """
        基于网格点上的尺寸值计算 Discrete Laplace-Beltrami Operator。
        
        参数:
        weighting_scheme (str): 权重计算方案，可选 'cotangent' (余切权重) 或 'uniform' (均匀权重)
        
        返回:
        L (scipy.sparse.csr_matrix): 离散 Laplace-Beltrami 算子矩阵 (稀疏矩阵)
        """
        n_vertices = self.vertices.shape[0]
        
        # 如果没有提供尺寸值，使用默认值1.0
        if self.sizing_values is None:
            sizing_values = torch.ones(n_vertices, dtype=self.vertices.dtype, device=self.vertices.device)
        else:
            sizing_values = self.sizing_values
        
        # 转换为numpy数组以便处理
        if torch.is_tensor(self.vertices):
            vertices_np = self.vertices.cpu().numpy()
        else:
            vertices_np = self.vertices
            
        if torch.is_tensor(self.faces):
            faces_np = self.faces.cpu().numpy()
        else:
            faces_np = self.faces
            
        if torch.is_tensor(sizing_values):
            sizing_np = sizing_values.cpu().numpy()
        else:
            sizing_np = sizing_values
        
        # 构建邻接矩阵和权重矩阵
        rows, cols, weights = [], [], []
        
        for i in range(n_vertices):
            # 获取顶点i的邻居
            neighbors = self._get_vertex_neighbors(i)
            
            if weighting_scheme == 'cotangent':
                # 预计算余切权重矩阵
                if not hasattr(self, '_cotangent_weights') or self._cotangent_weights is None:
                    self._precompute_cotangent_weights(vertices_np)
                
                # 使用预计算的余切权重
                for i in range(n_vertices):
                    neighbors = self._vertex_neighbors[i]
                    total_weight = 0.0
                    
                    for j in neighbors:
                        # 获取边的余切权重
                        edge_key = tuple(sorted((i, j)))
                        weight = self._cotangent_weights.get(edge_key, 0.0)
                        
                        # 根据尺寸值调整权重
                        avg_sizing = (sizing_np[i] + sizing_np[j]) / 2.0
                        adjusted_weight = weight / (avg_sizing * avg_sizing)
                        
                        total_weight += adjusted_weight
                        rows.append(i)
                        cols.append(j)
                        weights.append(-adjusted_weight)
                
                # 对角元素
                rows.append(i)
                cols.append(i)
                weights.append(total_weight)
                
            
        
        # 创建稀疏矩阵
        L = coo_matrix((weights, (rows, cols)), shape=(n_vertices, n_vertices))
        L = L.tocsr()
        
        return L

    def _precompute_vertex_neighbors(self):
        """预计算所有顶点的邻居关系"""
        n_vertices = self.vertices.shape[0]
        self._vertex_neighbors = [[] for _ in range(n_vertices)]
        
        faces_np = self.faces.cpu().numpy() if torch.is_tensor(self.faces) else self.faces
        
        for face in faces_np:
            v0, v1, v2 = face
            self._vertex_neighbors[v0].extend([v1, v2])
            self._vertex_neighbors[v1].extend([v0, v2])
            self._vertex_neighbors[v2].extend([v0, v1])
        
        # 去重并排序
        for i in range(n_vertices):
            self._vertex_neighbors[i] = list(set(self._vertex_neighbors[i]))

    def _get_vertex_neighbors(self, vertex_idx):
        """获取顶点的所有邻居顶点索引"""
        neighbors = set()
        for face in self.faces:
            if vertex_idx in face:
                for v in face:
                    if v != vertex_idx:
                        neighbors.add(v)
        return list(neighbors)

    def _precompute_cotangent_weights(self, vertices):
        """预计算所有边的余切权重"""
        self._cotangent_weights = {}
        faces_np = self.faces.cpu().numpy() if torch.is_tensor(self.faces) else self.faces
        
        # 遍历所有面片计算余切权重
        for face in faces_np:
            v0, v1, v2 = face
            p0, p1, p2 = vertices[v0], vertices[v1], vertices[v2]
            
            # 计算三个角的余切值
            cot0 = self._compute_cotangent(p1, p0, p2)
            cot1 = self._compute_cotangent(p2, p1, p0)
            cot2 = self._compute_cotangent(p0, p2, p1)
            
            # 为每条边累加余切权重
            edges = [
                (min(v0, v1), max(v0, v1), cot2),
                (min(v1, v2), max(v1, v2), cot0),
                (min(v2, v0), max(v2, v0), cot1)
            ]
            
            for edge in edges:
                v_min, v_max, cot = edge
                edge_key = (v_min, v_max)
                self._cotangent_weights[edge_key] = self._cotangent_weights.get(edge_key, 0.0) + cot / 2.0

    def _compute_cotangent(self, a, b, c):
        """计算角b的余切值（向量a->b和c->b的夹角）"""
        vec1 = a - b
        vec2 = c - b
        
        # 计算余弦和正弦
        dot_product = np.dot(vec1, vec2)
        cross_norm = np.linalg.norm(np.cross(vec1, vec2))
        
        # 避免除零错误
        if cross_norm < 1e-12:
            return 0.0
        
        return dot_product / cross_norm

    def recalculate_size(self, target_mesh):
        points_arr = np.array(self.vertices)
        mesh_vertices = target_mesh.vertices.numpy()
        mesh_faces = target_mesh.faces.numpy()
        mesh_size_values = target_mesh.sizing_values
        
        
        # 查询每个点的最近顶点
        tree = cKDTree(target_mesh.vertices.numpy())
        _, point_indices = tree.query(points_arr)
        
        # # 预计算所有三角形的法向量
        # v0s = mesh_vertices[mesh_faces[:, 1]] - mesh_vertices[mesh_faces[:, 0]]
        # v1s = mesh_vertices[mesh_faces[:, 2]] - mesh_vertices[mesh_faces[:, 1]]
        # normals = np.cross(v0s, v1s)
        # normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        normals = target_mesh.face_normal
        
        vertex_to_faces = target_mesh._vertex_face_map

        # 为每个点找到包含它的三角形
        results_s = np.zeros(len(points_arr))
        results_d = np.zeros(len(points_arr))
        
        # 为所有点找到包含它们的三角形
        all_containing_tris = []
        max_tris = 0
        
        for point_idx in point_indices:
            # containing_tris = vertex_to_faces.get(point_idx, [])
            containing_tris = vertex_to_faces[point_idx]
            all_containing_tris.append(containing_tris)
            max_tris = max(max_tris, len(containing_tris))
        
        # 创建填充数组用于向量化计算
        tri_mask = np.zeros((len(points_arr), max_tris), dtype=bool)
        tri_indices = -np.ones((len(points_arr), max_tris), dtype=int)
        
        for i, tris in enumerate(all_containing_tris):
            if tris:
                tri_mask[i, :len(tris)] = True
                tri_indices[i, :len(tris)] = tris
        
        # 批量计算距离和重心坐标
        valid_points_mask = tri_mask.any(axis=1)
        
        if np.any(valid_points_mask):
            # 获取有效点的索引
            valid_indices = np.where(valid_points_mask)[0]
            
            # 为所有有效点批量计算
            for i in valid_indices:
                containing_tris = all_containing_tris[i]
                if not containing_tris:
                    continue
                    
                point = points_arr[i]
                tri_verts = mesh_vertices[mesh_faces[containing_tris]]
                
                # 计算距离
                ds = np.einsum('ij,ij->i', normals[containing_tris], point - tri_verts[:, 0])
                abs_ds = np.abs(ds)
                
                # 批量计算重心坐标
                UVW = uvw_vectorized(point, target_mesh, mesh_faces[containing_tris])
                valid_mask = np.all((UVW >= -1e-8) & (UVW <= 1+1e-8), axis=1)
                
                if np.any(valid_mask):
                    valid_ds = np.where(valid_mask, abs_ds, np.inf)
                    min_idx = np.argmin(valid_ds)
                    min_tri = containing_tris[min_idx]
                    
                    projection_point = torch.from_numpy(point) - ds[min_idx] * normals[min_tri]
                    results_s[i] = barycentric_interpolation(projection_point, 
                                                            mesh_faces[min_tri], 
                                                            target_mesh)
                    results_d[i] = abs_ds[min_idx]
                else:
                    results_s[i] = mesh_size_values[point_indices[i]]
                    results_d[i] = np.linalg.norm(point - mesh_vertices[point_indices[i]])
        
        # 处理没有包含三角形的点
        no_tris_mask = ~valid_points_mask
        if np.any(no_tris_mask):
            no_tris_indices = np.where(no_tris_mask)[0]
            for i in no_tris_indices:
                point = points_arr[i]
                point_idx = point_indices[i]
                results_s[i] = mesh_size_values[point_idx]
                results_d[i] = np.linalg.norm(point - mesh_vertices[point_idx])
        
        self.sizing_values = torch.tensor(results_s).unsqueeze(1)

    def recalculate_size1(self, target_mesh):
        # 初始化时间统计字典
        time_stats = {
            'total': 0,
            'data_preparation': 0,
            'kdtree_build': 0,
            'kdtree_query': 0,
            'normal_precomputation': 0,
            'main_loop': 0,
            'containing_tris_find': 0,
            'distance_calculation': 0,
            'barycentric_uvw': 0,
            'valid_mask_check': 0,
            'min_distance_selection': 0,
            'projection_interpolation': 0,
            'fallback_case': 0
        }
        
        total_start = time.time()
        
        # 1. 数据准备
        t0 = time.time()
        points_arr = np.array(self.vertices)
        mesh_vertices = target_mesh.vertices.numpy()
        mesh_faces = target_mesh.faces.numpy()
        mesh_size_values = target_mesh.size_value.numpy() if hasattr(target_mesh, 'size_value') else np.zeros(len(mesh_vertices))
        time_stats['data_preparation'] = time.time() - t0
        
        # 2. 构建KDTree
        t0 = time.time()
        tree = cKDTree(target_mesh.vertices.numpy())
        time_stats['kdtree_build'] = time.time() - t0
        
        # 3. KDTree查询
        t0 = time.time()
        _, point_indices = tree.query(points_arr)
        time_stats['kdtree_query'] = time.time() - t0
        
        # 4. 预计算法向量
        t0 = time.time()
        v0s = mesh_vertices[mesh_faces[:, 1]] - mesh_vertices[mesh_faces[:, 0]]
        v1s = mesh_vertices[mesh_faces[:, 2]] - mesh_vertices[mesh_faces[:, 1]]
        normals = np.cross(v0s, v1s)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        time_stats['normal_precomputation'] = time.time() - t0
        
        # 初始化结果数组
        results_s = np.zeros(len(points_arr))
        results_d = np.zeros(len(points_arr))
        
        # 5. 主循环
        main_loop_start = time.time()
        for i, (point, point_idx) in enumerate(zip(points_arr, point_indices)):
            # 5.1 找到包含该顶点的所有三角形
            t0 = time.time()
            # containing_tris1 = np.where(np.any(mesh_faces == point_idx, axis=1))[0]
            containing_tris = target_mesh._vertex_face_map[point_idx]
            time_stats['containing_tris_find'] += time.time() - t0
            
            if len(containing_tris) == 0:
                # 回退情况
                t0 = time.time()
                results_s[i] = mesh_size_values[point_idx]
                results_d[i] = np.linalg.norm(point - mesh_vertices[point_idx])
                time_stats['fallback_case'] += time.time() - t0
                continue
                
            # 5.2 计算点到三角形平面的距离
            t0 = time.time()
            tri_verts = mesh_vertices[mesh_faces[containing_tris]]
            ds = np.einsum('ij,ij->i', normals[containing_tris], point - tri_verts[:, 0])
            abs_ds = np.abs(ds)
            time_stats['distance_calculation'] += time.time() - t0
            
            # 5.3 计算重心坐标
            t0 = time.time()
            # Uvw = np.array([uvw(point, target_mesh, mesh_faces[tri_idx]) 
            #             for tri_idx in containing_tris])
            Uvw = uvw_vectorized(point, target_mesh, mesh_faces[containing_tris])
            time_stats['barycentric_uvw'] += time.time() - t0
            
            # 5.4 检查有效掩码
            t0 = time.time()
            valid_mask = np.all((Uvw >= -1e-8) & (Uvw <= 1+1e-8), axis=1)
            time_stats['valid_mask_check'] += time.time() - t0
            
            if np.any(valid_mask):
                # 5.5 选择最小距离
                t0 = time.time()
                valid_ds = np.where(valid_mask, abs_ds, np.inf)
                min_idx = np.argmin(valid_ds)
                min_tri = containing_tris[min_idx]
                time_stats['min_distance_selection'] += time.time() - t0
                
                # 5.6 投影和插值
                t0 = time.time()
                projection_point = point - ds[min_idx] * normals[min_tri]
                results_s[i] = barycentric_interpolation(projection_point, 
                                                        mesh_faces[min_tri], 
                                                        target_mesh)
                results_d[i] = abs_ds[min_idx]
                time_stats['projection_interpolation'] += time.time() - t0
            else:
                # 回退情况
                t0 = time.time()
                results_s[i] = mesh_size_values[point_idx]
                results_d[i] = np.linalg.norm(point - mesh_vertices[point_idx])
                time_stats['fallback_case'] += time.time() - t0
        
        time_stats['main_loop'] = time.time() - main_loop_start
        
        # 最终处理
        t0 = time.time()
        self.sizing_values = torch.tensor(results_s).unsqueeze(1)
        time_stats['tensor_conversion'] = time.time() - t0
        
        # 总时间
        time_stats['total'] = time.time() - total_start
        print(" ")

    # def recalculate_size2(self, target_mesh):
    #     points_arr = np.array(self.vertices)
    #     mesh_vertices = target_mesh.vertices.numpy()
    #     mesh_faces = target_mesh.faces.numpy()
    #     mesh_size_values = target_mesh.sizing_values
        
        
    #     # 查询每个点的最近顶点
    #     tree = cKDTree(target_mesh.vertices.numpy())
    #     _, point_indices = tree.query(points_arr)
        
    #     normals = target_mesh.face_normal
    #     vertex_to_faces = target_mesh._vertex_face_map

    #     # 为每个点找到包含它的三角形
    #     results_s = np.zeros(len(points_arr))
    #     results_d = np.zeros(len(points_arr))
        
    #     # 为所有点找到包含它们的三角形
    #     all_containing_tris = []
    #     max_tris = 0

    #     for points in 

    def recalculate_size2(self, mesh):
        # 将 pyvista 网格转换为 trimesh 网格
        trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        t1 = time.time()
        index_point, distance, index_face = trimesh_mesh.nearest.on_surface(self.vertices)
        print(f"Cost {time.time() - t1}s")
        face_normal = mesh.face_normal
        # 计算点到三角面片所在平面的距离
        d = torch.sum(face_normal[index_face] * (self.vertices - index_point), dim=1)

        # 计算投影点
        projections = self.vertices - d.unsqueeze(1) * face_normal[index_face]

        self.sizing_values = barycentric_interpolation_batch(projections, mesh.faces[index_face], mesh)

    def recalculate_size_one(self, mesh, v_index):
        # 将 pyvista 网格转换为 trimesh 网格
        trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        index_point, distance, index_face = trimesh_mesh.nearest.on_surface(self.vertices[v_index].unsqueeze(0))
        face_normal = mesh.face_normal
        # 计算点到三角面片所在平面的距离
        d = torch.sum(face_normal[index_face] * (self.vertices[v_index] - index_point), dim=1)

        # 计算投影点
        projections = self.vertices[v_index] - d.unsqueeze(1) * face_normal[index_face]

        self.sizing_values[v_index] = barycentric_interpolation_batch(projections, mesh.faces[index_face], mesh)
    
    def recalculate_size_one1(self, target_mesh, v_index):
        points_arr = np.array(self.vertices[v_index].unsqueeze(0))
        mesh_vertices = target_mesh.vertices.numpy()
        mesh_faces = target_mesh.faces.numpy()
        mesh_size_values = target_mesh.sizing_values
        
        
        # 查询每个点的最近顶点
        tree = cKDTree(target_mesh.vertices.numpy())
        _, point_indices = tree.query(points_arr)
        
        # # 预计算所有三角形的法向量
        # v0s = mesh_vertices[mesh_faces[:, 1]] - mesh_vertices[mesh_faces[:, 0]]
        # v1s = mesh_vertices[mesh_faces[:, 2]] - mesh_vertices[mesh_faces[:, 1]]
        # normals = np.cross(v0s, v1s)
        # normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        normals = target_mesh.face_normal
        
        vertex_to_faces = target_mesh._vertex_face_map

        # 为每个点找到包含它的三角形
        results_s = np.zeros(len(points_arr))
        results_d = np.zeros(len(points_arr))
        
        # 为所有点找到包含它们的三角形
        all_containing_tris = []
        max_tris = 0
        
        for point_idx in point_indices:
            # containing_tris = vertex_to_faces.get(point_idx, [])
            containing_tris = vertex_to_faces[point_idx]
            all_containing_tris.append(containing_tris)
            max_tris = max(max_tris, len(containing_tris))
        
        # 创建填充数组用于向量化计算
        tri_mask = np.zeros((len(points_arr), max_tris), dtype=bool)
        tri_indices = -np.ones((len(points_arr), max_tris), dtype=int)
        
        for i, tris in enumerate(all_containing_tris):
            if tris:
                tri_mask[i, :len(tris)] = True
                tri_indices[i, :len(tris)] = tris
        
        # 批量计算距离和重心坐标
        valid_points_mask = tri_mask.any(axis=1)
        
        if np.any(valid_points_mask):
            # 获取有效点的索引
            valid_indices = np.where(valid_points_mask)[0]
            
            # 为所有有效点批量计算
            for i in valid_indices:
                containing_tris = all_containing_tris[i]
                if not containing_tris:
                    continue
                    
                point = points_arr[i]
                tri_verts = mesh_vertices[mesh_faces[containing_tris]]
                
                # 计算距离
                ds = np.einsum('ij,ij->i', normals[containing_tris], point - tri_verts[:, 0])
                abs_ds = np.abs(ds)
                
                # 批量计算重心坐标
                UVW = uvw_vectorized(point, target_mesh, mesh_faces[containing_tris])
                valid_mask = np.all((UVW >= -1e-8) & (UVW <= 1+1e-8), axis=1)
                
                if np.any(valid_mask):
                    valid_ds = np.where(valid_mask, abs_ds, np.inf)
                    min_idx = np.argmin(valid_ds)
                    min_tri = containing_tris[min_idx]
                    
                    projection_point = torch.from_numpy(point) - ds[min_idx] * normals[min_tri]
                    results_s[i] = barycentric_interpolation(projection_point, 
                                                            mesh_faces[min_tri], 
                                                            target_mesh)
                    results_d[i] = abs_ds[min_idx]
                else:
                    results_s[i] = mesh_size_values[point_indices[i]]
                    results_d[i] = np.linalg.norm(point - mesh_vertices[point_indices[i]])
        
        # 处理没有包含三角形的点
        no_tris_mask = ~valid_points_mask
        if np.any(no_tris_mask):
            no_tris_indices = np.where(no_tris_mask)[0]
            for i in no_tris_indices:
                point = points_arr[i]
                point_idx = point_indices[i]
                results_s[i] = mesh_size_values[point_idx]
                results_d[i] = np.linalg.norm(point - mesh_vertices[point_idx])
        
        self.sizing_values[v_index] = torch.tensor(results_s[0])




    def recalculate_size_gpu(self, target_mesh, device='cuda'):
        """
        GPU加速版本的重计算尺寸函数
        """
        # 将数据移到GPU
        points_arr = self.vertices.numpy() if torch.is_tensor(self.vertices) else np.array(self.vertices)
        mesh_vertices = target_mesh.vertices.numpy()
        mesh_faces = target_mesh.faces.numpy()
        mesh_size_values = target_mesh.sizing_values
        
        # 使用KDTree查询最近顶点
        tree = cKDTree(mesh_vertices)
        _, point_indices = tree.query(points_arr)
        
        # 预计算所有三角形的法向量
        v0 = mesh_vertices[mesh_faces[:, 0]]
        v1 = mesh_vertices[mesh_faces[:, 1]]
        v2 = mesh_vertices[mesh_faces[:, 2]]
        
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        normals = np.cross(v0v1, v0v2)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
        vertex_to_faces = target_mesh._vertex_face_map
        
        # 初始化结果数组
        results_s = np.zeros(len(points_arr))
        results_d = np.zeros(len(points_arr))
        
        # 为所有点找到包含它们的三角形
        all_containing_tris = [vertex_to_faces[idx] for idx in point_indices]
        
        # 批量处理有三角形的点
        batch_size = 500  # 减小批次大小以避免内存问题
        
        for batch_start in range(0, len(points_arr), batch_size):
            batch_end = min(batch_start + batch_size, len(points_arr))
            batch_points = points_arr[batch_start:batch_end]
            batch_point_indices = point_indices[batch_start:batch_end]
            batch_containing_tris = all_containing_tris[batch_start:batch_end]
            
            # 处理这个批次中的每个点
            for i_in_batch in range(len(batch_points)):
                global_idx = batch_start + i_in_batch
                point = batch_points[i_in_batch]
                nearest_vertex_idx = batch_point_indices[i_in_batch]
                containing_tris = batch_containing_tris[i_in_batch]
                
                if not containing_tris:
                    # 没有包含三角形，使用最近顶点
                    results_s[global_idx] = mesh_size_values[nearest_vertex_idx]
                    results_d[global_idx] = np.linalg.norm(point - mesh_vertices[nearest_vertex_idx])
                    continue
                
                # 为这个点找到最佳三角形
                best_tri_idx = None
                best_distance = float('inf')
                best_uvw = None
                
                for tri_idx in containing_tris:
                    # 获取三角形顶点
                    face = mesh_faces[tri_idx]
                    tri_verts = mesh_vertices[face]
                    
                    # 计算距离
                    normal = normals[tri_idx]
                    d = np.dot(normal, point - tri_verts[0])
                    abs_d = abs(d)
                    
                    # 计算重心坐标
                    uvw = calculate_uvw(point, tri_verts)
                    
                    # 检查点是否在三角形内（考虑数值误差）
                    if np.all(uvw >= -1e-8) and np.all(uvw <= 1.0 + 1e-8) and abs(uvw.sum() - 1.0) < 1e-8:
                        if abs_d < best_distance:
                            best_distance = abs_d
                            best_tri_idx = tri_idx
                            best_uvw = uvw
                
                if best_tri_idx is not None:
                    # 点在三角形内，进行插值
                    face = mesh_faces[best_tri_idx]
                    results_s[global_idx] = (best_uvw[0] * mesh_size_values[face[0]] +
                                            best_uvw[1] * mesh_size_values[face[1]] +
                                            best_uvw[2] * mesh_size_values[face[2]])
                    results_d[global_idx] = best_distance
                else:
                    # 点不在任何三角形内，使用最近顶点
                    results_s[global_idx] = mesh_size_values[nearest_vertex_idx]
                    results_d[global_idx] = np.linalg.norm(point - mesh_vertices[nearest_vertex_idx])
        
        self.sizing_values = torch.tensor(results_s, device=device).unsqueeze(1)

    def L1_size(self, target_mesh):
        points_arr = np.array(self.vertices)
        mesh_vertices = target_mesh.vertices.numpy()
        mesh_faces = target_mesh.faces.numpy()
        mesh_size_values = target_mesh.sizing_values
        
        
        # 查询每个点的最近顶点
        tree = cKDTree(target_mesh.vertices.numpy())
        _, point_indices = tree.query(points_arr)
        
        # 预计算所有三角形的法向量
        normals = np.array(target_mesh.face_normal)
        
        # 为每个点找到包含它的三角形
        results_s = np.zeros(len(points_arr))
        results_d = np.zeros(len(points_arr))
        
        for i, (point, point_idx) in enumerate(zip(points_arr, point_indices)):
            # 找到包含该顶点的所有三角形
            # containing_tris = np.where(np.any(mesh_faces == point_idx, axis=1))[0]
            containing_tris = target_mesh._vertex_face_map[point_idx]

            
            if len(containing_tris) == 0:
                results_s[i] = mesh_size_values[point_idx].item()
                results_d[i] = np.linalg.norm(point - mesh_vertices[point_idx])
                continue
                
            # 计算点到这些三角形平面的距离
            tri_verts = mesh_vertices[mesh_faces[containing_tris]]
            ds = np.einsum('ij,ij->i', normals[containing_tris], point - tri_verts[:, 0])
            abs_ds = np.abs(ds)
            
            # 计算重心坐标
            # Uvw = np.array([uvw(point, target_mesh, mesh_faces[tri_idx]) 
            #                for tri_idx in containing_tris])
            Uvw = uvw_vectorized(point, target_mesh, mesh_faces[containing_tris])
            valid_mask = np.all((Uvw >= -1e-8) & (Uvw <= 1+1e-8), axis=1)
            
            if np.any(valid_mask):
                # 选择有效投影中距离最小的
                valid_ds = np.where(valid_mask, abs_ds, np.inf)
                min_idx = np.argmin(valid_ds)
                min_tri = containing_tris[min_idx]
                
                # 计算投影点
                projection_point = point - ds[min_idx] * normals[min_tri]
                
                # 插值
                results_s[i] = barycentric_interpolation(projection_point, 
                                                             mesh_faces[min_tri], 
                                                             target_mesh)
                results_d[i] = abs_ds[min_idx]
            else:
                results_s[i] = mesh_size_values[point_idx].item()
                results_d[i] = np.linalg.norm(point - mesh_vertices[point_idx])
                
        return torch.tensor(results_s).unsqueeze(1), torch.tensor(results_d)

    def query(self, points):
        tree = cKDTree(self.vertices)
        t1 = time.time()
        for point in points:
            _, index = tree.query(point)
        print(f"cost {time.time() - t1}s")
    

    

    
    



if __name__ == '__main__':
    # mesh = CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/datasets/test_case/213_0.vtk")
    # new_mesh = simplify_mesh_with_kdtree_projection(mesh, int(len(mesh.faces) * 0.12158))
    # new_mesh.recalculate_size(mesh)
    # new_mesh = smooth_mesh_sizing(new_mesh, len(new_mesh.vertices), 1.2, torch.min(new_mesh.sizing_values))
    # new_mesh.writeVTK("simplified.vtk")

    mesh = CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/datasets/training/target/238_target.vtk")

    print(mesh.get_max_size(1))

    # mesh = CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/datasets/Bkgm0630/data/212_0.vtk")
    # mesh1 = mesh.collapsing_edge_id1
    # f_v, f_e = mesh.calculate_feature_points1()
    # visual_e = []
    # for e in f_e:
    #     for i, edge in enumerate(mesh.edges):
    #         if e[0] == edge[0] and e[1] == edge[1]:
    #             visual_e.append(i)
    # # visual_e = torch.tensor(visual_e)
    # mesh.visual_edge(visual_e, "visual_feature_edge.vtk")
    # mesh = CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/datasets/test_case/221_stl_bkgm.vtk")
    # mesh1 = CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/datasets/test_case/268_stl_bkgm.vtk")
    # mesh.sizing_values = torch.tensor(smooth_sizing_function(mesh, beta=1.2, tol = 1e-3)).unsqueeze(1)
    # start_time = time.time()
    # try:
    #     mesh.sizing_values = torch.tensor(smooth_sizing_function(mesh, beta=1.5, tol = 1e-1)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time}s")
    # except:
    #     print("failed")
    #     print(f"Cost {time.time() - start_time}s")

    # start_time1 = time.time()
    # try:
    #     mesh1.sizing_values = torch.tensor(smooth_sizing_function(mesh1, beta=1.5, tol = 1e-2)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time1}s")
    # except:
    #     print("failed")
    #     print(f"Cost {time.time() - start_time1}s")

    # mesh1 = CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/experiment/compare/110_component/110_simply_trad.vtk")   #268_simply_trad  256_simply_trad  238_simply_trad1
    # mesh2 = CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/experiment/compare/213_aircraft/213_simply_trad1.vtk")   #110_simply_trad  213_simply_trad1  221_simply_trad1.
    # mesh3 = CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/experiment/compare/221_missile/221_simply_trad1.vtk")
    # start_time1 = time.time()
    # try:
    #     mesh1.sizing_values = torch.tensor(smooth_sizing_function(mesh1)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time1}s")
    # except:
    #     print("mesh1 failed")
    #     print(f"Cost {time.time() - start_time1}s")
    # start_time2 = time.time()
    # try:
    #     mesh1.sizing_values = torch.tensor(smooth_sizing_function(mesh2)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time2}s")
    # except:
    #     print("mesh2 failed")
    #     print(f"Cost {time.time() - start_time2}s")
    # start_time3 = time.time()
    # try:
    #     mesh1.sizing_values = torch.tensor(smooth_sizing_function(mesh3)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time3}s")
    # except:
    #     print("mesh3 failed")
    #     print(f"Cost {time.time() - start_time3}s")

    # mesh1 = CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/experiment/compare/110_component/110_simply3.vtk")    #268_simply4  256_simply2 238_simply2
    # mesh2 = CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/experiment/compare/213_aircraft/213_simply.vtk")      #110_simply3  213_simply  221_simply4
    # mesh3 = CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/experiment/compare/221_missile/221_simply4.vtk")
    # start_time1 = time.time()
    # try:
    #     mesh1.sizing_values = torch.tensor(smooth_sizing_function(mesh1)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time1}s")
    # except:
    #     print("mesh1 failed")
    #     print(f"Cost {time.time() - start_time1}s")
    # start_time2 = time.time()
    # try:
    #     mesh1.sizing_values = torch.tensor(smooth_sizing_function(mesh2)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time2}s")
    # except:
    #     print("mesh2 failed")
    #     print(f"Cost {time.time() - start_time2}s")
    # start_time3 = time.time()
    # try:
    #     mesh1.sizing_values = torch.tensor(smooth_sizing_function(mesh3)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time3}s")
    # except:
    #     print("mesh3 failed")
    #     print(f"Cost {time.time() - start_time3}s")


    # mesh1 = CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/datasets/test_case/110_0.vtk")
    # mesh2 = CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/datasets/test_case/212_0.vtk")
    # feature = mesh2.compute_edge_features()
    # mesh3 = CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/datasets/test_case/221_0.vtk")
    # mesh4 = CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/datasets/test_case/268_1.vtk")
    # mesh5 = CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/datasets/test_case/256_0.vtk")
    # start_time1 = time.time()
    # try:
    #     mesh1.sizing_values = torch.tensor(smooth_sizing_function(mesh1)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time1}s")
    # except:
    #     print("mesh1 failed")
    #     print(f"Cost {time.time() - start_time1}s")
    # start_time2 = time.time()
    # try:
    #     mesh1.sizing_values = torch.tensor(smooth_sizing_function(mesh2)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time2}s")
    # except:
    #     print("mesh2 failed")
    #     print(f"Cost {time.time() - start_time2}s")
    # start_time3 = time.time()
    # try:
    #     mesh1.sizing_values = torch.tensor(smooth_sizing_function(mesh3)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time3}s")
    # except:
    #     print("mesh3 failed")
    #     print(f"Cost {time.time() - start_time3}s")
    # start_time4 = time.time()
    # try:
    #     mesh1.sizing_values = torch.tensor(smooth_sizing_function(mesh4)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time4}s")
    # except:
    #     print("mesh3 failed")
    #     print(f"Cost {time.time() - start_time4}s")
    # start_time5 = time.time()
    # try:
    #     mesh1.sizing_values = torch.tensor(smooth_sizing_function(mesh5)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time5}s")
    # except:
    #     print("mesh3 failed")
    #     print(f"Cost {time.time() - start_time5}s")

    mesh1 = CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/238_1.vtk")
    mesh2 = CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/experiment/compare/238_circuit_3/238_simply1.vtk")
    # mesh3 = CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/experiment/compare/238_circuit_3/238_stl_bkgm.vtk")
    # mesh = CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/110_0.vtk")
    # t1 = time.time()
    # mesh.recalculate_size1(mesh1)
    # print(f"Cost {time.time() - t1}s")
    # t2 = time.time()
    # mesh.recalculate_size1(mesh2)
    # print(f"Cost {time.time() - t2}s")
    # t3 = time.time()
    # mesh.recalculate_size1(mesh3)
    # print(f"Cost {time.time() - t3}s")
    # t4 = time.time()
    # mesh.recalculate_size_one1(mesh1, 5)
    # print(f"Cost {time.time() - t4}s")
    # mesh.writeVTK("result.vtk")



    # LBO_v, LBO_e = mesh2.compute_LBO()
    # print(" ")

    # lpls = []
    # color = []

    # # for i, point in enumerate(mesh.vertices):
    # #     value = abs(suanzi[i, i])
    # #     lpls.append(value)
    # # lpls_array = np.array(lpls)

    # Mesh1 = meshio.Mesh(
    #         points=mesh2.vertices.detach().numpy(),
    #         cells=[
    #             ("triangle", mesh2.faces.detach().numpy())                    # 边
    #         ],
    #         point_data={
    #             "Laplace-Beltrami-Operater":LBO_v
    #         }
    #     )
    # meshio.write("visual_lpls_vertex0828.vtk", Mesh1)

    # for i, edge in enumerate(mesh2.edges):
    #     value = LBO_e[i]
    #     lpls.append(value)
    #     if value <= 0.005:
    #         color.append([1, 0, 0])
    #     else:
    #         color.append([0, 0, 1])
    # color_array = np.array(color)
    # lpls_array = np.array(lpls)
    # Mesh2 = meshio.Mesh(
    #         points=mesh2.vertices.detach().numpy(),
    #         cells=[
    #             ("line", mesh2.edges.detach().numpy())                     # 边
    #         ],
    #         cell_data={
    #             # 每个cell data项应该只有一个数据块，与cell blocks数量一致
    #             "color": [color_array],  # 边颜色数据
    #             "Laplace-Beltrami-Operater": [lpls_array]  # 拉普拉斯值数据
    #         }
    #     )
    # meshio.write("visual_lpls.vtk", Mesh2)

    # points = generate_unique_surface_points()
    # print(f"采样得到的唯一点数: {len(points)}")
    # tree = cKDTree(mesh1.vertices)
    # t1 = time.time()
    # for point in points:
    #     _, index = tree.query(point)
    # print(f"cost {time.time() - t1}s")

    



