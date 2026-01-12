from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.TopoDS import topods, TopoDS_Face
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRep import BRep_Tool
import meshio
import numpy as np
import os, sys
from collections import defaultdict
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
from OCC.Core.Extrema import Extrema_ExtPS

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from models.layers.CustomMesh import CustomMesh

def print_face_ids(shape):
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    while exp.More():
        face = topods.Face(exp.Current())
        # 这里简单地将面的序号作为面 ID
        print(f"面 ID: {face_id}, 面对象: {face}")
        face_id += 1
        exp.Next()


def read_step_file(file_path):
    """
    读取 STEP 文件并返回形状对象
    :param file_path: STEP 文件路径
    :return: 形状对象
    """
    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path)
    if status == 1:
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape
    else:
        print("无法读取 STEP 文件")
        return None


def project_point_on_face(point, face):
    """将点投影到几何曲面上，并返回投影点及其 UV 参数"""
    # 1. 将拓扑面转换为几何曲面（关键！）
    topo_face = topods.Face(face)  # 拓扑面转换
    surf = BRep_Tool.Surface(topo_face)  # 提取几何曲面（Handle<Geom_Surface>）
    
    # 2. 创建投影器（传入待投影点和几何曲面）
    gp_point = gp_Pnt(point[0], point[1], point[2])  # numpy 点转 gp_Pnt
    proj = GeomAPI_ProjectPointOnSurf(gp_point, surf)  # 核心投影操作
    
    # 3. 获取投影结果（点和 UV 参数）
    if proj.NbPoints() > 0:
        projected_point = proj.NearestPoint()
        # 获取投影点的 UV 参数（曲面的参数坐标）
        u, v = proj.Parameters(1)  # 投影点的 UV 参数（索引从 1 开始）
        return projected_point, (u, v)
    # 3. 标准投影失败，使用Extrema计算最近点
    # print(f"警告：精确投影失败，使用最近点替代 (原始点: {point})")
    
    # 3.1 创建曲面适配器
    surf_adaptor = GeomAdaptor_Surface(surf)
    
    # 3.2 设置极值计算器
    extrema = Extrema_ExtPS(gp_point, surf_adaptor, 1e-20, 1e-20)
    # if not extrema.IsDone() or extrema.NbExt() == 0:
    #     print(f"错误：无法计算最近点 (原始点: {point})")
    #     return None, None
    
    # 3.3 获取最近点
    min_dist = float('inf')
    best_point = None
    best_uv = (0.0, 0.0)
    
    for i in range(1, extrema.NbExt()+1):
        dist = extrema.SquareDistance(i)
        if dist < min_dist:
            min_dist = dist
            best_point = extrema.Point(i).Value()
            best_uv = extrema.Point(i).Parameter()
    
    return best_point, best_uv

    # projected_point = proj.NearestPoint()
    # u, v = proj.Parameters(1)  # 投影点的 UV 参数（索引从 1 开始）
    # return projected_point, (u, v)



def calculate_curvature(surf_handle, point, uv_params):
    """
    计算点在面上的曲率
    :param point: 点
    :param face: 面
    :return: 主曲率和高斯曲率
    """
    u, v = uv_params
    # 构造曲率属性对象（参数：几何曲面句柄、导数阶数、公差）
    props = GeomLProp_SLProps(surf_handle, 1, 1e-10)  # 1 阶导数，公差 1e-6
    props.SetParameters(u, v)  # 设置曲面的 UV 参数（关键！）
    

    k1 = props.MinCurvature()
    k2 = props.MaxCurvature()
    kg = props.GaussianCurvature()
    km = props.MeanCurvature()
    return k1, k2, kg, km


def calculate_face_center(face, mesh):
    """
    计算面的中心坐标
    :param face: 面的索引
    :param mesh: 网格对象
    :return: 面的中心坐标
    """
    tris = mesh.faces[face]
    num_points = len(tris)
    center = np.array([0.0, 0.0, 0.0])
    for point_index in tris:
        center += np.array(mesh.vertices[point_index])
    center /= num_points * 3
    return center


def find_closest_surface(center, shape):
    """
    找到距离点最近的表面
    :param center: 点的坐标
    :param shape: 形状对象
    :return: 最近表面的索引
    """
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    min_distance = 100000.0
    closest_surface_index = None
    while exp.More():
        face = topods.Face(exp.Current())
        # surf = BRep_Tool.Surface(face)
        projected_point, _ = project_point_on_face(center, face)
        if projected_point is not None:
            distance = np.linalg.norm(np.array([projected_point.X(), projected_point.Y(), projected_point.Z()]) - center)
            if distance < min_distance:
                min_distance = distance
                closest_surface_index = exp.Current()
        exp.Next()
    return closest_surface_index


def map_mesh_to_shape(mesh, shape):
    """
    将 mesh 的面 ID 映射到 shape 的面 ID
    :param mesh: 网格对象
    :param shape: 形状对象
    :return: 面 ID 映射字典
    """
    surface_id_map = {}
    for face_index, region in enumerate(mesh.surface_id):
        region_tuple = tuple(region)
        if region_tuple not in surface_id_map:
            center = calculate_face_center(face_index, mesh)
            closest_surface_index = find_closest_surface(center, shape)
            surface_id_map[region] = closest_surface_index
    for i, r in enumerate(mesh.surface_id):
        mesh.surface_id[i] = surface_id_map[r]
    return mesh

def main():
    file_path = '/home/zhuxunyang/coding/banding_detect/datasets/model0501/135/135.step'
    vtk_path = '/home/zhuxunyang/coding/banding_detect/datasets/model0501/135/135_bkgm.vtk'
    shape = read_step_file(file_path)
    mesh = CustomMesh.from_vtk(vtk_path)
    if not shape:
        return
    else:
        print_face_ids(shape)
    
    # mesh = map_mesh_to_shape(mesh, shape)

    # 生成示例网格点（假设模型在合理坐标范围内，避免投影失败）
    grid_points = mesh.vertices.numpy()
    grid_tris = mesh.faces.numpy()
    
    geosurf2tris = []
    geosurf2point = defaultdict(set)
    for i, face in enumerate(grid_tris):
        center = calculate_face_center(i, mesh)
        closest_surface_index = find_closest_surface(center, shape)
        geosurf2tris.append(closest_surface_index)
    for n, tris in enumerate(grid_tris):
        for point_index in tris:
            geosurf2point[point_index].add(geosurf2tris[n])
    geosurf2point = {key: list(value) for key, value in geosurf2point.items()}

    for i, point in enumerate(grid_points):
        tmp_max = 0
        for geosurf in geosurf2point[i]:
            topo_face = topods.Face(geosurf)  # 提前转换为拓扑面
            surf_handle = BRep_Tool.Surface(topo_face)  # 获取几何曲面句柄（供曲率计算使用）
            projected_point, uv_params = project_point_on_face(point.tolist(), geosurf)
            if projected_point and uv_params is not None:
                k1, k2, kg, km = calculate_curvature(surf_handle, projected_point, uv_params)
                # print(f"{i} 投影点曲率: 主曲率1={k1:.4f}, 主曲率2={k2:.4f}, 高斯曲率={kg:.4f}, 平均曲率={km:.4f}")
                root = np.sqrt(km * km - kg)
                curvature = max(np.abs(km) + root, np.abs(km) - root)
                tmp_max = max(np.abs(k1), np.abs(k2))
        print(f"{i} 投影点曲率:{max(abs(k1), abs(k2))}")



    # # 遍历模型的所有面
    # explorer = TopExp_Explorer(shape, TopAbs_FACE)
    # while explorer.More():
    #     face = explorer.Current()
    #     topo_face = topods.Face(face)  # 提前转换为拓扑面
    #     surf_handle = BRep_Tool.Surface(topo_face)  # 获取几何曲面句柄（供曲率计算使用）
        
    #     for point in grid_points:
    #         projected_point, uv_params = project_point_on_face(point.tolist(), face)
    #         if projected_point and uv_params:
    #             # 计算曲率（传入几何曲面句柄、投影点、UV 参数）
    #             k1, k2, kg = calculate_curvature(surf_handle, projected_point, uv_params)
    #             print(f"投影点曲率: 主曲率1={k1:.4f}, 主曲率2={k2:.4f}, 高斯曲率={kg:.4f}")
    #     explorer.Next()


if __name__ == "__main__":
    main()

    # file_path = '/home/zhuxunyang/coding/banding_detect/datasets/model0501/103/103.step'
    # shape = read_step_file(file_path)
    # test_point = [1.241, 2.942, 3.738]
    # explorer = TopExp_Explorer(shape, TopAbs_FACE)
    # while explorer.More():
    #     face = explorer.Current()
    #     projected, uv = project_point_on_face(test_point, face)
    #     print("测试结果:", projected, uv)


