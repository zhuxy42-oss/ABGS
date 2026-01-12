import os
import numpy as np
from models.layers import CustomMesh
import torch
import re
from concurrent.futures import ThreadPoolExecutor
import pyvista as pv
import meshio
import subprocess

def collapse_edge(mesh, edge_id):
    # 获取边的顶点索引
    edges = mesh.extract_all_edges()
    if edge_id >= edges.n_cells:
        raise ValueError(f"边ID {edge_id} 超出范围 (总边数: {len(edges)})")
    edge_cell = edges.get_cell(edge_id)
    v1, v2 = edge_cell.point_ids
    if v1 == v2:
        raise ValueError("边的两个顶点不能相同")
    
    if v2 < v1:
        v1, v2 = v2, v1

    # 计算新顶点位置
    vertices = mesh.points
    new_vertex_pos = (vertices[v1] + vertices[v2]) / 2

    # 找到与边相关的面
    related_faces = []
    face_cells = mesh.cells_dict[5]
    for i, face in enumerate(face_cells):
        if v1 in face or v2 in face:
            related_faces.append((i, face))

    distance_sum = 0.0
    for _, face in related_faces:
        a, b, c = face
        va = vertices[a]
        vb = vertices[b]
        vc = vertices[c]

        # 计算面法线
        ab = vb - va
        ac = vc - va
        normal = np.cross(ab, ac)

        # 跳过退化面
        if np.linalg.norm(normal) < 1e-8:
            continue

        # 计算平面方程参数
        d = -np.dot(va, normal)

        # 计算点到面的距离
        distance = np.abs(np.dot(new_vertex_pos, normal) + d) / np.linalg.norm(normal)
        distance_sum += distance

    # 合并顶点
    new_vertices = np.delete(vertices, v2, axis=0)
    new_vertices[v1] = new_vertex_pos

    # 修改面的顶点索引
    new_face_cells = face_cells.copy()
    new_face_cells[new_face_cells == v2] = v1
    new_face_cells[new_face_cells > v2] -= 1

    # 移除退化面
    valid_faces = []
    for face in new_face_cells:
        if len(np.unique(face)) == 3:
            valid_faces.append(face)
    valid_faces = np.array(valid_faces)

    # 构建新的面数组（符合 pyvista 格式）
    new_faces = []
    for face in valid_faces:
        new_faces.extend([3] + face.tolist())
    new_faces = np.array(new_faces)

    # 创建新的网格
    new_mesh = pv.PolyData(new_vertices, new_faces)

    return new_mesh, distance_sum

def point_to_triangle_projection(point, mesh):
    """
    计算点在三角面片上的投影点
    :param point: 待投影的点
    :param triangle: 三角面片的三个顶点
    :return: 投影点
    """
    
    point = np.array(point)
    closest_cell = mesh.find_closest_cell(point)
    cell = mesh.get_cell(closest_cell)
    
    # 现在可以安全访问
    index0 = cell.point_ids[0]
    index1 = cell.point_ids[1]
    index2 = cell.point_ids[2]

    # 计算三角面片的法向量
    v0 = mesh.points[index1] - mesh.points[index0]
    v1 = mesh.points[index2] - mesh.points[index1]
    normal = np.cross(v0, v1)
    normal = normal / np.linalg.norm(normal)

    # 计算点到三角面片所在平面的距离
    d = np.dot(normal, point - mesh.points[index0])

    # # 计算投影点
    # projection = point - d * normal

    # s = barycentric_coordinates(projection, mesh, cell)

    return np.abs(d)


def barycentric_coordinates(point, mesh, cell):
    """
    计算点在三角面片中的重心坐标
    :param point: 待计算的点
    :param triangle: 三角面片的三个顶点
    :return: 重心坐标 (u, v, w)
    """
    index0 = cell.point_ids[0]
    index1 = cell.point_ids[1]
    index2 = cell.point_ids[2]
    v0 = mesh.points[index1] - mesh.points[index0]
    v1 = mesh.points[index2] - mesh.points[index0]
    v2 = point - mesh.points[index0]
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    # if not (0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1):
    #     raise ValueError("点不在三角形内部")
    # print(mesh.active_scalars[index0])
    return u * mesh.active_scalars[index0] + v * mesh.active_scalars[index1] + w * mesh.active_scalars[index2]

def group_vtk_files_with_numeric_prefix(directory, target_keyword="target", numeric=True):
    files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.endswith('.vtk')
    ]
    
    pattern = re.compile(r'^(.*)_.*_(.*)\.vtk$')
    # pattern = re.compile(r'^(\d+)_')  # 匹配开头数字 + 下划线

    groups = {}
    
    for filename in files:
        match = pattern.search(filename)
        if match:
            first_name, last_num = match.group(1), match.group(2)
            first_name, last_num = str(first_name), int(last_num)

            key = (first_name, last_num)

            if key not in groups:
                groups[key] = {"normal": [], "target": []}
            
            # 根据是否含关键词分类
            if target_keyword.lower() in filename.lower():
                groups[key]["target"].append(filename)
            else:
                groups[key]["normal"].append(filename)
    
    # 合并每组中的文件（普通文件在前，target文件在后）
    sorted_groups = []
    for key in sorted(groups.keys()):
        combined = groups[key]["normal"] + groups[key]["target"]
        if combined:  # 忽略空组
            sorted_groups.append(combined)
    
    return sorted_groups

def read_vtk_files(directory):
    normal_vtk_files = []
    target_vtk_files = []

    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.vtk'):
            file_path = os.path.join(directory, filename)
            if 'target' in filename.lower():
                target_vtk_files.append(file_path)
            else:
                normal_vtk_files.append(file_path)

    # 读取普通 VTK 文件
    normal_meshes = []
    for file in normal_vtk_files:
        try:
            mesh = CustomMesh.CustomMesh.from_vtk(file)
            normal_meshes.append(mesh)
            print(file, "read success")
        except Exception as e:
            print(f"读取 {file} 时出错: {e}")

    # 读取带 target 的 VTK 文件
    target_meshes = []
    for file in target_vtk_files:
        try:
            mesh = CustomMesh.CustomMesh.from_vtk(file)
            target_meshes.append(mesh)
            print(file, "read success")
        except Exception as e:
            print(f"读取 {file} 时出错: {e}")

    return normal_meshes, target_meshes

def One_Hot(labels, origin_mesh):
        one_hots = []
        for label in labels:
            one_hot = torch.zeros(len(origin_mesh))
            one_hot[labels] = 1
            one_hots.append(one_hot)
        
        return torch.stack(one_hots, dim=0)

def get_info_vista(mesh):
    # 读取点信息
    points = mesh.points
    # print(points.size())
    print("点信息：")
    print(points)

    # 读取线信息
    edges = mesh.extract_all_edges()
    # edge_cells = edges.cell.reshape(-1, 3)[:, 1:]  # 调整形状以获取每条边的两个顶点索引
    # print(edge_cells.size())
    print("\n线信息（每条线的顶点索引）：")
    for i in range(edges.n_cells):
        print(edges.get_cell(i).point_ids)

    # 读取面信息
    faces = mesh.cells_dict[5]  # 调整形状以获取每个面的顶点索引
    # print(faces.size())
    print("\n面信息（每个面的顶点索引）：")
    print(faces)

def get_label_QEM(dir):
    red = [1, 0, 0]
    blue = [0, 0, 1]
    mesh = CustomMesh.CustomMesh.from_vtk1(dir)
    dis_loss = []
    num = 0
    for i in range(len(mesh.edges)):
        new_mesh, dis = mesh.collapsing_edge(i)
        dis_loss.append(dis.detach().numpy())
        print("edge {:<4} QEM Loss: {:<10.4f}".format(i, dis))
    
    # 获取 dis 值最小的 n 条边的索引
    dis_loss = np.array(dis_loss)
    min_n_indices = np.argsort(dis_loss)[:len(mesh.edges) // 5]

    # 根据索引为边分配颜色
    colors = []
    for i in range(len(mesh.edges)):
        if i in min_n_indices:
            colors.append(red)
        else:
            colors.append(blue)

    # 将边转换为线单元（meshio 中的 "line" 类型）
    cells = [("line", mesh.edges.detach().numpy())]

    # 颜色数据作为单元格标量数据
    cell_data = {"Color": [colors]}

    # 创建网格对象
    meshio_mesh = meshio.Mesh(
        points=mesh.vertices.detach().numpy(),
        cells=cells,
        cell_data=cell_data
    )

    # 保存为 VTK 文件
    meshio_mesh.write("colored_edges.vtk")


def get_labels(list, directory):
    BLACK = 0    
    YELLOW = 1
    save_directory = "/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/sf/label_prob/"
    for group in list:
        target_path = os.path.join(directory, group[-1]) 
        for i in range(len(group) - 1):
            data_path = os.path.join(directory, group[i])
            name_part = group[i].split("_")
            data_mesh = CustomMesh.CustomMesh.from_vtk(data_path)
            save_path1 = "/home/zhuxunyang/coding/bkgm_simplification/middle.vtk"
            save_path2 = "/home/zhuxunyang/coding/bkgm_simplification/middle_sf.vtk"
            geo_path = f"/home/zhuxunyang/coding/bkgm_simplification/datasets/model/{name_part[0]}/{name_part[0]}.step"
            target_mesh_pv = pv.read(target_path)
            parts = data_path.split("_")
            num = parts[-2] + "_" + parts[-1].replace(".vtk", "")
            prob = []
            loss1_list = []
            size_diff_list = []
            for j in range(len(data_mesh.edges)):
                v1, v2 = data_mesh.edges[j]
                new_mesh, _ , past_element , _ = data_mesh.collapsing_edge_id(j)
                dis = point_to_triangle_projection(data_mesh.vertices[v1], target_mesh_pv)
                new_mesh.writeVTK(save_path1)
                theta = 0.1 * int(parts[-1][0]) + 0.1
                command = ['./test_sizefield1', save_path1, save_path2, '1.2', '10', str(theta), geo_path]
                try:
                    result = subprocess.run(command, check=True, text=True, capture_output=True)
                    print("size field generate success")
                    # print(result.stdout)
                except subprocess.CalledProcessError as e:
                    print("size field generate failed")
                    print(e.stderr)

                mesh = CustomMesh.CustomMesh.from_vtk(save_path2)
                post_element = 0
                related_face = []
                for n, face in enumerate(mesh.faces):
                    if v1 in face:
                        related_face.append(face)
                        post_element += mesh.area_num[n]
            
                loss1 = dis
                loss2 = (post_element - past_element) / past_element
                if(loss2 < 0):
                    loss2 = 0
                loss = loss1 + loss2
                loss1_list.append(loss1)
                size_diff_list.append(loss2)

                print("Mesh {:<5} edge {:<4}/{:<4} sum loss: {:<10.4f} QEM Loss: {:<10.4f} Size: {:<10.4f}=({:<8}-{:<8})/{:<8} ".format(
                    group[i], j, len(data_mesh.edges), loss, loss1, loss2, post_element, past_element, past_element
                ))
                prob.append(loss)
            # probs = torch.softmax(torch.stack(prob, dim=0), dim=0)\
            
            probs = np.stack(prob)
            max_value = np.max(probs)
            min_value = np.min(probs)
            probs = (probs - min_value) / max_value
            # probs = 0.5 * probs + 0.5
            print(np.max(probs).item(), np.min(probs).item())

            save_path = os.path.join(save_directory, name_part[0] + "_" + name_part[1] + "_" + name_part[2] + ".pt")
            torch.save(torch.tensor(probs), save_path)
            print("Save success!")

def size_loss(data_mesh, target_mesh):
    sum_dis = 0
    size_loss = 0
    for i in range(len(data_mesh.points)):
        dis, size = point_to_triangle_projection(data_mesh.points[i], target_mesh)
        sum_dis += dis
        size_loss += np.abs(size - data_mesh.sizevalue[i])
    print("Distance Loss:", sum_dis, "Size Loss:", size_loss)
    return sum_dis + size_loss


def label_mesh():
    BLUE = 0    # 蓝色编码（长边）
    RED = 1      # 红色编码（短边）
    mesh = pv.read("cubic_2.vtk")
    edges = mesh.extract_all_edges()
    # 初始化颜色数组 (0=蓝, 1=红)
    edge_colors = np.full(edges.n_cells, RED, dtype=np.int32)

    # 计算每条边长度并标记颜色
    for i in range(edges.n_cells):
        cell = edges.GetCell(i)
        pt1, pt2 = cell.GetPointIds().GetId(0), cell.GetPointIds().GetId(1)
        length = np.linalg.norm(edges.points[pt1] - edges.points[pt2])
        if length > 1:
            edge_colors[i] = BLUE

    # 添加颜色数据 (必须作为Cell Data!)
    edges.cell_data["EdgeType"] = edge_colors
    edges.save("colored_edges_by_length.vtk")

if __name__ == "__main__":
    directory = '/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/sf/data'  # 替换为实际的目录路径
    list = group_vtk_files_with_numeric_prefix(directory)
    for group in list:
        print(group)
    get_labels(list, directory)




    # label_mesh()
    # mesh = pv.read('/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/new_cubic/Backgroundmesh_cubic_2.vtk')
    # get_info_vista(mesh)
    # get_label_QEM('/home/zhuxunyang/coding/bkgm_simplification/cylinder_1.vtk')


    # normal_meshes, target_meshes = read_vtk_files(directory)
    # for mesh in normal_meshes:
    #     print("Num of edge:", len(mesh.edges))
    # label = get_label(normal_meshes, target_meshes[0])
    # torch.save(label, 'one_hot_labels.pt')
    # print(f"读取到的普通 VTK 文件数量: {len(normal_meshes)}")
    # print(f"读取到的带 target 的 VTK 文件数量: {len(target_meshes)}")