import h5py
import os
import numpy as np
from models.layers import CustomMesh
import torch
import re
from concurrent.futures import ThreadPoolExecutor
# import pyvista as pv
import meshio
# import trimesh
import subprocess
import shutil
from util.calculate_gradient import map_triangle_to_2d, calculate_gradinet
from util.bending import integrate_over_triangle
from util.tensor_replace import replace_tensor
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool  # 使用线程池替代进程池
import subprocess
from functools import partial
import time
import itertools
import traceback

def generate_edge_combinations(mesh, top_n):
    """
    根据网格边长计算前n个最长的边，生成所有可能的01排列组合
    
    参数:
        mesh: 网格对象，需包含vertices和edges属性
        top_n: 选择前n个最长的边
    
    返回:
        combinations: 形状为(k, m)的张量，k是组合数，m是边总数
        selected_edge_indices: 被选中的前n个最长边的索引列表
    """
    # 计算所有边的长度
    edge_lengths = []
    for edge in mesh.edges:
        v1, v2 = edge
        length = torch.norm(mesh.vertices[v1] - mesh.vertices[v2])
        edge_lengths.append(length)
    
    edge_lengths = torch.tensor(edge_lengths)

    if top_n > 20:
        top_n = 20
    
    # 获取前n个最长边的索引
    _, selected_edge_indices = torch.topk(edge_lengths, k=top_n)
    selected_edge_indices = selected_edge_indices.tolist()
    
    # 生成所有可能的01组合（只针对这n个边）
    binary_combinations = [comb for comb in itertools.product([0, 1], repeat=top_n) if sum(comb) > 0]
    
    # 创建全零张量，然后设置选中的边
    m = len(mesh.edges)
    combinations = torch.zeros((len(binary_combinations), m), dtype=torch.float32)
    
    for i, comb in enumerate(binary_combinations):
        for j, edge_idx in enumerate(selected_edge_indices):
            combinations[i, edge_idx] = comb[j]
    
    # 添加batch维度 (1, m)
    combinations = combinations.unsqueeze(1)  # 形状变为 (k, 1, m)
    
    return combinations, selected_edge_indices

def find_adjacent_edges_except_target(mesh, v1, v2):
    # 1. 查找包含 v1 和 v2 的邻接面
    adjacent_faces = []
    for face in mesh.faces:
        if v1 in face and v2 in face:
            adjacent_faces.append(face)
    # adjacent_faces = mesh.faces[np.any((mesh.faces == v1) & (mesh.faces == v2), axis=1)]
    if not adjacent_faces:
        return []  # 没有邻接面
    
    # 2. 提取邻接面的所有边（并去重）
    all_edges = set()
    for face in adjacent_faces:
        # 生成面的 3 条边（注意顺序统一，避免重复）
        edges = [
            tuple(sorted((face[0].item(), face[1].item()))),
            tuple(sorted((face[1].item(), face[2].item()))),
            tuple(sorted((face[2].item(), face[0].item())))
        ]
        all_edges.update(edges)
    
    # 3. 移除目标边 (v1, v2)
    # target_edge = tuple(sorted((v1.item(), v2.item())))
    # all_edges.discard(target_edge)

    all_edges = list(all_edges)
    edge_to_index = {tuple(sorted((edge[0].item(), edge[1].item()))): idx for idx, edge in enumerate(mesh.edges)}
    edge_indices = [edge_to_index[edge] for edge in all_edges if edge in edge_to_index]

    
    return all_edges, edge_indices

def detect_non_manifold_edge(root, file_name):
    vtk_file = root + '/' + file_name
    mesh = CustomMesh.CustomMesh.from_vtk(vtk_file)

    print(file_name, " detecting")
    
    manifold = True
    for edge in mesh.edges:
        v1, v2 = edge
        # 找到包含该边的两个三角形
        adjacent_faces = []
        for i, face in enumerate(mesh.faces):
            if v1 in face and v2 in face:
                adjacent_faces.append(i)
        
        if len(adjacent_faces) != 2:
            manifold = False
            break
    
    if manifold:
        print(file_name, "Manifold mesh")
    else:
        print(file_name, "Non manifold mesh")

def process_edge(args, data_mesh, geo_file, file_name):
    try:
        j, save_path1, save_path2 = args
        
        v1, v2 = data_mesh.edges[j]
        face_index = []
        for i, face in enumerate(data_mesh.faces):
            if v1 in face and v2 in face:
                face_index.append(i)
        
        try:
            new_mesh, past_elements, post_face = data_mesh.split_edge(j)
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            # return None
            raise Exception(f"Edge splitting failed for edge {j} is no-manifold")  # 抛出异常以终止父进程

        new_mesh.writeVTK(save_path1)

        # 这部分必须串行执行
        command = ['./test_sizefield2', save_path1, save_path2, '1.2', '0', '0.1', geo_file]
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            # print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"{file_name} size field generate failed because edge {j}")
            print(e.stderr)
            # return None
            raise Exception(f"Size field generation failed for edge {j}")  # 抛出异常以终止父进程

        splited_mesh = CustomMesh.CustomMesh.from_vtk(save_path2)
        
        post_elements = 0.0
        for face in post_face:
            coord_3d = torch.stack([splited_mesh.vertices[face[0]], splited_mesh.vertices[face[1]], splited_mesh.vertices[face[2]]])
            coord_2d = map_triangle_to_2d(coord_3d)
            size = np.array([splited_mesh.sizing_values[face[0]], splited_mesh.sizing_values[face[1]], splited_mesh.sizing_values[face[2]]])
            num = integrate_over_triangle(coord_2d.numpy(), size)
            post_elements += num
        
        ratio = 0
        if past_elements[0] / post_elements[0] < 1:
            ratio = 1
        else:
            ratio = past_elements[0] / post_elements[0]

        print(f"File {file_name} Edge {j}/{len(data_mesh.edges)} Points: {data_mesh.edges[j][0].item()} {data_mesh.edges[j][1].item()} Post num: {post_elements[0]} Past num: {past_elements[0]} Ratio: {ratio}\n")
        
        return ratio
    except Exception as e:
        # 捕获所有异常并重新抛出，以便父进程可以处理
        raise Exception(f"Error processing edge {j}: {str(e)}")

def label_bending(root, file_name, num_processes=None):
    try:
        red = [1, 0, 0]
        blue = [0, 0, 1]
        vtk_file = os.path.join(root, f"{file_name}_bkgm.vtk")
        geo_name = file_name.split('_')[0]
        geo_file = os.path.join(f"/home/zhuxunyang/coding/banding_detect/datasets/model0521/{geo_name}", f"{geo_name}.step")
        
        data_mesh = CustomMesh.CustomMesh.from_vtk(vtk_file)
        # past_num_mesh = data_mesh.area_num
        # data_mesh.get_all_info()
        
        if num_processes is None:
            # num_processes = cpu_count()  # 默认使用所有CPU核心
            num_processes = 16
        
        # 准备边缘处理参数
        edge_args = []
        for j in range(len(data_mesh.edges)):
            save_path1 = f"/home/zhuxunyang/coding/banding_detect/temporary_banding/split_{file_name}_{j}.vtk"
            save_path2 = f"/home/zhuxunyang/coding/banding_detect/temporary_banding/split_sf_{file_name}_{j}.vtk"
            edge_args.append((j, save_path1, save_path2))
        
        # 使用partial固定不变参数
        process_edge_partial = partial(process_edge, 
                                    data_mesh=data_mesh,
                                    geo_file=geo_file,
                                    file_name=file_name)
        
        # 使用多进程处理边缘
        ratios = []

        # 使用多线程处理边缘
        # with ThreadPool(num_processes) as pool:  # 改为ThreadPool
        #     results = pool.map(process_edge_partial, edge_args)
        with ThreadPool(num_processes) as pool:
            try:
                results = pool.map(process_edge_partial, edge_args)
            except Exception as e:
                print(f"❌ 处理文件 {file_name} 时发生错误，终止该文件的处理: {str(e)}")
                pool.terminate()  # 终止线程池
                raise  # 重新抛出异常，完全停止当前文件的处理
        
        # 收集有效结果
        ratios = [r for r in results if r is not None]
        
        # 保存结果
        try:
            # 1. 处理张量并保存 .pt 文件
            banding_q = torch.tensor(np.stack(ratios))
            torch.save(banding_q, os.path.join(root, f"{file_name}.pt"))
            
            # 2. 生成颜色列表
            colors = []
            for r in ratios:
                colors.append(red if r > 1.20 else blue)
            
            # 3. 生成网格数据并保存 .vtk 文件
            cells = [("line", data_mesh.edges.detach().numpy())]
            cell_data = {"Color": [colors], "banding_condition":[ratios]}
            meshio_mesh = meshio.Mesh(
                points=data_mesh.vertices.detach().numpy(),
                cells=cells,
                cell_data=cell_data
            )
            meshio_mesh.write(os.path.join(root, f"colored_edge_banding_{file_name}.vtk"))
            
            print(f"成功处理文件：{os.path.join(root, file_name)}")
        
        except Exception as e:
            print(f"❌ 处理文件 {file_name} 时发生错误：{str(e)}")
            print("-" * 60)
            traceback.print_exc()  # 打印完整的堆栈跟踪信息
            print("-" * 60)
    except Exception as e:
        # print(f"Detect no-manifold edge in {file_name}")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60)
        return  # 直接返回，不继续执行

def remove_directory_content(directory):
    try:
        # 遍历指定目录下的所有文件和子目录
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                # 删除文件
                os.remove(file_path)
                print(f"已删除文件: {file_path}")
    except Exception as e:
        print(f"删除文件或目录时出现错误: {e}")

def process_file(root, file):
    if file.endswith('_bkgm.vtk'):
        root_part = root.split('/')
        Root_dir = '/'.join(root_part[:-1])
        pt_root = os.path.join(Root_dir, "label")
        visual_root = os.path.join(Root_dir, "visual")
        file_name = file.split('_bkgm.vtk')[0]
        pt_file = os.path.join(pt_root, f"{file_name}.pt")
        colored_file = os.path.join(visual_root, f"colored_edge_banding_{file_name}.vtk")
        
        # 检查文件是否已存在（可选）
        if os.path.exists(pt_file) and os.path.exists(colored_file):
            print(f"{file}: has been labeled")
            return
        else:
            try:
                label_bending(root, file_name)
            except Exception as e:
                print(f"处理文件 {os.path.join(root, file)} 时发生错误：{str(e)}")
                print("-" * 60)
                traceback.print_exc()  # 打印完整的堆栈跟踪信息
                print("-" * 60)

def copy_file(root, file, target_dir):
    """
    处理文件并将其复制到目标目录
    
    参数:
        root: 当前文件所在的根目录
        file: 文件名
        target_dir: 目标目录路径
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    if file.endswith('_bkgm.vtk'):
        file_name = file.split('_bkgm.vtk')[0]
        bkgm_file = pt_file = os.path.join(root, f"{file_name}_bkgm.vtk")
        pt_file = os.path.join(root, f"{file_name}.pt")
        colored_file = os.path.join(root, f"colored_edge_banding_{file_name}.vtk")

        bkgm_dir = os.path.join(target_dir, "data")
        pt_dir = os.path.join(target_dir, "label")
        colored_dir = os.path.join(target_dir, "visual")
        
        # 复制 bkgm 文件，并在目标目录中重命名（如 111_bkgm.vtk → 111_0_bkgm.vtk）
        if os.path.exists(bkgm_file):
            new_bkgm_name = f"{file_name}_-1_bkgm.vtk"
            new_bkgm_path = os.path.join(bkgm_dir, new_bkgm_name)
            shutil.copy2(bkgm_file, new_bkgm_path)
            print(f"已复制并重命名: {bkgm_file} -> {new_bkgm_path}")
        else:
            print(f"文件不存在: {bkgm_file}")
        
        # 复制 pt 文件
        if os.path.exists(pt_file):
            new_pt_name = f"{file_name}_-1.pt"
            new_pt_path = os.path.join(pt_dir, new_pt_name)
            shutil.copy2(pt_file, new_pt_path)
            print(f"已复制: {pt_file} -> {new_pt_name}")
        else:
            print(f"文件不存在: {pt_file}")
        
        # 复制 colored 文件
        if os.path.exists(colored_file):
            new_color_name = f"colored_edge_banding_{file_name}_-1_pt.vtk"
            new_color_path = os.path.join(colored_dir, new_color_name)
            shutil.copy2(colored_file, new_color_path)
            print(f"已复制: {colored_file} -> {colored_dir}")
        else:
            print(f"文件不存在: {colored_file}")

def expand_data(root_dir):
    bkgm_list = []
    pt_list = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('-1_bkgm.vtk'):
                # 提取文件名中的前两个数字（假设格式为 "数字_数字_..."）
                parts = re.split(r'[_\.]', file)
                if len(parts) >= 2:
                    # 提取前两个数字部分（例如 "111_222" → ["111", "222"]）
                    num1 = parts[0]
                    num2 = parts[1]
                    
                    sort_key = (int(num1), int(num2))  # 用于排序的元组
                    bkgm_list.append((sort_key, os.path.join(root, file)))
            
            elif file.endswith('-1.pt'):
                # 同样处理 .pt 文件
                parts = re.split(r'[_\.]', file)
                if len(parts) >= 2:
                    num1 = parts[0]
                    num2 = parts[1]
                    sort_key = (int(num1), int(num2))
                    pt_list.append((sort_key, os.path.join(root, file)))
                        
    
    # 对 bkgm_list 和 pt_list 进行排序（按前两个数字升序）
    bkgm_list.sort(key=lambda x: x[0])
    pt_list.sort(key=lambda x: x[0])
    
    # 提取排序后的文件路径（去掉 sort_key）
    sorted_bkgm = [item[1] for item in bkgm_list]
    sorted_pt = [item[1] for item in pt_list]

    for i in range(len(sorted_bkgm)):
        mesh = CustomMesh.CustomMesh.from_vtk(sorted_bkgm[i])
        label = torch.load(sorted_pt[i])
        name = re.split("/", sorted_pt[i])[-1].split('.')[0].split('_')[0]
        number = re.split("/", sorted_pt[i])[-1].split('.')[0].split('_')[-1]
        valid_indices = torch.where(label < 1.2)[0]
        selected_indices = valid_indices[torch.randperm(len(valid_indices))[:len(valid_indices) // 5]]
        for n, index in enumerate(selected_indices):
            new_mesh, _, _ = mesh.split_edge(index)
            save_path1 = f"/home/zhuxunyang/coding/banding_detect/temporary_banding/split_{name}_{index}.vtk"
            # save_path2 = f"/home/zhuxunyang/coding/banding_detect/temporary_banding/split_sf_{name}_{n}.vtk"
            save_path2 = f"/home/zhuxunyang/coding/banding_detect/datasets/model0609/{name}_{index}.vtk"
            geo_file = f"/home/zhuxunyang/coding/banding_detect/datasets/model0521/{name}/{name}.step"
            new_mesh.writeVTK(save_path1)

            command = ['./test_sizefield2', save_path1, save_path2, '1.2', '0', '0.1', geo_file]
            try:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(e.stderr)
                return None

            this_mesh = CustomMesh.CustomMesh.from_vtk(save_path2)

            #找到该边临近的四条边的索引
            v1, v2 = mesh.edges[index]
            _, adjust_edge_index = find_adjacent_edges_except_target(mesh, v1, v2)
            mask = np.ones(len(mesh.edges), dtype=bool)
            mask[adjust_edge_index] = False

            origin_label = np.array(label)[mask]

            # last_eight_edges = mesh.edge[-8:]
            edge_indices = list(range(len(this_mesh.edges))[-8:])
            
            new_label = []
            for e in edge_indices:
                mesh1, past_num, past_faces = this_mesh.split_edge(e)
                save_path3 = f"/home/zhuxunyang/coding/banding_detect/temporary_banding/split_{name}_{index}_{e}.vtk"
                save_path4 = f"/home/zhuxunyang/coding/banding_detect/temporary_banding/split_sf_{name}_{index}_{e}.vtk"
                mesh1.writeVTK(save_path3)
                command = ['./test_sizefield2', save_path3, save_path4, '1.2', '0', '0.1', geo_file]
                try:
                    result = subprocess.run(command, check=True, text=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(e.stderr)
                    return None
                mesh1 = CustomMesh.CustomMesh.from_vtk(save_path4)
                post_num = 0
                for face in past_faces:
                    coord_3d = torch.stack([mesh1.vertices[face[0]], mesh1.vertices[face[1]], mesh1.vertices[face[2]]])
                    coord_2d = map_triangle_to_2d(coord_3d)
                    size = np.array([mesh1.sizing_values[face[0]], mesh1.sizing_values[face[1]], mesh1.sizing_values[face[2]]])
                    num = integrate_over_triangle(coord_2d.numpy(), size)
                    post_num += num
                r = past_num / post_num
                new_label.append(r)
            new_label = np.array(new_label)
            origin_label = torch.from_numpy(np.concatenate([np.expand_dims(origin_label, axis=1), new_label]))

            red = [1, 0, 0]
            blue = [0, 0, 1]
            colors = []
            for r in origin_label:
                colors.append(red if r > 1.20 else blue)

            cells = [("line", this_mesh.edges.detach().numpy())]
            cell_data = {"Color": [colors], "banding_condition":[origin_label]}
            meshio_mesh = meshio.Mesh(
                points=this_mesh.vertices.detach().numpy(),
                cells=cells,
                cell_data=cell_data
            )
            meshio_mesh.write(os.path.join(root_dir, "visual", f"colored_edge_banding_{name}_{n}.vtk"))
            print(f"{name}_{n} processed")

def expand_data1(root_dir):
    bkgm_list = []
    pt_list = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('-1_bkgm.vtk'):
                # 提取文件名中的前两个数字（假设格式为 "数字_数字_..."）
                parts = re.split(r'[_\.]', file)
                if len(parts) >= 2:
                    # 提取前两个数字部分（例如 "111_222" → ["111", "222"]）
                    num1 = parts[0]
                    num2 = parts[1]
                    
                    sort_key = (int(num1), int(num2))  # 用于排序的元组
                    bkgm_list.append((sort_key, os.path.join(root, file)))
            
            elif file.endswith('-1.pt'):
                # 同样处理 .pt 文件
                parts = re.split(r'[_\.]', file)
                if len(parts) >= 2:
                    num1 = parts[0]
                    num2 = parts[1]
                    sort_key = (int(num1), int(num2))
                    pt_list.append((sort_key, os.path.join(root, file)))
                        
    
    # 对 bkgm_list 和 pt_list 进行排序（按前两个数字升序）
    bkgm_list.sort(key=lambda x: x[0])
    pt_list.sort(key=lambda x: x[0])
    
    # 提取排序后的文件路径（去掉 sort_key）
    sorted_bkgm = [item[1] for item in bkgm_list]
    sorted_pt = [item[1] for item in pt_list]

    for i in range(len(sorted_bkgm)):
        mesh = CustomMesh.CustomMesh.from_vtk(sorted_bkgm[i])
        read_label = torch.load(sorted_pt[i])
        name = re.split("/", sorted_pt[i])[-1].split('.')[0].split('_')[0]
        number = re.split("/", sorted_pt[i])[-1].split('.')[0].split('_')[-1]
        banding_label, _ = generate_edge_combinations(mesh, len(mesh.edges) // 100)
        if banding_label.size(0) > 40:
            banding_label = banding_label[0:40, :, :]
        
        generated_file = 1
        for n, label in enumerate(banding_label):
            if generated_file > 20:
                break
            expand_vtk_path = os.path.join(root_dir, "data", f"{name}_{n}_bkgm.vtk")
            expand_pt_path = os.path.join(root_dir, "label", f"{name}_{n}.pt")
            expand_visual_path = os.path.join(root_dir, "visual", f"colored_edge_banding_{name}_{n}.vtk")
            if os.path.exists(expand_pt_path) and os.path.exists(expand_visual_path):
                print(f"{name}_{n} file exist")
                generated_file += 1
                continue
            try:
                new_mesh, effect_edge_id, new_edge_info = mesh.solve_banding1(label)
                save_path1 = f"/home/zhuxunyang/coding/banding_detect/temporary_banding/split_{name}_{n}.vtk"
                save_path2 = expand_vtk_path
                geo_file = f"/home/zhuxunyang/coding/banding_detect/datasets/model0521/{name}/{name}.step"
                new_mesh.writeVTK(save_path1)

                command = ['./test_sizefield2', save_path1, save_path2, '1.2', '0', '0.1', geo_file]
                try:
                    result = subprocess.run(command, check=True, text=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    print(e.stderr)
                    return None

                this_mesh = CustomMesh.CustomMesh.from_vtk(save_path2)

                # mask = np.ones(len(mesh.edges), dtype=bool)
                # mask[effect_edge_id] = False

                # deleted_label = np.array(read_label)[mask]

                
                new_label = []
                for e in new_edge_info:
                    mesh1, past_num, past_faces = this_mesh.split_edge(e)
                    save_path3 = f"/home/zhuxunyang/coding/banding_detect/temporary_banding/split_{name}_{n}_{e}.vtk"
                    save_path4 = f"/home/zhuxunyang/coding/banding_detect/temporary_banding/split_sf_{name}_{n}_{e}.vtk"
                    mesh1.writeVTK(save_path3)
                    command = ['./test_sizefield2', save_path3, save_path4, '1.2', '0', '0.1', geo_file]
                    try:
                        result = subprocess.run(command, check=True, text=True, capture_output=True)
                    except subprocess.CalledProcessError as e:
                        print(e.stderr)
                        return None
                    mesh1 = CustomMesh.CustomMesh.from_vtk(save_path4)
                    post_num = 0
                    for face in past_faces:
                        coord_3d = torch.stack([mesh1.vertices[face[0]], mesh1.vertices[face[1]], mesh1.vertices[face[2]]])
                        coord_2d = map_triangle_to_2d(coord_3d)
                        size = np.array([mesh1.sizing_values[face[0]], mesh1.sizing_values[face[1]], mesh1.sizing_values[face[2]]])
                        num = integrate_over_triangle(coord_2d.numpy(), size)
                        post_num += num
                    r = max(past_num[0] / post_num[0], 1.0)
                    new_label.append(r)
            except:
                print(f"{name}_{n} file process failed")
                continue
            new_label = np.array(new_label)

            # save_label = torch.from_numpy(np.concatenate([deleted_label, new_label])).T
            try:
                save_label = replace_tensor(read_label, effect_edge_id, new_edge_info, new_label)
            except:
                continue

            torch.save(torch.tensor(save_label), expand_pt_path)

            # visual_label = np.array(save_label)

            red = [1, 0, 0]
            blue = [0, 0, 1]
            colors = []
            for r in save_label:
                colors.append(red if r > 1.20 else blue)

            cells = [("line", this_mesh.edges.detach().numpy())]
            cell_data = {"Color": [colors], "banding_condition":[save_label]}
            meshio_mesh = meshio.Mesh(
                points=this_mesh.vertices.detach().numpy(),
                cells=cells,
                cell_data=cell_data
            )
            # this_mesh.writeVTK(os.path.join(root_dir, "data", f"{name}_{n}_bkgm.vtk"))
            meshio_mesh.write(expand_visual_path)
            print(f"{name}_{n} processed")
            generated_file += 1

def check_data(root_dir, dry_run=False):
    bkgm_list = []
    pt_list = []
    visual_list = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('_bkgm.vtk'):
                # 提取文件名中的前两个数字（假设格式为 "数字_数字_..."）
                parts = re.split(r'[_\.]', file)
                if len(parts) >= 2:
                    # 提取前两个数字部分（例如 "111_222" → ["111", "222"]）
                    num1 = parts[0]
                    num2 = parts[1]
                    
                    sort_key = (int(num1), int(num2))  # 用于排序的元组
                    bkgm_list.append((sort_key, os.path.join(root, file)))
            
            elif file.endswith('.pt'):
                # 同样处理 .pt 文件
                parts = re.split(r'[_\.]', file)
                if len(parts) >= 2:
                    num1 = parts[0]
                    num2 = parts[1]
                    sort_key = (int(num1), int(num2))
                    pt_list.append((sort_key, os.path.join(root, file)))

            elif file.endswith('.vtk'):
                parts = re.split(r'[_\.]', file)
                if len(parts) >= 2:
                    num1 = parts[-3]
                    num2 = parts[-2]
                    sort_key = (int(num1), int(num2))
                    visual_list.append((sort_key, os.path.join(root, file)))

     # 第二阶段：找出所有三个列表中共有的sort_key
    bkgm_keys = set(item[0] for item in bkgm_list)
    pt_keys = set(item[0] for item in pt_list)
    visual_keys = set(item[0] for item in visual_list)
    
    # 计算交集
    common_keys = bkgm_keys & pt_keys & visual_keys

    # 第三阶段：过滤列表，只保留共同的sort_key
    filtered_bkgm = [item for item in bkgm_list if item[0] in common_keys]
    filtered_pt = [item for item in pt_list if item[0] in common_keys]
    filtered_visual = [item for item in visual_list if item[0] in common_keys]

     # 第四阶段：删除不符合条件的文件
    def delete_files(file_list, filtered_list, file_type):
        files_to_delete = [item for item in file_list if item[0] not in common_keys]
        print(f"找到 {len(files_to_delete)} 个不符合条件的{file_type}文件")
        
        for sort_key, file_path in files_to_delete:
            try:
                if dry_run:
                    print(f"[干运行] 会删除: {file_path}")
                else:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
            except Exception as e:
                print(f"删除失败: {file_path}, 错误: {e}")
    
    # 执行删除操作
    delete_files(bkgm_list, filtered_bkgm, "_bkgm.vtk")
    delete_files(pt_list, filtered_pt, ".pt")
    delete_files(visual_list, filtered_visual, "color.vtk")
    
    return bkgm_list, pt_list, visual_list
    

def visual_generate(mesh, label, name):
    data_mesh = CustomMesh.CustomMesh.from_vtk(mesh)
    banding_condition = torch.load(label)
    red = [1, 0, 0]
    blue = [0, 0, 1]
    colors = []
    for r in banding_condition:
        colors.append(red if r > 1.20 else blue)

    cells = [("line", data_mesh.edges.detach().numpy())]
    cell_data = {"Color": [colors], "banding_condition":[banding_condition]}
    meshio_mesh = meshio.Mesh(
        points=data_mesh.vertices.detach().numpy(),
        cells=cells,
        cell_data=cell_data
    )
    meshio_mesh.write(os.path.join("/home/zhuxunyang/coding/banding_detect/datasets/model0609/visual", f"colored_edge_banding_{name}.vtk"))

def check_data1(root_dir):
    bkgm_list = []
    pt_list = []
    visual_list = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('_bkgm.vtk'):
                # 提取文件名中的前两个数字（假设格式为 "数字_数字_..."）
                parts = re.split(r'[_\.]', file)
                if len(parts) >= 2:
                    # 提取前两个数字部分（例如 "111_222" → ["111", "222"]）
                    num1 = parts[0]
                    num2 = parts[1]
                    if num2 == '-1':
                        os.remove(os.path.join(root, file))
                        print(f"{os.path.join(root, file)} deleted")
                        continue
                    
                    sort_key = (int(num1), int(num2))  # 用于排序的元组
                    bkgm_list.append((sort_key, os.path.join(root, file)))
            
            elif file.endswith('.pt'):
                # 同样处理 .pt 文件
                parts = re.split(r'[_\.]', file)
                if len(parts) >= 2:
                    num1 = parts[0]
                    num2 = parts[1]
                    if num2 == '-1':
                        os.remove(os.path.join(root, file))
                        print(f"{os.path.join(root, file)} deleted")
                        continue
                    sort_key = (int(num1), int(num2))
                    pt_list.append((sort_key, os.path.join(root, file)))

            elif file.endswith('.vtk'):
                parts = re.split(r'[_\.]', file)
                if len(parts) >= 2:
                    num1 = parts[-3]
                    num2 = parts[-2]
                    if num2 == '-1':
                        os.remove(os.path.join(root, file))
                        print(f"{os.path.join(root, file)} deleted")
                        continue
                    sort_key = (int(num1), int(num2))
                    visual_list.append((sort_key, os.path.join(root, file)))

    # bkgm_list.sort(key=lambda x: x[0])
    # pt_list.sort(key=lambda x: x[0])
    # visual_list.sort(key=lambda x: x[0])


    # length = len(bkgm_list)

    # for i in range(length):
    #     mesh = CustomMesh.CustomMesh.from_vtk(bkgm_list[i][1])
    #     label = torch.load(pt_list[i][1])
    #     visual = meshio.read(visual_list[i][1])
    #     # print(len(mesh.edges), label.shape, visual.cell_data["banding_condition"][0].shape)
    #     true_edge_len = len(mesh.edges)
    #     pt_len = label.shape[0]
    #     visual_len = visual.cell_data["banding_condition"][0].shape[0]
    #     if true_edge_len != pt_len and len(mesh.edges) != visual_len:
    #         print("Error file", bkgm_list[i][0], true_edge_len, pt_len, visual_len)
    #         os.remove(bkgm_list[i][1], pt_list[i][1], visual_list[i][1])
    #     elif true_edge_len == pt_len and len(mesh.edges) != visual_len:
    #         print("Error file", bkgm_list[i][0], true_edge_len, pt_len, visual_len)
    #         continue
    #     elif true_edge_len != pt_len and len(mesh.edges) == visual_len:
    #         print("Error file", bkgm_list[i][0], true_edge_len, pt_len, visual_len)
    #         label = torch.from_numpy(visual.cell_data["banding_condition"][0])
    #         torch.save(label, (pt_list[i][1]))
    #     else:
    #         print("Right file", bkgm_list[i][0], true_edge_len, pt_len, visual_len)
    #         continue



if __name__ == "__main__":
    # remove_directory_content("/home/zhuxunyang/coding/banding_detect/temporary_banding")
    # root_dir = '/home/zhuxunyang/coding/banding_detect/datasets/model_single'
    
    # # 收集所有需要处理的文件
    # files_to_process = []
    # for root, dirs, files in os.walk(root_dir):
    #     for file in files:
    #         if file.endswith('_bkgm.vtk'):
    #             files_to_process.append((root, file))
    # # 使用多进程处理不同文件
    # with Pool(cpu_count()) as pool:
    #     pool.starmap(process_file, files_to_process)


    # files_to_process = []
    # for root, dirs, files in os.walk(root_dir):
    #     for file in files:
    #         if file.endswith('_bkgm.vtk'):
    #             # files_to_process.append((root, file))
    #             copy_file(root, file, "/home/zhuxunyang/coding/banding_detect/datasets/model0609")

    check_data1("/home/zhuxunyang/coding/banding_detect/datasets/model_expanded")

    # expand_data1("/home/zhuxunyang/coding/banding_detect/datasets/model_single0625")

    # visual_generate("/home/zhuxunyang/coding/banding_detect/datasets/model0609/data/213_-1_bkgm.vtk", "/home/zhuxunyang/coding/banding_detect/datasets/model0609/label/213_-1.pt", "213_-1")