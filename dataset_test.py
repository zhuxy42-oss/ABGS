import h5py
import os
import numpy as np
from models.layers import CustomMesh
import torch
import re
from concurrent.futures import ThreadPoolExecutor
import pyvista as pv
import meshio
import trimesh
import subprocess
import shutil

from multiprocessing import Pool, cpu_count
import subprocess
from functools import partial

def detect_non_manifold_edge(root, file_name):
    vtk_file = root + '/' + file_name
    mesh = CustomMesh.CustomMesh.from_vtk(vtk_file)

    # print(file_name, " detecting")
    
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
        return True
    else:
        print(file_name, "Non manifold mesh")
        return False

def process_vtk_files(root_dir):
    """
    处理VTK文件的主函数
    
    参数:
        root_dir: 要搜索的根目录路径
    """
    # 限制只搜索两级子目录
    max_depth = 2
    
    # 遍历目录树
    for root, dirs, files in os.walk(root_dir):
        # 计算当前深度
        current_depth = root[len(root_dir):].count(os.sep)
        if current_depth > max_depth:
            continue
        
        for file in files:
            if file.endswith('_stl.vtk'):
                # 获取完整文件路径
                stl_file_path = os.path.join(root, file)
                bkgm_file_name = file.replace("_stl.vtk", "_bkgm.vtk")
                geo_file_name = file.replace("_stl.vtk", ".step")
                # 生成对应的 _bkgm.vtk 文件的完整路径
                bkgm_file_path = os.path.join(root, bkgm_file_name)
                geo_file_path = os.path.join(root, geo_file_name)

                command = ['./test_sizefield', stl_file_path, bkgm_file_path, '1.2', '0', '0.1', geo_file_path]
                # print(command)
                try:
                    result = subprocess.run(command, check=True, text=True, capture_output=True)
                    print("size field generate success", bkgm_file_name)
                    # print(result.stdout)
                except subprocess.CalledProcessError as e:
                    print("size field generate failed", bkgm_file_name)
                    print(e.stderr)
                    

def label_bending(root, file_name):
    red = [1, 0, 0]
    blue = [0, 0, 1]
    vtk_file = root + '/' + file_name + '_bkgm.vtk'
    geo_file = root + '/' + file_name + '.step'
    data_mesh = CustomMesh.CustomMesh.from_vtk(vtk_file)
    past_num_mesh = data_mesh.area_num
    data_mesh.get_all_info()
    ratios = []
    for j in range(len(data_mesh.edges)):
        save_path1 = "/home/zhuxunyang/coding/bkgm_simplification/split.vtk"
        save_path2 = "/home/zhuxunyang/coding/bkgm_simplification/new_sf.vtk"
        v1, v2 = data_mesh.edges[j]
        face_index = []
        for i, face in enumerate(data_mesh.faces):
            if v1 in face and v2 in face:
                face_index.append(i)

        new_mesh = data_mesh.split_edge(j)
        if new_mesh.is_empty:
            print(file_name, "error")
            continue
        new_mesh.writeVTK(save_path1)
 
        command = ['./test_sizefield2', save_path1, save_path2, '1.2', '0', '0.1', geo_file]
        # print(command)
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            # print("size field generate success")
        except subprocess.CalledProcessError as e:
            print(f"{file_name} size field generate failed because edge {j}")
            print(e.stderr)

        splited_mesh = CustomMesh.CustomMesh.from_vtk(save_path2)
        # splited_mesh.get_all_info()
        splited_mesh.writeVTK(save_path2)
        post_num_mesh = splited_mesh.area_num
        
        post_num = post_num_mesh[-1] + post_num_mesh[-2] + post_num_mesh[-3] + post_num_mesh[-4]
        past_num = past_num_mesh[face_index[0]] + past_num_mesh[face_index[1]]
        ratio = 0
        if past_num / post_num < 1:
            ratio = 1
            ratios.append(ratio)
        else:
            ratio = past_num / post_num
            ratios.append(ratio)

        print("File name", file_name, "Edge", j, "Points:", data_mesh.edges[j][0].item(), data_mesh.edges[j][1].item(), "Post num:", post_num, "Past num:", past_num, "Ratio:", ratio)
        os.remove(save_path1)
        os.remove(save_path2)

    try:
        # 1. 处理张量并保存 .pt 文件
        banding_q = torch.tensor(np.stack(ratios))
        torch.save(banding_q, root + "/" + file_name + ".pt")
        
        # 2. 生成颜色列表（处理可能的索引错误或数据类型错误）
        colors = []
        for i, r in enumerate(ratios):
            if r > 1.20:
                colors.append(red)
            else:
                colors.append(blue)
        
        # 3. 生成网格数据并保存 .vtk 文件（处理 meshio 相关异常）
        cells = [("line", data_mesh.edges.detach().numpy())]
        cell_data = {"Color": [colors]}
        meshio_mesh = meshio.Mesh(
            points=data_mesh.vertices.detach().numpy(),
            cells=cells,
            cell_data=cell_data
        )
        meshio_mesh.write(root + "/colored_edge_banding_" + file_name + ".vtk")
        
        print(f"成功处理并保存文件：{file_name}")
    
    except torch.TensorError as e:
        print(f"张量处理错误: {e}")
    except RuntimeError as e:
        print(f"PyTorch 运行时错误（如文件写入失败）: {e}")
    except IndexError as e:
        print(f"索引错误（可能是 ratios 数据格式不匹配）: {e}")
    except meshio.MeshioException as e:
        print(f"meshio 网格操作错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")

if __name__ == "__main__":
    # directory = '/home/zhuxunyang/coding/bkgm_simplification/datasets/model/25'  # 替换为实际的目录路径
    # file = '25'
    # label_bending(directory, file)

    # root = '/home/zhuxunyang/coding/banding_detect/datasets/model0501/184'    #88 92
    # file = '184_bkgm.vtk'
    # detect_non_manifold_edge(root, file)

    root = '/home/zhuxunyang/coding/banding_detect/datasets/model0501/184'
    file = '184_bkgm.vtk'
    geo = '184.step'
    output_path = '/home/zhuxunyang/coding/bkgm_simplification/temporary_banding/middle_sf.vtk'
    split_path = '/home/zhuxunyang/coding/bkgm_simplification/temporary_banding/sf.vtk'
    vtk_file = root + '/' + file
    geo_file = root + '/' + geo
    mesh = CustomMesh.CustomMesh.from_vtk(vtk_file)
    
    potential_banding = []

    for j in range(len(mesh.edges)):
        new_mesh = mesh.split_edge(j)
        new_mesh.writeVTK(output_path)
        command = ['./test_sizefield2', output_path, split_path, '1000000', '0', '0.1', geo_file]
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"{file} size field generate failed because edge {j}")
            print(e.stderr)

        splited_mesh = CustomMesh.CustomMesh.from_vtk(split_path)
        new_index = len(splited_mesh.vertices) - 1
        
        for face in splited_mesh.faces:
            if new_index in face:
                non_new_index_indices = [index for index in face if index != new_index]
                for index in non_new_index_indices:
                    rito = splited_mesh.sizing_values[index] - splited_mesh.sizing_values[new_index] / splited_mesh.sizing_values[new_index]
                    if rito > 1.2:
                        potential_banding.append(j)
                        break
                    break
                
    print(potential_banding)






 
    # max_edge = 0
    # root_dir = '/home/zhuxunyang/coding/banding_detect/datasets/model0501/'         #model0501
    # num = 0
    # max_file = -1
    # for root, dirs, files in os.walk(root_dir):
    #     for file in files:
    #         if file.endswith('_bkgm.vtk'):
    #             file_name = file.split('_bkgm.vtk')[0]  # 去掉后缀

    #             # 构造待检查的文件路径
    #             pt_file = os.path.join(root, f"{file_name}.pt")
    #             colored_file = os.path.join(root, f"colored_edge_banding_{file_name}.vtk")
                
    #             # detect_non_manifold_edge(root, file)

    #             if os.path.exists(pt_file) & os.path.exists(colored_file):
    #                 label = torch.load(pt_file)
    #                 banding_num = 0
    #                 for Bool in label:
    #                     if Bool > 1.2:
    #                         banding_num += 1
    #                 print(f"{file}: {banding_num / len(label)}")
                            
    #                 # print(f"{file}: {len(label)}")
    #                 # if(len(label) > max_edge):
    #                 #     max_edge = len(label)
    #                 #     max_file = file_name
    #                 # print(f"{file}: has been labeled")
    #                 break
    #             else:
    #                 print(f"{file}: hasn't been labeled")
    #                 num = num + 1
    #                 # detect_non_manifold_edge(root, file)

    #                 break
  
    # #             # if os.path.exists(pt_file) & os.path.exists(colored_file):
    # #             #     # detect_non_manifold_edge(root, file)
    # #             #     print(f"{file}: has been labeled")
    # #             #     break

    # #             try:
    # #                 # 核心处理逻辑：调用 label_bending 函数
    # #                 label_bending(root, file_name)
    # #                 print(f"成功处理文件：{os.path.join(root, file)}")
                    
    # #             except Exception as e:
    # #                 # 捕获并处理可能出现的异常（可根据需求细分异常类型）
    # #                 print(f"处理文件 {os.path.join(root, file)} 时发生错误：")
    # #                 print(f"错误类型：{type(e).__name__}")
    # #                 print(f"错误信息：{str(e)}")
    # #                 # 可添加额外处理逻辑（如跳过文件、记录日志等）
    # #                 continue  # 跳过当前出错文件，继续处理下一个


    # print(num)
    # print(max_file, max_edge)




