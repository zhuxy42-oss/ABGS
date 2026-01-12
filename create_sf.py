import subprocess
from pathlib import Path
import os
from models.layers import CustomMesh
import re
import shutil

def detect_non_manifold_edge(root, file_name):
    vtk_file = root + '/' + file_name
    mesh = CustomMesh.CustomMesh.from_vtk(vtk_file)
    manifold = True
    non_manifold_edges = []
    for n, edge in enumerate(mesh.edges):
        v1, v2 = edge
        # 找到包含该边的两个三角形
        adjacent_faces = []
        adjacent_faces_id = []
        for i, face in enumerate(mesh.faces):
            if v1 in face and v2 in face:
                adjacent_faces.append(i)
        
        if len(adjacent_faces) != 2:
            manifold = False
            non_manifold_edges.append(tuple((n, (v1.item(), v2.item()), adjacent_faces)))
    
    if manifold:
        print(file_name, "Manifold mesh")
    else:
        print(file_name, "Non manifold mesh")
        for non_manifold_edge_info in non_manifold_edges:
            print(non_manifold_edge_info)

def create_sf(root, file_name):
    red = [1, 0, 0]
    blue = [0, 0, 1]
    stl_file = root + '/' + file_name + '_stl.vtk'
    bkgm_file = root + '/' + file_name + '_bkgm.vtk'
    geo_file = root + '/' + file_name + '.step'
    command = ['./test_sizefield', stl_file, bkgm_file, '1.2', '0', '0.1', geo_file]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"size field {file_name} generate success")
    except subprocess.CalledProcessError as e:
        print(f"size field {file_name} generate failed")
        print(e.stderr)

def replace_number_in_path(path_str, old_num, new_num):
    """替换路径字符串中的数字"""
    # 使用正则表达式替换数字，确保只替换完整的数字
    pattern = r'\b' + re.escape(str(old_num)) + r'\b'
    return re.sub(pattern, str(new_num), path_str)

def flatten_and_rename(source_dir, target_dir, original_num, new_num):
    """
    将所有数字文件夹及其内部文件复制到目标目录，并替换文件名中的数字
    
    Args:
        source_dir (str): 源目录（包含数字文件夹）
        target_dir (str): 目标目录（所有文件将直接放在这里）
        original_num (int/str): 要替换的原始数字（如 `123`）
        new_num (int/str): 替换后的新数字（如 `789`）
    """
    os.makedirs(target_dir, exist_ok=True)  # 确保目标目录存在
    
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        
        if not os.path.isdir(folder_path) or not folder_name.isdigit():
            continue  # 跳过非数字文件夹
        
        # 遍历文件夹内的所有文件
        for root, dirs, files in os.walk(folder_path):
            # 忽略名为 'cache' 的目录
            if 'cache' in dirs:
                dirs.remove('cache')
            for file_name in files:
                src_file = os.path.join(root, file_name)
                
                # 构造新文件名（替换数字）
                new_file_name = (
                    file_name
                    .replace(str(folder_name), str(int(folder_name) + 80))  # 文件名替换
                )
                
                if not os.path.isdir(os.path.join(target_dir, str(int(folder_name) + 80))):
                    os.makedirs(os.path.join(target_dir, str(int(folder_name) + 80)))
                    print(f"文件夹已创建: {os.path.join(target_dir, str(int(folder_name) + 80))}")

                dst_file = os.path.join(target_dir, str(int(folder_name) + 80), new_file_name)
                
                # 复制文件（避免覆盖已有文件）
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied: {src_file} → {dst_file}")
                else:
                    print(f"Skipped (already exists): {dst_file}")

if __name__ == "__main__":
    # root_dir = '/home/zhuxunyang/coding/banding_detect/datasets/new_bkgm'
    # for root, dirs, files in os.walk(root_dir):
    #     for file in files:
    #         if file.endswith('_bkgm.vtk'):
    #             file_name = file.split('_bkgm.vtk')[0]  # 去掉后缀
    #             # create_sf(root, file_name)
    #             detect_non_manifold_edge(root, file_name + "_bkgm.vtk")

    # detect_non_manifold_edge("/home/zhuxunyang/coding/banding_detect/datasets/model0609/data", "214_-1_bkgm.vtk")

    # pt_file = "/home/zhuxunyang/coding/banding_detect/datasets/model0609/label/212_-1.pt"
    # colored_file = "/home/zhuxunyang/coding/banding_detect/datasets/model0609/visual/colored_edge_banding_212_-1.vtk"
    # print(os.path.exists(pt_file))
    # print(os.path.exists(colored_file))

    flatten_and_rename("/home/zhuxunyang/coding/banding_detect/datasets/banding_dataset/val", "/home/zhuxunyang/coding/banding_detect/datasets/banding_dataset/val", 161, 201)