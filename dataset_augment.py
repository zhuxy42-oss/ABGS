import os
import numpy as np
from models.layers import CustomMesh
import torch




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


if __name__ == "__main__":
    remove_directory_content("/home/zhuxunyang/coding/bkgm_simplification/temporary_banding")
    root_dir = '/home/zhuxunyang/coding/banding_detect/datasets/model0501'
    
    # 收集所有需要处理的文件
    files_to_process = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('_bkgm.vtk'):
                # files_to_process.append((root, file))
                file_name = file.split('_bkgm.vtk')[0]
                pt_file = os.path.join(root, f"{file_name}.pt")
                vtk_path = os.path.join(root, file)
                mesh = CustomMesh.CustomMesh.fromvtk(vtk_path)
                label = torch.load(pt_file)