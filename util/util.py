from __future__ import print_function
import torch
import numpy as np
import os
import re


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

MESH_EXTENSIONS = [
    '.obj',
    '.vtk'
]

def has_nan_or_inf(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()

def is_mesh_file(filename):
    return any(filename.endswith(extension) for extension in MESH_EXTENSIONS)

def pad_tensor(x, target_size, mode='zeros', align='left'):
    """
    将 (1,n) 张量填充到 (1,m)
    
    参数：
        x: 输入张量 (1, n)
        target_size: 目标长度 m
        mode: 填充模式 ('zeros', 'edge', 'value')
        align: 对齐方式 ('left', 'center', 'right')
        value: 若 mode='value'，指定填充值
    """
    n = x.size(1)
    assert target_size >= n, "目标大小必须 ≥ 原始长度"
    
    if mode == 'zeros':
        pad_value = 1
    elif mode =='one':
        pad_value = 1
    elif mode == 'edge':
        pad_value = x[:, -1] if align == 'left' else x[:, 0]
    
    # 计算填充量
    if align == 'left':
        left_pad = 0
        right_pad = target_size - n
    elif align == 'right':
        left_pad = target_size - n
        right_pad = 0
    else:  # center
        left_pad = (target_size - n) // 2
        right_pad = target_size - n - left_pad
    
    # 执行填充
    padded = torch.zeros(1, target_size, dtype=x.dtype, device=x.device)
    padded[:, left_pad:left_pad+n] = x
    
    if mode != 'zeros':
        padded[:, :left_pad] = pad_value
        padded[:, left_pad+n:] = pad_value
    
    return padded

def pad(input_arr, target_length, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)

def smooth_map(x, mid=1.2):
    """
    平滑映射函数，支持 NumPy 数组和 PyTorch 张量输入：
    - x=1 时映射到 0
    - x=mid 时映射到 0.5
    - x > mid 时映射到 (0.5, 1]
    - 在 mid 处可微（左右导数相等）
    """
    k = mid / (mid - 1)
    c = (mid ** k) / 2

    # 检查输入类型
    if isinstance(x, (int, float)):
        # 标量输入
        if x <= mid:
            return 0.5 / (mid - 1) * (x - 1)
        else:
            return 1 - c / (x ** k)
    elif isinstance(x, np.ndarray):
        # NumPy 数组输入
        return np.where(x <= mid,
                        0.5 / (mid - 1) * (x - 1),
                        1 - c / (x ** k))
    elif isinstance(x, torch.Tensor):
        # PyTorch 张量输入
        return torch.where(x <= mid,
                           0.5 / (mid - 1) * (x - 1),
                           1 - c / (x ** k))
    else:
        raise TypeError("输入类型必须是标量、NumPy 数组或 PyTorch 张量")

def seg_accuracy(predicted, ssegs, meshes):
    correct = 0
    ssegs = ssegs.squeeze(-1)
    correct_mat = ssegs.gather(2, predicted.cpu().unsqueeze(dim=2))
    for mesh_id, mesh in enumerate(meshes):
        correct_vec = correct_mat[mesh_id, :mesh.edges_count, 0]
        edge_areas = torch.from_numpy(mesh.get_edge_areas())
        correct += (correct_vec.float() * edge_areas).sum()
    return correct

def print_network(net):
    """Print the total number of parameters in the network
    Parameters:
        network
    """
    print('---------- Network initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')

def get_heatmap_color(value, minimum=0, maximum=1):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


def normalize_np_array(np_array):
    min_value = np.min(np_array)
    max_value = np.max(np_array)
    return (np_array - min_value) / (max_value - min_value)


def calculate_entropy(np_array):
    entropy = 0
    np_array /= np.sum(np_array)
    for a in np_array:
        if a != 0:
            entropy -= a * np.log(a)
    entropy /= np.log(np_array.shape[0])
    return entropy

def make_dataset(path):
        meshes = []
        target = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if fname.endswith('_bkgm.vtk'):
                    path = os.path.join(root, fname)
                    meshes.append(path)
                elif fname.endswith('.pt'):
                    path = os.path.join(root, fname)
                    target.append(path)
                else:
                    continue


        def get_sort_key(filename):
            # 使用正则表达式匹配文件名开头的数字（格式如 "123_xxx" 或 "123_xxx.yyy"）
            match = re.search(r'^(\d+)', filename)
            if match:
                num = int(match.group(1))  # 提取开头的数字并转为整数
                return num
            return float('inf')

        
        print(len(meshes), len(target))

        meshes = sorted(meshes, key=get_sort_key)
        target = sorted(target, key=get_sort_key)

         
        return meshes, target
