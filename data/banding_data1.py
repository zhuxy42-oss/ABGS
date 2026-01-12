import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad, pad_tensor, smooth_map
import numpy as np
from models.layers.mesh import Mesh
from models.layers.CustomMesh import CustomMesh
import meshio
import re
from numpy.core.multiarray import _reconstruct
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import pickle

class BandingData(DataLoader):

    def __init__(self, opt):
        # BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root = "/home/zhuxunyang/coding/banding_detect/datasets/banding_dataset/train"
        self.val = "/home/zhuxunyang/coding/banding_detect/datasets/banding_dataset/val"
        self.paths, self.target = self.make_dataset(self.root)
        self.val_path, self.val_target = self.make_dataset(self.val)
        train_len = len(self.paths)
        val_len = len(self.val_path)
        self.size = len(self.paths)
        self.train_data = []
        self.val_data = []
        self.train_label = []
        self.val_label = []
        for i in range(train_len):
            mesh = Mesh(file=self.paths[i], opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
            target = (torch.load(self.target[i], weights_only=True) - 1).unsqueeze(0).requires_grad_(True).to(self.device).float() 


        opt.input_nc = 9


    # def __getitem__(self, index):
    #     path = self.paths[index]
    #     target_path = self.target[index]
    #     mesh = Mesh(file=path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
    #     meta = {}
    #     meta['mesh'] = mesh
    #     target = (torch.load(target_path, weights_only=True) - 1).unsqueeze(0).requires_grad_(True).to(self.device).float() 
    #     meta['regression_target'] = target
    #     meta['classify_target'] = (target > 0.4).float()
    #     edge_features = mesh.extract_features()
    #     data = Data(x=torch.from_numpy(mesh.size_value), edge_index=torch.from_numpy(mesh.edges), edge_attr=torch.from_numpy(edge_features), pos=torch.from_numpy(mesh.vs))
    #     meta['data'] = data

    #     val_path = self.val_path[index]
    #     val_target_path = self.val_target[index]
    #     val_mesh = Mesh(file=val_path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
    #     meta['val_mesh'] = val_mesh
    #     val_target = (torch.load(val_target_path, weights_only=True).unsqueeze(0) - 1).requires_grad_(True).to(self.device).float() 
    #     meta['val_regression_target'] = val_target
    #     meta['val_classify_target'] = (val_target > 0.4).float()
    #     val_edge_features = val_mesh.extract_features()
    #     val_data = Data(x=torch.from_numpy(val_mesh.size_value), edge_index=torch.from_numpy(val_mesh.edges), edge_attr=torch.from_numpy(val_edge_features), pos=torch.from_numpy(val_mesh.vs))
    #     meta['val_data'] =val_data
    #     return meta
    
    def get(self, idx):

        
        # 加载图数据和标签
        try:
            mesh = Mesh(file=self.paths[idx], opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
            edge_features = mesh.extract_features()
            graph_data = Data(x=torch.from_numpy(mesh.size_value), edge_index=torch.from_numpy(mesh.edges), edge_attr=torch.from_numpy(edge_features), pos=torch.from_numpy(mesh.vs))
            label = torch.load(self.target[idx])

            val_mesh = Mesh(file=self.val_path[idx], opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
            val_edge_features = val_mesh.extract_features()
            val_graph_data = Data(x=torch.from_numpy(val_mesh.size_value), edge_index=torch.from_numpy(val_mesh.edges), edge_attr=torch.from_numpy(val_edge_features), pos=torch.from_numpy(val_mesh.vs))
            val_label = torch.load(self.val_target[idx])
            
            # 将标签添加到图数据中
            graph_data.y = label
            val_graph_data.y = val_label
            
            return (graph_data, val_graph_data)

            
        except Exception as e:
            print(f"加载数据时出错 (索引 {idx}): {e}")
            return None

    def __len__(self):
        return self.size

    @staticmethod
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
            # 使用正则表达式匹配文件名开头的两个数字部分（格式如 "123_456_xxx" 或 "123_456.yyy"）
            filename = filename.split('/')[-1]
            # match = re.search(r'^(\d+)_(\d+)', filename)
            match = re.search(r'^([+-]?\d+)_([+-]?\d+)', filename)
            if match:
                # 提取开头的两个数字并转为元组 (int1, int2) 作为排序依据
                num1 = int(match.group(1))
                num2 = int(match.group(2))
                return (num1, num2)
            return (float('inf'), float('inf'))

        
        # print(len(meshes), len(target))

        meshes = sorted(meshes, key=get_sort_key)
        target = sorted(target, key=get_sort_key)

         
        return meshes, target

def load_or_compute_stats(all_node_features, all_edge_features, save_dir='norm_params'):
    os.makedirs(save_dir, exist_ok=True)
    node_file = os.path.join(save_dir, '/home/zhuxunyang/coding/banding_detect/para/node_stats.pkl')
    edge_file = os.path.join(save_dir, '/home/zhuxunyang/coding/banding_detect/para/edge_stats.pkl')

    # 尝试加载已有参数
    if os.path.exists(node_file) and os.path.exists(edge_file):
        with open(node_file, 'rb') as f:
            node_mean, node_std = pickle.load(f)
        with open(edge_file, 'rb') as f:
            edge_mean, edge_std = pickle.load(f)
        return node_mean, node_std, edge_mean, edge_std

    node_mean = np.mean(all_node_features, axis=0)
    node_std = np.std(all_node_features, axis=0) + 1e-8
    edge_mean = np.mean(all_edge_features, axis=1)
    edge_std = np.std(all_edge_features, axis=1) + 1e-8

    with open(node_file, 'wb') as f:
        pickle.dump((node_mean, node_std), f)
    with open(edge_file, 'wb') as f:
        pickle.dump((edge_mean, edge_std), f)
    return node_mean, node_std, edge_mean, edge_std

def gather_graph(path, opt, device, batch_size):
    vtk_path, pt_path = make_dataset(path)
    train_len = len(vtk_path)
    graph_list = []
    for i in range(train_len):
        mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
        target = smooth_map(torch.load(pt_path[i], weights_only=True), 1.2).unsqueeze(0).requires_grad_(True).to(device).float() 
        edge_features = mesh.extract_features()
        data = Data(x=torch.from_numpy(mesh.size_value).float(), edge_index=torch.from_numpy(mesh.edges).transpose(1, 0).long(), edge_attr=torch.from_numpy(edge_features).transpose(1, 0).float(), pos=torch.from_numpy(mesh.vs)).to(device)
        data.y = target.transpose(1, 0)
        data.y1 = (target.transpose(1, 0) > 0.5).float()
        graph_list.append(data)
    loader = DataLoader(graph_list, batch_size=batch_size)
    return loader

def gather_graph_normal(path, opt, device, batch_size, norm_params=None):
    vtk_path, pt_path = make_dataset(path)
    # for i in range(len(pt_path)):
    #     print(vtk_path[i], pt_path[i])
    train_len = len(vtk_path)
    graph_list = []
    
    # 第一次遍历：仅当未提供归一化参数时收集特征
    if norm_params is None:
        all_node_features, all_edge_features = [], []
        for i in range(train_len):
            mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
            all_node_features.append(mesh.size_value)
            all_edge_features.append(mesh.extract_features())
        

        # 计算并保存全局归一化参数（或逐样本）
        node_mean, node_std, edge_mean, edge_std = load_or_compute_stats(np.concatenate(all_node_features, axis=0), np.concatenate(all_edge_features, axis=1))
    else:
        node_mean, node_std, edge_mean, edge_std = norm_params  # 使用传入参数

    # 第二次遍历：构建归一化的图数据
    for i in range(train_len):
        mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
        label = torch.load(pt_path[i])
        target = smooth_map(label, 1.2).unsqueeze(0).to(device).float()
        
        edge_len = len(mesh.edges)
        target_edge_len = label.shape[0]
        if edge_len != target_edge_len:
            print(edge_len, target_edge_len)
            print(f"{vtk_path[i]} and {pt_path[i]} error")
            continue

        edge_features = mesh.extract_features()

        GEO = vtk_path[i].split('/')[-1].split('_')[0]
            
        # 归一化（假设使用全局参数）
        norm_node = (mesh.size_value - node_mean) / node_std
        norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
        
        
    
        data = Data(
            x=torch.from_numpy(norm_node).float(),
            edge_index=torch.from_numpy(mesh.edges).T.long(),
            edge_attr=torch.from_numpy(norm_edge).T.float(),
            pos=torch.from_numpy(mesh.vs),
            y=target.T,
            y1=(target.T > 0.5).float(),
            # mesh=CustomMesh.from_vtk(vtk_path[i]),
            geo = GEO
        ).to(device)
        graph_list.append(data)
    
    return DataLoader(graph_list, batch_size=batch_size, shuffle=True), (node_mean, node_std, edge_mean, edge_std)


def gather_graph_normal1(path, opt, device, batch_size, norm_params=None):
    vtk_path, pt_path = make_dataset(path)
    for i in range(len(pt_path)):
        print(vtk_path[i], pt_path[i])
    train_len = len(vtk_path)
    graph_list = []

    all_node_features, all_edge_features = [], []

    # 第一次遍历：仅当未提供归一化参数时收集特征
    if norm_params is None:
        for i in range(train_len):
            mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
            print(vtk_path[i], "loaded")
            all_node_features.append(mesh.size_value)
            all_edge_features.append(mesh.extract_features())
        

        # 计算并保存全局归一化参数（或逐样本）
        node_mean, node_std, edge_mean, edge_std = load_or_compute_stats(np.concatenate(all_node_features, axis=0), np.concatenate(all_edge_features, axis=1))
    else:
        node_mean, node_std, edge_mean, edge_std = norm_params  # 使用传入参数

    # 第二次遍历：构建归一化的图数据
    for i in range(train_len):
        if len(all_node_features) != 0 and len(all_edge_features) != 0:
            target = smooth_map(torch.load(pt_path[i]), 1.2).unsqueeze(0).to(device).float()
            edge_features = all_edge_features[i]
            GEO = vtk_path[i].split('/')[-1].split('_')[0]
            norm_node = (all_node_features[i] - node_mean) / node_std
            norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
        else:
            mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
            target = smooth_map(torch.load(pt_path[i]), 1.2).unsqueeze(0).to(device).float()
            edge_features = mesh.extract_features()
            GEO = vtk_path[i].split('/')[-1].split('_')[0]
            
            # 归一化（假设使用全局参数）
            norm_node = (mesh.size_value - node_mean) / node_std
            norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)

        

        
        data = Data(
            x=torch.from_numpy(norm_node).float(),
            edge_index=torch.from_numpy(mesh.edges).T.long(),
            edge_attr=torch.from_numpy(norm_edge).T.float(),
            pos=torch.from_numpy(mesh.vs),
            y=target.T,
            y1=(target.T > 0.5).float(),
            # mesh=CustomMesh.from_vtk(vtk_path[i]),
            geo = GEO
        ).to(device)
        graph_list.append(data)
    
    return DataLoader(graph_list, batch_size=batch_size, shuffle=True), (node_mean, node_std, edge_mean, edge_std)


def gather_graph_normal2(path, opt, device, batch_size, norm_params=None):
    vtk_path, pt_path = make_dataset(path)
    train_len = len(vtk_path)
    graph_list = []
    
    # 第一次遍历：仅当未提供归一化参数时收集特征
    if norm_params is None:
        all_node_features, all_edge_features = [], []
        for i in range(train_len):
            mesh = CustomMesh.from_vtk(vtk_path[i])
            all_node_features.append(mesh.sizing_values)
            all_edge_features.append(mesh.compute_edge_features())

        # 计算并保存全局归一化参数（或逐样本）
        node_mean, node_std, edge_mean, edge_std = load_or_compute_stats(np.concatenate(all_node_features, axis=0), np.concatenate(all_edge_features, axis=1))
    else:
        node_mean, node_std, edge_mean, edge_std = norm_params  # 使用传入参数

    # 第二次遍历：构建归一化的图数据
    for i in range(train_len):
        # mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
        mesh = CustomMesh.from_vtk(vtk_path[i])
        label = torch.load(pt_path[i])
        target = smooth_map(label, 1.02).unsqueeze(0).to(device).float()
        
        edge_len = len(mesh.edges)
        target_edge_len = label.shape[0]
        if edge_len != target_edge_len:
            print(edge_len, target_edge_len)
            print(f"{vtk_path[i]} and {pt_path[i]} error")
            continue

        edge_features = mesh.compute_edge_features()

        GEO = vtk_path[i].split('/')[-1].split('_')[0]
            
        # 归一化（假设使用全局参数）
        norm_node = (mesh.sizing_values - node_mean) / node_std
        norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
    
        data = Data(
            x=norm_node.float(),
            edge_index=mesh.edges.T.long(),
            edge_attr=norm_edge.T.float(),
            pos=mesh.vertices,
            y=target.T,
            y1=(target.T > 0.5).float(),
            # mesh=CustomMesh.from_vtk(vtk_path[i]),
            geo = GEO
        ).to(device)
        graph_list.append(data)
    
    return DataLoader(graph_list, batch_size=batch_size, shuffle=True), (node_mean, node_std, edge_mean, edge_std)
