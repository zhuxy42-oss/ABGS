import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
import numpy as np
from models.layers.mesh import Mesh
from models.layers.CustomMesh import CustomMesh
import meshio
import re
from numpy.core.multiarray import _reconstruct
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle
import meshio

class SimplificationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root = opt.dataroot
        self.val = opt.valroot
        self.paths, self.target, self.visual = self.make_dataset(self.root)
        self.val_path, self.val_target, self.val_visual = self.make_dataset(self.val)
        self.size = len(self.paths)
        # self.get_mean_std()
        # self.get_val_mean_std()
        opt.input_nc = 5


    def __getitem__(self, index):
        path = self.paths[index]
        # target_path = self.target[index]
        visual_path = self.visual[index]
        mesh = Mesh(file=path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
        meta = {}
        meta['mesh'] = mesh
        original_mesh = CustomMesh.from_vtk(path)
        original_mesh.edges = mesh.edges
        meta['original_mesh'] = original_mesh
        meta['mean'] = self.mean
        meta['std'] = self.std
        # meta['target'] = torch.load(target_path, weights_only=True).unsqueeze(0).to(self.device)
        edge_target = meshio.read(visual_path)
        label = edge_target.cell_data['size_loss'][0] + edge_target.cell_data['distance_loss'][0] * 3
        meta['target'] = torch.from_numpy((label - np.min(label)) / (np.max(label) - np.min(label))).unsqueeze(0).to(self.device)
        edge_features = mesh.extract_features()
        meta['edge_features'] = pad(edge_features, self.opt.ninput_edges)
        # meta['edge_features'] = (edge_features - self.mean) / self.std

        val_path = self.val_path[index]
        # val_target_path = self.val_target[index]
        val_visual_path = self.val_visual[index]
        val_mesh = Mesh(file=val_path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
        meta['val_mesh'] = val_mesh
        val_original_mesh = CustomMesh.from_vtk(val_path)
        val_original_mesh.edges = val_mesh.edges
        meta['val_original_mesh'] = val_original_mesh
        # meta['val_target'] = torch.load(val_target_path, weights_only=True).unsqueeze(0).to(self.device)
        val_edge_target = meshio.read(val_visual_path)
        val_label = val_edge_target.cell_data['size_loss'][0] + val_edge_target.cell_data['distance_loss'][0] * 3
        meta['val_target'] = torch.from_numpy((val_label - np.min(val_label)) / (np.max(val_label) - np.min(val_label))).unsqueeze(0).to(self.device)
        val_edge_features = val_mesh.extract_features()
        meta['val_edge_features'] = pad(val_edge_features, self.opt.ninput_edges)
        # meta['val_edge_features'] = (val_edge_features - self.mean) / self.std

        return meta

    def __len__(self):
        return self.size

    
    @staticmethod
    def make_dataset(path):
        meshes = []
        target = []
        visual = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if 'target' in fname:
                    continue
                if 'colored' in fname:
                    path = os.path.join(root, fname)
                    visual.append(path)
                elif is_mesh_file(fname) and 'colored' not in fname:
                    path = os.path.join(root, fname)
                    meshes.append(path)
                elif fname.endswith('.pt'):
                    path = os.path.join(root, fname)
                    target.append(path)
        
        # def get_sort_key(filename):
        #     parts = filename.split("_")
        #     num1 = int(parts[-2])          # 提取倒数第二个部分作为 num1
        #     num2 = int(parts[-1].split(".")[0])  # 提取最后部分并移除 .vtk 作为 num2
        #     return (num1, num2)

        def get_sort_key(filename):
            # 使用正则表达式匹配文件名末尾的 _num1_num2.扩展名 模式
            match = re.search(r'(\d+)_(\d+)\.\w+$', filename)
            if match:
                # num1 = int(match.group(1))
                num2 = int(match.group(2))
                return (num2)
        
        print(len(meshes), len(target))

        meshes = sorted(meshes, key=get_sort_key)
        target = sorted(target, key=get_sort_key)
        visual = sorted(visual, key=get_sort_key)

         
        return meshes, target, visual

def make_dataset(path):
    meshes = []
    targets = []
    vtk_targets = []
    assert os.path.isdir(path), '%s is not a valid directory' % path

    # # 只遍历path的直接子目录
    # for entry in sorted(os.scandir(path), key=lambda x: x.name):
    #     if not entry.is_dir():
    #         continue  # 跳过非目录文件
            
        # # 处理子目录中的文件
        # for fname in sorted(os.listdir(entry.path)):
        #     full_path = os.path.join(entry.path, fname)
        #     match = re.search(r'(?:\w+_)*(\d+)_(\d+)\.\w+$', fname)
        #     if match:
        #         if '.vtk' and 'color' in fname:
        #             vtk_targets.append(full_path)
        #         elif fname.endswith('.pt'):
        #             targets.append(full_path)
        #         else:
        #             meshes.append(full_path)

    for root, _, fnames in sorted(os.walk(path)):
        for fname in fnames:
            if '.vtk' and 'colored' in fname:
                path = os.path.join(root, fname)
                vtk_targets.append(path)
            elif fname.endswith('.vtk') and 'colored' not in fname and 'target' not in fname:
                path = os.path.join(root, fname)
                meshes.append(path)

    def get_sort_key(filename):
        # 使用正则表达式匹配文件名开头的两个数字部分（格式如 "123_456_xxx" 或 "123_456.yyy"）
        filename = filename.split('/')[-1]
        match = re.search(r'(\d+)_(\d+)(?=[_.][^_.]*$|$)', filename)
        if match:
            # 提取开头的两个数字并转为元组 (int1, int2) 作为排序依据
            num1 = int(match.group(1))
            num2 = int(match.group(2))
            return (num1, num2)
        return (float('inf'), float('inf'))

    # # 定义排序函数
    # def get_sort_key(filename):
    #     # 从完整路径中提取文件名
    #     basename = os.path.basename(filename)
    #     # match = re.search(r'(\d+)_(\d+)\.\w+$', basename)
    #     match = re.search(r'(?:\w+_)*(\d+)_(\d+)\.\w+$', basename)
    #     if match:
    #         return (match.group(0), match.group(1))  # 按第二个数字排序
    #     return 0  # 默认值

    # 排序
    meshes = sorted(meshes, key=get_sort_key)
    vtk_targets = sorted(vtk_targets, key=get_sort_key)

    print(f"Found {len(meshes)} mesh files and {len(vtk_targets)} vtk target files")
    return meshes, vtk_targets


def load_or_compute_stats(all_node_features, all_edge_features, save_dir='norm_params'):
    os.makedirs(save_dir, exist_ok=True)
    node_file = os.path.join(save_dir, '/home/zhuxunyang/coding/simply/para/node_stats.pkl')
    edge_file = os.path.join(save_dir, '/home/zhuxunyang/coding/simply/para/edge_stats.pkl')

    # 尝试加载已有参数
    if os.path.exists(node_file) and os.path.exists(edge_file):
        with open(node_file, 'rb') as f:
            node_mean, node_std = pickle.load(f)
            print("Node para load success")
        with open(edge_file, 'rb') as f:
            edge_mean, edge_std = pickle.load(f)
            print("Edge para load success")
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
        target = torch.load(pt_path[i], weights_only=True).unsqueeze(0).requires_grad_(True).to(device).float() 
        edge_features = mesh.extract_features()
        data = Data(x=torch.from_numpy(mesh.size_value).float(), edge_index=torch.from_numpy(mesh.edges).transpose(1, 0).long(), edge_attr=torch.from_numpy(edge_features).transpose(1, 0).float(), pos=torch.from_numpy(mesh.vs)).to(device)
        data.y = target.transpose(1, 0)
        data.y1 = (target.transpose(1, 0) > 0.5).float()
        graph_list.append(data)
    loader = DataLoader(graph_list, batch_size=batch_size)
    return loader

def gather_graph_normal(path, opt, device, batch_size, norm_params=None):
    vtk_path, vtk_target_path = make_dataset(path)
    train_len = len(vtk_path)
    graph_list = []
    
    # 第一次遍历：仅当未提供归一化参数时收集特征
    if norm_params is None:
        all_node_features, all_edge_features = [], []
        for i in range(train_len):
            try:
                mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
                edge_target = meshio.read(vtk_target_path[i])
                edge_len = mesh.edges_count
                target_edge_len = edge_target.cells_dict['line'].shape[0]
                if edge_len != target_edge_len:
                    print(edge_len, target_edge_len)
                    print(f"{vtk_path[i]} and {vtk_target_path[i]} error")
                    # os.remove(vtk_path[i])
                    # os.remove(vtk_target_path[i])
                    continue
                else:
                    # print(f"{vtk_path[i]} and {vtk_target_path[i]} right")
                    all_node_features.append(mesh.size_value)
                    all_edge_features.append(mesh.extract_features())
            except:
                continue
        

        # 计算并保存全局归一化参数（或逐样本）
        node_mean, node_std, edge_mean, edge_std = load_or_compute_stats(np.concatenate(all_node_features, axis=0), np.concatenate(all_edge_features, axis=1))
    else:
        node_mean, node_std, edge_mean, edge_std = norm_params  # 使用传入参数

    # 第二次遍历：构建归一化的图数据
    for i in range(train_len):
        try:
            mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
            edge_target = meshio.read(vtk_target_path[i])
            if mesh.edges_count != edge_target.cell_data['label0-1'][0].size:
                print(f"warning different edge count between {vtk_path[i]} and {vtk_target_path[i]}")
                continue
            
            edge_features = mesh.extract_features()
            GEO = vtk_path[i].split('/')[-1].split('_')[0]

            edge_len = len(mesh.edges)
            
            # 归一化（假设使用全局参数）
            norm_node = (mesh.size_value - node_mean) / node_std
            norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
            
            sizeloss = edge_target.cell_data['size_loss'][0]
            distance_loss = edge_target.cell_data['distance_loss'][0]
            label = np.abs(sizeloss) * 0 + distance_loss * 1
            label = (label - np.min(label)) / (np.max(label) - np.min(label))
            # if label01.dtype.byteorder != '=':  # '='表示系统字节序
            #     label01 = label01.byteswap().newbyteorder('=')
            target_flat = torch.tensor(label, dtype=torch.float32)

            # 将小于0.1的值的索引设为0，其他设为1
            one_hot = torch.zeros_like(target_flat)
            one_hot[target_flat < 0.05] = 1

            data = Data(
                x=torch.from_numpy(norm_node).float(),
                edge_index=torch.from_numpy(mesh.edges).T.long(),
                edge_attr=torch.from_numpy(norm_edge).T.float(),
                pos=torch.from_numpy(mesh.vs),
                y=target_flat.unsqueeze(1),
                y1=one_hot.unsqueeze(1),
                length = edge_len,
                mesh = CustomMesh.from_vtk(vtk_path[i]),
                geo = GEO
            ).to(device)
            graph_list.append(data)
        except:
            continue
    
    return DataLoader(graph_list, batch_size=batch_size, shuffle=True), (node_mean, node_std, edge_mean, edge_std)


def gather_graph_normal1(path, opt, device, batch_size, norm_params=None):
    vtk_path, vtk_target_path = make_dataset(path)
    train_len = len(vtk_path)
    graph_list = []
    mesh_list = []
    target_mesh_list = []
    # node_mean, node_std, edge_mean, edge_std = None
    
    # 第一次遍历：仅当未提供归一化参数时收集特征
    if norm_params is None:
        all_node_features, all_edge_features = [], []
        for i in range(train_len):
            try:
                # mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
                mesh = CustomMesh.from_vtk(vtk_path[i])
                edge_points = mesh._get_edge_points()
                edge_vectors = mesh.vertices[edge_points[:, 0]] - mesh.vertices[edge_points[:, 1]]
                edges_lengths_raw = np.linalg.norm(edge_vectors, ord=2, axis=1)
                zero_edge_mask = edges_lengths_raw < 1e-8
                zero_edge_count = np.sum(zero_edge_mask)
                if zero_edge_count > 0:
                    continue
                lbo_vertex, lbo_edge = mesh.compute_LBO1()
                mesh_list.append(mesh)
                edge_target = meshio.read(vtk_target_path[i])
                target_mesh_list.append(edge_target)
                edge_len = len(mesh.edges)
                target_edge_len = edge_target.cells_dict['line'].shape[0]
                if edge_len != target_edge_len:
                    print(edge_len, target_edge_len)
                    print(f"{vtk_path[i]} and {vtk_target_path[i]} error")
                    # os.remove(vtk_path[i])
                    # os.remove(vtk_target_path[i])
                    continue
                else:
                    print(f"{vtk_path[i]} and {vtk_target_path[i]} right")
                    all_node_features.append(torch.cat((mesh.sizing_values, torch.from_numpy(lbo_vertex).unsqueeze(1)), dim=1))
                    all_edge_features.append(torch.cat((mesh.compute_edge_features(), torch.from_numpy(lbo_edge).unsqueeze(0)), dim=0))
                    print(" ")
            except:
                continue
        

        # 计算并保存全局归一化参数（或逐样本）
        node_feature = np.concatenate(all_node_features, axis=0)
        edge_feature = np.concatenate(all_edge_features, axis=1)
        node_mean, node_std, edge_mean, edge_std = load_or_compute_stats(node_feature, edge_feature)
    else:
        node_mean, node_std, edge_mean, edge_std = norm_params  # 使用传入参数
        print(" ")

    mesh_list.clear()

    # 第二次遍历：构建归一化的图数据
    for i in range(train_len):
        # if mesh_list is None:
        #     # mesh = Mesh(file=vtk_path[i], opt=opt, hold_history=True, export_folder=opt.export_folder)
        #     mesh = CustomMesh.from_vtk(vtk_path[i])
        #     edge_target = CustomMesh.from_vtk(vtk_target_path[i])
        # else:
        #     mesh = mesh_list[i]
        #     edge_target = target_mesh_list[i]
        
        mesh = CustomMesh.from_vtk(vtk_path[i])
        edge_target = meshio.read(vtk_target_path[i])
        lbo_vertex, lbo_edge = mesh.compute_LBO1()
        edge_points = mesh._get_edge_points()
        edge_vectors = mesh.vertices[edge_points[:, 0]] - mesh.vertices[edge_points[:, 1]]
        edges_lengths_raw = np.linalg.norm(edge_vectors, ord=2, axis=1)
        zero_edge_mask = edges_lengths_raw < 1e-8
        zero_edge_count = np.sum(zero_edge_mask)
        if zero_edge_count > 0:
            continue
        if len(mesh.edges) != edge_target.cell_data['label0-1'][0].size:
            print(f"warning different edge count between {vtk_path[i]} and {vtk_target_path[i]}")
            continue
            
        edge_features = mesh.compute_edge_features()
        GEO = vtk_path[i].split('/')[-1].split('_')[0]

        edge_len = len(mesh.edges)
            
        # 归一化（假设使用全局参数）
        norm_node = torch.from_numpy((np.concatenate([mesh.sizing_values, np.expand_dims(lbo_vertex, axis=1)], axis=1) - node_mean) / node_std)
        norm_edge = torch.from_numpy((np.concatenate([edge_features, np.expand_dims(lbo_edge, axis=0)], axis=0) - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1))
            
        sizeloss = edge_target.cell_data['size_loss'][0]
        distance_loss = edge_target.cell_data['distance_loss'][0]

        # sizeloss = np.where(sizeloss < 0, 0, sizeloss)
        # sizeloss = sizeloss - min(sizeloss)
        sizeloss = np.abs(sizeloss)

        label = sizeloss * 5 + distance_loss * 1
        label = (label - np.min(label)) / (np.max(label) - np.min(label))
        # if label01.dtype.byteorder != '=':  # '='表示系统字节序
        #     label01 = label01.byteswap().newbyteorder('=')
        target_flat = torch.tensor(label, dtype=torch.float32)
        threshold = torch.quantile(target_flat, 0.3)


        # 将小于0.1的值的索引设为0，其他设为1
        count = 0
        one_hot = torch.zeros_like(target_flat)
        # 向量化条件判断
        condition = ((target_flat <= threshold)).bool()
        one_hot[condition] = 1
        count = condition.sum().item()
        # print(i, " ", count, "/", len(one_hot))

        data = Data(
            x=norm_node.float(),
            edge_index=mesh.edges.T.long(),
            edge_attr=norm_edge.T.float(),
            pos=mesh.vertices,
            y=target_flat.unsqueeze(1),
            y1=one_hot.unsqueeze(1),
            length = edge_len,
            mesh = mesh,
            geo = GEO
        ).to(device)
        graph_list.append(data)
    
    return DataLoader(graph_list, batch_size=batch_size, shuffle=True), (node_mean, node_std, edge_mean, edge_std)