import meshio
import time
import numpy as np
from data.classification_data import ClassificationData
from options.base_options import BaseOptions
from options.test_options import TestOptions
from models.layers import CustomMesh
from torch_geometric.loader import DataLoader
from models import create_model
from util.writer import Writer
from util.util import make_dataset
from test import run_test
import torch
import torch.nn as nn
from models.networks import SimplificationLoss, EdgeCrossEntropyLoss, ListNetLoss, SpearmanLoss, M2MRegressionLoss, EdgeRankingGNN, EdgeClassificationGNN1, EdgeRankingGNN2, EdgeRankingGNN2_Ablation, EdgeRankingGNN2_Ablation1, EdgeRankingGNN_Ablation_0109
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from models.layers.mesh import Mesh
from torch_geometric.data import Data
from data.simplification_data import gather_graph, gather_graph_normal
import subprocess
from data.simplification_data import load_or_compute_stats
import re
import os
import pickle
from util.triangle_sample import sampleTriangle
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from models.layers.CustomMesh import angle_between_normals, angle_between_normals_batch, uvw_vectorized
from collections import defaultdict
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from functools import partial
import gc
import matplotlib.pyplot as plt
import multiprocessing
import random
import torch.nn.functional as F
# from util.nlp_smooth import smooth_sizing_function
from util.xiao_nlp1 import smooth_mesh_sizing

def final_process(path, geo_path):
    command1 = ['./remesh', '--input', path, '--eps', '1e-4', '--envelope-dis', '5e-4', '--max-pass', '100', '--output', path,'--split-num', '0', '--collapse-num', '0']
    try:
        result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
        print("remesh success")
    except subprocess.CalledProcessError as e:
        print("remesh fail")
        print(e.stderr)

    command2 = ['./test_sizefield_noIDmap', path, path, '1.2', '0', "0.1", geo_path]
    try:
        result2 = subprocess.run(command2, check=True, text=True, capture_output=True)
        # print(result2.stdout)
        print("create size field success")
    except subprocess.CalledProcessError as e:
        print("create size field fail")
        print(result2.stdout)
        print(e.stderr)

def convexity(mesh, id):
    ids = list(np.array(id))
    face_ids = []
    if_convex = []
    for v_id in ids:
        face_ids.append(mesh._vertex_face_map[v_id])
    vertex_normal = mesh.vertex_normal[ids]

    for i, ids in enumerate(face_ids):
        face_normals = mesh.face_normal[ids]
        this_vertex_normal = vertex_normal[i]
        flag = True
        for normal in face_normals:
            result = this_vertex_normal.dot(normal)
            if result < 0:
                flag = False
                break
        if_convex.append(flag)
    return torch.tensor(if_convex)

def vertex_convexity_consistency(mesh, edge_ids):
    edge_ids_tensor = torch.tensor(edge_ids, device=mesh.edges.device)
    edges = mesh.edges[edge_ids_tensor]
    v1 = edges[:, 0]
    v2 = edges[:, 1]
    if_convex1 = convexity(v1)
    if_convex2 = convexity(v2)
    return torch.any(if_convex1 == if_convex2)
    
def plot_tend(collapse_ranks, l1_losses, save_path):
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(collapse_ranks, l1_losses, marker='o', linestyle='-', linewidth=2, markersize=8)

    # 设置标题和轴标签
    plt.title('L1 Loss vs Collapse Rank', fontsize=16, fontweight='bold')
    plt.xlabel('Collapse Rank', fontsize=14)
    plt.ylabel('L1 Loss', fontsize=14)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()
    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {os.path.abspath(save_path)}")

    # 显示图表
    plt.show()
    plt.close()


def filter_feature_edge(mesh, edge_ids):
    true_collapse_edges = []
    for id in edge_ids:
        v1 = mesh.edges[id][0].item()
        v2 = mesh.edges[id][1].item()
        angle = angle_between_normals(mesh.get_vertex_normal(v1), mesh.get_vertex_normal(v2), degrees=True)
        if angle > 10:
            continue
        else:
            true_collapse_edges.append(id)
    return true_collapse_edges

def filter_feature_edge1(mesh, edge_ids):
    """
    加速版特征边过滤函数（向量化实现）
    
    参数:
        mesh: 网格对象（需包含edges、feature_point属性）
        edge_ids: 待过滤的边索引列表
        
    返回:
        List[int]: 满足条件的边索引列表
    """
    # 转换为Tensor并获取所有需要处理的顶点
    edge_ids_tensor = torch.tensor(edge_ids, device=mesh.edges.device)
    edges = mesh.edges[edge_ids]  # (n, 2)
    
    vertex_normals = mesh.vertex_normal
    
    # 向量化判断条件
    v1 = list(np.array(edges[:, 0]))
    v2 = list(np.array(edges[:, 1]))
    
    # 条件1：两个顶点都是特征点
    is_feature_pair = (mesh.feature_point[v1] & mesh.feature_point[v2])
    
    # 条件2：法向夹角>2度
    normals1 = vertex_normals[v1]
    normals2 = vertex_normals[v2]
    
    # 向量化计算夹角（单位向量点积）
    dots = (normals1 * normals2).sum(dim=1)
    dots = torch.clamp(dots, -1.0, 1.0)
    angles_deg = torch.rad2deg(torch.acos(dots))
    angle = (angles_deg < 15)     #10

    convex1 = mesh.convex[v1]
    convex2 = mesh.convex[v2]
    convex = ((convex1 == convex2) & (convex1 != 0))
    # mask = ~(is_feature_pair) | (is_feature_pair & angle & convex)
    mask = (~(is_feature_pair) & angle) | (is_feature_pair & angle & convex)

    return edge_ids_tensor[mask].tolist()


def filter_feature_edge2(mesh, edge_ids, _angle):

    edge_ids_tensor = torch.tensor(edge_ids, device=mesh.edges.device)
    edges = mesh.edges[edge_ids]  # (n, 2)
    
    vertex_normals = mesh.vertex_normal
    
    # 向量化判断条件
    v1 = list(np.array(edges[:, 0]))
    v2 = list(np.array(edges[:, 1]))


    
    # 条件1：两个顶点都是特征点
    # is_feature_pair = ~(mesh.feature_point[v1] ^ mesh.feature_point[v2])
    is_feature_pair = ((mesh.feature_point[v1] == mesh.feature_point[v2]))
    
    # 条件2：法向夹角>2度
    normals1 = vertex_normals[v1]
    normals2 = vertex_normals[v2]
    
    # 向量化计算夹角（单位向量点积
    dots = (normals1 * normals2).sum(dim=1)
    angles_deg = torch.rad2deg(torch.acos(dots))
    angle = angles_deg < _angle
    convex1 = mesh.convex[v1]
    convex2 = mesh.convex[v2]
    convex = (convex1 == convex2) & (convex1 != 0)
    mask = (is_feature_pair & angle & convex)

    return edge_ids_tensor[mask].tolist()

def filter_overlap(mesh, edge_id):
    v1 = mesh.edges[edge_id][0].item()
    v2 = mesh.edges[edge_id][1].item()
    new_mesh, _, _, _, post_face = mesh.collapsing_edge_new1(v1, v2)
    if new_mesh.if_manifold():
        flag = True
        return flag
    for face in post_face:
        adjacent_face_id, face_id = new_mesh.get_adjacent_face(face)
        for tris_id in adjacent_face_id:
            if tris_id is None:
                continue
            if new_mesh.check_triangle_overlap(face_id, tris_id):
                return True
    return False

# def process_edges_no_parallel(true_collapse_edges, mesh, target_mesh, tree1, size_threshold, dis_threshold):
#     operate_edge_ids = []
#     for i, edge in enumerate(true_collapse_edges):
#         collapse_one_start = time.time()
#         # v1, v2 = mesh.edges[edge]
#         # new_mesh, _, _, _, post_face = mesh.collapsing_edge_new(v1, v2)
#         new_mesh, v, post_face = mesh.collapsing_edge_id(edge)
#         collapse_one_time += (time.time() - collapse_one_start)
            
#         if new_mesh is None:
#             continue

#         re_size_start = time.time()
#         new_mesh.recalculate_size_one1(mesh, v)
#         # new_mesh.recalculate_size(mesh)
#         re_size_time += (time.time() - re_size_start)
            
#         # 构建KDTree
#         kdtree_start = time.time()
#         tree2 = cKDTree(new_mesh.vertices.numpy())
#         kdtree_time += time.time() - kdtree_start
            
#         # 三角形采样
#         tri_start = time.time()
#         sampled_points = []
#         for tri in post_face:
#             sample_p = sampleTriangle([
#                 new_mesh.vertices[tri[0]].numpy(),
#                 new_mesh.vertices[tri[1]].numpy(), 
#                 new_mesh.vertices[tri[2]].numpy()
#             ])
#             sampled_points.extend(sample_p)
#         inner_sample_time += time.time() - tri_start

#         if len(sampled_points) == 0:
#             break
            
#         # 投影计算
#         project_start = time.time()
#         target_s, target_d = batch_project1(sampled_points, target_mesh, tree1)
#         data_s, data_d = batch_project1(sampled_points, new_mesh, tree2)
#         project_time += time.time() - project_start
        
#         calculate_loss_start = time.time()
#         valid_mask = ~np.isnan(target_s) & ~np.isnan(data_s) & ~np.isnan(target_d) & ~np.isnan(data_d) &(target_s != 0) & (data_s != 0)   
#         size_losses = np.where(valid_mask, data_s / (target_s + 1e-8), np.nan)
#         dis_losses = np.where(valid_mask, target_d, np.nan)
#         size_losses[size_losses < 1] = 1
#         loss1 = np.nanmean(size_losses)
#         loss2 = max(dis_losses)
#         calculate_loss_time += time.time() - calculate_loss_start
            
#         if loss1 < size_threshold and loss2 < dis_threshold:
#             operate_edge_ids.append(edge)
#             print(f"{i}:{edge} accept cost {time.time() - collapse_one_start}s")



def process_edges_no_parallel(true_collapse_edges, mesh, target_mesh, tree1, size_threshold, dis_threshold):
    total_times = {
        'collapse': 0,
        'resize': 0,
        'kdtree': 0,
        'sample': 0,
        'project': 0,
        'loss': 0
    }
    
    operate_edge_ids = []
    last_print_time = time.time()
    last_message = ""


    
    for i, edge in enumerate(true_collapse_edges):
        # 边折叠时间
        collapse_start = time.time()
        new_mesh, v, post_face = mesh.collapsing_edge_id(edge)
        total_times['collapse'] += time.time() - collapse_start
            
        if new_mesh is None:
            continue

        # 重新计算大小时间
        resize_start = time.time()
        new_mesh.recalculate_size_one1(mesh, v)
        total_times['resize'] += time.time() - resize_start
            
        # 构建KDTree时间
        kdtree_start = time.time()
        tree2 = cKDTree(new_mesh.vertices.numpy())
        total_times['kdtree'] += time.time() - kdtree_start
            
        # 三角形采样时间
        sample_start = time.time()
        sampled_points = []
        for tri in post_face:
            sample_p = sampleTriangle([
                new_mesh.vertices[tri[0]].numpy(),
                new_mesh.vertices[tri[1]].numpy(), 
                new_mesh.vertices[tri[2]].numpy()
            ])
            sampled_points.extend(sample_p)
        total_times['sample'] += time.time() - sample_start

        if len(sampled_points) == 0:
            break
            
        # 投影计算时间
        project_start = time.time()
        target_s, target_d = batch_project1(sampled_points, target_mesh, tree1)
        data_s, data_d = batch_project1(sampled_points, new_mesh, tree2)
        total_times['project'] += time.time() - project_start
        
        # 损失计算时间
        loss_start = time.time()
        valid_mask = ~np.isnan(target_s) & ~np.isnan(data_s) & ~np.isnan(target_d) & ~np.isnan(data_d) & (target_s != 0) & (data_s != 0)   
        size_losses = np.where(valid_mask, data_s / (target_s + 1e-8), np.nan)
        dis_losses = np.where(valid_mask, target_d, np.nan)
        # size_losses[size_losses < 1] = 1
        # loss1 = max(size_losses)
        loss1 = np.average(np.abs(size_losses - 1))
        loss2 = max(dis_losses)
        
        total_times['loss'] += time.time() - loss_start
            
        # if loss1 < size_threshold and loss2 < dis_threshold:
        #     operate_edge_ids.append(edge)
        #     print(f"{i}:{edge} accept cost {time.time() - collapse_start}s")

        # 实时更新处理进度
        current_time = time.time()
        if current_time - last_print_time > 0.1:  # 每0.1秒更新一次
            progress_msg = f"Processing edge {i+1}/{len(true_collapse_edges)} - Accepted: {len(operate_edge_ids)}"
            # 清除上一行并输出新内容
            print('\r' + ' ' * len(last_message) + '\r' + progress_msg, end='', flush=True)
            last_message = progress_msg
            last_print_time = current_time
            
        if loss1 < size_threshold and loss2 < dis_threshold:
            operate_edge_ids.append(edge)
            # # 清除进度条，显示接受信息，然后恢复进度条
            # accept_msg = f"{i}:{edge} accept cost {time.time() - collapse_start:.2f}s"
            # print('\r' + ' ' * len(last_message) + '\r' + accept_msg)
            # # 重新显示进度条
            # print('\r' + last_message, end='', flush=True)

    # 处理完成后清除进度条
    print('\r' + ' ' * len(last_message) + '\r', end='')
    
    # 打印最终统计信息
    print(f"\n处理完成！总共处理了 {len(true_collapse_edges)} 条边，接受了 {len(operate_edge_ids)} 条边")
    
    # 打印各阶段耗时统计
    # print("\n=== 时间统计 ===")
    # for key, value in total_times.items():
    #     print(f"{key}: {value:.4f} seconds")
    
    return operate_edge_ids, total_times

def process_edge(edge, mesh, target_mesh, tree1):
    """处理单个边的折叠操作和损失计算"""
    try:
        v1, v2 = mesh.edges[edge]
        new_mesh, _, _, _, post_face = mesh.collapsing_edge_new(v1, v2)
        new_mesh.recalculate_size(mesh)
        tree2 = cKDTree(new_mesh.vertices.numpy())
        
        # 采样点生成
        sampled_points = []
        for tri in post_face:
            sample_p = sampleTriangle([
                new_mesh.vertices[tri[0]].numpy(),
                new_mesh.vertices[tri[1]].numpy(), 
                new_mesh.vertices[tri[2]].numpy()
            ])
            sampled_points.extend(sample_p)
        
        # 投影计算
        target_s, target_d = batch_project1(sampled_points, target_mesh, tree1)
        data_s, data_d = batch_project1(sampled_points, new_mesh, tree2)
        
        # 有效性检查
        valid_mask = (~np.isnan(target_s) & ~np.isnan(data_s) & 
                      ~np.isnan(target_d) & ~np.isnan(data_d) &
                      (target_s != 0) & (data_s != 0)) 
        
        # 损失计算
        size_losses = np.where(valid_mask, data_s / (target_s + 1e-6), np.nan)
        dis_losses = np.where(valid_mask, target_d, np.nan)
        size_losses[size_losses < 1] = 1
        
        loss1 = np.nanmean(size_losses)
        loss2 = np.nanmean(dis_losses)
        
        return edge, loss1, loss2, loss1 < 1.3 and loss2 < 5.0, None
    except Exception as e:
        return edge, None, None, False, str(e)

def parallel_process_edges(true_collapse_edges, mesh, target_mesh, tree1, max_workers=None):
    """使用线程池并行处理边缘"""
    operate_edge_ids = []
    # 使用线程池而不是进程池
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_edge = {
            executor.submit(process_edge, edge, mesh, target_mesh, tree1): (i, edge) 
            for i, edge in enumerate(true_collapse_edges)
        }
        
        # 处理完成的结果
        for future in as_completed(future_to_edge):
            i, edge = future_to_edge[future]
            try:
                edge_id, loss1, loss2, accepted, error = future.result()
                
                if error:
                    print(f"Error processing edge {edge}: {error}")
                elif accepted:
                    operate_edge_ids.append(edge_id)
                    print(f"{i}:{edge_id} accept")
                else:
                    print(f"{i}:{edge_id} size effect:{loss1:.4f} distance effect:{loss2:.4f}")
                    
            except Exception as e:
                print(f"Unexpected error with edge {edge}: {str(e)}")
    
    return operate_edge_ids

# 首先将循环体内的操作封装成一个函数
def process_edge1(edge, mesh, target_mesh, tree1, size_threshold, dis_threshold):
    result = {
        'edge': edge,
        'accept': False,
        'collapse_time': 0,
        'resize_time': 0,
        'kdtree_time': 0,
        'sample_time': 0,
        'project_time': 0,
        'loss_time': 0
    }
    
    # 边折叠
    collapse_one_start = time.time()
    new_mesh, v, post_face = mesh.collapsing_edge_id(edge)
    result['collapse_time'] = time.time() - collapse_one_start
    
    if new_mesh is None:
        return result
    
    # 重新计算大小
    re_size_start = time.time()
    new_mesh.recalculate_size_one1(mesh, v)
    result['resize_time'] = time.time() - re_size_start
    
    # 构建KDTree
    kdtree_start = time.time()
    tree2 = cKDTree(new_mesh.vertices.numpy())
    result['kdtree_time'] = time.time() - kdtree_start
    
    # 三角形采样
    tri_start = time.time()
    sampled_points = []
    for tri in post_face:
        sample_p = sampleTriangle([
            new_mesh.vertices[tri[0]].numpy(),
            new_mesh.vertices[tri[1]].numpy(), 
            new_mesh.vertices[tri[2]].numpy()
        ])
        sampled_points.extend(sample_p)
    result['sample_time'] = time.time() - tri_start
    
    if len(sampled_points) == 0:
        return result
    
    # 投影计算
    project_start = time.time()
    target_s, target_d = batch_project1(sampled_points, target_mesh, tree1)
    data_s, data_d = batch_project1(sampled_points, new_mesh, tree2)
    result['project_time'] = time.time() - project_start
    
    # 计算损失
    calculate_loss_start = time.time()
    valid_mask = ~np.isnan(target_s) & ~np.isnan(data_s) & ~np.isnan(target_d) & ~np.isnan(data_d) &(target_s != 0) & (data_s != 0)   
    size_losses = np.where(valid_mask, data_s / (target_s + 1e-8), np.nan)
    dis_losses = np.where(valid_mask, target_d, np.nan)
    size_losses[size_losses < 1] = 1
    loss1 = np.nanmean(size_losses)
    loss2 = max(dis_losses) if len(dis_losses) > 0 else float('inf')
    result['loss_time'] = time.time() - calculate_loss_start
    
    # 判断是否接受
    if loss1 < size_threshold and loss2 < dis_threshold:
        result['accept'] = True
    
    return result

# 主函数中使用进程池进行并行处理
def process_edges_in_parallel1(true_collapse_edges, mesh, target_mesh, tree1, size_threshold, dis_threshold, max_workers=None):
    operate_edge_ids = []
    total_times = {
        'collapse': 0,
        'resize': 0,
        'kdtree': 0,
        'sample': 0,
        'project': 0,
        'loss': 0
    }
    
    # 使用进程池
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_edge1, edge, mesh, target_mesh, tree1, size_threshold, dis_threshold): (i, edge) 
                  for i, edge in enumerate(true_collapse_edges)}
        
        # 处理完成的任务
        for future in as_completed(futures):
            i, edge = futures[future]
            try:
                result = future.result()
                
                # 累加时间
                total_times['collapse'] += result['collapse_time']
                total_times['resize'] += result['resize_time']
                total_times['kdtree'] += result['kdtree_time']
                total_times['sample'] += result['sample_time']
                total_times['project'] += result['project_time']
                total_times['loss'] += result['loss_time']
                
                # 如果接受该边，加入列表
                if result['accept']:
                    operate_edge_ids.append(edge)
                    # print(f"{i}: Edge {edge} accepted, cost {result['collapse_time']:.4f}s")
                    
            except Exception as e:
                print(f"Error processing edge {edge}: {str(e)}")
    
    return operate_edge_ids, total_times

def process_edge2(edge, mesh, target_mesh, tree1, size_threshold, dis_threshold):
    result = {
        'edge': edge,
        'accept': False,
        'collapse_time': 0,
        'resize_time': 0,
        'kdtree_time': 0,
        'sample_time': 0,
        'project_time': 0,
        'loss_time': 0
    }
    
    # 并行执行边折叠和重新计算大小
    collapse_one_start = time.time()
    
    # 使用线程池并行执行这两个操作
    with ThreadPoolExecutor(max_workers=16) as executor:
        # 提交边折叠任务
        collapse_future = executor.submit(mesh.collapsing_edge_id, edge)
        # 提交重新计算大小任务（但需要等待边折叠完成）
        # 注意：这里需要先获取边折叠结果，然后才能重新计算大小
        
        # 获取边折叠结果
        new_mesh, v1, post_face = collapse_future.result()
        result['collapse_time'] = time.time() - collapse_one_start
        
        if new_mesh is None:
            return result
        
        # 提交重新计算大小任务
        resize_future = executor.submit(new_mesh.recalculate_size_one1, mesh, v1)
        # resize_future = executor.submit(new_mesh.recalculate_size1, mesh)
        # 同时构建KDTree
        kdtree_future = executor.submit(cKDTree, new_mesh.vertices.numpy())
        
        # 等待重新计算大小完成
        resize_future.result()
        result['resize_time'] = time.time() - collapse_one_start - result['collapse_time']
        
        # 获取KDTree
        tree2 = kdtree_future.result()
        result['kdtree_time'] = time.time() - collapse_one_start - result['collapse_time'] - result['resize_time']
    
    # 三角形采样
    tri_start = time.time()
    sampled_points = []
    for tri in post_face:
        sample_p = sampleTriangle([
            new_mesh.vertices[tri[0]].numpy(),
            new_mesh.vertices[tri[1]].numpy(), 
            new_mesh.vertices[tri[2]].numpy()
        ])
        sampled_points.extend(sample_p)
    result['sample_time'] = time.time() - tri_start
    
    if len(sampled_points) == 0:
        return result
    
    # 并行执行两个投影计算
    project_start = time.time()
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     target_future = executor.submit(batch_project1, sampled_points, target_mesh, tree1)
    #     data_future = executor.submit(batch_project1, sampled_points, new_mesh, tree2)
        
    #     target_s, target_d = target_future.result()
    #     data_s, data_d = data_future.result()
    target_s, target_d = batch_project1(sampled_points, target_mesh, tree1)
    data_s, data_d = batch_project1(sampled_points, new_mesh, tree2)
    result['project_time'] = time.time() - project_start
    
    # 计算损失
    calculate_loss_start = time.time()
    valid_mask = ~np.isnan(target_s) & ~np.isnan(data_s) & ~np.isnan(target_d) & ~np.isnan(data_d) &(target_s != 0) & (data_s != 0)   
    size_losses = np.where(valid_mask, data_s / (target_s + 1e-8), np.nan)
    dis_losses = np.where(valid_mask, target_d, np.nan)
    size_losses[size_losses < 1] = 1
    loss1 = np.nanmean(size_losses)
    loss2 = max(dis_losses) if len(dis_losses) > 0 else float('inf')
    result['loss_time'] = time.time() - calculate_loss_start
    
    # 判断是否接受
    if loss1 < size_threshold and loss2 < dis_threshold:
        result['accept'] = True
    
    return result

# 主函数中使用进程池进行并行处理
def process_edges_in_parallel2(true_collapse_edges, mesh, target_mesh, tree1, size_threshold, dis_threshold, max_workers=None):
    operate_edge_ids = []
    total_times = {
        'collapse': 0,
        'resize': 0,
        'kdtree': 0,
        'sample': 0,
        'project': 0,
        'loss': 0
    }
    
    # 使用进程池
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_edge2, edge, mesh, target_mesh, tree1, size_threshold, dis_threshold): (i, edge) 
                  for i, edge in enumerate(true_collapse_edges)}
        
        # 处理完成的任务
        for future in as_completed(futures):
            i, edge = futures[future]
            try:
                result = future.result()
                
                # 累加时间
                total_times['collapse'] += result['collapse_time']
                total_times['resize'] += result['resize_time']
                total_times['kdtree'] += result['kdtree_time']
                total_times['sample'] += result['sample_time']
                total_times['project'] += result['project_time']
                total_times['loss'] += result['loss_time']
                
                # 如果接受该边，加入列表
                if result['accept']:
                    operate_edge_ids.append(edge)
                    # print(f"{i}: Edge {edge} accepted, cost {result['collapse_time']:.4f}s")
                    
            except Exception as e:
                print(f"Error processing edge {edge}: {str(e)}")
    
    return operate_edge_ids, total_times

def process_single_edge(args):
    """处理单条边的函数，可在独立进程中运行"""
    edge, mesh, target_mesh, tree1, size_threshold, dis_threshold = args
    
    times = {
        'collapse': 0,
        'resize': 0,
        'kdtree': 0,
        'sample': 0,
        'project': 0,
        'loss': 0
    }
    
    # 边折叠操作
    collapse_start = time.time()
    new_mesh, v, post_face = mesh.collapsing_edge_id(edge)
    times['collapse'] = time.time() - collapse_start
    
    if new_mesh is None:
        return edge, False, times
    
    # 重新计算大小
    resize_start = time.time()
    new_mesh.recalculate_size_one1(mesh, v)
    times['resize'] = time.time() - resize_start
    
    # 构建KDTree
    kdtree_start = time.time()
    tree2 = cKDTree(new_mesh.vertices.numpy())
    times['kdtree'] = time.time() - kdtree_start
    
    # 三角形采样
    sample_start = time.time()
    sampled_points = []
    for tri in post_face:
        sample_p = sampleTriangle([
            new_mesh.vertices[tri[0]].numpy(),
            new_mesh.vertices[tri[1]].numpy(), 
            new_mesh.vertices[tri[2]].numpy()
        ])
        sampled_points.extend(sample_p)
    times['sample'] = time.time() - sample_start

    if len(sampled_points) == 0:
        return edge, False, times
    
    # 投影计算
    project_start = time.time()
    target_s, target_d = batch_project1(sampled_points, target_mesh, tree1)
    data_s, data_d = batch_project1(sampled_points, new_mesh, tree2)
    times['project'] = time.time() - project_start
    
    # 损失计算
    loss_start = time.time()
    valid_mask = ~np.isnan(target_s) & ~np.isnan(data_s) & ~np.isnan(target_d) & ~np.isnan(data_d) & (target_s != 0) & (data_s != 0)   
    size_losses = np.where(valid_mask, data_s / (target_s + 1e-8), np.nan)
    dis_losses = np.where(valid_mask, target_d, np.nan)
    size_losses[size_losses < 1] = 1
    loss1 = np.nanmean(size_losses)
    loss2 = max(dis_losses) if len(dis_losses) > 0 else 0
    times['loss'] = time.time() - loss_start
    
    # 判断是否接受该边
    accept = loss1 < size_threshold and loss2 < dis_threshold
    return edge, accept, times

def process_edges_parallel(true_collapse_edges, mesh, target_mesh, tree1, size_threshold, dis_threshold, max_workers=None):
    """并行处理所有边，每条边独立处理"""
    # 初始化总时间统计
    total_times = {
        'collapse': 0,
        'resize': 0,
        'kdtree': 0,
        'sample': 0,
        'project': 0,
        'loss': 0
    }
    
    operate_edge_ids = []
    last_print_time = time.time()
    last_message = ""
    processed = 0
    total_edges = len(true_collapse_edges)
    
    # 如果未指定工作进程数，使用CPU核心数
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    print(f"使用 {max_workers} 个进程并行处理 {total_edges} 条边...")
    
    # 准备参数列表，确保每个任务的参数独立
    args_list = [
        (edge, mesh.clone_mesh(), target_mesh.clone_mesh(), tree1, size_threshold, dis_threshold) 
        for edge in true_collapse_edges
    ]
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_edge, args): i 
                  for i, args in enumerate(args_list)}
        
        # 处理完成的任务，按完成顺序处理
        for future in as_completed(futures):
            try:
                edge, accept, times = future.result()
                
                # 累加各阶段时间
                for key, value in times.items():
                    total_times[key] += value
                
                # 记录接受的边
                if accept:
                    operate_edge_ids.append(edge)
                
                # 更新进度显示
                processed += 1
                current_time = time.time()
                
                # 控制进度更新频率，避免过多IO操作
                if current_time - last_print_time > 0.1:
                    progress_msg = f"已处理: {processed}/{total_edges} 条边, 已接受: {len(operate_edge_ids)} 条"
                    # 清除上一行并打印新进度
                    print('\r' + ' ' * len(last_message) + '\r' + progress_msg, end='', flush=True)
                    last_message = progress_msg
                    last_print_time = current_time
            except Exception as e:
                print(f"\n处理边时发生错误: {str(e)}")
    
    # 清除进度条
    print('\r' + ' ' * len(last_message) + '\r', end='')
    
    # 打印最终统计信息
    print(f"处理完成！总共处理 {total_edges} 条边，接受 {len(operate_edge_ids)} 条边")
    # print("\n各阶段总耗时:")
    # for key, value in total_times.items():
    #     print(f"  {key}: {value:.4f} 秒")
    
    return operate_edge_ids, total_times

def find_nearest_triangles(mesh1, mesh2):
    """
    为mesh1的每个三角面片找到mesh2中最近的三角面片
    
    参数:
    mesh1 - 源网格 (CustomMesh对象)
    mesh2 - 目标网格 (CustomMesh对象)
    kdtree - 可选的预构建KD树 (加速查询)
    
    返回:
    nearest_triangles - 每个mesh1面片对应的最近mesh2面片索引列表
    distances - 对应的距离列表
    """
    # 计算mesh1所有面片的中心点
    mesh1_vertices = mesh1.vertices.numpy()
    mesh1_faces = mesh1.faces.numpy()
    mesh1_centroids = np.mean(mesh1_vertices[mesh1_faces], axis=1)
    
    # 计算mesh2所有面片的中心点和法向量
    mesh2_vertices = mesh2.vertices.numpy()
    mesh2_faces = mesh2.faces.numpy()
    mesh2_centroids = np.mean(mesh2_vertices[mesh2_faces], axis=1)
    
    kdtree = KDTree(mesh2_centroids)
    
    # 查询每个mesh1中心点的最近mesh2中心点
    _, nearest_indices = kdtree.query(mesh1_centroids)
    
    for i in range(len(nearest_indices)):
        mesh1.surface_id[i] = mesh2.surface_id[nearest_indices[i]]

def hausdorff_distance_max(mesh1_verts, mesh2_verts):
    """
    计算两个网格之间的双向豪斯多夫距离
    参数:
        mesh1_verts: (N,3) 第一个网格的顶点
        mesh2_verts: (M,3) 第二个网格的顶点
    返回:
        max_dist: 最大豪斯多夫距离
        mean_dist: 平均豪斯多夫距离
    """
    # 计算所有顶点对之间的距离矩阵 (N,M)
    dist_matrix = torch.cdist(mesh1_verts, mesh2_verts)
    
    # 单向豪斯多夫距离
    d_1_to_2 = dist_matrix.min(dim=1)[0]  # mesh1→mesh2
    # d_2_to_1 = dist_matrix.min(dim=0)[0]  # mesh2→mesh1
    
    # 双向豪斯多夫距离
    # max_dist = max(d_1_to_2.max(), d_2_to_1.max())
    
    # return max_dist.item()
    return d_1_to_2.max().item()

def hausdorff_distance_mean(mesh1_verts, mesh2_verts):
    """
    计算两个网格之间的双向豪斯多夫距离
    参数:
        mesh1_verts: (N,3) 第一个网格的顶点
        mesh2_verts: (M,3) 第二个网格的顶点
    返回:
        max_dist: 最大豪斯多夫距离
        mean_dist: 平均豪斯多夫距离
    """
    # 计算所有顶点对之间的距离矩阵 (N,M)
    dist_matrix = torch.cdist(mesh1_verts, mesh2_verts)
    
    # 单向豪斯多夫距离
    d_1_to_2 = dist_matrix.min(dim=1)[0]  # mesh1→mesh2
    d_2_to_1 = dist_matrix.min(dim=0)[0]  # mesh2→mesh1
    
    # 双向豪斯多夫距离
    mean_dist = (d_1_to_2.mean() + d_2_to_1.mean()) / 2
    
    return mean_dist.item()

def calculate_size_dis(data_mesh, target_mesh):
    sampled_points = []
    for tri in data_mesh.faces:
        sample_p = sampleTriangle([data_mesh.vertices[tri[0]].numpy(), data_mesh.vertices[tri[1]].numpy(), data_mesh.vertices[tri[2]].numpy()])
        sampled_points.extend(sample_p)
    tree1 = cKDTree(target_mesh.vertices.numpy())
    tree2 = cKDTree(data_mesh.vertices.numpy())
    target_s, target_d = batch_project(sampled_points, target_mesh, tree1)
    data_s, data_d = batch_project(sampled_points, data_mesh, tree2)
    valid_mask = ~np.isnan(target_s) & ~np.isnan(data_s) & ~np.isnan(target_d) & ~np.isnan(data_d)
    size_losses = np.where(valid_mask, target_s - data_s, np.nan)
    dis_losses = np.where(valid_mask, target_d, np.nan)
    loss1 = np.mean(size_losses)
    loss2 = np.mean(dis_losses)
    return loss1, loss2

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
    return u * sizes[0] + v * sizes[1] + w * sizes[2]

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

def batch_project(points, mesh, tree):
    points_arr = np.array(points)
    mesh_vertices = mesh.vertices.numpy()
    mesh_faces = mesh.faces.numpy()
    mesh_size_values = mesh.sizing_values.numpy()
        
    # 查询每个点的最近顶点
    _, point_indices = tree.query(points_arr)
        
    # 预计算所有三角形的法向量
    v0s = mesh_vertices[mesh_faces[:, 1]] - mesh_vertices[mesh_faces[:, 0]]
    v1s = mesh_vertices[mesh_faces[:, 2]] - mesh_vertices[mesh_faces[:, 1]]
    normals = np.cross(v0s, v1s)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
    # 为每个点找到包含它的三角形
    results_s = np.zeros(len(points_arr))
    results_d = np.zeros(len(points_arr))
        
    for i, (point, point_idx) in enumerate(zip(points_arr, point_indices)):
        # 找到包含该顶点的所有三角形
        containing_tris = np.where(np.any(mesh_faces == point_idx, axis=1))[0]
            
        if len(containing_tris) == 0:
            results_s[i] = mesh_size_values[point_idx]
            results_d[i] = np.linalg.norm(point - mesh_vertices[point_idx])
            continue
                
        # 计算点到这些三角形平面的距离
        tri_verts = mesh_vertices[mesh_faces[containing_tris]]
        ds = np.einsum('ij,ij->i', normals[containing_tris], point - tri_verts[:, 0])
        abs_ds = np.abs(ds)
            
        # 计算重心坐标
        # UVW = np.array([uvw(point, mesh, mesh_faces[tri_idx]) 
        #                 for tri_idx in containing_tris])
        UVW = uvw_vectorized(point, mesh, mesh_faces[containing_tris])
        valid_mask = np.all((UVW >= -1e-8) & (UVW <= 1+1e-8), axis=1)
            
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
                                                             mesh)
            results_d[i] = abs_ds[min_idx]
        else:
            results_s[i] = mesh_size_values[point_idx]
            results_d[i] = np.linalg.norm(point - mesh_vertices[point_idx])
                
    return results_s, results_d

def batch_project1(points, mesh, tree):
    points_arr = np.array(points)
    mesh_vertices = mesh.vertices.numpy()
    mesh_faces = mesh.faces.numpy()
    mesh_size_values = mesh.sizing_values.numpy()
    
    # 查询所有点的最近顶点
    _, point_indices = tree.query(points_arr)
    
    # 预计算所有三角形的法向量
    # v0s = mesh_vertices[mesh_faces[:, 1]] - mesh_vertices[mesh_faces[:, 0]]
    # v1s = mesh_vertices[mesh_faces[:, 2]] - mesh_vertices[mesh_faces[:, 1]]
    # normals = np.cross(v0s, v1s)
    # normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    normals = np.array(mesh.face_normal)

    # 预构建面-顶点映射（加速查找包含顶点的三角形）
    vertex_to_faces = mesh._vertex_face_map
    
    # 批量处理所有点
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
            UVW = uvw_vectorized(point, mesh, mesh_faces[containing_tris])
            valid_mask = np.all((UVW >= -1e-8) & (UVW <= 1+1e-8), axis=1)
            
            if np.any(valid_mask):
                valid_ds = np.where(valid_mask, abs_ds, np.inf)
                min_idx = np.argmin(valid_ds)
                min_tri = containing_tris[min_idx]
                
                projection_point = point - ds[min_idx] * normals[min_tri]
                results_s[i] = barycentric_interpolation(projection_point, 
                                                        mesh_faces[min_tri], 
                                                        mesh)
                results_d[i] = abs_ds[min_idx]
            else:
                # results_s[i] = mesh_size_values[point_indices[i]]
                results_s[i] = mesh_size_values[point_indices[i]].item()
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
    
    return results_s, results_d

def load_para():
    node_file = '/home/zhuxunyang/coding/bkgm_simplification/para/node_stats.pkl'
    edge_file = '/home/zhuxunyang/coding/bkgm_simplification/para/edge_stats.pkl'

    # 尝试加载已有参数
    if os.path.exists(node_file) and os.path.exists(edge_file):
        with open(node_file, 'rb') as f:
            node_mean, node_std = pickle.load(f)
        with open(edge_file, 'rb') as f:
            edge_mean, edge_std = pickle.load(f)
        return node_mean, node_std, edge_mean, edge_std

def sort_clean_edge(mesh, index):
    result1 = []
    used_vertices = set()
    for id in index:
        edge = mesh.edges[id]
        # 假设边用两个顶点表示，如 (vertex1, vertex2)
        vertex1, vertex2 = edge
        if vertex1 not in used_vertices and vertex2 not in used_vertices:
            result1.append(id.item())
            used_vertices.add(vertex1)
            used_vertices.add(vertex2)
    result = sorted(
        result1,
        key=lambda edge_id: max(mesh.edges[edge_id]),  # 取边的两个顶点中较大的值
        reverse=True  # 降序
    )
    return result

def sort_clean_edge1(mesh, edge_indices):
    """
    筛选并排序边，确保保留的边之间至少保持2-ring距离
    
    参数:
        mesh: 网格对象，包含edges属性
        edge_indices: 待处理的边索引列表
        
    返回:
        list: 排序后的边索引列表，满足2-ring距离条件
    """
    # 转换为边列表（假设mesh.edges是张量）
    edges = mesh.edges.cpu().numpy() if torch.is_tensor(mesh.edges) else mesh.edges
    edge_indices = [idx.item() if isinstance(idx, torch.Tensor) else idx for idx in edge_indices]
    
    # 构建顶点到边的映射
    vertex_edge_map = defaultdict(list)
    for edge_id in edge_indices:
        v1, v2 = edges[edge_id]
        vertex_edge_map[v1].append(edge_id)
        vertex_edge_map[v2].append(edge_id)
    
    result = []
    used_vertices = set()
    banned_vertices = set()  # 用于2-ring限制
    
    # 按顶点最大度排序（优先处理连接边多的顶点）
    sorted_edges = sorted(
        edge_indices,
        key=lambda eid: -max(len(vertex_edge_map[edges[eid][0]]), 
                            len(vertex_edge_map[edges[eid][1]]))
    )
    
    for edge_id in sorted_edges:
        v1, v2 = edges[edge_id]
        
        # 检查是否满足2-ring条件
        if (v1 not in used_vertices and 
            v2 not in used_vertices and
            v1 not in banned_vertices and
            v2 not in banned_vertices):
            
            result.append(edge_id)
            used_vertices.update([v1, v2])
            
            # 将邻接顶点加入禁止集合（1-ring）
            for v in [v1, v2]:
                for neighbor_edge in vertex_edge_map[v]:
                    nv1, nv2 = edges[neighbor_edge]
                    banned_vertices.add(nv1)
                    banned_vertices.add(nv2)
    
    # 最终按边的最大顶点降序排列
    return sorted(
        result,
        key=lambda eid: max(edges[eid]),
        reverse=True
    )

def sort_clean_edge2(mesh, edge_indices):
    """
    筛选并排序边，确保保留的边之间至少保持3-ring距离
    
    参数:
        mesh: 网格对象，包含edges属性
        edge_indices: 待处理的边索引列表
        
    返回:
        list: 排序后的边索引列表，满足3-ring距离条件
    """
    # 转换为边列表（假设mesh.edges是张量）
    edges = mesh.edges.cpu().numpy() if torch.is_tensor(mesh.edges) else mesh.edges
    edge_indices = [idx.item() if isinstance(idx, torch.Tensor) else idx for idx in edge_indices]
    
    # 构建顶点到边的映射
    vertex_edge_map = defaultdict(list)
    for edge_id in edge_indices:
        v1, v2 = edges[edge_id]
        vertex_edge_map[v1].append(edge_id)
        vertex_edge_map[v2].append(edge_id)
    
    result = []
    used_vertices = set()
    banned_vertices = set()  # 用于3-ring限制（禁止2-ring内的顶点）
    
    # 按顶点最大度排序（优先处理连接边多的顶点）
    sorted_edges = sorted(
        edge_indices,
        key=lambda eid: -max(len(vertex_edge_map[edges[eid][0]]), 
                            len(vertex_edge_map[edges[eid][1]]))
    )
    
    for edge_id in sorted_edges:
        v1, v2 = edges[edge_id]
        
        # 检查是否满足3-ring条件：顶点未被使用且不在禁止集合中
        if (v1 not in used_vertices and 
            v2 not in used_vertices and
            v1 not in banned_vertices and
            v2 not in banned_vertices):
            
            result.append(edge_id)
            used_vertices.update([v1, v2])
            
            # 将1-ring和2-ring顶点加入禁止集合
            for v in [v1, v2]:
                # 1-ring: 直接邻接顶点
                for neighbor_edge in vertex_edge_map[v]:
                    nv1, nv2 = edges[neighbor_edge]
                    banned_vertices.update([nv1, nv2])
                    # 2-ring: 邻接顶点的邻接顶点
                    for second_ring_edge in vertex_edge_map[nv1] + vertex_edge_map[nv2]:
                        sv1, sv2 = edges[second_ring_edge]
                        banned_vertices.update([sv1, sv2])
    
    # 最终按边的最大顶点降序排列
    return sorted(
        result,
        key=lambda eid: max(edges[eid]),
        reverse=True
    )

def create_example(vtk_path, opt, device, norm_params=None):
    print(CustomMesh.CustomMesh.from_vtk(vtk_path).if_manifold())
    mesh = Mesh(file=vtk_path, opt=opt, hold_history=True, export_folder=opt.export_folder)
    all_node_features, all_edge_features = [], []
    all_node_features.append(mesh.size_value)
    all_edge_features.append(mesh.extract_features())
    node_mean, node_std, edge_mean, edge_std = load_or_compute_stats(np.concatenate(all_node_features, axis=0), np.concatenate(all_edge_features, axis=1))
    edge_features = mesh.extract_features()
    GEO = vtk_path.split('/')[-1].split('_')[0]
    edge_len = len(mesh.edges)

    # 归一化（假设使用全局参数）
    norm_node = (mesh.size_value - node_mean) / node_std
    norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
    
    # 检查并转换顶点数据的字节序
    if mesh.vs.dtype.byteorder != '=':
        mesh.vs = mesh.vs.byteswap().newbyteorder()

    data = Data(
        x=torch.from_numpy(norm_node).float(),
        edge_index=torch.from_numpy(mesh.edges).T.long(),
        edge_attr=torch.from_numpy(norm_edge).T.float(),
        pos=torch.from_numpy(mesh.vs),
        length = edge_len,
        mesh = CustomMesh.CustomMesh.from_vtk(vtk_path),
        geo = GEO
    ).to(device)
    return DataLoader([data], batch_size=1, shuffle=True), (node_mean, node_std, edge_mean, edge_std)

def create_example1(vtk_path, opt, device, norm_params=None):
    mesh = CustomMesh.CustomMesh.from_vtk(vtk_path)
    # print(mesh.if_manifold())
    all_node_features, all_edge_features = [], []
    lbo_vertex, lbo_edge = mesh.compute_LBO()
    # all_node_features.append(mesh.sizing_values)
    # all_edge_features.append(mesh.compute_edge_features())
    all_node_features.append(torch.cat((mesh.sizing_values, torch.from_numpy(lbo_vertex).unsqueeze(1)), dim=1))
    all_edge_features.append(torch.cat((mesh.compute_edge_features(), torch.from_numpy(lbo_edge).unsqueeze(0)), dim=0))
    if norm_params is None:
        node_mean, node_std, edge_mean, edge_std = load_or_compute_stats(np.concatenate(all_node_features, axis=0), np.concatenate(all_edge_features, axis=1))
    else:
        node_mean, node_std, edge_mean, edge_std = norm_params
    edge_features = mesh.compute_edge_features()
    GEO = vtk_path.split('/')[-1].split('_')[0]
    edge_len = len(mesh.edges)

    # 归一化（假设使用全局参数）
    # norm_node = (mesh.sizing_values - node_mean) / node_std
    # norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
    
    norm_node = torch.from_numpy((np.concatenate([mesh.sizing_values, np.expand_dims(lbo_vertex, axis=1)], axis=1) - node_mean) / node_std)
    norm_edge = torch.from_numpy((np.concatenate([edge_features, np.expand_dims(lbo_edge, axis=0)], axis=0) - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1))
            

    data = Data(
        x=norm_node.float(),
        edge_index=mesh.edges.T.long(),
        edge_attr=norm_edge.T.float(),
        pos=mesh.vertices,
        length = edge_len,
        mesh = mesh,
        geo = GEO
    ).to(device)
    return DataLoader([data], batch_size=1, shuffle=True), (node_mean, node_std, edge_mean, edge_std)

def create_example2(vtk_path, opt, device, norm_params=None, label_vtk=None):
    mesh = CustomMesh.CustomMesh.from_vtk(vtk_path)
    # print(mesh.if_manifold())
    all_node_features, all_edge_features = [], []
    lbo_vertex, lbo_edge = mesh.compute_LBO()
    # all_node_features.append(mesh.sizing_values)
    # all_edge_features.append(mesh.compute_edge_features())
    all_node_features.append(torch.cat((mesh.sizing_values, torch.from_numpy(lbo_vertex).unsqueeze(1)), dim=1))
    all_edge_features.append(torch.cat((mesh.compute_edge_features(), torch.from_numpy(lbo_edge).unsqueeze(0)), dim=0))
    if norm_params is None:
        node_mean, node_std, edge_mean, edge_std = load_or_compute_stats(np.concatenate(all_node_features, axis=0), np.concatenate(all_edge_features, axis=1))
    else:
        node_mean, node_std, edge_mean, edge_std = norm_params
    edge_features = mesh.compute_edge_features()
    GEO = vtk_path.split('/')[-1].split('_')[0]
    edge_len = len(mesh.edges)

    # 归一化（假设使用全局参数）
    # norm_node = (mesh.sizing_values - node_mean) / node_std
    # norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
    
    norm_node = torch.from_numpy((np.concatenate([mesh.sizing_values, np.expand_dims(lbo_vertex, axis=1)], axis=1) - node_mean) / node_std)
    norm_edge = torch.from_numpy((np.concatenate([edge_features, np.expand_dims(lbo_edge, axis=0)], axis=0) - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1))
            
    label_mesh = meshio.read(label_vtk)
    data = Data(
        x=norm_node.float(),
        y=label_mesh.cell_data['label0-1'][0],
        edge_index=mesh.edges.T.long(),
        edge_attr=norm_edge.T.float(),
        pos=mesh.vertices,
        length = edge_len,
        mesh = mesh,
        geo = GEO
    ).to(device)
    return DataLoader([data], batch_size=1, shuffle=True), (node_mean, node_std, edge_mean, edge_std)

def test_regression():
    # 初始化选项
    opt = TestOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset, norm_params = gather_graph_normal("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/sf0613/val", opt, device, 1)

    model = EdgeRankingGNN(
        node_feat_dim=1,
        edge_feat_dim=5,
        hidden_dim=64
    ).to(device=device)
    total_steps = 0

    checkpoint = torch.load(
        "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Rank+Regression.pth",
        weights_only=False
    )

    model.load_state_dict(checkpoint)
    i = 0
    for val_batch in val_dataset:
        with torch.no_grad():
            edge_count = len(val_batch.mesh[0].edges)
            N = 0
            step = 0
            original_mesh = val_batch.mesh[0]
            while N < edge_count // 5:
                geo = val_batch.geo[0]
                val_out = model(val_batch)
                _, sort_pred = torch.topk(val_out[:, 0], 10, dim=0, largest=False)
                collapse_edges = sort_clean_edge(val_batch.mesh[0], sort_pred)
                print("We will collapse", len(collapse_edges), "edges")
                collapse_point = []
                for id in collapse_edges:
                    collapse_point.append(val_batch.mesh[0].edges[id])
                
                for points in collapse_point:
                    original_mesh, dis = original_mesh.collapsing_edge(points[0], points[1])


                    original_mesh.get_all_info()
                
                save_path1 = "/home/zhuxunyang/coding/bkgm_simplification/result/result_before_smooth" + str(i) + "_" + str(step) + ".vtk"
                save_path2 = "/home/zhuxunyang/coding/bkgm_simplification/result/result_after_smooth" + str(i) + "_" + str(step) + ".vtk"
                save_path3 = "/home/zhuxunyang/coding/bkgm_simplification/result/result_sizefield" + str(i) + "_" + str(step) + ".vtk"
                geo_path = f"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/model/{geo}/{geo}.step"
                original_mesh.writeVTK(save_path1)
                command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', '1e-4', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
                try:
                    result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
                    print("remesh success")
                except subprocess.CalledProcessError as e:
                    print("remesh fail")
                    print(e.stderr)

                command2 = ['./test_sizefield1', save_path2, save_path3, '1.2', '0', "0.1", geo_path]
                try:
                    result2 = subprocess.run(command2, check=True, text=True, capture_output=True)
                    # print(result2.stdout)
                    print("create size field success")
                except subprocess.CalledProcessError as e:
                    print("create size field fail")
                    print(e.stderr)
                
                # 更新 original_mesh 和 model_mesh
                original_mesh = CustomMesh.CustomMesh.from_vtk(save_path3)
                print("If mainfold:", original_mesh.if_manifold())
                model_mesh = Mesh(file=save_path3, opt=opt)
                
                # 提取新网格的特征并归一化
                node_mean, node_std, edge_mean, edge_std = norm_params
                edge_features = model_mesh.extract_features()
                norm_node = (model_mesh.size_value - node_mean) / node_std
                norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
                
                num_nodes = len(model_mesh.vs)
                # 更新 val_batch
                val_batch.x = torch.from_numpy(norm_node).float().to(device)
                val_batch.edge_index = torch.from_numpy(model_mesh.edges).T.long().to(device)
                val_batch.edge_attr = torch.from_numpy(norm_edge).T.float().to(device)
                val_batch.pos = torch.from_numpy(model_mesh.vs).to(device)
                val_batch.length = [len(model_mesh.edges)]
                val_batch.mesh = [original_mesh]  # 注意保持 batch 结构
                val_batch.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
                
                N = N + (len(collapse_edges) * 3)
                step += 1
            i += 1

def test_example(vtk_path, n):
    total_time = defaultdict(float)
    start_time = time.time()
    opt = TestOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, norm_params = create_example1(vtk_path, opt, device, load_para())
    model = EdgeRankingGNN(
        node_feat_dim=1,
        edge_feat_dim=7,
        hidden_dim=64
    ).to(device=device)


    total_steps = 0

    checkpoint = torch.load(
        # "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Regression_size5.pth",
        # "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Regression3.pth",
        "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Regression_dis1size0.pth",
        weights_only=False
    )

    Geo_path = None
    vtk = None
    no_manifold_times = 0
    name = re.split("/", vtk_path)[-1].split('.')[0]

    model.load_state_dict(checkpoint)
    i = 0
    if_end = False
    collapse_rank = 0
    for val_batch in dataset:
        with torch.no_grad():
            t0 = time.time()
            edge_count = len(val_batch.mesh[0].edges)
            N = 0
            step = 0
            geo = val_batch.geo[0]
            target_path = f"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/target/{geo}_target.vtk"
            target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
            tree1 = cKDTree(target_mesh.vertices.numpy())
            total_time['preprocessing'] = time.time() - t0

            begin_mesh = val_batch.mesh[0]

            flag = True
            times = 0
            # while N < edge_count // 2:
            while flag:
                iter_time = time.time()
                t0 = time.time()
                if times > 5:
                    Geo_path = geo_path
                    vtk = save_path3
                    flag = False
                    break
                val_out = model(val_batch)
                _, sort_pred = torch.topk(val_out[:, 0], n, dim=0, largest=False)
                total_time['edge_predict'] = time.time() - t0
                t0 = time.time()
                collapse_edges = sort_clean_edge1(val_batch.mesh[0], sort_pred)
                # val_batch.mesh[0].visual_edge(collapse_edges, "selected_edge.vtk")
                # collapse_edges = list(collapse_edges.detach().numpy())
                total_time['edge_filter1'] = time.time() - t0
                t0 = time.time()

                # true_collapse_edges = filter_feature_edge(val_batch.mesh[0], collapse_edges)
                true_collapse_edges = filter_feature_edge1(val_batch.mesh[0], collapse_edges)

                total_time['edge_filter2'] = time.time() - t0
                
                # edges = val_batch.mesh[0].edges.cpu().numpy()
                # point_set = set()
                # banned_vertices = set()
                # vertex_edge_map = defaultdict(list)
                # for i, edge in enumerate(edges):
                #     v1, v2 = edges[i]
                #     vertex_edge_map[v1].append(i)
                #     vertex_edge_map[v2].append(i)
                # edge_ids = []

                # #trad
                # for index in sort_pred:
                #     v1, v2 = edges[index]
                #     convex1 = val_batch.mesh[0].convex[v1]
                #     convex2 = val_batch.mesh[0].convex[v2]
                #     normals1 = val_batch.mesh[0].vertex_normal[v1]
                #     normals2 = val_batch.mesh[0].vertex_normal[v2]
                #     dots = normals1.dot(normals2)
                #     angles_deg = torch.rad2deg(torch.acos(dots))
                #     if (v1 not in point_set and 
                #         v2 not in point_set and 
                #         v1 not in banned_vertices and 
                #         v2 not in banned_vertices and 
                #         (val_batch.mesh[0].feature_point[v1] == val_batch.mesh[0].feature_point[v2]) and 
                #         ((convex1 == convex2) & (convex1 != 0)) and (angles_deg < angle)):
                #         edge_ids.append(index)
                #         point_set.update([v1, v2])
                #         for v in [v1, v2]:
                #             for neighbor_edge in vertex_edge_map[v]:
                #                 nv1, nv2 = edges[neighbor_edge]
                #                 banned_vertices.add(nv1)
                #                 banned_vertices.add(nv2)
                #     if len(edge_ids) >= n:
                #         break

                if len(true_collapse_edges) == 0:
                    print("No edge will collapse")
                    times += 1
                    continue
                else:
                    times = 0

                # print("We have collapsed", len(true_collapse_edges), "edges", val_batch.mesh[0].edges[true_collapse_edges])
                print("We have collapsed", len(true_collapse_edges), "/", n, "edges")
                original_mesh = val_batch.mesh[0]

                save_path1 = "/home/zhuxunyang/coding/bkgm_simplification/result/result_before_smooth_" + name + "_" + str(step) + ".vtk"
                save_path2 = "/home/zhuxunyang/coding/bkgm_simplification/result/result_after_smooth_" + name + "_" + str(step) + ".vtk"
                save_path3 = "/home/zhuxunyang/coding/bkgm_simplification/result/result_sizefield_" + name + "_" + str(step) + ".vtk"
                geo_path = f"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/target/{geo}.step"

                # original_mesh = original_mesh.collapse_multiple_edges(true_collapse_edges)
                save_path4 = f"/home/zhuxunyang/coding/bkgm_simplification/result/{step}_before_visual.vtk"
                original_mesh.visual_edge(true_collapse_edges, save_path4)
                
                t0 = time.time()
                # new_mesh, post_face = original_mesh.collapse_multiple_edges1(true_collapse_edges)
                new_mesh = original_mesh.collapse_multiple_edges(true_collapse_edges)
                # find_nearest_triangles(new_mesh, target_mesh)
                new_mesh.get_all_info()
                total_time['edge_collapse'] = time.time() - t0

                # if not new_mesh.if_manifold():
                #     original_mesh.writeVTK("/home/zhuxunyang/coding/bkgm_simplification/result/error_mesh.vtk")
                #     print("Detect no-manifold mesh, then we re-predict")
                #     no_manifold_times += 1
                #     if no_manifold_times > 3:
                #         n = int(n * 0.9)
                #         no_manifold_times = 0
                #     continue

                if False:
                    print("")
                
                # elif not new_mesh.detect_non_manifold_vertices():
                #     print("Detect no-manifold veretices, then we re-predict")
                #     continue

                # elif original_mesh.check_edge_face_intersections1():
                #     print("Detect face_intersections, then we re-predict")
                #     continue

                else:
                    save_path5 = f"/home/zhuxunyang/coding/bkgm_simplification/result/{step}_after_visual.vtk"
                    original_mesh.visual_edge(true_collapse_edges, save_path5)
                    t0 = time.time()
                    new_mesh.writeVTK(save_path1)
                    original_mesh = new_mesh
                    total_time['mesh_io'] = time.time() - t0
                    
                t0 = time.time()
                # command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', '1e-4', '--max-pass', '10', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
                command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', '1e-2', '--max-pass', '20', '--output', save_path2,'--split-num', '0', '--collapse-num', '0', '--feature-angle', '1']
                try:
                    result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
                    print("remesh success")
                except subprocess.CalledProcessError as e:
                    print("remesh fail")
                    print(e.stderr)
                    continue

                original_mesh = CustomMesh.CustomMesh.from_vtk(save_path2)

                original_mesh.recalculate_size(begin_mesh)

                original_mesh.writeVTK(save_path3)

                # command2 = ['./test_sizefield_noIDmap', save_path2, save_path3, '1.2', '0', "0.1", geo_path]
                # try:
                #     result2 = subprocess.run(command2, check=True, text=True, capture_output=True)
                #     # print(result2.stdout)
                #     print("create size field success", save_path3)
                # except subprocess.CalledProcessError as e:
                #     print("create size field fail")
                #     print(result2.stdout)
                #     print(e.stderr)
                #     continue

                total_time['postprocessing'] = time.time() - t0
                
                t0 = time.time()
                # 更新 original_mesh 和 model_mesh
                # original_mesh = CustomMesh.CustomMesh.from_vtk(save_path3)
                # if not new_mesh.if_manifold():
                #     print("Detect no-manifold mesh, then we re-predict")
                #     continue

                # model_mesh = Mesh(file=save_path3, opt=opt)

                # sampled_points = []
                # for tri in post_face:
                #     sample_p = sampleTriangle([original_mesh.vertices[tri[0]].numpy(), original_mesh.vertices[tri[1]].numpy(), original_mesh.vertices[tri[2]].numpy()])
                #     sampled_points.extend(sample_p)
                                            

                # if len(sampled_points) == 0:
                #     no_banding_rito = 1
                # else:
                #     tree2 = cKDTree(original_mesh.vertices.numpy())
                #     target_s, target_d = batch_project(sampled_points, target_mesh, tree1)
                #     data_s, data_d = batch_project(sampled_points, original_mesh, tree2)
                #     valid_mask = ~np.isnan(target_s) & ~np.isnan(data_s) & ~np.isnan(target_d) & ~np.isnan(data_d)
                #     true_count = np.count_nonzero(valid_mask)
                #     false_count = len(valid_mask) - true_count
                #     size_losses = np.where(valid_mask, data_s / (target_s + 1e-10), np.nan)
                #     mask = (size_losses < 1.3)
                #     banding_mask = (size_losses > 1.3) & (size_losses < 10)
                #     count = np.count_nonzero(mask)
                #     no_banding_rito = count / len(size_losses)

                # if no_banding_rito < 0.96:
                #     print(f"{no_banding_rito} Banding area is too large")
                #     os.remove(save_path1)
                #     os.remove(save_path2)
                #     os.remove(save_path3)
                #     os.remove(save_path4)
                #     continue

                # realtime_num_mesh = sum(original_mesh.num_mesh())
                # realtime_area = sum(original_mesh.get_surface_area()).item()
                # print(collapse_rank, "Real time condition", realtime_num_mesh, realtime_area, "Target condition:", target_num_mesh, target_area, "hausdorff_dis:", hausdorff_distance_max(target_mesh.vertices, original_mesh.vertices), "no_Banding_rito:", no_banding_rito)

                # print(collapse_rank, "hausdorff_dis:", hausdorff_distance_max(target_mesh.vertices, original_mesh.vertices), "no_Banding_rito:", no_banding_rito)
                print(collapse_rank, "hausdorff_dis:", hausdorff_distance_max(target_mesh.vertices, original_mesh.vertices))
                collapse_rank += 1
                
                # rito_num = np.abs(target_num_mesh - realtime_num_mesh) / target_num_mesh
                # rito_area = np.abs(target_area - realtime_area) / target_area

                # if rito_num > 0.1 or rito_area > 0.1:
                #     times = times + 1
                #     if times >= 3:
                #         flag = False
                #     continue


                # 提取新网格的特征并归一化
                node_mean, node_std, edge_mean, edge_std = norm_params
                edge_features = original_mesh.compute_edge_features()
                norm_node = (original_mesh.sizing_values - node_mean) / node_std
                norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
                
                num_nodes = len(original_mesh.vertices)
                # 更新 val_batch
                val_batch.x = norm_node.float().to(device)
                val_batch.edge_index = original_mesh.edges.T.long().to(device)
                val_batch.edge_attr = norm_edge.T.float().to(device)
                val_batch.pos = original_mesh.vertices.to(device)
                val_batch.length = [len(original_mesh.edges)]
                val_batch.mesh = [original_mesh]  # 注意保持 batch 结构
                val_batch.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
                N = N + (len(collapse_edges) * 3)
                total_time['data_update'] = time.time() - t0
                step += 1
                print(f"Collapsing rank {collapse_rank} succeed")
                total_time['per_iteration'] = time.time() - iter_time

                print("\nTime Profiling Results:")
                for key, value in total_time.items():
                    print(f"{key:20}: {value:.2f}s")

    final_process(vtk, Geo_path)
    end_time = time.time()
    print(f"Cost {end_time - start_time:.2f}s")

def test_traditional_method(vtk_path, size, dis, rito, save_path, remesh_envi):
    print("Tradition method testing")
    start_time = time.time()
    L1 = nn.L1Loss()
    collapse_ranks_list = []
    l1_losses_list = []
    distance_list = []
    end_mesh = None
    
    # 初始化时间统计变量
    total_time = 0
    load_mesh_time = 0
    laplace_time = 0
    filter_edges_time = 0
    collapse_eval_time = 0
    collapse_true_time = 0
    remesh_time = 0
    other_time = 0
    
    mesh_load_start = time.time()
    mesh = CustomMesh.CustomMesh.from_vtk(vtk_path)
    begin_mesh = mesh
    load_mesh_time += time.time() - mesh_load_start
    
    name = re.split("/", vtk_path)[-1].split('.')[0]
    geo = vtk_path.split('/')[-1].split('_')[0]
    step = 0
    collapse_rank = 0
    
    element_num = []
    query_cell_num = []

    target_load_start = time.time()
    target_path = f"/home/zhuxunyang/coding/simply/datasets/training/target/{geo}_target.vtk"
    target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
    tree1 = cKDTree(target_mesh.vertices.numpy())
    load_mesh_time += time.time() - target_load_start
    
    while True:
        # mesh = smooth_mesh_sizing(mesh, len(mesh.vertices), 1.2, torch.min(mesh.sizing_values))
        element_num.append(sum(mesh.num_mesh()).item())
        query_cell_num.append(len(mesh.faces))
        n_edges = mesh.edges.size(0)
        pre_select_num = int(n_edges * rito)   #25
        print(f"We will pre-select {pre_select_num} edges")

        iteration_start = time.time()
        laplace_start = time.time()

        _, lbo_edge = mesh.compute_LBO()
        laplace_time += time.time() - laplace_start

        filter_start = time.time()
        sorted_indices = np.argsort(lbo_edge)
            
        edges = mesh.edges.cpu().numpy()
        point_set = set()
        banned_vertices = set()
        vertex_edge_map = defaultdict(list)
        for i, edge in enumerate(mesh.edges):
            v1, v2 = edges[i]
            vertex_edge_map[v1].append(i)
            vertex_edge_map[v2].append(i)
        edge_ids = []

        #trad
        for index in sorted_indices:
            v1, v2 = edges[index]
            convex1 = mesh.convex[v1]
            convex2 = mesh.convex[v2]
            normals1 = mesh.vertex_normal[v1]
            normals2 = mesh.vertex_normal[v2]
            dots = normals1.dot(normals2)
            angles_deg = torch.rad2deg(torch.acos(dots))
            if (
                v1 not in point_set and 
                v2 not in point_set and 
                # v1 not in banned_vertices and 
                # v2 not in banned_vertices and 
                (~(mesh.feature_point[v1] & mesh.feature_point[v2]) | ((convex1 == convex2) & (convex1 != 0))) and 
                (angles_deg < 10)):
                edge_ids.append(index)
                point_set.update([v1, v2])
                for v in [v1, v2]:
                    for neighbor_edge in vertex_edge_map[v]:
                        nv1, nv2 = edges[neighbor_edge]
                        banned_vertices.add(nv1)
                        banned_vertices.add(nv2)
            if len(edge_ids) > pre_select_num:
                break
    

        # true_collapse_edges = filter_feature_edge2(mesh, edge_ids, angle)
        # mesh.visual_edge(edge_ids, "/home/zhuxunyang/coding/bkgm_simplification/visual_lpls_edge.vtk")
        true_collapse_edges = edge_ids
        filter_edges_time += time.time() - filter_start

        if len(true_collapse_edges) == 0:
            t0 = time.time()
            smooth_mesh_sizing(mesh, len(mesh.vertices), beta=1.2, h_min= torch.min(mesh.sizing_values))
            print(f"Smooth cost {time.time() - t0} s")
            mesh.writeVTK(f"{save_path}/final_simply.vtk")
            break

        # 3. 边折叠评估
        collapse_eval_start = time.time()
        operate_edge_ids = []
        collapse_one_time = 0
        re_size_time = 0
        inner_sample_time = 0
        kdtree_time = 0
        project_time = 0
        calculate_loss_time = 0

        print("Pre-selecting")
        # operate_edge_ids = parallel_process_edges(true_collapse_edges, mesh, target_mesh, tree1, max_workers=16)
        # operate_edge_ids, Time = process_edges_in_parallel1(true_collapse_edges, mesh, target_mesh, tree1, size, dis, max_workers=24)
        # operate_edge_ids, Time = process_edges_in_parallel2(true_collapse_edges, mesh, target_mesh, tree1, size, dis, max_workers=16)
        # print(f"Complete pre-selecting. Cost {time.time() - collapse_eval_start}s")
        operate_edge_ids, Time = process_edges_no_parallel(true_collapse_edges, mesh, target_mesh, tree1, size, dis)
        # operate_edge_ids, Time = process_edges_parallel(true_collapse_edges, mesh, target_mesh, tree1, size, dis)
        
        if len(operate_edge_ids) == 0:
            t0 = time.time()
            smooth_mesh_sizing(mesh, len(mesh.vertices), beta=1.2, h_min= torch.min(mesh.sizing_values))
            print(f"Smooth cost {time.time() - t0} s")
            mesh.writeVTK(f"{save_path}/final_simply.vtk")
            break

        collapse_one_time = Time['collapse']
        re_size_time = Time['resize']
        inner_sample_time = Time['sample']
        kdtree_time = Time['kdtree']
        project_time = Time['project']
        calculate_loss_time = Time['loss']

        collapse_eval_time += time.time() - collapse_eval_start
        print(f"Edge evaluation breakdown:")
        print(f"  - Collapse one time: {collapse_one_time:.3f}s")
        print(f"  - Resize time: {re_size_time:.3f}s")
        print(f"  - Sampling time: {inner_sample_time:.3f}s")
        print(f"  - KDTree time: {kdtree_time:.3f}s")
        print(f"  - Projection time: {project_time:.3f}s")
        print(f"  - Calculate loss time: {calculate_loss_time:.3f}s")

        print(f"We will collapse {len(operate_edge_ids)}/{len(edge_ids)} edges")
        
        

        save_path1 = save_path + "/result_before_smooth_" + name + "_" + str(step) + ".vtk"
        save_path2 = save_path + "/result_after_smooth_" + name + "_" + str(step) + ".vtk"
        save_path3 = save_path + "/result_sizefield_" + name + "_" + str(step) + ".vtk"
        save_path5 = f"{save_path}/{name}_{step}_after_visual.vtk"

        # save_path1 = "/home/zhuxunyang/coding/bkgm_simplification/result/result_before_smooth_" + name + "_" + str(step) + ".vtk"
        # save_path2 = "/home/zhuxunyang/coding/bkgm_simplification/result/result_after_smooth_" + name + "_" + str(step) + ".vtk"
        # save_path3 = "/home/zhuxunyang/coding/bkgm_simplification/result/result_sizefield_" + name + "_" + str(step) + ".vtk"
        # save_path5 = f"/home/zhuxunyang/coding/bkgm_simplification/result/{step}_after_visual.vtk"

        # 4. 执行边折叠
        collapse_start = time.time()
        
        operate_edge_ids = sorted(
            operate_edge_ids,
            key=lambda eid: max(edges[eid]),
            reverse=True
        )

        mesh.visual_edge(operate_edge_ids, save_path5)
        this_mesh = mesh
        try:
            mesh = mesh.collapse_multiple_edges2(operate_edge_ids)
        except:
            break
        # mesh, _ = mesh.collapse_multiple_edges1(operate_edge_ids)
        mesh.get_all_info()
        collapse_time = time.time() - collapse_start
        collapse_true_time += collapse_time

        # 5. 重网格化和后处理
        remesh_start = time.time()

        mesh.writeVTK(save_path1)

        # if collapse_rank % 1 == 0:
        # if True:
        #     command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', '1e-4', '--max-pass', '5', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
        #     # command1 = ['./remesh', '--input', save_path1, '--eps', '1e-6', '--envelope-dis', '4e-4', '--max-pass', '20', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
        #     try:
        #         remesh_cmd_start = time.time()
        #         result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
        #         remesh_cmd_time = time.time() - remesh_cmd_start
        #         mesh = CustomMesh.CustomMesh.from_vtk(save_path2)
        #         print("remesh success")
        #     except subprocess.CalledProcessError as e:
        #         print("remesh fail")
        #         print(e.stderr)
        #         mesh = CustomMesh.CustomMesh.from_vtk(save_path1)
        #         # continue

        command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', remesh_envi, '--max-pass', '20', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']    #3e-4
        # command1 = ['./remesh', '--input', save_path1, '--eps', '1e-6', '--envelope-dis', '4e-4', '--max-pass', '20', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
        try:
            remesh_cmd_start = time.time()
            result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
            remesh_cmd_time = time.time() - remesh_cmd_start
            mesh = CustomMesh.CustomMesh.from_vtk(save_path2)
            print("remesh success")
        except subprocess.CalledProcessError as e:
            print("remesh fail")
            print(e.stderr)
            mesh = CustomMesh.CustomMesh.from_vtk(save_path1)
            # continue

        mesh_load_start = time.time()
        mesh.recalculate_size(begin_mesh)
        mesh.writeVTK(save_path3)
        load_mesh_time += time.time() - mesh_load_start
        
        remesh_time += time.time() - remesh_start

        # print(f"Remesh breakdown:")
        # print(f"  - Remesh command time: {remesh_cmd_time:.3f}s")
        # print(f"  - Mesh reload time: {time.time() - mesh_load_start:.3f}s")

        result_size, result_dis = begin_mesh.L1_size(mesh)
        true_size = begin_mesh.sizing_values

        L1loss = L1(result_size, true_size).item()
        distance_loss = torch.sum(result_dis).item() / len(result_dis)
        l1_losses_list.append(L1loss)
        distance_list.append(distance_loss)
        collapse_ranks_list.append(collapse_rank)
        # plot_tend(collapse_ranks_list, l1_losses_list, f"{save_path}/{collapse_rank}_tend.png")
        print(collapse_rank, "L1 Loss:", L1loss, "Distance:", distance_loss, "hausdorff_dis:", hausdorff_distance_max(target_mesh.vertices, mesh.vertices))
        collapse_rank += 1
        step += 1
        
        iteration_time = time.time() - iteration_start
        other_time = iteration_time - (laplace_time + filter_edges_time + collapse_eval_time + remesh_time)
        
        print(f"\n=== Iteration {step} Time Breakdown ===")
        print(f"Total iteration time: {iteration_time:.3f}s")
        print(f"  - Laplace computation: {laplace_time:.3f}s ({laplace_time/iteration_time*100:.1f}%)")
        print(f"  - Edge filtering: {filter_edges_time:.3f}s ({filter_edges_time/iteration_time*100:.1f}%)")
        print(f"  - Collapse evaluation: {collapse_eval_time:.3f}s ({collapse_eval_time/iteration_time*100:.1f}%)")
        print(f"  - Collapse true: {collapse_true_time:.3f}s ({collapse_true_time/iteration_time*100:.1f}%)")
        print(f"  - Remeshing: {remesh_time:.3f}s ({remesh_time/iteration_time*100:.1f}%)")
        print(f"  - Other: {other_time:.3f}s ({other_time/iteration_time*100:.1f}%)")
        print("=" * 50 + "\n")
        
        # 重置本迭代的时间统计
        laplace_time = 0
        filter_edges_time = 0
        collapse_eval_time = 0
        collapse_true_time = 0
        remesh_time = 0
        other_time = 0
        print(f"{collapse_rank} cost {time.time() - start_time}s")
    
    
    total_time = time.time() - start_time
    print(f"\n=== Total Execution Time ===")
    print(f"Total time: {total_time:.3f}s")
    end_time = time.time()
    print(f"Cost {end_time - start_time:.2f}s")

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

    # 方法1：使用索引作为横坐标
    x_indices = range(len(query_cell_num))
    ax1.plot(x_indices, list(element_num), 'b-', linewidth=2, label='Element num')

    # 设置横坐标标签为query_cell_num的值，但保持顺序
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f'{x:.0f}' for x in query_cell_num], rotation=45)

    ax1.set_xlabel('Query cell')
    ax1.set_ylabel('Element num')
    ax1.set_title('Q-E')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 使用对数坐标，便于观察变化

    plt.tight_layout()

    # 保存图片
    path = f"{save_path}/convergence.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"收敛曲线图已保存至: {path}")

    # 保存数据到CSV文件
    data_to_save = {
        'list_index': list(range(len(query_cell_num))),  # 列表索引 0, 1, 2, ...
        'element_num': list(element_num),                # element_num值
        'query_cell_num': list(query_cell_num)           # 也可以保存query_cell_num
    }

    df = pd.DataFrame(data_to_save)
    csv_path = f"{save_path}/convergence_data.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"数据已保存至CSV文件: {csv_path}")


def test_traditional_classify(vtk_path, size, dis, angle, save_path):
    print("Tradition method testing")
    start_time = time.time()
    L1 = nn.L1Loss()
    collapse_ranks_list = []
    l1_losses_list = []
    distance_list = []
    
    # 初始化时间统计变量
    total_time = 0
    load_mesh_time = 0
    laplace_time = 0
    filter_edges_time = 0
    collapse_eval_time = 0
    collapse_true_time = 0
    remesh_time = 0
    other_time = 0
    
    mesh_load_start = time.time()
    mesh = CustomMesh.CustomMesh.from_vtk(vtk_path)
    begin_mesh = mesh
    load_mesh_time += time.time() - mesh_load_start
    
    name = re.split("/", vtk_path)[-1].split('.')[0]
    geo = vtk_path.split('/')[-1].split('_')[0]
    step = 0
    collapse_rank = 0
    
    target_load_start = time.time()
    target_path = f"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/target/{geo}_target.vtk"
    target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
    tree1 = cKDTree(target_mesh.vertices.numpy())
    load_mesh_time += time.time() - target_load_start
    
    while True:
        n_edges = mesh.edges.size(0)
        pre_select_num = int(n_edges / 5)   #25
        print(f"We will pre-select {pre_select_num} edges")

        iteration_start = time.time()
        laplace_start = time.time()

        _, lbo_edge = mesh.compute_LBO()
        laplace_time += time.time() - laplace_start

        filter_start = time.time()
        sorted_indices = np.argsort(lbo_edge)
            
        edges = mesh.edges.cpu().numpy()
        point_set = set()
        banned_vertices = set()
        vertex_edge_map = defaultdict(list)
        for i, edge in enumerate(mesh.edges):
            v1, v2 = edges[i]
            vertex_edge_map[v1].append(i)
            vertex_edge_map[v2].append(i)
        edge_ids = []

        #trad
        for index in sorted_indices:
            v1, v2 = edges[index]
            convex1 = mesh.convex[v1]
            convex2 = mesh.convex[v2]
            normals1 = mesh.vertex_normal[v1]
            normals2 = mesh.vertex_normal[v2]
            dots = normals1.dot(normals2)
            angles_deg = torch.rad2deg(torch.acos(dots))
            if (
                # v1 not in point_set and 
                # v2 not in point_set and 
                # v1 not in banned_vertices and 
                # v2 not in banned_vertices and 
                (~(mesh.feature_point[v1] & mesh.feature_point[v2]) | ((convex1 == convex2) & (convex1 != 0))) and 
                (angles_deg < angle)):
                edge_ids.append(index)
                point_set.update([v1, v2])
                for v in [v1, v2]:
                    for neighbor_edge in vertex_edge_map[v]:
                        nv1, nv2 = edges[neighbor_edge]
                        banned_vertices.add(nv1)
                        banned_vertices.add(nv2)
            if len(edge_ids) > pre_select_num:
                break
    

        # true_collapse_edges = filter_feature_edge2(mesh, edge_ids, angle)
        # mesh.visual_edge(edge_ids, "/home/zhuxunyang/coding/bkgm_simplification/visual_lpls_edge.vtk")
        true_collapse_edges = edge_ids
        filter_edges_time += time.time() - filter_start

        if len(true_collapse_edges) == 0:
            break

        # 3. 边折叠评估
        collapse_eval_start = time.time()
        operate_edge_ids = []
        collapse_one_time = 0
        re_size_time = 0
        inner_sample_time = 0
        kdtree_time = 0
        project_time = 0
        calculate_loss_time = 0

        print("Pre-selecting")
        # operate_edge_ids = parallel_process_edges(true_collapse_edges, mesh, target_mesh, tree1, max_workers=16)
        # operate_edge_ids, Time = process_edges_in_parallel1(true_collapse_edges, mesh, target_mesh, tree1, size, dis, max_workers=24)
        # operate_edge_ids, Time = process_edges_in_parallel2(true_collapse_edges, mesh, target_mesh, tree1, size, dis, max_workers=16)
        # print(f"Complete pre-selecting. Cost {time.time() - collapse_eval_start}s")
        operate_edge_ids, Time = process_edges_no_parallel(true_collapse_edges, mesh, target_mesh, tree1, size, dis)
        # operate_edge_ids, Time = process_edges_parallel(true_collapse_edges, mesh, target_mesh, tree1, size, dis)
        
        if len(operate_edge_ids) == 0:
            break

        collapse_one_time = Time['collapse']
        re_size_time = Time['resize']
        inner_sample_time = Time['sample']
        kdtree_time = Time['kdtree']
        project_time = Time['project']
        calculate_loss_time = Time['loss']

        collapse_eval_time += time.time() - collapse_eval_start
        print(f"Edge evaluation breakdown:")
        print(f"  - Collapse one time: {collapse_one_time:.3f}s")
        print(f"  - Resize time: {re_size_time:.3f}s")
        print(f"  - Sampling time: {inner_sample_time:.3f}s")
        print(f"  - KDTree time: {kdtree_time:.3f}s")
        print(f"  - Projection time: {project_time:.3f}s")
        print(f"  - Calculate loss time: {calculate_loss_time:.3f}s")

        print(f"We will collapse {len(operate_edge_ids)}/{len(edge_ids)} edges")
        
        

        save_path1 = save_path + "/result_before_smooth_" + name + "_" + str(step) + ".vtk"
        save_path2 = save_path + "/result_after_smooth_" + name + "_" + str(step) + ".vtk"
        save_path3 = save_path + "/result_sizefield_" + name + "_" + str(step) + ".vtk"
        save_path5 = f"{save_path}/{name}_{step}_after_visual.vtk"

        # save_path1 = "/home/zhuxunyang/coding/bkgm_simplification/result/result_before_smooth_" + name + "_" + str(step) + ".vtk"
        # save_path2 = "/home/zhuxunyang/coding/bkgm_simplification/result/result_after_smooth_" + name + "_" + str(step) + ".vtk"
        # save_path3 = "/home/zhuxunyang/coding/bkgm_simplification/result/result_sizefield_" + name + "_" + str(step) + ".vtk"
        # save_path5 = f"/home/zhuxunyang/coding/bkgm_simplification/result/{step}_after_visual.vtk"

        # 4. 执行边折叠
        collapse_start = time.time()
        
        operate_edge_ids = sorted(
            operate_edge_ids,
            key=lambda eid: max(edges[eid]),
            reverse=True
        )

        mesh.visual_edge(operate_edge_ids, save_path5)

def test_random_method(vtk_path, angle, save_path,n):
    print("Tradition method testing")
    start_time = time.time()
    L1 = nn.L1Loss()
    collapse_ranks_list = []
    l1_losses_list = []
    distance_list = []
    
    # 初始化时间统计变量
    total_time = 0
    load_mesh_time = 0
    laplace_time = 0
    filter_edges_time = 0
    collapse_eval_time = 0
    collapse_true_time = 0
    remesh_time = 0
    other_time = 0
    
    mesh_load_start = time.time()
    mesh = CustomMesh.CustomMesh.from_vtk(vtk_path)
    begin_mesh = mesh
    load_mesh_time += time.time() - mesh_load_start
    
    name = re.split("/", vtk_path)[-1].split('.')[0]
    geo = vtk_path.split('/')[-1].split('_')[0]
    step = 0
    collapse_rank = 0
    
    target_load_start = time.time()
    target_path = f"/home/zhuxunyang/coding/simply/datasets/training/target/{geo}_target.vtk"
    target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
    tree1 = cKDTree(target_mesh.vertices.numpy())
    load_mesh_time += time.time() - target_load_start
    
    element_num = []
    query_cell_num = []

    while begin_mesh.faces.size(0) / mesh.faces.size(0) < n:
        element_num.append(sum(mesh.num_mesh()).item())
        query_cell_num.append(len(mesh.faces))
        n_edges = mesh.edges.size(0)
        pre_select_num = int(n_edges / 20)
        print(f"We will pre-select {pre_select_num} edges")

        iteration_start = time.time()
        laplace_start = time.time()

        # _, lbo_edge = mesh.compute_LBO()
        # laplace_time += time.time() - laplace_start

        filter_start = time.time()
        # sorted_indices = np.argsort(lbo_edge)
            
        edges = mesh.edges.cpu().numpy()
        point_set = set()
        banned_vertices = set()
        vertex_edge_map = defaultdict(list)
        for i, edge in enumerate(mesh.edges):
            v1, v2 = edges[i]
            vertex_edge_map[v1].append(i)
            vertex_edge_map[v2].append(i)
        edge_ids = []

        select_history = set()
        #random
        while len(select_history) < n_edges:
            index = random.randint(0, n_edges - 1)
            if index in select_history:
                # print(f"{index} has selected")
                continue
            select_history.add(index)
            v1, v2 = edges[index]
            convex1 = mesh.convex[v1]
            convex2 = mesh.convex[v2]
            normals1 = mesh.vertex_normal[v1]
            normals2 = mesh.vertex_normal[v2]
            dots = normals1.dot(normals2)
            angles_deg = torch.rad2deg(torch.acos(dots))
            # if (v1 not in point_set and 
            #     v2 not in point_set):
                # (mesh.feature_point[v1] == mesh.feature_point[v2]) and   #v1 not in banned_vertices and v2 not in banned_vertice  ((convex1 == convex2) & (convex1 != 0)) and (angles_deg < angle))
            if (v1 not in point_set and 
                v2 not in point_set and 
                # v1 not in banned_vertices and 
                # v2 not in banned_vertices and 
                ((convex1 == convex2) & (convex1 != 0)) and 
                (angles_deg < angle)):
                edge_ids.append(index)
                point_set.update([v1, v2])
                for v in [v1, v2]:
                    for neighbor_edge in vertex_edge_map[v]:
                        nv1, nv2 = edges[neighbor_edge]
                        banned_vertices.add(nv1)
                        banned_vertices.add(nv2)
            if len(edge_ids) >= pre_select_num:
                break

        # true_collapse_edges = filter_feature_edge2(mesh, edge_ids, angle)
        true_collapse_edges = edge_ids
        filter_edges_time += time.time() - filter_start

        if len(true_collapse_edges) == 0:
            print("no edge")
            break

        # 3. 边折叠评估
        collapse_eval_start = time.time()
        collapse_one_time = 0
        re_size_time = 0
        inner_sample_time = 0
        kdtree_time = 0
        project_time = 0
        calculate_loss_time = 0

        print("Pre-selecting")
        # operate_edge_ids = parallel_process_edges(true_collapse_edges, mesh, target_mesh, tree1, max_workers=16)
        # operate_edge_ids, Time = process_edges_in_parallel1(true_collapse_edges, mesh, target_mesh, tree1, size, dis, max_workers=24)
        # operate_edge_ids, Time = process_edges_in_parallel2(true_collapse_edges, mesh, target_mesh, tree1, size, dis, max_workers=16)
        # print(f"Complete pre-selecting. Cost {time.time() - collapse_eval_start}s")
        # operate_edge_ids, Time = process_edges_no_parallel(true_collapse_edges, mesh, target_mesh, tree1, size, dis)
        # operate_edge_ids, Time = process_edges_parallel(true_collapse_edges, mesh, target_mesh, tree1, size, dis)
        
        operate_edge_ids = edge_ids
        if len(operate_edge_ids) == 0:
            break

        # collapse_one_time = Time['collapse']
        # re_size_time = Time['resize']
        # inner_sample_time = Time['sample']
        # kdtree_time = Time['kdtree']
        # project_time = Time['project']
        # calculate_loss_time = Time['loss']

        collapse_eval_time += time.time() - collapse_eval_start
        print(f"Edge evaluation breakdown:")
        print(f"  - Collapse one time: {collapse_one_time:.3f}s")
        print(f"  - Resize time: {re_size_time:.3f}s")
        print(f"  - Sampling time: {inner_sample_time:.3f}s")
        print(f"  - KDTree time: {kdtree_time:.3f}s")
        print(f"  - Projection time: {project_time:.3f}s")
        print(f"  - Calculate loss time: {calculate_loss_time:.3f}s")

        print(f"We will collapse {len(operate_edge_ids)}/{len(edge_ids)} edges")
        
        save_path1 = f"{save_path}/result_before_smooth_" + name + "_" + str(step) + ".vtk"
        save_path2 = f"{save_path}/result_after_smooth_" + name + "_" + str(step) + ".vtk"
        save_path3 = f"{save_path}/result_sizefield_" + name + "_" + str(step) + ".vtk"
        save_path5 = f"{save_path}/{step}_after_visual.vtk"

        # 4. 执行边折叠
        collapse_start = time.time()
        
        operate_edge_ids = sorted(
            operate_edge_ids,
            key=lambda eid: max(edges[eid]),
            reverse=True
        )

        mesh.visual_edge(operate_edge_ids, save_path5)
        mesh = mesh.collapse_multiple_edges(operate_edge_ids)
        # mesh, _ = mesh.collapse_multiple_edges1(operate_edge_ids)
        mesh.get_all_info()
        collapse_time = time.time() - collapse_start
        collapse_true_time += collapse_time

        # 5. 重网格化和后处理
        remesh_start = time.time()

        mesh.writeVTK(save_path1)

        command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', '1e-3', '--max-pass', '5', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
        # command1 = ['./remesh', '--input', save_path1, '--eps', '1e-6', '--envelope-dis', '4e-4', '--max-pass', '20', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
        try:
            remesh_cmd_start = time.time()
            result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
            remesh_cmd_time = time.time() - remesh_cmd_start
            print("remesh success")
        except subprocess.CalledProcessError as e:
            print("remesh fail")
            os.remove(save_path1)
            print(e.stderr)
            continue

        mesh_load_start = time.time()
        mesh = CustomMesh.CustomMesh.from_vtk(save_path2)
        mesh.recalculate_size(begin_mesh)
        mesh.writeVTK(save_path3)
        load_mesh_time += time.time() - mesh_load_start
        
        remesh_time += time.time() - remesh_start

        # print(f"Remesh breakdown:")
        # print(f"  - Remesh command time: {remesh_cmd_time:.3f}s")
        # print(f"  - Mesh reload time: {time.time() - mesh_load_start:.3f}s")

        result_size, result_dis = begin_mesh.L1_size(mesh)
        true_size = begin_mesh.sizing_values

        L1loss = L1(result_size, true_size).item()
        distance_loss = torch.sum(result_dis).item() / len(result_dis)
        l1_losses_list.append(L1loss)
        distance_list.append(distance_loss)
        collapse_ranks_list.append(collapse_rank)
        plot_tend(collapse_ranks_list, l1_losses_list, f"{save_path}/{collapse_rank}_tend.png")
        print(collapse_rank, "L1 Loss:", L1loss, "Distance:", distance_loss, "hausdorff_dis:", hausdorff_distance_max(target_mesh.vertices, mesh.vertices))
        collapse_rank += 1
        step += 1
        
        iteration_time = time.time() - iteration_start
        other_time = iteration_time - (laplace_time + filter_edges_time + collapse_eval_time + remesh_time)
        
        print(f"\n=== Iteration {step} Time Breakdown ===")
        print(f"Total iteration time: {iteration_time:.3f}s")
        print(f"  - Laplace computation: {laplace_time:.3f}s ({laplace_time/iteration_time*100:.1f}%)")
        print(f"  - Edge filtering: {filter_edges_time:.3f}s ({filter_edges_time/iteration_time*100:.1f}%)")
        print(f"  - Collapse evaluation: {collapse_eval_time:.3f}s ({collapse_eval_time/iteration_time*100:.1f}%)")
        print(f"  - Collapse true: {collapse_true_time:.3f}s ({collapse_true_time/iteration_time*100:.1f}%)")
        print(f"  - Remeshing: {remesh_time:.3f}s ({remesh_time/iteration_time*100:.1f}%)")
        print(f"  - Other: {other_time:.3f}s ({other_time/iteration_time*100:.1f}%)")
        print("=" * 50 + "\n")
        
        # 重置本迭代的时间统计
        laplace_time = 0
        filter_edges_time = 0
        collapse_eval_time = 0
        collapse_true_time = 0
        remesh_time = 0
        other_time = 0

    total_time = time.time() - start_time
    print(f"\n=== Total Execution Time ===")
    print(f"Total time: {total_time:.3f}s")
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

    # 方法1：使用索引作为横坐标
    x_indices = range(len(query_cell_num))
    ax1.plot(x_indices, list(element_num), 'b-', linewidth=2, label='Element num')

    # 设置横坐标标签为query_cell_num的值，但保持顺序
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f'{x:.0f}' for x in query_cell_num], rotation=45)

    ax1.set_xlabel('Query cell')
    ax1.set_ylabel('Element num')
    ax1.set_title('Q-E')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 使用对数坐标，便于观察变化

    plt.tight_layout()

    # 保存图片
    path = f"{save_path}/convergence.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"收敛曲线图已保存至: {path}")

    # 保存数据到CSV文件
    data_to_save = {
        'list_index': list(range(len(query_cell_num))),  # 列表索引 0, 1, 2, ...
        'element_num': list(element_num),                # element_num值
        'query_cell_num': list(query_cell_num)           # 也可以保存query_cell_num
    }

    df = pd.DataFrame(data_to_save)
    csv_path = f"{save_path}/convergence_data.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"数据已保存至CSV文件: {csv_path}")

def test_example1(vtk_path):
    opt = TestOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, norm_params = create_example(vtk_path, opt, device, load_para())
    model = EdgeRankingGNN(
        node_feat_dim=1,
        edge_feat_dim=5,
        hidden_dim=64
    ).to(device=device)


    total_steps = 0

    checkpoint = torch.load(
        # "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Regression3.pth",
        "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Regression_size.pth",
        weights_only=False
    )

    name = re.split("/", vtk_path)[-1].split('.')[0]

    model.load_state_dict(checkpoint)
    i = 0
    collapse_rank = 0
    if_end = False
    for val_batch in dataset:
        with torch.no_grad():
            edge_count = len(val_batch.mesh[0].edges)
            N = 0
            step = 0
            geo = val_batch.geo[0]
            target_path = f"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/target/{geo}_target.vtk"
            target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
            tree1 = cKDTree(target_mesh.vertices.numpy())
            target_num_mesh = sum(target_mesh.num_mesh())
            target_area = sum(target_mesh.get_surface_area()).item()
            print("Target condition:", target_num_mesh, target_area)
            realtime_num_mesh = sum(val_batch.mesh[0].num_mesh())
            realtime_area = sum(val_batch.mesh[0].get_surface_area()).item()
            print("Begin time condition", realtime_num_mesh, realtime_area)
            if not val_batch.mesh[0].if_manifold():
                print("No-manifold mesh")
                break
            flag = True
            times = 0
            # while N < edge_count // 2:
            while flag:
                val_out = model(val_batch)
                _, sort_pred = torch.topk(val_out[:, 0], 20, dim=0, largest=False)
                collapse_edges = list(np.array(sort_pred.cpu()))
                original_mesh = val_batch.mesh[0]
                collapse_edges_v = original_mesh.edges[collapse_edges]
                print("We will collapse", len(collapse_edges), "edges", collapse_edges)
                true_collasping_edge = 0
                i = 0

                # for n, edge in enumerate(collapse_edges):
                while i < len(collapse_edges):
                    i += 1
                    # v1, v2 = original_mesh.edges[collapse_edges[n]]
                    v1, v2 = collapse_edges_v[i - 1]
                    if v1.item() == v2.item():
                        print(f"{v1.item()} and {v2.item()} can't be same")
                        continue

                    sorted_edge = torch.tensor(sorted((v1.item(), v2.item())))
                    if not any(torch.equal(sorted_edge, edge) for edge in original_mesh.edges):
                        print(f"{v1.item()} and {v2.item()} is not in edges")
                        continue
                    if original_mesh.feature_point[v1] and original_mesh.feature_point[v2]:
                        print(f"Feature edge {v1.item()} and {v2.item()} collapsed reject")
                        continue

                    new_mesh, _, remain_v, delete_v, post_face = original_mesh.collapsing_edge_new(v1, v2)
                    if new_mesh is None:
                        print(f"{v1.item()} and {v2.item()} collapse failed")
                        continue
                    
                    if not new_mesh.if_manifold():
                        print(f"Collapsed {v1.item()} and {v2.item()} will cause no-manifold mesh, we won't operate")
                        continue
                    else:
                        # sampled_points = []
                        # for tri in post_face:
                        #     sample_p = sampleTriangle([new_mesh.vertices[tri[0]].numpy(), new_mesh.vertices[tri[1]].numpy(), new_mesh.vertices[tri[2]].numpy()])
                        #     sampled_points.extend(sample_p)
                        # if len(sampled_points) == 0:
                        #     print(f"warning {v1.item()} and {v2.item()}")
                        #     continue
                        # tree2 = cKDTree(new_mesh.vertices.numpy())
                        # target_s, target_d = batch_project(sampled_points, target_mesh, tree1)
                        # data_s, data_d = batch_project(sampled_points, new_mesh, tree2)
                        # valid_mask = ~np.isnan(target_s) & ~np.isnan(data_s) & ~np.isnan(target_d) & ~np.isnan(data_d)
                        # size_losses = np.where(valid_mask, target_s - data_s, np.nan)
                        # dis_losses = np.where(valid_mask, target_d, np.nan)
                        # loss1 = np.mean(size_losses)
                        # loss2 = np.mean(dis_losses)
                        loss1 = 0
                        loss2 = 0

                        if loss2 > 0.15:
                            print(f"{v1.item()} and {v2.item()} collapsed will cause huge loss will be skipped. Size_loss:{loss1} Distance_loss:{loss2}, {new_mesh.get_all_info()}")
                            continue
                        else:
                            true_collasping_edge = true_collasping_edge + 1
                            original_mesh = new_mesh
                            print(f"{v1.item()} and {v2.item()} collapsed will only cause Size_loss:{loss1} Distance_loss:{loss2}. hausdorff_dis:{hausdorff_distance_max(target_mesh.vertices, original_mesh.vertices)}")
                            collapse_rank += 1
                            collapse_edges_v[i:][collapse_edges_v[i:] > delete_v] -= 1
                            # original_mesh.writeVTK(f"/home/zhuxunyang/coding/bkgm_simplification/result/result_sizefield_{name}_rank{collapse_rank}_{v1.item()}and{v2.item()}_{loss1:.4f}_{loss2:.4f}.vtk")

                save_path1 = "/home/zhuxunyang/coding/bkgm_simplification/result/result_before_smooth_" + name + "_" + str(step) + ".vtk"
                save_path2 = "/home/zhuxunyang/coding/bkgm_simplification/result/result_after_smooth_" + name + "_" + str(step) + ".vtk"
                save_path3 = "/home/zhuxunyang/coding/bkgm_simplification/result/result_sizefield_" + name + "_" + str(step) + ".vtk"
                geo_path = f"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/target/{geo}.step"

                original_mesh.get_all_info()
                original_mesh.writeVTK(save_path1)
                
                command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', '1e-4', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
                try:
                    result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
                    print("remesh success")
                except subprocess.CalledProcessError as e:
                    print("remesh fail")
                    print(e.stderr)
                    continue

                command2 = ['./test_sizefield_noIDmap', save_path2, save_path3, '1.2', '0', "0.1", geo_path]
                try:
                    result2 = subprocess.run(command2, check=True, text=True, capture_output=True)
                    # print(result2.stdout)
                    print("create size field success")
                except subprocess.CalledProcessError as e:
                    print("create size field fail")
                    print(result2.stdout)
                    print(e.stderr)
                    continue
                
                # 更新 original_mesh 和 model_mesh
                original_mesh = CustomMesh.CustomMesh.from_vtk(save_path3)
                try:
                    model_mesh = Mesh(file=save_path3, opt=opt)
                except:
                    print("Construct Mesh error")
                    continue

                realtime_num_mesh = sum(original_mesh.num_mesh())
                realtime_area = sum(original_mesh.get_surface_area()).item()
                print("Real time condition", realtime_num_mesh, realtime_area, "Target condition:", target_num_mesh, target_area)
                
                rito_num = np.abs(target_num_mesh - realtime_num_mesh) / target_num_mesh
                rito_area = np.abs(target_area - realtime_area) / target_area

                if true_collasping_edge < 3:
                    times = times + 1
                    if times >= 3:
                        flag = False
                    continue


                # 提取新网格的特征并归一化
                node_mean, node_std, edge_mean, edge_std = norm_params
                edge_features = model_mesh.extract_features()
                norm_node = (model_mesh.size_value - node_mean) / node_std
                norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
                
                num_nodes = len(model_mesh.vs)
                # 更新 val_batch
                val_batch.x = torch.from_numpy(norm_node).float().to(device)
                val_batch.edge_index = torch.from_numpy(model_mesh.edges).T.long().to(device)
                val_batch.edge_attr = torch.from_numpy(norm_edge).T.float().to(device)
                val_batch.pos = torch.from_numpy(model_mesh.vs).to(device)
                val_batch.length = [len(model_mesh.edges)]
                val_batch.mesh = [original_mesh]  # 注意保持 batch 结构
                val_batch.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
                N = N + (len(collapse_edges) * 3)
                
                step += 1

def test_example2(vtk_path, save_path, max_score, rito, model_path, remesh_eps):
    plt.close('all')
    total_time = defaultdict(float)
    L1 = nn.L1Loss()
    collapse_ranks_list = []
    l1_losses_list = []
    distance_list = []
    start_time = time.time()
    opt = TestOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, norm_params = create_example1(vtk_path, opt, device, load_para())
    # model = EdgeRankingGNN(
    #     node_in_dim=2,
    #     edge_in_dim=8,
    #     hidden_dim=64
    # ).to(device=device)
    model = EdgeRankingGNN2(
        node_in_dim=2,
        edge_in_dim=8,
        hidden_dim=64
    ).to(device=device)

    mesh = CustomMesh.CustomMesh.from_vtk(vtk_path)
    begin_mesh = mesh

    total_steps = 0

    checkpoint = torch.load(model_path, weights_only=False)
    # checkpoint = torch.load(
    #     # "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_ablation_nonode_dis1Newsize3_1225.pth",
    #     # "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize3_1225.pth",
    #     "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_ablation_noedge_dis1Newsize3_1225.pth",
    #     weights_only=False
    # )

    Geo_path = None
    vtk = None
    no_manifold_times = 0
    name = re.split("/", vtk_path)[-1].split('.')[0]

    model.load_state_dict(checkpoint)
    i = 0
    if_end = False
    collapse_rank = 0
    element_num = []
    query_cell_num = []
    dis = []
    # try:
    for val_batch in dataset:
        with torch.no_grad():
            t0 = time.time()
            edge_count = len(val_batch.mesh[0].edges)
            N = 0
            step = 0
            geo = val_batch.geo[0]
            target_path = f"/home/zhuxunyang/coding/simply/datasets/training/target/{geo}_target.vtk"
            target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
            tree1 = cKDTree(target_mesh.vertices.numpy())
            total_time['preprocessing'] = time.time() - t0

            begin_mesh = val_batch.mesh[0]

            flag = True
            times = 0
            # while N < edge_count // 2:
            while flag:
                # val_batch.mesh[0] = smooth_mesh_sizing(val_batch.mesh[0], len(val_batch.mesh[0].vertices), 1.2, torch.min(val_batch.mesh[0].sizing_values))
                element_num.append(sum(val_batch.mesh[0].num_mesh()))
                query_cell_num.append(len(val_batch.mesh[0].faces))
                dis.append(hausdorff_distance_max(val_batch.mesh[0].vertices, target_mesh.vertices))
                iter_time = time.time()
                t0 = time.time()
                if times > 5:
                    Geo_path = geo_path
                    vtk = save_path3
                    flag = False
                    break
                val_out = model(val_batch)

                sort_score, sort_pred = torch.topk(val_out[:, 0], len(val_batch.mesh[0].edges), dim=0, largest=False)

                total_time['edge_predict'] = time.time() - t0
                t0 = time.time()
                # collapse_edges = sort_clean_edge1(val_batch.mesh[0], sort_pred)
                # total_time['edge_filter1'] = time.time() - t0
                # t0 = time.time()
                # true_collapse_edges = filter_feature_edge1(val_batch.mesh[0], collapse_edges)
                # total_time['edge_filter2'] = time.time() - t0
                
                edges = val_batch.mesh[0].edges.cpu().numpy()
                select_edge_num = int(len(edges) * rito)
                point_set = set()
                banned_vertices = set()
                vertex_edge_map = defaultdict(list)
                for i, edge in enumerate(edges):
                    v1, v2 = edges[i]
                    vertex_edge_map[v1].append(i)
                    vertex_edge_map[v2].append(i)
                edge_ids = []

                final_index = -1
                for index in sort_pred:
                    # if val_out[:, 0][index] > max_score:
                    #     break
                    v1, v2 = edges[index]
                    convex1 = val_batch.mesh[0].convex[v1]
                    convex2 = val_batch.mesh[0].convex[v2]
                    normals1 = val_batch.mesh[0].vertex_normal[v1]
                    normals2 = val_batch.mesh[0].vertex_normal[v2]
                    dots = normals1.dot(normals2)
                    angles_deg = torch.rad2deg(torch.acos(dots))
                    if (
                        v1 not in point_set and 
                        v2 not in point_set and 
                        # v1 not in banned_vertices and 
                        # v2 not in banned_vertices and 
                        (angles_deg < 10) and 
                        (~(val_batch.mesh[0].feature_point[v1] & val_batch.mesh[0].feature_point[v2]) | ((convex1 == convex2) & (convex1 != 0)))):
                        edge_ids.append(index.item())
                        final_index = index.item()
                        point_set.update([v1, v2])
                        for v in [v1, v2]:
                            for neighbor_edge in vertex_edge_map[v]:
                                nv1, nv2 = edges[neighbor_edge]
                                banned_vertices.add(nv1)
                                banned_vertices.add(nv2)
                    if len(edge_ids) >= select_edge_num:
                        break

                total_time['edge_filter2'] = time.time() - t0

                true_collapse_edges = edge_ids

                if len(true_collapse_edges) == 0:
                    print("No edge will collapse")
                    times += 1
                    continue
                else:
                    times = 0

                print("We have collapsed", len(true_collapse_edges), "/", select_edge_num, "edges", "lowest edge score:", sort_score[0].item(), "Largest edge score:", val_out[:, 0][final_index].item())
                original_mesh = val_batch.mesh[0]

                if val_out[:, 0][final_index].item() > max_score:
                    # t0 = time.time()
                    # smooth_mesh_sizing(original_mesh, len(original_mesh.vertices), beta=1.2, h_min= torch.min(original_mesh.sizing_values))
                    # print(f"Smooth cost {time.time() - t0} s")
                    # original_mesh.writeVTK(f"{save_path}/final_simply.vtk")
                    break
                save_path1 = f"{save_path}/result_before_smooth_" + name + "_" + str(step) + ".vtk"
                save_path2 = f"{save_path}/result_after_smooth_" + name + "_" + str(step) + ".vtk"
                save_path3 = f"{save_path}/result_sizefield_" + name + "_" + str(step) + ".vtk"
                geo_path = f"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/target/{geo}.step"

                # original_mesh = original_mesh.collapse_multiple_edges(true_collapse_edges)
                save_path4 = f"{save_path}/{name}_{step}_before_visual.vtk"
                # original_mesh.visual_edge(true_collapse_edges, save_path4)
                
                t0 = time.time()
                # new_mesh = original_mesh.collapse_multiple_edges2(true_collapse_edges)
                try:
                    new_mesh = original_mesh.collapse_multiple_edges2(true_collapse_edges)
                except:
                    # continue
                    break
                # find_nearest_triangles(new_mesh, target_mesh)
                new_mesh.get_all_info()
                total_time['edge_collapse'] = time.time() - t0

                if False:
                    print("")
                
                else:
                    save_path5 = f"{save_path}/{step}_after_visual.vtk"
                    # original_mesh.visual_edge(true_collapse_edges, save_path5)
                    t0 = time.time()
                    new_mesh.writeVTK(save_path1)
                    original_mesh = new_mesh
                    total_time['mesh_io'] = time.time() - t0
                    
                t0 = time.time()
                # command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', '1e-4', '--max-pass', '10', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
                if collapse_rank < 10000:
                    command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', str(remesh_eps), '--max-pass', '10', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
                    try:
                        result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
                        print("remesh success")
                        original_mesh = CustomMesh.CustomMesh.from_vtk(save_path2)
                    except subprocess.CalledProcessError as e:
                        print("remesh fail")
                        print(e.stderr)
                        original_mesh = CustomMesh.CustomMesh.from_vtk(save_path1)
                        # break

                original_mesh.recalculate_size(begin_mesh)
                
                total_time['postprocessing'] = time.time() - t0
                original_mesh.writeVTK(save_path3)
                
                t0 = time.time()
                collapse_rank += 1
                # result_size, result_dis = begin_mesh.L1_size(original_mesh)
                true_size = begin_mesh.sizing_values

                # L1loss = L1(result_size, true_size).item()
                # distance_loss = torch.sum(result_dis).item() / len(result_dis)
                # l1_losses_list.append(L1loss)
                # distance_list.append(distance_loss)
                collapse_ranks_list.append(collapse_rank)
                # plot_tend(collapse_ranks_list, l1_losses_list, f"{save_path}/{collapse_rank}_tend.png")
                # print(collapse_rank, "L1 Loss:", L1loss, "Distance:", distance_loss, "hausdorff_dis:", hausdorff_distance_max(target_mesh.vertices, original_mesh.vertices))


                # 提取新网格的特征并归一化
                node_mean, node_std, edge_mean, edge_std = norm_params
                edge_features = original_mesh.compute_edge_features()
                lbo_vertex, lbo_edge = original_mesh.compute_LBO()
                # norm_node = (original_mesh.sizing_values - node_mean) / node_std
                # norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
                lbo_v = torch.from_numpy(lbo_vertex).unsqueeze(1)
                lbo_e = torch.from_numpy(lbo_edge).unsqueeze(0)
                norm_node = (torch.concatenate([original_mesh.sizing_values, lbo_v], dim=1) - np.expand_dims(node_mean, axis=0)) / np.expand_dims(node_std, axis=0)
                norm_edge = (torch.concatenate([edge_features, lbo_e], dim=0) - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
                
                num_nodes = len(original_mesh.vertices)
                # 更新 val_batch
                val_batch.x = norm_node.float().to(device)
                val_batch.edge_index = original_mesh.edges.T.long().to(device)
                val_batch.edge_attr = norm_edge.T.float().to(device)
                val_batch.pos = original_mesh.vertices.to(device)
                val_batch.length = [len(original_mesh.edges)]
                val_batch.mesh = [original_mesh]  # 注意保持 batch 结构
                val_batch.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
                total_time['data_update'] = time.time() - t0
                step += 1
                print(f"Collapsing rank {collapse_rank} succeed")
                total_time['per_iteration'] = time.time() - iter_time

                print("\nTime Profiling Results:")
                for key, value in total_time.items():
                    print(f"{key:20}: {value:.2f}s")
                print(f"{collapse_rank} cost {time.time() - start_time}s")
    # except:
    #     print("end")
    end_time = time.time()
    print(f"Cost {end_time - start_time:.2f}s")

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

    # 方法1：使用索引作为横坐标
    x_indices = range(len(query_cell_num))
    ax1.plot(x_indices, list(element_num), 'b-', linewidth=2, label='Element num')

    # 设置横坐标标签为query_cell_num的值，但保持顺序
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f'{x:.0f}' for x in query_cell_num], rotation=45)

    ax1.set_xlabel('Query cell')
    ax1.set_ylabel('Element num')
    ax1.set_title('Q-E')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 使用对数坐标，便于观察变化

    plt.tight_layout()

    # 保存图片
    path = f"{save_path}/convergence.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"收敛曲线图已保存至: {path}")

    fig, ax3 = plt.subplots(1, 1, figsize=(12, 10))

    # 方法1：使用索引作为横坐标
    x_indices = range(len(query_cell_num))
    ax3.plot(x_indices, list(dis), 'b-', linewidth=2, label='Hausdorff dis')

    # 设置横坐标标签为Hausdorff的值，但保持顺序
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels([f'{x:.0f}' for x in query_cell_num], rotation=45)

    ax3.set_xlabel('Query cell')
    ax3.set_ylabel('Hausdorff dis')
    ax3.set_title('Q-E')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # 使用对数坐标，便于观察变化

    plt.tight_layout()

    # 保存图片
    path = f"{save_path}/convergence_Hausdorff.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"收敛曲线图已保存至: {path}")
    plt.close()

    # 保存数据到CSV文件
    data_to_save = {
        'list_index': list(range(len(query_cell_num))),  # 列表索引 0, 1, 2, ...
        'element_num': list(element_num),                # element_num值
        'query_cell_num': list(query_cell_num),           # 也可以保存query_cell_num
        'has_dis':list(dis)
    }

    df = pd.DataFrame(data_to_save)
    csv_path = f"{save_path}/convergence_data.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"数据已保存至CSV文件: {csv_path}")

def test_example2_ablation1(vtk_path, save_path, angle, max_score, rito, model_path, remesh_eps):
    plt.close('all')
    total_time = defaultdict(float)
    L1 = nn.L1Loss()
    collapse_ranks_list = []
    l1_losses_list = []
    distance_list = []
    start_time = time.time()
    opt = TestOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, norm_params = create_example1(vtk_path, opt, device, load_para())
    # model = EdgeRankingGNN(
    #     node_in_dim=2,
    #     edge_in_dim=8,
    #     hidden_dim=64
    # ).to(device=device)
    model = EdgeRankingGNN2_Ablation(
        node_in_dim=2,
        edge_in_dim=8,
        hidden_dim=64
    ).to(device=device)

    mesh = CustomMesh.CustomMesh.from_vtk(vtk_path)
    begin_mesh = mesh

    total_steps = 0

    checkpoint = torch.load(model_path, weights_only=False)
    # checkpoint = torch.load(
    #     # "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_ablation_nonode_dis1Newsize3_1225.pth",
    #     # "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize3_1225.pth",
    #     "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_ablation_noedge_dis1Newsize3_1225.pth",
    #     weights_only=False
    # )

    Geo_path = None
    vtk = None
    no_manifold_times = 0
    name = re.split("/", vtk_path)[-1].split('.')[0]

    model.load_state_dict(checkpoint)
    i = 0
    if_end = False
    collapse_rank = 0
    element_num = []
    query_cell_num = []
    dis = []
    # try:
    for val_batch in dataset:
        with torch.no_grad():
            t0 = time.time()
            edge_count = len(val_batch.mesh[0].edges)
            N = 0
            step = 0
            geo = val_batch.geo[0]
            target_path = f"/home/zhuxunyang/coding/simply/datasets/training/target/{geo}_target.vtk"
            target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
            tree1 = cKDTree(target_mesh.vertices.numpy())
            total_time['preprocessing'] = time.time() - t0

            begin_mesh = val_batch.mesh[0]

            flag = True
            times = 0
            # while N < edge_count // 2:
            while flag:
                element_num.append(sum(val_batch.mesh[0].num_mesh()))
                query_cell_num.append(len(val_batch.mesh[0].faces))
                dis.append(hausdorff_distance_max(val_batch.mesh[0].vertices, target_mesh.vertices))
                iter_time = time.time()
                t0 = time.time()
                if times > 5:
                    Geo_path = geo_path
                    vtk = save_path3
                    flag = False
                    break
                val_out = model(val_batch, ablation = 'no_edge')

                sort_score, sort_pred = torch.topk(val_out[:, 0], len(val_batch.mesh[0].edges), dim=0, largest=False)

                total_time['edge_predict'] = time.time() - t0
                t0 = time.time()
                # collapse_edges = sort_clean_edge1(val_batch.mesh[0], sort_pred)
                # total_time['edge_filter1'] = time.time() - t0
                # t0 = time.time()
                # true_collapse_edges = filter_feature_edge1(val_batch.mesh[0], collapse_edges)
                # total_time['edge_filter2'] = time.time() - t0
                
                edges = val_batch.mesh[0].edges.cpu().numpy()
                select_edge_num = int(len(edges) * rito)
                point_set = set()
                banned_vertices = set()
                vertex_edge_map = defaultdict(list)
                for i, edge in enumerate(edges):
                    v1, v2 = edges[i]
                    vertex_edge_map[v1].append(i)
                    vertex_edge_map[v2].append(i)
                edge_ids = []

                final_index = -1
                for index in sort_pred:
                    # if val_out[:, 0][index] > max_score:
                    #     break
                    v1, v2 = edges[index]
                    convex1 = val_batch.mesh[0].convex[v1]
                    convex2 = val_batch.mesh[0].convex[v2]
                    normals1 = val_batch.mesh[0].vertex_normal[v1]
                    normals2 = val_batch.mesh[0].vertex_normal[v2]
                    dots = normals1.dot(normals2)
                    angles_deg = torch.rad2deg(torch.acos(dots))
                    if (
                        v1 not in point_set and 
                        v2 not in point_set and 
                        # v1 not in banned_vertices and 
                        # v2 not in banned_vertices and 
                        (angles_deg < angle) and 
                        (~(val_batch.mesh[0].feature_point[v1] & val_batch.mesh[0].feature_point[v2]) | ((convex1 == convex2) & (convex1 != 0)))):
                        edge_ids.append(index.item())
                        final_index = index.item()
                        point_set.update([v1, v2])
                        for v in [v1, v2]:
                            for neighbor_edge in vertex_edge_map[v]:
                                nv1, nv2 = edges[neighbor_edge]
                                banned_vertices.add(nv1)
                                banned_vertices.add(nv2)
                    if len(edge_ids) >= select_edge_num:
                        break

                total_time['edge_filter2'] = time.time() - t0

                true_collapse_edges = edge_ids

                if len(true_collapse_edges) == 0:
                    print("No edge will collapse")
                    times += 1
                    continue
                else:
                    times = 0

                print("We have collapsed", len(true_collapse_edges), "/", select_edge_num, "edges", "lowest edge score:", sort_score[0].item(), "Largest edge score:", val_out[:, 0][final_index].item())
                original_mesh = val_batch.mesh[0]

                if val_out[:, 0][final_index].item() > max_score:
                    # t0 = time.time()
                    # smooth_mesh_sizing(original_mesh, len(original_mesh.vertices), beta=1.2, h_min= torch.min(original_mesh.sizing_values))
                    # print(f"Smooth cost {time.time() - t0} s")
                    # original_mesh.writeVTK(f"{save_path}/final_simply.vtk")
                    break
                save_path1 = f"{save_path}/result_before_smooth_" + name + "_" + str(step) + ".vtk"
                save_path2 = f"{save_path}/result_after_smooth_" + name + "_" + str(step) + ".vtk"
                save_path3 = f"{save_path}/result_sizefield_" + name + "_" + str(step) + ".vtk"
                geo_path = f"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/target/{geo}.step"

                # original_mesh = original_mesh.collapse_multiple_edges(true_collapse_edges)
                save_path4 = f"{save_path}/{name}_{step}_before_visual.vtk"
                original_mesh.visual_edge(true_collapse_edges, save_path4)
                
                t0 = time.time()
                # new_mesh = original_mesh.collapse_multiple_edges2(true_collapse_edges)
                try:
                    new_mesh = original_mesh.collapse_multiple_edges2(true_collapse_edges)
                except:
                    continue
                # find_nearest_triangles(new_mesh, target_mesh)
                new_mesh.get_all_info()
                total_time['edge_collapse'] = time.time() - t0

                if False:
                    print("")
                
                else:
                    save_path5 = f"{save_path}/{step}_after_visual.vtk"
                    original_mesh.visual_edge(true_collapse_edges, save_path5)
                    t0 = time.time()
                    new_mesh.writeVTK(save_path1)
                    original_mesh = new_mesh
                    total_time['mesh_io'] = time.time() - t0
                    
                t0 = time.time()
                # command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', '1e-4', '--max-pass', '10', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
                if collapse_rank < 10000:
                    command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', str(remesh_eps), '--max-pass', '10', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
                    try:
                        result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
                        print("remesh success")
                        original_mesh = CustomMesh.CustomMesh.from_vtk(save_path2)
                    except subprocess.CalledProcessError as e:
                        print("remesh fail")
                        print(e.stderr)
                        original_mesh = CustomMesh.CustomMesh.from_vtk(save_path1)
                        # break

                original_mesh.recalculate_size(begin_mesh)
                total_time['postprocessing'] = time.time() - t0
                original_mesh.writeVTK(save_path3)
                
                t0 = time.time()
                collapse_rank += 1
                result_size, result_dis = begin_mesh.L1_size(original_mesh)
                true_size = begin_mesh.sizing_values

                L1loss = L1(result_size, true_size).item()
                distance_loss = torch.sum(result_dis).item() / len(result_dis)
                l1_losses_list.append(L1loss)
                distance_list.append(distance_loss)
                collapse_ranks_list.append(collapse_rank)
                # plot_tend(collapse_ranks_list, l1_losses_list, f"{save_path}/{collapse_rank}_tend.png")
                print(collapse_rank, "L1 Loss:", L1loss, "Distance:", distance_loss, "hausdorff_dis:", hausdorff_distance_max(target_mesh.vertices, original_mesh.vertices))


                # 提取新网格的特征并归一化
                node_mean, node_std, edge_mean, edge_std = norm_params
                edge_features = original_mesh.compute_edge_features()
                lbo_vertex, lbo_edge = original_mesh.compute_LBO()
                # norm_node = (original_mesh.sizing_values - node_mean) / node_std
                # norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
                lbo_v = torch.from_numpy(lbo_vertex).unsqueeze(1)
                lbo_e = torch.from_numpy(lbo_edge).unsqueeze(0)
                norm_node = (torch.concatenate([original_mesh.sizing_values, lbo_v], dim=1) - np.expand_dims(node_mean, axis=0)) / np.expand_dims(node_std, axis=0)
                norm_edge = (torch.concatenate([edge_features, lbo_e], dim=0) - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
                
                num_nodes = len(original_mesh.vertices)
                # 更新 val_batch
                val_batch.x = norm_node.float().to(device)
                val_batch.edge_index = original_mesh.edges.T.long().to(device)
                val_batch.edge_attr = norm_edge.T.float().to(device)
                val_batch.pos = original_mesh.vertices.to(device)
                val_batch.length = [len(original_mesh.edges)]
                val_batch.mesh = [original_mesh]  # 注意保持 batch 结构
                val_batch.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
                total_time['data_update'] = time.time() - t0
                step += 1
                print(f"Collapsing rank {collapse_rank} succeed")
                total_time['per_iteration'] = time.time() - iter_time

                print("\nTime Profiling Results:")
                for key, value in total_time.items():
                    print(f"{key:20}: {value:.2f}s")
                print(f"{collapse_rank} cost {time.time() - start_time}s")
    # except:
    #     print("end")
    end_time = time.time()
    print(f"Cost {end_time - start_time:.2f}s")

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

    # 方法1：使用索引作为横坐标
    x_indices = range(len(query_cell_num))
    ax1.plot(x_indices, list(element_num), 'b-', linewidth=2, label='Element num')

    # 设置横坐标标签为query_cell_num的值，但保持顺序
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f'{x:.0f}' for x in query_cell_num], rotation=45)

    ax1.set_xlabel('Query cell')
    ax1.set_ylabel('Element num')
    ax1.set_title('Q-E')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 使用对数坐标，便于观察变化

    plt.tight_layout()

    # 保存图片
    path = f"{save_path}/convergence.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"收敛曲线图已保存至: {path}")

    fig, ax3 = plt.subplots(1, 1, figsize=(12, 10))

    # 方法1：使用索引作为横坐标
    x_indices = range(len(query_cell_num))
    ax3.plot(x_indices, list(dis), 'b-', linewidth=2, label='Hausdorff dis')

    # 设置横坐标标签为Hausdorff的值，但保持顺序
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels([f'{x:.0f}' for x in query_cell_num], rotation=45)

    ax3.set_xlabel('Query cell')
    ax3.set_ylabel('Hausdorff dis')
    ax3.set_title('Q-E')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # 使用对数坐标，便于观察变化

    plt.tight_layout()

    # 保存图片
    path = f"{save_path}/convergence_Hausdorff.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"收敛曲线图已保存至: {path}")
    plt.close()

    # 保存数据到CSV文件
    data_to_save = {
        'list_index': list(range(len(query_cell_num))),  # 列表索引 0, 1, 2, ...
        'element_num': list(element_num),                # element_num值
        'query_cell_num': list(query_cell_num),           # 也可以保存query_cell_num
        'has_dis':list(dis)
    }

    df = pd.DataFrame(data_to_save)
    csv_path = f"{save_path}/convergence_data.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"数据已保存至CSV文件: {csv_path}")

def test_example2_ablation2(vtk_path, save_path, angle, max_score, rito, model_path, remesh_eps):
    plt.close('all')
    total_time = defaultdict(float)
    L1 = nn.L1Loss()
    collapse_ranks_list = []
    l1_losses_list = []
    distance_list = []
    start_time = time.time()
    opt = TestOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, norm_params = create_example1(vtk_path, opt, device, load_para())
    # model = EdgeRankingGNN(
    #     node_in_dim=2,
    #     edge_in_dim=8,
    #     hidden_dim=64
    # ).to(device=device)
    model = EdgeRankingGNN2_Ablation(
        node_in_dim=2,
        edge_in_dim=8,
        hidden_dim=64
    ).to(device=device)

    mesh = CustomMesh.CustomMesh.from_vtk(vtk_path)
    begin_mesh = mesh

    total_steps = 0

    checkpoint = torch.load(model_path, weights_only=False)
    # checkpoint = torch.load(
    #     # "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_ablation_nonode_dis1Newsize3_1225.pth",
    #     # "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize3_1225.pth",
    #     "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_ablation_noedge_dis1Newsize3_1225.pth",
    #     weights_only=False
    # )

    Geo_path = None
    vtk = None
    no_manifold_times = 0
    name = re.split("/", vtk_path)[-1].split('.')[0]

    model.load_state_dict(checkpoint)
    i = 0
    if_end = False
    collapse_rank = 0
    element_num = []
    query_cell_num = []
    dis = []
    # try:
    for val_batch in dataset:
        with torch.no_grad():
            t0 = time.time()
            edge_count = len(val_batch.mesh[0].edges)
            N = 0
            step = 0
            geo = val_batch.geo[0]
            target_path = f"/home/zhuxunyang/coding/simply/datasets/training/target/{geo}_target.vtk"
            target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
            tree1 = cKDTree(target_mesh.vertices.numpy())
            total_time['preprocessing'] = time.time() - t0

            begin_mesh = val_batch.mesh[0]

            flag = True
            times = 0
            # while N < edge_count // 2:
            while flag:
                element_num.append(sum(val_batch.mesh[0].num_mesh()))
                query_cell_num.append(len(val_batch.mesh[0].faces))
                dis.append(hausdorff_distance_max(val_batch.mesh[0].vertices, target_mesh.vertices))
                iter_time = time.time()
                t0 = time.time()
                if times > 5:
                    Geo_path = geo_path
                    vtk = save_path3
                    flag = False
                    break
                val_out = model(val_batch, ablation = 'no_node')

                sort_score, sort_pred = torch.topk(val_out[:, 0], len(val_batch.mesh[0].edges), dim=0, largest=False)

                total_time['edge_predict'] = time.time() - t0
                t0 = time.time()
                # collapse_edges = sort_clean_edge1(val_batch.mesh[0], sort_pred)
                # total_time['edge_filter1'] = time.time() - t0
                # t0 = time.time()
                # true_collapse_edges = filter_feature_edge1(val_batch.mesh[0], collapse_edges)
                # total_time['edge_filter2'] = time.time() - t0
                
                edges = val_batch.mesh[0].edges.cpu().numpy()
                select_edge_num = int(len(edges) * rito)
                point_set = set()
                banned_vertices = set()
                vertex_edge_map = defaultdict(list)
                for i, edge in enumerate(edges):
                    v1, v2 = edges[i]
                    vertex_edge_map[v1].append(i)
                    vertex_edge_map[v2].append(i)
                edge_ids = []

                final_index = -1
                for index in sort_pred:
                    # if val_out[:, 0][index] > max_score:
                    #     break
                    v1, v2 = edges[index]
                    convex1 = val_batch.mesh[0].convex[v1]
                    convex2 = val_batch.mesh[0].convex[v2]
                    normals1 = val_batch.mesh[0].vertex_normal[v1]
                    normals2 = val_batch.mesh[0].vertex_normal[v2]
                    dots = normals1.dot(normals2)
                    angles_deg = torch.rad2deg(torch.acos(dots))
                    if (
                        v1 not in point_set and 
                        v2 not in point_set and 
                        # v1 not in banned_vertices and 
                        # v2 not in banned_vertices and 
                        (angles_deg < angle) and 
                        (~(val_batch.mesh[0].feature_point[v1] & val_batch.mesh[0].feature_point[v2]) | ((convex1 == convex2) & (convex1 != 0)))):
                        edge_ids.append(index.item())
                        final_index = index.item()
                        point_set.update([v1, v2])
                        for v in [v1, v2]:
                            for neighbor_edge in vertex_edge_map[v]:
                                nv1, nv2 = edges[neighbor_edge]
                                banned_vertices.add(nv1)
                                banned_vertices.add(nv2)
                    if len(edge_ids) >= select_edge_num:
                        break

                total_time['edge_filter2'] = time.time() - t0

                true_collapse_edges = edge_ids

                if len(true_collapse_edges) == 0:
                    print("No edge will collapse")
                    times += 1
                    continue
                else:
                    times = 0

                print("We have collapsed", len(true_collapse_edges), "/", select_edge_num, "edges", "lowest edge score:", sort_score[0].item(), "Largest edge score:", val_out[:, 0][final_index].item())
                original_mesh = val_batch.mesh[0]

                if val_out[:, 0][final_index].item() > max_score:
                    # t0 = time.time()
                    # smooth_mesh_sizing(original_mesh, len(original_mesh.vertices), beta=1.2, h_min= torch.min(original_mesh.sizing_values))
                    # print(f"Smooth cost {time.time() - t0} s")
                    # original_mesh.writeVTK(f"{save_path}/final_simply.vtk")
                    break
                save_path1 = f"{save_path}/result_before_smooth_" + name + "_" + str(step) + ".vtk"
                save_path2 = f"{save_path}/result_after_smooth_" + name + "_" + str(step) + ".vtk"
                save_path3 = f"{save_path}/result_sizefield_" + name + "_" + str(step) + ".vtk"
                geo_path = f"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/target/{geo}.step"

                # original_mesh = original_mesh.collapse_multiple_edges(true_collapse_edges)
                save_path4 = f"{save_path}/{name}_{step}_before_visual.vtk"
                original_mesh.visual_edge(true_collapse_edges, save_path4)
                
                t0 = time.time()
                # new_mesh = original_mesh.collapse_multiple_edges2(true_collapse_edges)
                try:
                    new_mesh = original_mesh.collapse_multiple_edges2(true_collapse_edges)
                except:
                    continue
                # find_nearest_triangles(new_mesh, target_mesh)
                new_mesh.get_all_info()
                total_time['edge_collapse'] = time.time() - t0

                if False:
                    print("")
                
                else:
                    save_path5 = f"{save_path}/{step}_after_visual.vtk"
                    original_mesh.visual_edge(true_collapse_edges, save_path5)
                    t0 = time.time()
                    new_mesh.writeVTK(save_path1)
                    original_mesh = new_mesh
                    total_time['mesh_io'] = time.time() - t0
                    
                t0 = time.time()
                # command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', '1e-4', '--max-pass', '10', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
                if collapse_rank < 10000:
                    command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', str(remesh_eps), '--max-pass', '10', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
                    try:
                        result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
                        print("remesh success")
                        original_mesh = CustomMesh.CustomMesh.from_vtk(save_path2)
                    except subprocess.CalledProcessError as e:
                        print("remesh fail")
                        print(e.stderr)
                        original_mesh = CustomMesh.CustomMesh.from_vtk(save_path1)
                        # break

                original_mesh.recalculate_size(begin_mesh)
                total_time['postprocessing'] = time.time() - t0
                original_mesh.writeVTK(save_path3)
                
                t0 = time.time()
                collapse_rank += 1
                result_size, result_dis = begin_mesh.L1_size(original_mesh)
                true_size = begin_mesh.sizing_values

                L1loss = L1(result_size, true_size).item()
                distance_loss = torch.sum(result_dis).item() / len(result_dis)
                l1_losses_list.append(L1loss)
                distance_list.append(distance_loss)
                collapse_ranks_list.append(collapse_rank)
                # plot_tend(collapse_ranks_list, l1_losses_list, f"{save_path}/{collapse_rank}_tend.png")
                print(collapse_rank, "L1 Loss:", L1loss, "Distance:", distance_loss, "hausdorff_dis:", hausdorff_distance_max(target_mesh.vertices, original_mesh.vertices))


                # 提取新网格的特征并归一化
                node_mean, node_std, edge_mean, edge_std = norm_params
                edge_features = original_mesh.compute_edge_features()
                lbo_vertex, lbo_edge = original_mesh.compute_LBO()
                # norm_node = (original_mesh.sizing_values - node_mean) / node_std
                # norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
                lbo_v = torch.from_numpy(lbo_vertex).unsqueeze(1)
                lbo_e = torch.from_numpy(lbo_edge).unsqueeze(0)
                norm_node = (torch.concatenate([original_mesh.sizing_values, lbo_v], dim=1) - np.expand_dims(node_mean, axis=0)) / np.expand_dims(node_std, axis=0)
                norm_edge = (torch.concatenate([edge_features, lbo_e], dim=0) - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
                
                num_nodes = len(original_mesh.vertices)
                # 更新 val_batch
                val_batch.x = norm_node.float().to(device)
                val_batch.edge_index = original_mesh.edges.T.long().to(device)
                val_batch.edge_attr = norm_edge.T.float().to(device)
                val_batch.pos = original_mesh.vertices.to(device)
                val_batch.length = [len(original_mesh.edges)]
                val_batch.mesh = [original_mesh]  # 注意保持 batch 结构
                val_batch.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
                total_time['data_update'] = time.time() - t0
                step += 1
                print(f"Collapsing rank {collapse_rank} succeed")
                total_time['per_iteration'] = time.time() - iter_time

                print("\nTime Profiling Results:")
                for key, value in total_time.items():
                    print(f"{key:20}: {value:.2f}s")
                print(f"{collapse_rank} cost {time.time() - start_time}s")
    # except:
    #     print("end")
    end_time = time.time()
    print(f"Cost {end_time - start_time:.2f}s")

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

    # 方法1：使用索引作为横坐标
    x_indices = range(len(query_cell_num))
    ax1.plot(x_indices, list(element_num), 'b-', linewidth=2, label='Element num')

    # 设置横坐标标签为query_cell_num的值，但保持顺序
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f'{x:.0f}' for x in query_cell_num], rotation=45)

    ax1.set_xlabel('Query cell')
    ax1.set_ylabel('Element num')
    ax1.set_title('Q-E')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 使用对数坐标，便于观察变化

    plt.tight_layout()

    # 保存图片
    path = f"{save_path}/convergence.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"收敛曲线图已保存至: {path}")

    fig, ax3 = plt.subplots(1, 1, figsize=(12, 10))

    # 方法1：使用索引作为横坐标
    x_indices = range(len(query_cell_num))
    ax3.plot(x_indices, list(dis), 'b-', linewidth=2, label='Hausdorff dis')

    # 设置横坐标标签为Hausdorff的值，但保持顺序
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels([f'{x:.0f}' for x in query_cell_num], rotation=45)

    ax3.set_xlabel('Query cell')
    ax3.set_ylabel('Hausdorff dis')
    ax3.set_title('Q-E')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # 使用对数坐标，便于观察变化

    plt.tight_layout()

    # 保存图片
    path = f"{save_path}/convergence_Hausdorff.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"收敛曲线图已保存至: {path}")
    plt.close()

    # 保存数据到CSV文件
    data_to_save = {
        'list_index': list(range(len(query_cell_num))),  # 列表索引 0, 1, 2, ...
        'element_num': list(element_num),                # element_num值
        'query_cell_num': list(query_cell_num)   ,        # 也可以保存query_cell_num
        'has_dis':list(dis)
    }

    df = pd.DataFrame(data_to_save)
    csv_path = f"{save_path}/convergence_data.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"数据已保存至CSV文件: {csv_path}")

def test_example2_ablation3(vtk_path, save_path, angle, max_score, rito, model_path, remesh_eps, ablation_mode):
    plt.close('all')
    total_time = defaultdict(float)
    L1 = nn.L1Loss()
    collapse_ranks_list = []
    l1_losses_list = []
    distance_list = []
    start_time = time.time()
    opt = TestOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, norm_params = create_example1(vtk_path, opt, device, load_para())
    # model = EdgeRankingGNN(
    #     node_in_dim=2,
    #     edge_in_dim=8,
    #     hidden_dim=64
    # ).to(device=device)
    model = EdgeRankingGNN2_Ablation(
        node_in_dim=2,
        edge_in_dim=8,
        hidden_dim=64
    ).to(device=device)

    mesh = CustomMesh.CustomMesh.from_vtk(vtk_path)
    begin_mesh = mesh

    total_steps = 0

    checkpoint = torch.load(model_path, weights_only=False)
    # checkpoint = torch.load(
    #     # "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_ablation_nonode_dis1Newsize3_1225.pth",
    #     # "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize3_1225.pth",
    #     "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_ablation_noedge_dis1Newsize3_1225.pth",
    #     weights_only=False
    # )

    Geo_path = None
    vtk = None
    no_manifold_times = 0
    name = re.split("/", vtk_path)[-1].split('.')[0]

    model.load_state_dict(checkpoint)
    i = 0
    if_end = False
    collapse_rank = 0
    element_num = []
    query_cell_num = []
    dis = []
    # try:
    for val_batch in dataset:
        with torch.no_grad():
            t0 = time.time()
            edge_count = len(val_batch.mesh[0].edges)
            N = 0
            step = 0
            geo = val_batch.geo[0]
            target_path = f"/home/zhuxunyang/coding/simply/datasets/training/target/{geo}_target.vtk"
            target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
            tree1 = cKDTree(target_mesh.vertices.numpy())
            total_time['preprocessing'] = time.time() - t0

            begin_mesh = val_batch.mesh[0]

            flag = True
            times = 0
            # while N < edge_count // 2:
            while flag:
                element_num.append(sum(val_batch.mesh[0].num_mesh()))
                query_cell_num.append(len(val_batch.mesh[0].faces))
                dis.append(hausdorff_distance_max(val_batch.mesh[0].vertices, target_mesh.vertices))
                iter_time = time.time()
                t0 = time.time()
                if times > 5:
                    Geo_path = geo_path
                    vtk = save_path3
                    flag = False
                    break
                val_out = model(val_batch, ablation = 'no_edge')

                sort_score, sort_pred = torch.topk(val_out[:, 0], len(val_batch.mesh[0].edges), dim=0, largest=False)

                total_time['edge_predict'] = time.time() - t0
                t0 = time.time()
                # collapse_edges = sort_clean_edge1(val_batch.mesh[0], sort_pred)
                # total_time['edge_filter1'] = time.time() - t0
                # t0 = time.time()
                # true_collapse_edges = filter_feature_edge1(val_batch.mesh[0], collapse_edges)
                # total_time['edge_filter2'] = time.time() - t0
                
                edges = val_batch.mesh[0].edges.cpu().numpy()
                select_edge_num = int(len(edges) * rito)
                point_set = set()
                banned_vertices = set()
                vertex_edge_map = defaultdict(list)
                for i, edge in enumerate(edges):
                    v1, v2 = edges[i]
                    vertex_edge_map[v1].append(i)
                    vertex_edge_map[v2].append(i)
                edge_ids = []

                final_index = -1
                for index in sort_pred:
                    # if val_out[:, 0][index] > max_score:
                    #     break
                    v1, v2 = edges[index]
                    convex1 = val_batch.mesh[0].convex[v1]
                    convex2 = val_batch.mesh[0].convex[v2]
                    normals1 = val_batch.mesh[0].vertex_normal[v1]
                    normals2 = val_batch.mesh[0].vertex_normal[v2]
                    dots = normals1.dot(normals2)
                    angles_deg = torch.rad2deg(torch.acos(dots))
                    if (
                        v1 not in point_set and 
                        v2 not in point_set and 
                        # v1 not in banned_vertices and 
                        # v2 not in banned_vertices and 
                        (angles_deg < angle) and 
                        (~(val_batch.mesh[0].feature_point[v1] & val_batch.mesh[0].feature_point[v2]) | ((convex1 == convex2) & (convex1 != 0)))):
                        edge_ids.append(index.item())
                        final_index = index.item()
                        point_set.update([v1, v2])
                        for v in [v1, v2]:
                            for neighbor_edge in vertex_edge_map[v]:
                                nv1, nv2 = edges[neighbor_edge]
                                banned_vertices.add(nv1)
                                banned_vertices.add(nv2)
                    if len(edge_ids) >= select_edge_num:
                        break

                total_time['edge_filter2'] = time.time() - t0

                true_collapse_edges = edge_ids

                if len(true_collapse_edges) == 0:
                    print("No edge will collapse")
                    times += 1
                    continue
                else:
                    times = 0

                print("We have collapsed", len(true_collapse_edges), "/", select_edge_num, "edges", "lowest edge score:", sort_score[0].item(), "Largest edge score:", val_out[:, 0][final_index].item())
                original_mesh = val_batch.mesh[0]

                if val_out[:, 0][final_index].item() > max_score:
                    # t0 = time.time()
                    # smooth_mesh_sizing(original_mesh, len(original_mesh.vertices), beta=1.2, h_min= torch.min(original_mesh.sizing_values))
                    # print(f"Smooth cost {time.time() - t0} s")
                    # original_mesh.writeVTK(f"{save_path}/final_simply.vtk")
                    break
                save_path1 = f"{save_path}/result_before_smooth_" + name + "_" + str(step) + ".vtk"
                save_path2 = f"{save_path}/result_after_smooth_" + name + "_" + str(step) + ".vtk"
                save_path3 = f"{save_path}/result_sizefield_" + name + "_" + str(step) + ".vtk"
                geo_path = f"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/target/{geo}.step"

                # original_mesh = original_mesh.collapse_multiple_edges(true_collapse_edges)
                save_path4 = f"{save_path}/{name}_{step}_before_visual.vtk"
                original_mesh.visual_edge(true_collapse_edges, save_path4)
                
                t0 = time.time()
                # new_mesh = original_mesh.collapse_multiple_edges2(true_collapse_edges)
                try:
                    new_mesh = original_mesh.collapse_multiple_edges2(true_collapse_edges)
                except:
                    continue
                # find_nearest_triangles(new_mesh, target_mesh)
                new_mesh.get_all_info()
                total_time['edge_collapse'] = time.time() - t0

                if False:
                    print("")
                
                else:
                    save_path5 = f"{save_path}/{step}_after_visual.vtk"
                    original_mesh.visual_edge(true_collapse_edges, save_path5)
                    t0 = time.time()
                    new_mesh.writeVTK(save_path1)
                    original_mesh = new_mesh
                    total_time['mesh_io'] = time.time() - t0
                    
                t0 = time.time()
                # command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', '1e-4', '--max-pass', '10', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
                if collapse_rank < 10000:
                    command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', str(remesh_eps), '--max-pass', '10', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
                    try:
                        result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
                        print("remesh success")
                        original_mesh = CustomMesh.CustomMesh.from_vtk(save_path2)
                    except subprocess.CalledProcessError as e:
                        print("remesh fail")
                        print(e.stderr)
                        original_mesh = CustomMesh.CustomMesh.from_vtk(save_path1)
                        # break

                original_mesh.recalculate_size(begin_mesh)
                total_time['postprocessing'] = time.time() - t0
                original_mesh.writeVTK(save_path3)
                
                t0 = time.time()
                collapse_rank += 1
                result_size, result_dis = begin_mesh.L1_size(original_mesh)
                true_size = begin_mesh.sizing_values

                L1loss = L1(result_size, true_size).item()
                distance_loss = torch.sum(result_dis).item() / len(result_dis)
                l1_losses_list.append(L1loss)
                distance_list.append(distance_loss)
                collapse_ranks_list.append(collapse_rank)
                plot_tend(collapse_ranks_list, l1_losses_list, f"{save_path}/{collapse_rank}_tend.png")
                print(collapse_rank, "L1 Loss:", L1loss, "Distance:", distance_loss, "hausdorff_dis:", hausdorff_distance_max(target_mesh.vertices, original_mesh.vertices))


                # 提取新网格的特征并归一化
                node_mean, node_std, edge_mean, edge_std = norm_params
                edge_features = original_mesh.compute_edge_features()
                lbo_vertex, lbo_edge = original_mesh.compute_LBO()
                # norm_node = (original_mesh.sizing_values - node_mean) / node_std
                # norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
                lbo_v = torch.from_numpy(lbo_vertex).unsqueeze(1)
                lbo_e = torch.from_numpy(lbo_edge).unsqueeze(0)
                norm_node = (torch.concatenate([original_mesh.sizing_values, lbo_v], dim=1) - np.expand_dims(node_mean, axis=0)) / np.expand_dims(node_std, axis=0)
                norm_edge = (torch.concatenate([edge_features, lbo_e], dim=0) - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
                
                num_nodes = len(original_mesh.vertices)
                # 更新 val_batch
                val_batch.x = norm_node.float().to(device)
                val_batch.edge_index = original_mesh.edges.T.long().to(device)
                val_batch.edge_attr = norm_edge.T.float().to(device)
                val_batch.pos = original_mesh.vertices.to(device)
                val_batch.length = [len(original_mesh.edges)]
                val_batch.mesh = [original_mesh]  # 注意保持 batch 结构
                val_batch.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
                total_time['data_update'] = time.time() - t0
                step += 1
                print(f"Collapsing rank {collapse_rank} succeed")
                total_time['per_iteration'] = time.time() - iter_time

                print("\nTime Profiling Results:")
                for key, value in total_time.items():
                    print(f"{key:20}: {value:.2f}s")
                print(f"{collapse_rank} cost {time.time() - start_time}s")
    # except:
    #     print("end")
    end_time = time.time()
    print(f"Cost {end_time - start_time:.2f}s")

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

    # 方法1：使用索引作为横坐标
    x_indices = range(len(query_cell_num))
    ax1.plot(x_indices, list(element_num), 'b-', linewidth=2, label='Element num')

    # 设置横坐标标签为query_cell_num的值，但保持顺序
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f'{x:.0f}' for x in query_cell_num], rotation=45)

    ax1.set_xlabel('Query cell')
    ax1.set_ylabel('Element num')
    ax1.set_title('Q-E')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 使用对数坐标，便于观察变化

    plt.tight_layout()

    # 保存图片
    path = f"{save_path}/convergence.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"收敛曲线图已保存至: {path}")

    fig, ax3 = plt.subplots(1, 1, figsize=(12, 10))

    # 方法1：使用索引作为横坐标
    x_indices = range(len(query_cell_num))
    ax3.plot(x_indices, list(dis), 'b-', linewidth=2, label='Hausdorff dis')

    # 设置横坐标标签为Hausdorff的值，但保持顺序
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels([f'{x:.0f}' for x in query_cell_num], rotation=45)

    ax3.set_xlabel('Query cell')
    ax3.set_ylabel('Hausdorff dis')
    ax3.set_title('Q-E')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # 使用对数坐标，便于观察变化

    plt.tight_layout()

    # 保存图片
    path = f"{save_path}/convergence_Hausdorff.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"收敛曲线图已保存至: {path}")

    # 保存数据到CSV文件
    data_to_save = {
        'list_index': list(range(len(query_cell_num))),  # 列表索引 0, 1, 2, ...
        'element_num': list(element_num),                # element_num值
        'query_cell_num': list(query_cell_num),           # 也可以保存query_cell_num
        'has_dis':list(dis)
    }

    df = pd.DataFrame(data_to_save)
    csv_path = f"{save_path}/convergence_data.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"数据已保存至CSV文件: {csv_path}")

def test_example_ablation(vtk_path, save_path, max_score, rito, model_path, remesh_eps, mode=None):
    plt.close('all')
    total_time = defaultdict(float)
    L1 = nn.L1Loss()
    collapse_ranks_list = []
    l1_losses_list = []
    distance_list = []
    start_time = time.time()
    opt = TestOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, norm_params = create_example1(vtk_path, opt, device, load_para())
    if mode == "nogine":
        model = EdgeRankingGNN_Ablation_0109(use_gine=False).to(device=device)
    elif mode == "noglobal":
        model = EdgeRankingGNN_Ablation_0109(use_global=False).to(device=device)
    elif mode == "noencoder_v":
        model = EdgeRankingGNN_Ablation_0109(node_encoder_type='linear').to(device=device)
    elif mode == "noencoder_e":
        model = EdgeRankingGNN_Ablation_0109(edge_encoder_type='linear').to(device=device)
    else:
        model = EdgeRankingGNN2(node_in_dim=2, edge_in_dim=8, hidden_dim=64, num_layers=2).to(device=device)

    mesh = CustomMesh.CustomMesh.from_vtk(vtk_path)
    begin_mesh = mesh

    total_steps = 0

    checkpoint = torch.load(model_path, weights_only=False)
    # checkpoint = torch.load(
    #     # "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_ablation_nonode_dis1Newsize3_1225.pth",
    #     # "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize3_1225.pth",
    #     "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_ablation_noedge_dis1Newsize3_1225.pth",
    #     weights_only=False
    # )

    Geo_path = None
    vtk = None
    no_manifold_times = 0
    name = re.split("/", vtk_path)[-1].split('.')[0]

    model.load_state_dict(checkpoint)
    i = 0
    if_end = False
    collapse_rank = 0
    element_num = []
    query_cell_num = []
    dis = []
    # try:
    final_mesh = None
    for val_batch in dataset:
        with torch.no_grad():
            t0 = time.time()
            edge_count = len(val_batch.mesh[0].edges)
            N = 0
            step = 0
            geo = val_batch.geo[0]
            target_path = f"/home/zhuxunyang/coding/simply/datasets/training/target/{geo}_target.vtk"
            target_mesh = CustomMesh.CustomMesh.from_vtk(target_path)
            tree1 = cKDTree(target_mesh.vertices.numpy())
            total_time['preprocessing'] = time.time() - t0

            begin_mesh = val_batch.mesh[0]

            flag = True
            times = 0
            # while N < edge_count // 2:
            while flag:
                # val_batch.mesh[0] = smooth_mesh_sizing(val_batch.mesh[0], len(val_batch.mesh[0].vertices), 1.2, torch.min(val_batch.mesh[0].sizing_values))
                element_num.append(sum(val_batch.mesh[0].num_mesh()))
                query_cell_num.append(len(val_batch.mesh[0].faces))
                dis.append(hausdorff_distance_max(val_batch.mesh[0].vertices, target_mesh.vertices))
                iter_time = time.time()
                t0 = time.time()
                if times > 5:
                    Geo_path = geo_path
                    vtk = save_path3
                    flag = False
                    final_mesh = val_batch.mesh[0]
                    break
                val_out = model(val_batch)

                sort_score, sort_pred = torch.topk(val_out[:, 0], len(val_batch.mesh[0].edges), dim=0, largest=False)

                total_time['edge_predict'] = time.time() - t0
                t0 = time.time()
                # collapse_edges = sort_clean_edge1(val_batch.mesh[0], sort_pred)
                # total_time['edge_filter1'] = time.time() - t0
                # t0 = time.time()
                # true_collapse_edges = filter_feature_edge1(val_batch.mesh[0], collapse_edges)
                # total_time['edge_filter2'] = time.time() - t0
                
                edges = val_batch.mesh[0].edges.cpu().numpy()
                select_edge_num = int(len(edges) * rito)
                point_set = set()
                banned_vertices = set()
                vertex_edge_map = defaultdict(list)
                for i, edge in enumerate(edges):
                    v1, v2 = edges[i]
                    vertex_edge_map[v1].append(i)
                    vertex_edge_map[v2].append(i)
                edge_ids = []

                final_index = -1
                for index in sort_pred:
                    # if val_out[:, 0][index] > max_score:
                    #     break
                    v1, v2 = edges[index]
                    convex1 = val_batch.mesh[0].convex[v1]
                    convex2 = val_batch.mesh[0].convex[v2]
                    normals1 = val_batch.mesh[0].vertex_normal[v1]
                    normals2 = val_batch.mesh[0].vertex_normal[v2]
                    dots = normals1.dot(normals2)
                    angles_deg = torch.rad2deg(torch.acos(dots))
                    if (
                        v1 not in point_set and 
                        v2 not in point_set and 
                        # v1 not in banned_vertices and 
                        # v2 not in banned_vertices and 
                        (angles_deg < 10) and 
                        (~(val_batch.mesh[0].feature_point[v1] & val_batch.mesh[0].feature_point[v2]) | ((convex1 == convex2) & (convex1 != 0)))):
                        edge_ids.append(index.item())
                        final_index = index.item()
                        point_set.update([v1, v2])
                        for v in [v1, v2]:
                            for neighbor_edge in vertex_edge_map[v]:
                                nv1, nv2 = edges[neighbor_edge]
                                banned_vertices.add(nv1)
                                banned_vertices.add(nv2)
                    if len(edge_ids) >= select_edge_num:
                        break

                total_time['edge_filter2'] = time.time() - t0

                true_collapse_edges = edge_ids

                if len(true_collapse_edges) == 0:
                    print("No edge will collapse")
                    times += 1
                    final_mesh = val_batch.mesh[0]
                    break
                else:
                    times = 0

                print("We have collapsed", len(true_collapse_edges), "/", select_edge_num, "edges", "lowest edge score:", sort_score[0].item(), "Largest edge score:", val_out[:, 0][final_index].item())
                original_mesh = val_batch.mesh[0]

                if val_out[:, 0][final_index].item() > max_score:
                    # t0 = time.time()
                    # smooth_mesh_sizing(original_mesh, len(original_mesh.vertices), beta=1.2, h_min= torch.min(original_mesh.sizing_values))
                    # print(f"Smooth cost {time.time() - t0} s")
                    # original_mesh.writeVTK(f"{save_path}/final_simply.vtk")
                    final_mesh = original_mesh
                    break
                save_path1 = f"{save_path}/result_before_smooth_" + name + "_" + str(step) + ".vtk"
                save_path2 = f"{save_path}/result_after_smooth_" + name + "_" + str(step) + ".vtk"
                save_path3 = f"{save_path}/result_sizefield_" + name + "_" + str(step) + ".vtk"
                geo_path = f"/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/target/{geo}.step"

                # original_mesh = original_mesh.collapse_multiple_edges(true_collapse_edges)
                save_path4 = f"{save_path}/{name}_{step}_before_visual.vtk"
                # original_mesh.visual_edge(true_collapse_edges, save_path4)
                
                t0 = time.time()
                # new_mesh = original_mesh.collapse_multiple_edges2(true_collapse_edges)
                try:
                    new_mesh = original_mesh.collapse_multiple_edges2(true_collapse_edges)
                except:
                    # continue
                    final_mesh = original_mesh
                    break
                # find_nearest_triangles(new_mesh, target_mesh)
                new_mesh.get_all_info()
                total_time['edge_collapse'] = time.time() - t0

                if False:
                    print("")
                
                else:
                    save_path5 = f"{save_path}/{step}_after_visual.vtk"
                    # original_mesh.visual_edge(true_collapse_edges, save_path5)
                    t0 = time.time()
                    new_mesh.writeVTK(save_path1)
                    original_mesh = new_mesh
                    total_time['mesh_io'] = time.time() - t0
                    
                t0 = time.time()
                # command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', '1e-4', '--max-pass', '10', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
                if collapse_rank < 10000:
                    command1 = ['./remesh', '--input', save_path1, '--eps', '1e-4', '--envelope-dis', str(remesh_eps), '--max-pass', '10', '--output', save_path2,'--split-num', '0', '--collapse-num', '0']
                    try:
                        result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
                        print("remesh success")
                        original_mesh = CustomMesh.CustomMesh.from_vtk(save_path2)
                    except subprocess.CalledProcessError as e:
                        print("remesh fail")
                        print(e.stderr)
                        original_mesh = CustomMesh.CustomMesh.from_vtk(save_path1)
                        # break

                original_mesh.recalculate_size(begin_mesh)
                
                total_time['postprocessing'] = time.time() - t0
                original_mesh.writeVTK(save_path3)
                
                t0 = time.time()
                collapse_rank += 1
                # result_size, result_dis = begin_mesh.L1_size(original_mesh)
                true_size = begin_mesh.sizing_values

                # L1loss = L1(result_size, true_size).item()
                # distance_loss = torch.sum(result_dis).item() / len(result_dis)
                # l1_losses_list.append(L1loss)
                # distance_list.append(distance_loss)
                collapse_ranks_list.append(collapse_rank)
                # plot_tend(collapse_ranks_list, l1_losses_list, f"{save_path}/{collapse_rank}_tend.png")
                # print(collapse_rank, "L1 Loss:", L1loss, "Distance:", distance_loss, "hausdorff_dis:", hausdorff_distance_max(target_mesh.vertices, original_mesh.vertices))


                # 提取新网格的特征并归一化
                node_mean, node_std, edge_mean, edge_std = norm_params
                edge_features = original_mesh.compute_edge_features()
                lbo_vertex, lbo_edge = original_mesh.compute_LBO()
                # norm_node = (original_mesh.sizing_values - node_mean) / node_std
                # norm_edge = (edge_features - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
                lbo_v = torch.from_numpy(lbo_vertex).unsqueeze(1)
                lbo_e = torch.from_numpy(lbo_edge).unsqueeze(0)
                norm_node = (torch.concatenate([original_mesh.sizing_values, lbo_v], dim=1) - np.expand_dims(node_mean, axis=0)) / np.expand_dims(node_std, axis=0)
                norm_edge = (torch.concatenate([edge_features, lbo_e], dim=0) - np.expand_dims(edge_mean, axis=1)) / np.expand_dims(edge_std, axis=1)
                
                num_nodes = len(original_mesh.vertices)
                # 更新 val_batch
                val_batch.x = norm_node.float().to(device)
                val_batch.edge_index = original_mesh.edges.T.long().to(device)
                val_batch.edge_attr = norm_edge.T.float().to(device)
                val_batch.pos = original_mesh.vertices.to(device)
                val_batch.length = [len(original_mesh.edges)]
                val_batch.mesh = [original_mesh]  # 注意保持 batch 结构
                val_batch.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
                total_time['data_update'] = time.time() - t0
                step += 1
                print(f"Collapsing rank {collapse_rank} succeed")
                total_time['per_iteration'] = time.time() - iter_time

                print("\nTime Profiling Results:")
                for key, value in total_time.items():
                    print(f"{key:20}: {value:.2f}s")
                print(f"{collapse_rank} cost {time.time() - start_time}s")
            final_mesh = smooth_mesh_sizing(final_mesh, len(final_mesh.vertices), 1.2, torch.min(final_mesh.sizing_values))
            final_mesh.writeVTK(f"{save_path}/final" + name + ".vtk")
            print("smooth complete")
    # except:
    #     print("end")
    end_time = time.time()
    print(f"Cost {end_time - start_time:.2f}s")

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

    # 方法1：使用索引作为横坐标
    x_indices = range(len(query_cell_num))
    ax1.plot(x_indices, list(element_num), 'b-', linewidth=2, label='Element num')

    # 设置横坐标标签为query_cell_num的值，但保持顺序
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f'{x:.0f}' for x in query_cell_num], rotation=45)

    ax1.set_xlabel('Query cell')
    ax1.set_ylabel('Element num')
    ax1.set_title('Q-E')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 使用对数坐标，便于观察变化

    plt.tight_layout()

    # 保存图片
    path = f"{save_path}/convergence.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"收敛曲线图已保存至: {path}")

    fig, ax3 = plt.subplots(1, 1, figsize=(12, 10))

    # 方法1：使用索引作为横坐标
    x_indices = range(len(query_cell_num))
    ax3.plot(x_indices, list(dis), 'b-', linewidth=2, label='Hausdorff dis')

    # 设置横坐标标签为Hausdorff的值，但保持顺序
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels([f'{x:.0f}' for x in query_cell_num], rotation=45)

    ax3.set_xlabel('Query cell')
    ax3.set_ylabel('Hausdorff dis')
    ax3.set_title('Q-E')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # 使用对数坐标，便于观察变化

    plt.tight_layout()

    # 保存图片
    path = f"{save_path}/convergence_Hausdorff.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"收敛曲线图已保存至: {path}")
    plt.close()

    # 保存数据到CSV文件
    data_to_save = {
        'list_index': list(range(len(query_cell_num))),  # 列表索引 0, 1, 2, ...
        'element_num': list(element_num),                # element_num值
        'query_cell_num': list(query_cell_num),           # 也可以保存query_cell_num
        'has_dis':list(dis)
    }

    df = pd.DataFrame(data_to_save)
    csv_path = f"{save_path}/convergence_data.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"数据已保存至CSV文件: {csv_path}")

def check_model(vtk_path):
    opt = TestOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, norm_params = create_example1(vtk_path, opt, device, load_para())
    model = EdgeRankingGNN(
        node_feat_dim=2,
        edge_feat_dim=8,
        hidden_dim=64
    ).to(device=device)


    checkpoint = torch.load(
        # "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Regression_size5.pth",
        # "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Regression3.pth",
        "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Regression1_dis1Newsize3.pth",
        weights_only=False
    )

    model.load_state_dict(checkpoint)

    for val_batch in dataset:
        with torch.no_grad():
            val_out = model(val_batch)
            sort_score, sort_pred = torch.topk(val_out[:, 0], len(val_batch.mesh[0].edges) / 20, dim=0, largest=False)

def classify_compare(pred_vtk, target_vtk):
    pred = meshio.read(pred_vtk)
    target = meshio.read(target_vtk)
    length = len(pred.cell_data['Color'][0])
    count = 0
    true_count = 0
    for i in range(length):
        print(pred.cell_data['Color'][0][i], target.cell_data['Color'][0][i])
        if np.array_equal(pred.cell_data['Color'][0][i], target.cell_data['Color'][0][i]) and np.array_equal(pred.cell_data['Color'][0][i], [1, 0, 0]):
            count += 1
        if np.array_equal(pred.cell_data['Color'][0][i], [1, 0, 0]):
            true_count += 1

    print("Accuracy:", count / true_count)


    print(" ")

def test_classify(vtk_path):
    total_time = defaultdict(float)
    L1 = nn.L1Loss()
    collapse_ranks_list = []
    l1_losses_list = []
    distance_list = []
    start_time = time.time()
    opt = TestOptions().parse()
    print('Running Test')
    opt.serial_batches = True  # no shuffle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, norm_params = create_example1(vtk_path, opt, device, load_para())
    model = EdgeClassificationGNN1(
        node_feat_dim=2,
        edge_feat_dim=8,
        hidden_dim=64
    ).to(device=device)

    mesh = CustomMesh.CustomMesh.from_vtk(vtk_path)
    begin_mesh = mesh

    total_steps = 0

    checkpoint = torch.load(
        # "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Regression_size5.pth",
        # "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Regression3.pth",
        # "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Regression_dis1newsize1.pth",
        "/home/zhuxunyang/coding/bkgm_simplification/checkpoints/debug/GNN_Classify1_dis1Newsize1.pth",
        weights_only=False
    )

    Geo_path = None
    vtk = None
    no_manifold_times = 0
    name = re.split("/", vtk_path)[-1].split('.')[0]

    model.load_state_dict(checkpoint)
    i = 0
    if_end = False
    collapse_rank = 0
    for val_batch in dataset:
        with torch.no_grad():
            val_out = F.sigmoid(model(val_batch)).squeeze(1)
            print(val_out)
            classify_result = (val_out > 0.5).float().to("cpu")
            indices_1 = np.where(classify_result == 1)[0]
            print(indices_1)
            mesh.visual_edge(list(indices_1), "/home/zhuxunyang/coding/bkgm_simplification/result.vtk")


if __name__ == "__main__":
    # mesh = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/datasets/test_case/256_0.vtk")
    # # mesh.sizing_values = torch.tensor(smooth_sizing_function(mesh), beta=1.5, tol = 1e-3).unsqueeze(1)
    # mesh = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/datasets/test_case/256_0.vtk")
    # start_time = time.time()
    # try:
        
    #     mesh.sizing_values = torch.tensor(smooth_sizing_function(mesh, beta=1.5, tol = 1e-0)).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time}s")
    # except:
    #     print("failed")
    #     print(f"Cost {time.time() - start_time}s")

    # start_time = time.time()
    # try:
        
    #     mesh.sizing_values = torch.tensor(smooth_sizing_function(mesh), beta=1.2, tol = 1e-0).unsqueeze(1)
    #     print(f"Cost {time.time() - start_time}s")
    # except:
    #     print("failed")
    #     print(f"Cost {time.time() - start_time}s")
    # classify_compare("/home/zhuxunyang/coding/bkgm_simplification/result/213_0_0_before_visual.vtk", "/home/zhuxunyang/coding/bkgm_simplification/result/213_0_0_after_visual.vtk")

    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/110_0.vtk", "/home/zhuxunyang/coding/simply/result_110_GCN_5", 10, 0.22, 0.1, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 1e-3)
    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/110_0.vtk", "/home/zhuxunyang/coding/simply/result_110_GCN_10", 10, 0.22, 0.1)
    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/110_0.vtk", "/home/zhuxunyang/coding/simply/result_110_GCN_12.5", 10, 0.22, 0.125)

    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/213_0.vtk", "/home/zhuxunyang/coding/simply/result_213_GCN_5", 10, 0.10, 0.10, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 2e-4)    #0.05
    # test_example2_ablation1("/home/zhuxunyang/coding/simply/datasets/test_case/213_0.vtk", "/home/zhuxunyang/coding/simply/result_213_GCN_5_ablation_edge", 10, 0.035, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 2e-4)
    # test_example2_ablation2("/home/zhuxunyang/coding/simply/datasets/test_case/213_0.vtk", "/home/zhuxunyang/coding/simply/result_213_GCN_5_ablation_node", 10, 0.035, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 2e-4)
    
    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/110_0.vtk", "/home/zhuxunyang/coding/simply/result_110_GCN_5_1229", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize3_1225.pth", 1e-3)    #0.05
    # test_example2_ablation1("/home/zhuxunyang/coding/simply/datasets/test_case/110_0.vtk", "/home/zhuxunyang/coding/simply/result_110_GCN_5_ablation_edge", 10, 0.8, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize3_1225.pth", 1e-3)
    # test_example2_ablation2("/home/zhuxunyang/coding/simply/datasets/test_case/110_0.vtk", "/home/zhuxunyang/coding/simply/result_110_GCN_5_ablation_node", 10, 0.8, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize3_1225.pth", 1e-3)
    
    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/268_1.vtk", "/home/zhuxunyang/coding/simply/result_268_GCN_5", 10, 0.37, 0.10, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 2e-3)    #0.05
    # test_example2_ablation1("/home/zhuxunyang/coding/simply/datasets/test_case/268_1.vtk", "/home/zhuxunyang/coding/simply/result_268_GCN_5_ablation_edge", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 2e-3)
    # test_example2_ablation2("/home/zhuxunyang/coding/simply/datasets/test_case/268_1.vtk", "/home/zhuxunyang/coding/simply/result_268_GCN_5_ablation_node", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 2e-3)

    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/221_0.vtk", "/home/zhuxunyang/coding/simply/result_221_GCN_5_1", 10, 0.19, 0.1, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 5e-4)    #0.05
    # test_example2_ablation1("/home/zhuxunyang/coding/simply/datasets/test_case/221_0.vtk", "/home/zhuxunyang/coding/simply/result_221_GCN_5_ablation_edge", 10, 0.07, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth",5e-4)
    # test_example2_ablation2("/home/zhuxunyang/coding/simply/datasets/test_case/221_0.vtk", "/home/zhuxunyang/coding/simply/result_221_GCN_5_ablation_node", 10, 0.07, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 5e-4)

    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/256_0.vtk", "/home/zhuxunyang/coding/simply/result_256_GCN_5", 10, 0.25, 0.10, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 2e-4)    #0.05
    # test_example2_ablation1("/home/zhuxunyang/coding/simply/datasets/test_case/256_0.vtk", "/home/zhuxunyang/coding/simply/result_256_GCN_5_ablation_edge", 10, 0.19, 0.025, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 2e-4)
    # test_example2_ablation2("/home/zhuxunyang/coding/simply/datasets/test_case/256_0.vtk", "/home/zhuxunyang/coding/simply/result_256_GCN_5_ablation_node", 10, 0.19, 0.025, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 2e-4)

    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/238_1.vtk", "/home/zhuxunyang/coding/simply/result_238_GCN_5", 10, 0.25, 0.1, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize1_1226.pth", 3e-4)    #0.05
    # test_example2_ablation1("/home/zhuxunyang/coding/simply/datasets/test_case/238_1.vtk", "/home/zhuxunyang/coding/simply/result_238_GCN_5_ablation_edge", 10, 0.25, 0.1, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize1_1226.pth", 3e-4)
    # test_example2_ablation2("/home/zhuxunyang/coding/simply/datasets/test_case/238_1.vtk", "/home/zhuxunyang/coding/simply/result_238_GCN_5_ablation_node", 10, 0.25, 0.1, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize1_1226.pth", 3e-4)

    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/286_0.vtk", "/home/zhuxunyang/coding/simply/result_286_GCN_5", 10, 0.17, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 3e-3)    #0.05
    # test_example2_ablation1("/home/zhuxunyang/coding/simply/datasets/test_case/286_0.vtk", "/home/zhuxunyang/coding/simply/result_286_GCN_5_ablation_edge", 10, 0.17, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 3e-3)
    # test_example2_ablation2("/home/zhuxunyang/coding/simply/datasets/test_case/286_0.vtk", "/home/zhuxunyang/coding/simply/result_286_GCN_5_ablation_node", 10, 0.17, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 3e-3)

    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/221_0.vtk", "/home/zhuxunyang/coding/simply/result_221_GCN_5", 10, 0.09, 0.05)
    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/256_0.vtk", "/home/zhuxunyang/coding/simply/result_256_GCN_5", 10, 0.14, 0.05)
    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/283_0.vtk", "/home/zhuxunyang/coding/simply/result_283_GCN_5", 10, 0.18, 0.05)
    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/284_0.vtk", "/home/zhuxunyang/coding/simply/result_284_GCN_5", 10, 0.3, 0.05)
    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/286_0.vtk", "/home/zhuxunyang/coding/simply/result_286_GCN_5", 10, 0.2, 0.05)
    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/238_1.vtk", "/home/zhuxunyang/coding/simply/result_238_GCN_5", 10, 0.14, 0.05)

    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/260_0.vtk", "/home/zhuxunyang/coding/simply/result_260_GCN_5", 10, 0.15, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 2e-4)

    # test_example2("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/238_1.vtk", 100, 5)
    # test_example2("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/256_0.vtk", 100, 10)
    # test_example2("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/221_0.vtk", 100, 10)
    # test_example2("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/213_0.vtk", 100, 10)
    # test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/110_0.vtk", "/home/zhuxunyang/coding/simply/result_110_GCN_12.5", 10, 0.225)
    # test_example2("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/238_1.vtk", 100, 10)
    # test_example2("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/268_1.vtk", 100, 10)

    # test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/110_0.vtk", 0.095, 1, 10, "/home/zhuxunyang/coding/simply/result_110_Trad_12.5.", '1e-3')
    # test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/268_1.vtk", 0.2, 0.5, 10, 0.125, "/home/zhuxunyang/coding/simply/result3", '2e-3')
    
    
    # test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/213_0.vtk", 0.10, 150, 20, 0.125, "/home/zhuxunyang/coding/simply/result1", '3e-4')
    # test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/238_1.vtk", 0.095, 0.5, 10, 0.125, "/home/zhuxunyang/coding/simply/result7", '2e-3')
    # test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/221_0.vtk", 0.10, 10, 10, 0.125, "/home/zhuxunyang/coding/simply/result2", '3e-4')
    # test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/284_0.vtk", 0.095, 1, 10, "/home/zhuxunyang/coding/simply/result8", '5e-3')
    # test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/285_0.vtk", 0.095, 1, 10, "/home/zhuxunyang/coding/simply/result8", '5e-3')
    # test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/286_0.vtk", 0.3, 0.05, 10, "/home/zhuxunyang/coding/simply/result9", '3e-3')
    # test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/256_0.vtk", 0.095, 0.5, 10, 0.125, "/home/zhuxunyang/coding/simply/result6", '2e-3')


    # test_traditional_method("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/256_0.vtk", 2.5, 0.5, 10, "/home/zhuxunyang/coding/bkgm_simplification/result")
    # test_traditional_method("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/110_0.vtk", 1.2, 10, 10, "/home/zhuxunyang/coding/bkgm_simplification/result")    #50
    # test_traditional_method("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/221_0.vtk", 1.5, 10, 20, "/home/zhuxunyang/coding/bkgm_simplification/result")
    
    # test_traditional_classify("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/213_0.vtk", 3.0, 100, 20, "/home/zhuxunyang/coding/bkgm_simplification/result")
    # test_traditional_classify("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/221_0.vtk", 1.5, 10, 20, "/home/zhuxunyang/coding/bkgm_simplification/result")
    # test_traditional_classify("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/268_1.vtk", 2.5, 0.5, 10, "/home/zhuxunyang/coding/bkgm_simplification/result1")
    # test_traditional_classify("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/256_0.vtk", 2.5, 0.5, 10, "/home/zhuxunyang/coding/bkgm_simplification/result")

    # test_random_method("/home/zhuxunyang/coding/simply/datasets/test_case/110_0.vtk", 10, "/home/zhuxunyang/coding/simply/result_110_Random", 5)
    # test_random_method("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/238_1.vtk", 1.3, 0.5, 5)

    # test_classify("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/110_0.vtk")

    # mesh = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/213_0.vtk")
    # mesh1 = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/target/213_target.vtk")
    # process_edges_no_parallel([21154, 21159], mesh, mesh1, cKDTree(mesh1.vertices.numpy()), 3.0, 100)

    # test_example1("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/test_case/212_0.vtk")

    # mesh = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/result/result_sizefield_232_0_2.vtk")
    # # mesh.visualize_convexity("convex.vtk")
    # print(mesh.face_normal[28271])
    # result = filter_feature_edge2(mesh, 341, 8638)
    # print(mesh.is_convex_vertex(341), mesh.is_convex_vertex(8638))

    # mesh = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/datasets/training/target/213_target.vtk")
    # mesh1 = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/experiment/213_stl_bkgm.vtk")
    # mesh2 = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/experiment/213_simply.vtk")
    # mesh3 = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/experiment/213_0.vtk")
    # mesh4 = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/experiment/213_simply_trad.vtk")
    # mesh5 = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/experiment/213_simply_qem.vtk")
    # dis1 = hausdorff_distance_max(mesh1.vertices, mesh.vertices)
    # dis2 = hausdorff_distance_max(mesh2.vertices, mesh.vertices)
    # dis3 = hausdorff_distance_max(mesh3.vertices, mesh.vertices)
    # dis4 = hausdorff_distance_max(mesh4.vertices, mesh.vertices)
    # dis5 = hausdorff_distance_max(mesh5.vertices, mesh.vertices)
    # print(dis1, dis2, dis3, dis4, dis5)


    # mesh = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/datasets/training/target/238_target.vtk")
    # max_size = mesh.get_max_size(1)
    # mesh1 = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/ablation_study/238_ablation_whole.vtk")
    # mesh2 = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/ablation_study/238_ablation_nogine.vtk")
    # mesh3 = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/ablation_study/238_ablation_noglobal.vtk")
    # mesh4 = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/ablation_study/238_ablation_noencoder_v.vtk")
    # mesh5 = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/ablation_study/238_ablation_noencoder_e.vtk")
    # dis1 = hausdorff_distance_max(mesh1.vertices, mesh.vertices)
    # dis2 = hausdorff_distance_max(mesh2.vertices, mesh.vertices)
    # dis3 = hausdorff_distance_max(mesh3.vertices, mesh.vertices)
    # dis4 = hausdorff_distance_max(mesh4.vertices, mesh.vertices)
    # dis5 = hausdorff_distance_max(mesh5.vertices, mesh.vertices)
    # print(238 ,dis1 / max_size, dis2 / max_size, dis3 / max_size, dis4 / max_size, dis5 / max_size)


    # mesh1 = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/datasets/training/target/238_target.vtk")
    # mesh2 = CustomMesh.CustomMesh.from_vtk("/home/zhuxunyang/coding/simply/qem_result/238_simply_qem.vtk")
    # print(hausdorff_distance_max(mesh2.vertices, mesh1.vertices))
    # final_process("/home/zhuxunyang/coding/bkgm_simplification/result/result_sizefield_213_0_31.vtk", "/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/target/213.step")

    # print(mesh.convex[309], mesh.convex[9332])
    # print(convexity(mesh, torch.tensor([309, 9332])))

    # mesh1  = CustomMesh.CustomMesh.from_vtk(f"/home/zhuxunyang/coding/simply/result_221_ablation_whole/result_sizefield_221_0_16.vtk")

    # model = 221
    # mesh1  = CustomMesh.CustomMesh.from_vtk(f"/home/zhuxunyang/coding/simply/experiment/{model}_0.vtk")
    # mesh2  = CustomMesh.CustomMesh.from_vtk(f"/home/zhuxunyang/coding/simply/experiment/{model}_simply.vtk")
    # mesh3  = CustomMesh.CustomMesh.from_vtk(f"/home/zhuxunyang/coding/simply/experiment/{model}_simply_trad.vtk")
    # mesh4  = CustomMesh.CustomMesh.from_vtk(f"/home/zhuxunyang/coding/simply/experiment/{model}_stl_bkgm.vtk")
    # mesh5  = CustomMesh.CustomMesh.from_vtk(f"/home/zhuxunyang/coding/simply/experiment/{model}_simply_qem.vtk")
    # try:
    #     t0 = time.time()
    #     mesh1 = smooth_mesh_sizing(mesh1, len(mesh1.vertices), 1.2, torch.min(mesh1.sizing_values))
    #     print(f"{time.time() - t0} s")
    # except:
    #     print("error")
    # mesh1.writeVTK(f"/home/zhuxunyang/coding/simply/smooth_result/{model}_0_smoothed.vtk")
    # mesh1.writeVTK(f"/home/zhuxunyang/coding/simply/result_221_ablation_whole/result_sizefield_221_0_16.vtk")

    # try:
    #     t0 = time.time()
    #     mesh2 = smooth_mesh_sizing(mesh2, len(mesh2.vertices), 1.2, torch.min(mesh2.sizing_values))
    #     print(f"{time.time() - t0} s")
    # except:
    #     print("error")
    # mesh2.writeVTK(f"/home/zhuxunyang/coding/simply/smooth_result/{model}_simply_smoothed.vtk")

    # try:
    #     t0 = time.time()
    #     mesh3 = smooth_mesh_sizing(mesh3, len(mesh3.vertices), 1.2, torch.min(mesh3.sizing_values))
    #     print(f"{time.time() - t0} s")
    # except:
    #     print("error")
    # mesh3.writeVTK(f"/home/zhuxunyang/coding/simply/smooth_result/{model}_simply_trad_smoothed.vtk")

    # try:
    #     t0 = time.time()
    #     mesh4 = smooth_mesh_sizing(mesh4, len(mesh4.vertices), 1.2, torch.min(mesh4.sizing_values))
    #     print(f"{time.time() - t0} s")
    # except:
    #     print("error")
    # mesh4.writeVTK(f"/home/zhuxunyang/coding/simply/smooth_result/{model}_stl_bkgm_smoothed.vtk")

    # try:
    #     t0 = time.time()
    #     mesh5 = smooth_mesh_sizing(mesh5, len(mesh5.vertices), 1.2, torch.min(mesh5.sizing_values))
    #     print(f"{time.time() - t0} s")
    # except:
    #     print("error")
    # mesh5.writeVTK(f"/home/zhuxunyang/coding/simply/smooth_result/{model}_simply_qem_smoothed.vtk")


    test_example2("/home/zhuxunyang/coding/simply/datasets/test_case/110_0.vtk", "/home/zhuxunyang/coding/simply/110_gcn_draw_5", 0.32, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize3_1225.pth", 1e-3)
    test_traditional_method("/home/zhuxunyang/coding/simply/datasets/test_case/110_0.vtk", 1, 1, 0.125, "/home/zhuxunyang/coding/simply/110_trad_draw_125", '1e-3')

    test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/286_0.vtk", "/home/zhuxunyang/coding/simply/result_286_ablation_whole", 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 3e-3)
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/286_0.vtk", "/home/zhuxunyang/coding/simply/result_286_ablation_nogine", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_nogine.pth", 3e-3, "nogine")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/286_0.vtk", "/home/zhuxunyang/coding/simply/result_286_ablation_noglobal", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noglobal.pth", 3e-3, "noglobal")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/286_0.vtk", "/home/zhuxunyang/coding/simply/result_286_ablation_noencoder_v", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noencoder_v.pth", 3e-3, "noencoder_v")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/286_0.vtk", "/home/zhuxunyang/coding/simply/result_286_ablation_noencoder_e", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noencoder_e.pth", 3e-3, "noencoder_e")

    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/221_0.vtk", "/home/zhuxunyang/coding/simply/result_221_ablation_whole", 10, 0.066, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 1e-3)
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/221_0.vtk", "/home/zhuxunyang/coding/simply/result_221_ablation_nogine", 10, 0.068, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_nogine.pth", 5e-4, "nogine")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/221_0.vtk", "/home/zhuxunyang/coding/simply/result_221_ablation_noglobal", 10, 0.068, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noglobal.pth", 5e-4, "noglobal")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/221_0.vtk", "/home/zhuxunyang/coding/simply/result_221_ablation_noencoder_v", 10, 0.068, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noencoder_v.pth", 5e-4, "noencoder_v")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/221_0.vtk", "/home/zhuxunyang/coding/simply/result_221_ablation_noencoder_e", 10, 0.068, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noencoder_e.pth", 5e-4, "noencoder_e")

    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/213_0.vtk", "/home/zhuxunyang/coding/simply/result_213_ablation_whole", 10, 0.04, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 2e-4)
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/213_0.vtk", "/home/zhuxunyang/coding/simply/result_213_ablation_nogine", 10, 0.04, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_nogine.pth", 2e-4, "nogine")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/213_0.vtk", "/home/zhuxunyang/coding/simply/result_213_ablation_noglobal", 10, 0.04, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noglobal.pth", 2e-4, "noglobal")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/213_0.vtk", "/home/zhuxunyang/coding/simply/result_213_ablation_noencoder_v", 10, 0.04, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noencoder_v.pth", 2e-4, "noencoder_v")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/213_0.vtk", "/home/zhuxunyang/coding/simply/result_213_ablation_noencoder_e", 10, 0.04, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noencoder_e.pth", 2e-4, "noencoder_e")

    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/268_1.vtk", "/home/zhuxunyang/coding/simply/result_268_ablation_whole", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 1e-3)
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/268_1.vtk", "/home/zhuxunyang/coding/simply/result_268_ablation_nogine", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_nogine.pth", 2e-4, "nogine")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/268_1.vtk", "/home/zhuxunyang/coding/simply/result_268_ablation_noglobal", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noglobal.pth", 2e-4, "noglobal")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/268_1.vtk", "/home/zhuxunyang/coding/simply/result_268_ablation_noencoder_v", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noencoder_v.pth", 2e-4, "noencoder_v")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/268_1.vtk", "/home/zhuxunyang/coding/simply/result_268_ablation_noencoder_e", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noencoder_e.pth", 2e-4, "noencoder_e")

    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/256_0.vtk", "/home/zhuxunyang/coding/simply/result_256_ablation_whole", 10, 0.22, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 2e-4)
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/256_0.vtk", "/home/zhuxunyang/coding/simply/result_256_ablation_nogine", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_nogine.pth", 2e-4, "nogine")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/256_0.vtk", "/home/zhuxunyang/coding/simply/result_256_ablation_noglobal", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noglobal.pth", 2e-4, "noglobal")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/256_0.vtk", "/home/zhuxunyang/coding/simply/result_256_ablation_noencoder_v", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noencoder_v.pth", 2e-4, "noencoder_v")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/256_0.vtk", "/home/zhuxunyang/coding/simply/result_256_ablation_noencoder_e", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noencoder_e.pth", 2e-4, "noencoder_e")

    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/238_1.vtk", "/home/zhuxunyang/coding/simply/result_238_ablation_whole", 10, 0.18, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_1228.pth", 3e-4)
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/238_1.vtk", "/home/zhuxunyang/coding/simply/result_238_ablation_nogine", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_nogine.pth", 3e-4, "nogine")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/238_1.vtk", "/home/zhuxunyang/coding/simply/result_238_ablation_noglobal", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noglobal.pth", 3e-4, "noglobal")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/238_1.vtk", "/home/zhuxunyang/coding/simply/result_238_ablation_noencoder_v", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noencoder_v.pth", 3e-4, "noencoder_v")
    # test_example_ablation("/home/zhuxunyang/coding/simply/datasets/test_case/238_1.vtk", "/home/zhuxunyang/coding/simply/result_238_ablation_noencoder_e", 10, 0.25, 0.05, "/home/zhuxunyang/coding/simply/checkpoints/debug/GNN_Regression2_dis1Newsize5_ablation_noencoder_e.pth", 3e-4, "noencoder_e")


    