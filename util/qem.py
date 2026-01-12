import torch
import numpy as np
import heapq
import os
import subprocess

class BatchQEMSimplifier:
    def __init__(self, mesh):
        """
        基于 Batch 处理的 QEM 简化器
        :param mesh: 初始 CustomMesh 实例
        """
        self.current_mesh = mesh

    def _compute_face_quadrics(self, vertices, faces):
        """
        计算所有面片的 Kp 矩阵 (N_faces, 4, 4)
        """
        # 获取三角形顶点
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # 计算法向量
        e1 = v1 - v0
        e2 = v2 - v0
        normals = np.cross(e1, e2)
        areas = np.linalg.norm(normals, axis=1)
        
        # 避免除以零
        valid_mask = areas > 1e-12
        normals[valid_mask] /= areas[valid_mask, np.newaxis]
        
        # 平面方程系数 [a, b, c, d]
        # ax + by + cz + d = 0  => d = -(n . v)
        d = -np.sum(normals * v0, axis=1)
        
        # 构造 p = [a, b, c, d]
        p = np.column_stack((normals, d)) # (N_faces, 4)
        
        # Kp = p * p.T -> 利用广播机制批量计算
        # (N, 4, 1) * (N, 1, 4) -> (N, 4, 4)
        Kp = p[:, :, np.newaxis] @ p[:, np.newaxis, :]
        
        # 可选：按面积加权
        return Kp # * areas[:, np.newaxis, np.newaxis]

    def _compute_vertex_quadrics(self, vertices, faces):
        """
        计算所有顶点的 Q 矩阵
        """
        n_verts = len(vertices)
        Kp = self._compute_face_quadrics(vertices, faces)
        Q_v = np.zeros((n_verts, 4, 4))
        
        # 累加每个面片的 Kp 到其三个顶点
        # 使用 np.add.at 进行非缓冲累加
        for i in range(3):
            np.add.at(Q_v, faces[:, i], Kp)
            
        return Q_v

    def _calculate_edge_costs(self, mesh, Q_v):
        """
        计算所有边的坍缩代价
        返回: 排序后的边列表 [(cost, edge_id, v1, v2), ...]
        """
        edges = mesh.edges.cpu().numpy()
        vertices = mesh.vertices.cpu().numpy()
        
        # 获取边的两个端点
        v1_indices = edges[:, 0]
        v2_indices = edges[:, 1]
        
        # Q_bar = Q1 + Q2
        Q_bar = Q_v[v1_indices] + Q_v[v2_indices]
        
        # 简化策略：计算中点坍缩的代价
        # v_mid = (v1 + v2) / 2
        mid_points = (vertices[v1_indices] + vertices[v2_indices]) / 2.0
        
        # 扩展为齐次坐标 [x, y, z, 1]
        mid_points_h = np.column_stack((mid_points, np.ones(len(mid_points))))
        
        # Cost = v.T * Q * v
        # (N, 1, 4) @ (N, 4, 4) @ (N, 4, 1) -> (N, 1, 1)
        temp = np.einsum('ni,nij->nj', mid_points_h, Q_bar)
        costs = np.einsum('nj,nj->n', temp, mid_points_h)
        
        # --- 已移除特征点保护逻辑 ---
        # 无论边的两端点是否是特征点，代价均只由 QEM 误差决定
            
        # 组合结果并排序
        # 格式: (cost, edge_id, v1, v2)
        edge_data = []
        for i in range(len(costs)):
            edge_data.append((costs[i], i, v1_indices[i], v2_indices[i]))
            
        # 按代价排序
        edge_data.sort(key=lambda x: x[0])
        
        return edge_data

    def simplify(self, target_face_count, batch_ratio=0.1, work_dir="./temp_remesh"):
        """
        执行简化循环，包含 Batch Collapse 和 External Remesh
        :param target_face_count: 目标面片数量
        :param batch_ratio: 每轮尝试坍缩当前边总数的比例
        :param work_dir: 临时文件存储目录
        """
        iteration = 0
        
        # 确保工作目录存在
        if not os.path.exists(work_dir):
            try:
                os.makedirs(work_dir)
            except OSError as e:
                print(f"Error creating directory {work_dir}: {e}")
                return self.current_mesh
        
        # 保存初始网格作为参考（如果需要用于重投影或其他用途）
        begin_mesh = self.current_mesh
        
        while len(self.current_mesh.faces) > target_face_count:
            iteration += 1
            print(f"--- Iteration {iteration} (Current Faces: {len(self.current_mesh.faces)}) ---")

            # -------------------------------------------------
            # 1. 计算代价并筛选独立集 (Batch Collapse)
            # -------------------------------------------------
            mesh = self.current_mesh
            vertices_np = mesh.vertices.cpu().numpy()
            faces_np = mesh.faces.cpu().numpy()
            
            # 计算 Q 矩阵
            Q_v = self._compute_vertex_quadrics(vertices_np, faces_np)
            # 计算边代价（不再包含特征点保护）
            sorted_edges = self._calculate_edge_costs(mesh, Q_v)
            
            if not sorted_edges:
                print("No edges left to collapse.")
                break

            # 筛选独立边
            edges_to_collapse = []
            occupied_vertices = set()
            max_collapse_per_round = max(1, int(len(mesh.edges) * batch_ratio))
            
            for cost, edge_id, v1, v2 in sorted_edges:
                if v1 in occupied_vertices or v2 in occupied_vertices:
                    continue
                edges_to_collapse.append(edge_id)
                occupied_vertices.add(v1)
                occupied_vertices.add(v2)
                if len(edges_to_collapse) >= max_collapse_per_round:
                    break
            
            if not edges_to_collapse:
                print("Could not find valid independent edges.")
                break
                
            # 执行坍缩
            print(f"Collapsing {len(edges_to_collapse)} edges...")
            collapsed_mesh = mesh.collapse_multiple_edges2(edges_to_collapse)
            
            if collapsed_mesh is None or (hasattr(collapsed_mesh, 'is_empty') and collapsed_mesh.is_empty):
                print("Collapse step failed.")
                break

            # -------------------------------------------------
            # 2. 执行外部 Remesh 操作
            # -------------------------------------------------
            
            # 使用绝对路径以确保 subprocess 能正确找到文件
            save_path1 = os.path.abspath(os.path.join(work_dir, f"iter_{iteration}_input.vtk"))
            save_path2 = os.path.abspath(os.path.join(work_dir, f"iter_{iteration}_output.vtk"))
            
            # 保存当前坍缩后的网格
            try:
                collapsed_mesh.writeVTK(save_path1)
            except Exception as e:
                print(f"Failed to write input VTK: {e}")
                break
            
            # 构造命令
            command1 = [
                './remesh', 
                '--input', save_path1, 
                '--eps', '1e-4', 
                '--envelope-dis', '1e-0', 
                '--max-pass', '10', 
                '--output', save_path2,
                '--split-num', '0', 
                '--collapse-num', '0', 
                '--feature-angle', '45'
            ]
            
            remesh_success = False
            try:
                # 执行命令
                result1 = subprocess.run(command1, check=True, text=True, capture_output=True)
                print("remesh success")
                remesh_success = True
            except subprocess.CalledProcessError as e:
                print("remesh fail")
                print(f"Command: {' '.join(command1)}")
                print(f"Stderr: {e.stderr}")
                # remesh 失败时可以选择中止，或者继续使用 collapsed_mesh
                break
            except FileNotFoundError:
                print("Error: './remesh' executable not found.")
                break

            # 重新加载 Remesh 后的网格
            if remesh_success:
                if os.path.exists(save_path2):
                    try:
                        # 动态获取当前网格的类，用于加载 VTK
                        MeshClass = self.current_mesh.__class__
                        remeshed_mesh = MeshClass.from_vtk(save_path2)
                        
                        # 假设你需要重新计算尺寸场或其他属性（如果 CustomMesh 有此方法）
                        if hasattr(remeshed_mesh, 'recalculate_size'):
                             remeshed_mesh.recalculate_size(begin_mesh)
                        
                        # 更新当前网格
                        self.current_mesh = remeshed_mesh
                        print(f"Reloaded mesh. New face count: {len(self.current_mesh.faces)}")
                    except Exception as e:
                        print(f"Failed to reload mesh from {save_path2}: {e}")
                        break
                else:
                    print(f"Error: Output file {save_path2} was not generated.")
                    break
            
            # -------------------------------------------------
            # 3. 检查停止条件
            # -------------------------------------------------
            # 如果面数非常少，防止无限循环
            if len(self.current_mesh.faces) <= target_face_count:
                print("Target face count reached.")
                break
            
        return self.current_mesh