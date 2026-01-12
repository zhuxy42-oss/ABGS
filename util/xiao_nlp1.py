import numpy as np
import torch
import cyipopt
from scipy.sparse import coo_matrix

class SizingFieldSmoother:
    def __init__(self, mesh, active_node_count, beta=1.2, h_min=0.1):
        self.mesh = mesh
        self.beta = beta
        self.h_min = h_min
        self.n_nodes_active = active_node_count
        
        # --- 1. 数据切片 ---
        self.vertices = mesh.vertices[:self.n_nodes_active].detach().cpu().numpy()
        
        if mesh.sizing_values is None:
            raise ValueError("Mesh sizing_values cannot be None.")
        
        self.h0 = mesh.sizing_values[:self.n_nodes_active].detach().cpu().numpy().flatten()
        
        # --- 2. 拓扑过滤 ---
        raw_faces = mesh.faces.detach().cpu().numpy()
        if raw_faces.shape[0] == 3 and raw_faces.shape[1] > 3:
            raw_faces = raw_faces.T
            
        valid_mask = (raw_faces < self.n_nodes_active).all(axis=1)
        self.faces = raw_faces[valid_mask]
        self.n_elements = len(self.faces)
        
        # --- 3. 预计算 K 矩阵和雅可比稀疏结构 ---
        self.Ks = self._precompute_K_matrices()
        
        # 预计算雅可比矩阵的行与列索引 (Sparsity Structure)
        # 结构是固定的，只有数值会变，这样能极大加速 IPOPT
        self.jac_rows, self.jac_cols = self._precompute_jacobian_structure()

    def _precompute_K_matrices(self):
        """计算几何矩阵 K"""
        Ks = np.zeros((self.n_elements, 3, 3))
        for i, face in enumerate(self.faces):
            p0, p1, p2 = self.vertices[face]
            e1, e2 = p1 - p0, p2 - p0
            normal = np.cross(e1, e2)
            n_norm = np.linalg.norm(normal)
            if n_norm < 1e-12: continue
            
            u = e1 / np.linalg.norm(e1)
            w = normal / n_norm
            v = np.cross(w, u)
            
            x1, y1 = np.linalg.norm(e1), 0.0
            vec2 = p2 - p0
            x2, y2 = np.dot(vec2, u), np.dot(vec2, v)
            
            Area = 0.5 * np.abs(x1 * y2)
            if Area < 1e-12: continue
            
            b0, c0 = -y2, x2 - x1
            b1, c1 = y2, -x2
            b2, c2 = 0, x1
            bs, cs = [b0, b1, b2], [c0, c1, c2]
            
            factor = 1.0 / (4.0 * Area**2)
            for r in range(3):
                for c in range(3):
                    Ks[i, r, c] = (bs[r]*bs[c] + cs[r]*cs[c]) * factor
        return Ks

    def _precompute_jacobian_structure(self):
        """
        一次性计算出雅可比矩阵非零元素的坐标 (Row, Col)。
        IPOPT 需要知道稀疏矩阵的结构。
        """
        rows = []
        cols = []
        # 每个单元贡献 3 个非零元素（对应三角形的 3 个顶点）
        for i in range(self.n_elements):
            idx = self.faces[i]
            for local_j in range(3):
                # idx[local_j] 一定 < n_nodes_active，因为做了拓扑过滤
                rows.append(i)
                cols.append(idx[local_j])
        return np.array(rows, dtype=int), np.array(cols, dtype=int)

    def solve(self, verbose=0):
        """
        使用 CyIpopt 定义问题并求解
        """
        
        # --- 定义内部类或闭包来适配 CyIpopt 的接口 ---
        class IpoptProblem:
            def __init__(self, outer):
                self.outer = outer
                self.limit = np.log(outer.beta)**2
                
            def objective(self, h):
                """目标函数"""
                return 0.5 * np.sum((h - self.outer.h0)**2)

            def gradient(self, h):
                """目标函数的梯度 (Dense Vector)"""
                return h - self.outer.h0

            def constraints(self, h):
                """
                约束函数值 (Vector of size n_elements)
                约束形式: limit - h^T K h >= 0
                """
                cons = np.zeros(self.outer.n_elements)
                for i in range(self.outer.n_elements):
                    idx = self.outer.faces[i]
                    h_local = h[idx]
                    grad_sq_norm = h_local @ self.outer.Ks[i] @ h_local
                    cons[i] = self.limit - grad_sq_norm
                return cons

            def jacobian(self, h):
                """
                约束雅可比矩阵的非零数值 (Flat Array)
                注意：必须与 jacobianstructure 返回的索引顺序严格对应
                """
                # 预分配数组，长度等于非零元素个数
                data = np.zeros(len(self.outer.jac_rows))
                
                # 这里的循环顺序必须与 _precompute_jacobian_structure 完全一致
                ptr = 0
                for i in range(self.outer.n_elements):
                    idx = self.outer.faces[i]
                    h_local = h[idx]
                    
                    # 导数: -2 * K * H
                    deriv = -2.0 * (self.outer.Ks[i] @ h_local)
                    
                    for local_j in range(3):
                        data[ptr] = deriv[local_j]
                        ptr += 1
                return data

            def jacobianstructure(self):
                """返回稀疏矩阵的坐标 (Row indices, Col indices)"""
                return (self.outer.jac_rows, self.outer.jac_cols)

            # Hessian 近似使用 L-BFGS，不需要手动实现 hessian() 和 hessianstructure()
            # 这是 IPOPT 的一大优势

        # --- 准备 IPOPT 参数 ---
        problem_obj = IpoptProblem(self)
        
        # 变量边界 (Box Bounds): h_min <= h <= h0
        lb = np.full(self.n_nodes_active, self.h_min)
        ub = self.h0
        
        # 约束边界 (Constraint Bounds): 0 <= (limit - grad^2) <= inf
        # 即满足 limit - grad^2 >= 0
        cl = np.zeros(self.n_elements)
        cu = np.full(self.n_elements, 2.0e19) # 代表无穷大
        
        # 初始猜测
        x0 = self.h0.copy()
        
        # 创建 IPOPT 问题实例
        nlp = cyipopt.Problem(
            n=self.n_nodes_active,
            m=self.n_elements,
            problem_obj=problem_obj,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
        )
        
        # --- 设置求解器选项 ---
        # 关键设置: limited-memory 表示使用 L-BFGS 近似 Hessian，极大降低内存和计算量
        nlp.add_option('hessian_approximation', 'limited-memory')
        # nlp.add_option('mu_strategy', 'adaptive') # 自适应罚参数策略，通常收敛更快
        nlp.add_option('tol', 1e-4)
        nlp.add_option('max_iter', 5000)
        # nlp.add_option('constr_viol_tol', 1e-8)
        # nlp.add_option('compl_inf_tol', 1e-8)
        # nlp.add_option('limited_memory_max_history', 20)
        # nlp.add_option('mu_strategy', 'monotone')
        
        # 根据 verbose 设置输出级别
        if verbose == 0:
            nlp.add_option('print_level', 0)
        else:
            nlp.add_option('print_level', 5) # 5 是默认详细程度

        # --- 求解 ---
        if verbose > 1:
            print(f"IPOPT solving... Active Nodes: {self.n_nodes_active}, Constraints: {self.n_elements}")
            
        h_optimized, info = nlp.solve(x0)
        
        if verbose > 0:
            print(f"Optimization finished. Status: {info['status_msg']}")
            # 打印一些性能数据
            # print(f"Objective value: {info['obj_val']:.6f}")

        # --- 结果回写 ---
        device = self.mesh.vertices.device
        dtype = self.mesh.sizing_values.dtype
        
        new_active_values = torch.from_numpy(h_optimized).to(device=device, dtype=dtype)
        
        if self.mesh.sizing_values.dim() == 1:
            self.mesh.sizing_values[:self.n_nodes_active] = new_active_values
        else:
            self.mesh.sizing_values[:self.n_nodes_active, 0] = new_active_values
            
        return self.mesh

def smooth_mesh_sizing(custom_mesh_instance, active_node_count, beta=1.2, h_min=0.1):
    smoother = SizingFieldSmoother(custom_mesh_instance, active_node_count, beta=beta, h_min=h_min)
    return smoother.solve()