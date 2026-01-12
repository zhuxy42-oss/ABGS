import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import torch

def smooth_sizing_function(mesh, beta=1.2, method='trust-constr', tol=1e-6):
    """
    平滑尺寸函数以满足梯度约束。
    
    参数:
        mesh: CustomMesh实例，包含顶点、面和初始尺寸值。
        beta: 渐进因子，控制尺寸变化率。
        h_min: 最小允许尺寸值。
        method: 优化方法，可选'trust-constr'或'SLSQP'。
        tol: 优化容忍度。
    
    返回:
        smoothed_sizing: 平滑后的尺寸值数组。
    """
    vertices = mesh.vertices.cpu().numpy() if torch.is_tensor(mesh.vertices) else mesh.vertices
    faces = mesh.faces.cpu().numpy() if torch.is_tensor(mesh.faces) else mesh.faces
    h0 = mesh.sizing_values.cpu().numpy()
    h0 = h0.flatten()
    n_nodes = len(vertices)

    # 分别提取x、y、z坐标
    x_coords = mesh.vertices[:, 0]
    y_coords = mesh.vertices[:, 1]
    z_coords = mesh.vertices[:, 2]
    
    # 计算每个轴的最小值和最大值
    x_min = torch.min(x_coords).item()
    x_max = torch.max(x_coords).item()
    y_min = torch.min(y_coords).item()
    y_max = torch.max(y_coords).item()
    z_min = torch.min(z_coords).item()
    z_max = torch.max(z_coords).item()

    L = max(x_max - x_min, max(y_max - y_min, z_max - z_min))

    
    h_min=L * 4 / 100000
    h_max= L / 10
    
    # 计算每个三角形的K矩阵和面积
    K_matrices, areas = compute_K_matrices(vertices, faces)
    
    # 定义目标函数
    def objective(x):
        return np.sum((x - h0) ** 2)
    
    # 定义目标函数的梯度和Hessian
    def objective_grad(x):
        # 确保返回一维数组
        return 2 * (x - h0).flatten()

    
    def objective_hess(x):
        return 2 * np.eye(n_nodes)
    
    # 定义梯度约束函数
    def gradient_constraint(x):
        constraints = []
        for i, (face, K, A) in enumerate(zip(faces, K_matrices, areas)):
            H = x[face]
            value = H @ K @ H - (np.log(beta) ** 2)  
            constraints.append(value)
        return np.array(constraints)
    
    # 定义梯度约束的Jacobian（稀疏结构）
    def gradient_constraint_jacobian(x):
        jac_data = []
        jac_row = []
        jac_col = []
        
        for i, (face, K, A) in enumerate(zip(faces, K_matrices, areas)):
            H = x[face]
            grad = 2 * K @ H
            for j, idx in enumerate(face):
                jac_data.append(grad[j])
                jac_row.append(i)
                jac_col.append(idx)
        
        # 确保索引是整数
        jac_row = np.array(jac_row, dtype=int)
        jac_col = np.array(jac_col, dtype=int)
        jac_data = np.array(jac_data)
        
        # 使用scipy的稀疏矩阵格式
        from scipy.sparse import csr_matrix
        return csr_matrix((jac_data, (jac_row, jac_col)), shape=(len(faces), n_nodes))


    
    # 变量边界
    lb = h_min * np.ones(n_nodes)
    ub = h_max * np.ones(n_nodes)
    bounds = Bounds(lb, ub)
    
    # 非线性约束
    n_constraints = len(faces)
    constraint = NonlinearConstraint(
        gradient_constraint, -np.inf, 0, jac=gradient_constraint_jacobian
    )
    
    # 初始猜测
    x0 = h0.copy().flatten()
    
    print("x0 shape:", np.shape(x0))



    # 求解优化问题
    result = minimize(
        objective,
        x0,
        method=method,
        jac=objective_grad,
        hess=objective_hess,
        constraints=[constraint],
        bounds=bounds,
        options={'verbose': 1, 'maxiter': 10000, 'gtol': tol}
    )
    
    if result.success:
        smoothed_sizing = result.x
    else:
        raise RuntimeError("优化失败: " + result.message)
    
    return smoothed_sizing

def compute_K_matrices(vertices, faces):
    """
    计算每个三角形单元的K矩阵和面积。
    
    参数:
        vertices: 顶点数组。
        faces: 面数组。
    
    返回:
        K_matrices: 每个三角形的K矩阵列表。
        areas: 每个三角形的面积列表。
    """
    K_matrices = []
    areas = []
    for face in faces:
        A, B, C = vertices[face]
        # 计算向量AB和AC
        AB = B - A
        AC = C - A
        # 计算法向量和面积
        normal = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(normal)
        areas.append(area)
        # 计算几何系数b和c
        b0 = B[1] - C[1]
        b1 = C[1] - A[1]
        b2 = A[1] - B[1]
        c0 = C[0] - B[0]
        c1 = A[0] - C[0]
        c2 = B[0] - A[0]
        # 构建K矩阵
        K = np.array([
            [b0*b0 + c0*c0, b0*b1 + c0*c1, b0*b2 + c0*c2],
            [b1*b0 + c1*c0, b1*b1 + c1*c1, b1*b2 + c1*c2],
            [b2*b0 + c2*c0, b2*b1 + c2*c1, b2*b2 + c2*c2]
        ]) / (4 * area ** 2)
        K_matrices.append(K)
    return K_matrices, areas