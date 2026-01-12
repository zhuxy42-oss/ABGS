import numpy as np
from scipy.integrate import tplquad, dblquad

GAUSS_POINTS = {
    1: ([(1/3, 1/3)], [1]),
    3: ([(1/6, 1/6), (2/3, 1/6), (1/6, 2/3)], [1/3] * 3),
    4: ([(1/3, 1/3), (0.2, 0.2), (0.6, 0.2), (0.2, 0.6)], [-27/48] + [25/48] * 3),
    6: ([(0.10128650732346, 0.10128650732346), (0.10128650732346, 0.79742698535309), (0.79742698535309, 0.10128650732346),
         (0.47014206410511, 0.05971587178977), (0.47014206410511, 0.47014206410511), (0.05971587178977, 0.47014206410511)],
        [0.06296959, 0.06296959, 0.06296959, 0.06619708, 0.06619708, 0.06619708])
}

# 定义三维三角形的三个顶点坐标
vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 5, 0]])

# 定义三个顶点的尺寸值
h_values = np.array([[1], [2], [3]])


# # 定义线性插值函数，用于计算三角面片中任意点 p 的尺寸值 h(p)
# def interpolate_h(u, v):
#     # 三角面片中的点 p 可以表示为 p = (1 - u - v) * vertices[0] + u * vertices[1] + v * vertices[2]
#     # 其中 u >= 0, v >= 0, u + v <= 1
#     h = (1 - u - v) * h_values[0] + u * h_values[1] + v * h_values[2]
#     return h


# # 定义被积函数 1/h(p)
# def integrand(u, v):
#     h = interpolate_h(u, v)
#     return 1 / h


# # 定义积分的上下限
# # 对于三维三角面片，u 的范围是 [0, 1]，v 的范围是 [0, 1 - u]，w 的范围是 [0, 1 - u - v]
# def v_upper(u):
#     return 1 - u

# # 进行三重积分
# result, _ = dblquad(integrand, 0, 1, lambda u: 0, lambda u: 1 - u)

# A, B, C = vertices
# AB = B - A
# AC = C - A
# J = np.linalg.norm(np.cross(AB, AC)) * 0.5

# print(f"积分结果: {result * J}")

def calculate_triangle_integral(vertices, h_values):
    # 定义线性插值函数，用于计算三角面片中任意点 p 的尺寸值 h(p)
    def interpolate_h(u, v):
        # 三角面片中的点 p 可以表示为 p = (1 - u - v) * vertices[0] + u * vertices[1] + v * vertices[2]
        # 其中 u >= 0, v >= 0, u + v <= 1
        h = (1 - u - v) * h_values[0] + u * h_values[1] + v * h_values[2]
        return h

    # 定义被积函数 1/h(p)
    def integrand(u, v):
        h = interpolate_h(u, v)
        return 1 / (h**2)

    # 定义积分的上下限
    # 对于三维三角面片，u 的范围是 [0, 1]，v 的范围是 [0, 1 - u]，w 的范围是 [0, 1 - u - v]
    def v_upper(u):
        return 1 - u

    # 进行二重积分
    result, _ = dblquad(integrand, 0, 1, lambda u: 0, lambda u: 1 - u)

    A, B, C = vertices
    AB = B - A
    AC = C - A
    J = np.linalg.norm(np.cross(AB, AC)) * 0.5

    return result * J * 2


def integrate_over_triangle(vertices, h_values):
        """
        计算三角面片上的 ∫(1/h(p)) dA
        Args:
            vertices: 三角形顶点坐标 [(x1, y1), (x2, y2), (x3, y3)]
            h_values: 三个顶点的尺寸值 [h1, h2, h3]
        Returns:
            积分结果
        """
        A, B, C = vertices
        # h_A, h_B, h_C = h_values
        h_A = h_values[0]
        h_B = h_values[1]
        h_C = h_values[2]

        # 计算雅可比行列式
        AB = B - A
        AC = C - A
        J = np.linalg.norm(np.cross(AB, AC)) / 2

        # 使用 Dunavant 5 阶积分规则（7 个点）
        points = [
            (0.33333333333333, 0.33333333333333),
            (0.10128650732346, 0.10128650732346),
            (0.10128650732346, 0.79742698535309),
            (0.79742698535309, 0.10128650732346),
            (0.47014206410511, 0.05971587178977),
            (0.47014206410511, 0.47014206410511),
            (0.05971587178977, 0.47014206410511)
        ]
        weights = [
            0.1125,  # 0.225 * 0.5
            0.06296959, 0.06296959, 0.06296959,  # 0.12593918 * 0.5
            0.06619708, 0.06619708, 0.06619708    # 0.13239415 * 0.5
        ]

        # num_points = len(points)
        # weights = [1 / num_points] * num_points

        # points, weights = GAUSS_POINTS.get(6, GAUSS_POINTS[6])
        
        integral_sum = 0.0
        for (u, v), w in zip(points, weights):
            # 计算 h(u, v) 通过重心坐标插值
            h = h_A * (1 - u - v) + h_B * u + h_C * v
            integral_sum += w * (1.0 / (h * h))

        # 乘以雅可比行列式得到实际积分值
        return integral_sum * J
    

# print(integrate_over_triangle(vertices, h_values))
# print(calculate_triangle_integral(vertices, h_values))