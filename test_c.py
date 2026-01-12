import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
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

    



# print(smooth_map(1))
# print(smooth_map(1.2))
# print(smooth_map(1e10))
print(smooth_map(torch.tensor([1, 1.1, 1.2, 1.3, 1.4, 2.0, 1e10])))


# 测试映射函数
x_values = np.linspace(0.8, 3, 500)
y1_values = [smooth_map(x) for x in x_values]
y2_value = [F.tanh(torch.tensor(x)).item() for x in x_values]

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(x_values, y1_values, color='blue', label='Custom Mapping')
plt.plot(x_values, y2_value, color='red', label='Sigmoid')
plt.axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=1.2, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.5)
plt.title('Custom Mapping with Tanh')
plt.xlabel('Input Value')
plt.ylabel('Mapped Value')
plt.grid(True)
plt.savefig('custom_mapping_plot.png')
plt.show()


