import torch
import kaolin as kal

# 自定义顶点张量
vertices = torch.tensor([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0]
], dtype=torch.float32)

# 自定义面张量
faces = torch.tensor([
    [0, 1, 2],
    [1, 3, 2]
], dtype=torch.long)

# 创建一个简单的网格对象
mesh = kal.io.obj.Mesh(vertices=vertices, faces=faces)

# 打印网格信息
print("Vertices:", mesh.vertices)
print("Faces:", mesh.faces)