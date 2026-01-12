import torch
import kaolin as kal
import numpy as np
import sys
import os
import meshio
from kaolin.rep import SurfaceMesh

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from util.calculate_gradient import map_triangle_to_2d, calculate_gradinet

class CustomMesh(SurfaceMesh):
    def __init__(self, vertices, faces, edge_topology=None, sizing_values=None):
        """
        初始化 CustomMesh 类。

        参数:
        vertices (torch.Tensor): 顶点坐标张量。
        faces (torch.Tensor): 面拓扑张量。
        edge_topology (torch.Tensor, 可选): 边拓扑张量。默认为 None。
        sizing_values (torch.Tensor, 可选): 尺寸值张量。默认为 None。
        """
        super().__init__(vertices=vertices, faces=faces)
        if edge_topology == None:
            self.create_edge_index()
        else:
            self.edges = edge_topology
        self.sizing_values = sizing_values
        self.batching = SurfaceMesh.Batching.LIST
        

    def __getattr__(self, name):
        if name not in self.__dict__:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return self.__dict__[name]

    @classmethod
    def from_vtk(cls, vtk_path: str):
        """
        从 VTK 文件初始化 CustomMesh 的类方法。

        参数:
        vtk_path (str): VTK 文件的路径。
        sizing_field (str, 可选): VTK 文件中用于尺寸值的点数据字段名。默认为 None。

        返回:
        CustomMesh: 通过 VTK 文件初始化的实例。

        异常:
        ValueError: 如果 VTK 文件不包含三角形面片或指定字段不存在。
        """
        # 读取 VTK 文件
        mesh = meshio.read(vtk_path)

        vertices = torch.tensor(mesh.points, dtype=torch.float32)

        if "triangle" not in mesh.cells_dict:
            raise ValueError("VTK 文件必须包含三角形面片数据（'triangle'类型单元）")
        faces = torch.tensor(mesh.cells_dict["triangle"], dtype=torch.long)
        
        Edges = torch.cat([
            faces[:, [0, 1]],  # 边 v0-v1
            faces[:, [1, 2]],  # 边 v1-v2
            faces[:, [2, 0]]   # 边 v2-v0
        ], dim=0)

        # 统一边方向（小索引在前）
        edges_sorted, _ = torch.sort(Edges, dim=1)
        
        # 去重并保持顺序
        edges = torch.unique(edges_sorted, dim=0)

        if "sizing_value" not in mesh.point_data:
            raise ValueError("VTK 文件不是背景网格")
        sizing_values = torch.tensor(mesh.point_data['sizing_value'], dtype=torch.float32)
        mesh = cls(vertices=vertices, faces=faces, edge_topology=edges, sizing_values=sizing_values)
        mesh.batching = SurfaceMesh.Batching.LIST
        return mesh

    def clone_mesh(self):
        new_mesh = CustomMesh(self.vertices, self.faces, self.edges, self.sizing_values)
        return new_mesh

    def get_all_info(self):
        """
        获取点、面拓扑、边拓扑和尺寸信息。

        返回:
        tuple: 包含顶点、面、边拓扑和尺寸值的元组。
        """
        print("There has ", len(self.vertices), " vertices ", len(self.faces), " faces ", len(self.edges), " edges ", len(self.sizing_values), " has point size")

    def get_surface_area(self):
        sum = 0.0
        for tri in self.faces:
            v0, v1, v2 = self.vertices[tri]
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross_product = torch.linalg.cross(edge1, edge2)
            area = 0.5 * torch.linalg.vector_norm(cross_product)
            sum += area
        return sum.item()

    def create_edge_index(self):
        edges = torch.cat([
            self.faces[:, [0, 1]],  # 边 v0-v1
            self.faces[:, [1, 2]],  # 边 v1-v2
            self.faces[:, [2, 0]]   # 边 v2-v0
        ], dim=0)

        # 统一边方向（小索引在前）
        edges_sorted, _ = torch.sort(self.edges, dim=1)

        # 去重并保持顺序
        unique_edges = torch.unique(edges_sorted, dim=0)
        
        self.edges = unique_edges.long()

    def collapse_edge(self, edge_id):
        if edge_id >= len(self.edges):
            raise ValueError(f"边ID {edge_id} 超出范围 (总边数: {len(self.edges)})")
        
        v1, v2 = self.edges[edge_id]
        if v1 == v2:
            raise ValueError("边的两个顶点不能相同")

        # 计算新顶点位置
        new_vertex_pos = (self.vertices[v1] + self.vertices[v2]) / 2

        if v1 > v2:
            v1, v2 = v2, v1

        self.vertices = torch.cat([self.vertices[:v2], self.vertices[v2+1:]])
        self.vertices[v1] = new_vertex_pos

        # face_mask = torch.ones(len(self.faces), dtype=torch.bool)
        # for i, tri in enumerate(self.faces):
        #     if v1 in tri and v2 in tri:
        #         face_mask[i] = False
        #         print("face collapse", i)
        #         continue
        #     for j in range(3):
        #         if self.faces[i][j] > v2:
        #             self.faces[i][j] = self.faces[i][j] - 1
        # self.faces = self.faces[face_mask]

        # 修改顶点索引
        new_faces = self.faces.clone()
        new_faces[new_faces == v2] = v1
        new_faces[new_faces > v2] -= 1

        # 检查退化面
        unique_counts = torch.tensor([torch.unique(tri).size(0) for tri in new_faces])
        non_degenerate_mask = unique_counts == 3
        self.faces = new_faces[non_degenerate_mask]

        new_size_value = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
        
        self.sizing_values = torch.cat([self.sizing_values[:v2], self.sizing_values[v2+1:]])
        self.sizing_values[v1] = new_size_value

        edges = torch.cat([self.faces[:, [0,1]], 
                         self.faces[:, [1,2]],
                         self.faces[:, [2,0]],
                         self.faces[:, [1,0]],
                         self.faces[:, [2,1]],
                         self.faces[:, [0,2]]], dim=0)
        sorted_edges, _ = torch.sort(edges, dim=1)
        self.edges = torch.unique(sorted_edges, dim=0)
    
    def collapse_edges(self, edge_ids):
        for id in edge_ids:
            self.collapse_edge(id)
    
    def save_to_vtk(self, file_path):
        points = self.vertices.numpy()
        cells = [("triangle", self.faces.numpy())]
        point_data = {"sizing_value": self.sizing_values.numpy()}

        mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data)
        mesh.write(file_path)

    def get_gradinet(self):
        sum = 0
        for tri in self.faces:
            v0, v1, v2 = tri
            point_value = torch.tensor([self.sizing_values[v0], self.sizing_values[v1], self.sizing_values[v2]])
            point3d_coord =  point3d_coord = torch.cat((self.vertices[v0].unsqueeze(0), self.vertices[v1].unsqueeze(0), self.vertices[v2].unsqueeze(0)), dim=0)
            
            point2d_coord = map_triangle_to_2d(point3d_coord)
            grad = calculate_gradinet(point2d_coord, point_value)
            sum += torch.abs(grad)
        return (sum / self.faces.shape[0]).item()



# if __name__ == '__main__':
#     mesh = CustomMesh.from_vtk("/home/zhuxunyang/coding/bkgm_simplification/datasets/bkgm/training/Backgroundmesh_cylinder_nosmooth58.vtk")
#     original_mesh = mesh.clone_mesh()
#     original_mesh.get_all_info()

#     mesh.collapse_edges([0, 10, 100, 120])
#     print("collapse success")
#     mesh.get_all_info()
#     # mesh.save_to_vtk("/home/zhuxunyang/coding/bkgm_simplification/result.vtk")

#     print(mesh.get_gradinet())
#     print(mesh.get_surface_area())



    


