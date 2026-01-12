from tempfile import mkstemp
from shutil import move
import torch
import numpy as np
import os
from models.layers.mesh_union import MeshUnion
from models.layers.mesh_prepare import fill_mesh, fill_mesh_from_vtk


class Mesh:

    def __init__(self, file=None, opt=None, hold_history=False, export_folder=''):
        self.vs = self.v_mask = self.filename = self.features = self.edge_areas = self.size_value = None
        self.edges = self.gemm_edges = self.sides = None
        self.pool_count = 0
        # fill_mesh(self, file, opt)
        fill_mesh_from_vtk(self, file, opt)
        self.export_folder = export_folder
        self.history_data = None
        if hold_history:
            self.init_history()
        self.export()

    def collapsing_edge(self, edge_id):
        if edge_id >= len(self.edges):
            raise ValueError(f"边ID {edge_id} 超出范围 (总边数: {len(self.edges)})")
        
        v1, v2 = self.edges[edge_id]
        if v1 == v2:
            raise ValueError("边的两个顶点不能相同")

        # 计算新顶点位置
        new_vertex_pos = (self.vs[v1] + self.vs[v2]) / 2

        # 更新顶点位置
        self.vs[v1] = new_vertex_pos
        self.v_mask[v2] = False

        # 修改边中的顶点索引
        mask = self.edges == v2
        self.edges[mask] = v1

        # 移除与 v2 相关的边信息
        self.remove_edge(edge_id)
        for v in [v1, v2]:
            self.ve[v] = [e for e in self.ve[v] if e != edge_id]
        
        related_faces = []
        for face in self.faces:
            if v1 in face or v2 in face:
                related_faces.append(face)
        
        distance_sum = 0.0
        for face in related_faces:
            a, b, c = face
            va = self.vs[a]
            vb = self.vs[b]
            vc = self.vs[c]
            
            # 计算面法线
            ab = vb - va
            ac = vc - va
            normal = torch.linalg.cross(ab, ac)
            
            # 跳过退化面
            if torch.norm(normal) < 1e-8:
                continue
            
            # 计算平面方程参数
            d = -torch.dot(va, normal)
            
            # 计算点到面的距离
            distance = torch.abs(torch.dot(new_vertex_pos, normal) + d) / torch.norm(normal)
            distance_sum += distance

        new_vertices = torch.cat([self.vs[:v2], self.vs[v2+1:]])
        new_vertices[v1] = new_vertex_pos

        # 修改顶点索引
        new_faces = self.faces.clone()
        new_faces[new_faces == v2] = v1
        new_faces[new_faces > v2] -= 1

        # 检查退化面
        unique_counts = torch.tensor([torch.unique(tri).size(0) for tri in new_faces])
        non_degenerate_mask = unique_counts == 3
        newfaces = new_faces[non_degenerate_mask]

        new_size_value = (self.sizing_values[v1] + self.sizing_values[v2]) / 2
        new_sizing_values = torch.cat([self.sizing_values[:v2], self.sizing_values[v2+1:]])
        new_sizing_values[v1] = new_size_value





        return distance_sum

    def extract_features(self):
        return self.features

    def merge_vertices(self, edge_id):
        self.remove_edge(edge_id)
        edge = self.edges[edge_id]
        v_a = self.vs[edge[0]]
        v_b = self.vs[edge[1]]
        # update pA
        v_a.__iadd__(v_b)
        v_a.__itruediv__(2)
        self.v_mask[edge[1]] = False
        mask = self.edges == edge[1]
        self.ve[edge[0]].extend(self.ve[edge[1]])
        self.edges[mask] = edge[0]

    def remove_vertex(self, v):
        self.v_mask[v] = False

    def remove_edge(self, edge_id):
        vs = self.edges[edge_id]
        for v in vs:
            if edge_id not in self.ve[v]:
                print(self.ve[v])
                print(self.filename)
            self.ve[v].remove(edge_id)

    def clean(self, edges_mask, groups):
        edges_mask = edges_mask.astype(bool)
        torch_mask = torch.from_numpy(edges_mask.copy())
        self.gemm_edges = self.gemm_edges[edges_mask]
        self.edges = self.edges[edges_mask]
        self.sides = self.sides[edges_mask]
        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        self.gemm_edges[:, :] = new_indices[self.gemm_edges[:, :]]
        for v_index, ve in enumerate(self.ve):
            update_ve = []
            # if self.v_mask[v_index]:
            for e in ve:
                update_ve.append(new_indices[e])
            new_ve.append(update_ve)
        self.ve = new_ve
        self.__clean_history(groups, torch_mask)
        self.pool_count += 1
        self.export()


    def export(self, file=None, vcolor=None):
        if file is None:
            if self.export_folder:
                filename, file_extension = os.path.splitext(self.filename)
                file = '%s/%s_%d%s' % (self.export_folder, filename, self.pool_count, file_extension)
            else:
                return
        faces = []
        vs = self.vs[self.v_mask]
        gemm = np.array(self.gemm_edges)
        new_indices = np.zeros(self.v_mask.shape[0], dtype=np.int32)
        new_indices[self.v_mask] = np.arange(0, np.ma.where(self.v_mask)[0].shape[0])
        for edge_index in range(len(gemm)):
            cycles = self.__get_cycle(gemm, edge_index)
            for cycle in cycles:
                faces.append(self.__cycle_to_face(cycle, new_indices))
        with open(file, 'w+') as f:
            for vi, v in enumerate(vs):
                vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2]) if vcolor is not None else ''
                f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], vcol))
            for face_id in range(len(faces) - 1):
                f.write("f %d %d %d\n" % (faces[face_id][0] + 1, faces[face_id][1] + 1, faces[face_id][2] + 1))
            f.write("f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))
            for edge in self.edges:
                f.write("\ne %d %d" % (new_indices[edge[0]] + 1, new_indices[edge[1]] + 1))

    def export_segments(self, segments):
        if not self.export_folder:
            return
        cur_segments = segments
        for i in range(self.pool_count + 1):
            filename, file_extension = os.path.splitext(self.filename)
            file = '%s/%s_%d%s' % (self.export_folder, filename, i, file_extension)
            fh, abs_path = mkstemp()
            edge_key = 0
            with os.fdopen(fh, 'w') as new_file:
                with open(file) as old_file:
                    for line in old_file:
                        if line[0] == 'e':
                            new_file.write('%s %d' % (line.strip(), cur_segments[edge_key]))
                            if edge_key < len(cur_segments):
                                edge_key += 1
                                new_file.write('\n')
                        else:
                            new_file.write(line)
            os.remove(file)
            move(abs_path, file)
            if i < len(self.history_data['edges_mask']):
                cur_segments = segments[:len(self.history_data['edges_mask'][i])]
                cur_segments = cur_segments[self.history_data['edges_mask'][i]]

    def __get_cycle(self, gemm, edge_id):
        cycles = []
        for j in range(2):
            next_side = start_point = j * 2
            next_key = edge_id
            if gemm[edge_id, start_point] == -1:
                continue
            cycles.append([])
            for i in range(3):
                tmp_next_key = gemm[next_key, next_side]
                tmp_next_side = self.sides[next_key, next_side]
                tmp_next_side = tmp_next_side + 1 - 2 * (tmp_next_side % 2)
                gemm[next_key, next_side] = -1
                gemm[next_key, next_side + 1 - 2 * (next_side % 2)] = -1
                next_key = tmp_next_key
                next_side = tmp_next_side
                cycles[-1].append(next_key)
        return cycles

    def __cycle_to_face(self, cycle, v_indices):
        face = []
        for i in range(3):
            v = list(set(self.edges[cycle[i]]) & set(self.edges[cycle[(i + 1) % 3]]))[0]
            face.append(v_indices[v])
        return face

    def init_history(self):
        self.history_data = {
                               'groups': [],
                               'gemm_edges': [self.gemm_edges.copy()],
                               'occurrences': [],
                               'old2current': np.arange(self.edges_count, dtype=np.int32),
                               'current2old': np.arange(self.edges_count, dtype=np.int32),
                               'edges_mask': [torch.ones(self.edges_count,dtype=torch.bool)],
                               'edges_count': [self.edges_count],
                              }
        if self.export_folder:
            self.history_data['collapses'] = MeshUnion(self.edges_count)

    def union_groups(self, source, target):
        if self.export_folder and self.history_data:
            self.history_data['collapses'].union(self.history_data['current2old'][source], self.history_data['current2old'][target])
        return

    def remove_group(self, index):
        if self.history_data is not None:
            self.history_data['edges_mask'][-1][self.history_data['current2old'][index]] = 0
            self.history_data['old2current'][self.history_data['current2old'][index]] = -1
            if self.export_folder:
                self.history_data['collapses'].remove_group(self.history_data['current2old'][index])

    def get_groups(self):
        return self.history_data['groups'].pop()

    def get_occurrences(self):
        return self.history_data['occurrences'].pop()
    
    def __clean_history(self, groups, pool_mask):
        if self.history_data is not None:
            mask = self.history_data['old2current'] != -1
            self.history_data['old2current'][mask] = np.arange(self.edges_count, dtype=np.int32)
            self.history_data['current2old'][0: self.edges_count] = np.ma.where(mask)[0]
            if self.export_folder != '':
                self.history_data['edges_mask'].append(self.history_data['edges_mask'][-1].clone())
            self.history_data['occurrences'].append(groups.get_occurrences())
            self.history_data['groups'].append(groups.get_groups(pool_mask))
            self.history_data['gemm_edges'].append(self.gemm_edges.copy())
            self.history_data['edges_count'].append(self.edges_count)
    
    def unroll_gemm(self):
        self.history_data['gemm_edges'].pop()
        self.gemm_edges = self.history_data['gemm_edges'][-1]
        self.history_data['edges_count'].pop()
        self.edges_count = self.history_data['edges_count'][-1]

    def get_edge_areas(self):
        return self.edge_areas
