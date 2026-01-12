import numpy as np
import os, sys
import ntpath
import meshio
from collections import defaultdict
import torch

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from util.calculate_gradient import map_triangle_to_2d, calculate_gradinet
from util.bending import integrate_over_triangle, calculate_triangle_integral

def fill_mesh(mesh2fill, file: str, opt):
    load_path = get_mesh_path(file, opt.num_aug)
    if os.path.exists(load_path):
        mesh_data = np.load(load_path, encoding='latin1', allow_pickle=True)
    else:
        mesh_data = from_scratch(file, opt)
        np.savez_compressed(load_path, gemm_edges=mesh_data.gemm_edges, vs=mesh_data.vs, edges=mesh_data.edges,
                            edges_count=mesh_data.edges_count, ve=mesh_data.ve, v_mask=mesh_data.v_mask,
                            filename=mesh_data.filename, sides=mesh_data.sides,
                            edge_lengths=mesh_data.edge_lengths, edge_areas=mesh_data.edge_areas,
                            features=mesh_data.features, sizevalue=mesh_data.sizevalue)
    mesh2fill.vs = mesh_data['vs']
    mesh2fill.edges = mesh_data['edges']
    mesh2fill.gemm_edges = mesh_data['gemm_edges']
    mesh2fill.edges_count = int(mesh_data['edges_count'])
    mesh2fill.ve = mesh_data['ve']
    mesh2fill.v_mask = mesh_data['v_mask']
    mesh2fill.filename = str(mesh_data['filename'])
    mesh2fill.edge_lengths = mesh_data['edge_lengths']
    mesh2fill.edge_areas = mesh_data['edge_areas']
    mesh2fill.features = mesh_data['features']
    mesh2fill.sides = mesh_data['sides']

def fill_mesh_from_vtk(mesh2fill, file: str, opt):
    load_path = get_mesh_path(file, opt.num_aug)

    # # Check if npz file exists
    # if os.path.exists(load_path):
    #     # Load data from npz file
    #     mesh_data = np.load(load_path, allow_pickle=True)
    #     mesh2fill.vs = mesh_data['vs']
    #     mesh2fill.edges = mesh_data['edges']
    #     mesh2fill.gemm_edges = mesh_data['gemm_edges']
    #     mesh2fill.edges_count = int(mesh_data['edges_count'])
    #     mesh2fill.ve = mesh_data['ve']
    #     mesh2fill.v_mask = mesh_data['v_mask']
    #     mesh2fill.filename = str(mesh_data['filename'])
    #     mesh2fill.edge_lengths = mesh_data['edge_lengths']
    #     mesh2fill.edge_areas = mesh_data['edge_areas']
    #     mesh2fill.features = mesh_data['features']
    #     mesh2fill.sides = mesh_data['sides']
    #     mesh2fill.size_value = mesh_data['size_value']
    #     mesh_data.close()  # Remember to close the npz file
    #     return
    
    mesh_data = from_scratch_from_vtk(file, opt)
    
    np.savez_compressed(load_path, gemm_edges=mesh_data.gemm_edges, vs=mesh_data.vs, edges=mesh_data.edges,
                            edges_count=mesh_data.edges_count, ve=np.array(mesh_data.ve, dtype=object), v_mask=mesh_data.v_mask,
                            filename=mesh_data.filename, sides=mesh_data.sides,
                            edge_lengths=mesh_data.edge_lengths, edge_areas=mesh_data.edge_areas,
                            features=mesh_data.features, size_value=mesh_data.sizevalue)
    mesh2fill.vs = mesh_data['vs']
    mesh2fill.edges = mesh_data['edges']
    mesh2fill.gemm_edges = mesh_data['gemm_edges']
    mesh2fill.edges_count = int(mesh_data['edges_count'])
    mesh2fill.ve = mesh_data['ve']
    mesh2fill.v_mask = mesh_data['v_mask']
    mesh2fill.filename = str(mesh_data['filename'])
    mesh2fill.edge_lengths = mesh_data['edge_lengths']
    mesh2fill.edge_areas = mesh_data['edge_areas']
    mesh2fill.features = mesh_data['features']
    mesh2fill.sides = mesh_data['sides']
    mesh2fill.size_value = mesh_data['sizevalue']
    # mesh2fill.faces = mesh_data['faces']
    # mesh2fill.face_area = mesh_data['face_area']

def get_mesh_path(file: str, num_aug: int):
    filename, _ = os.path.splitext(file)
    dir_name = os.path.dirname(filename)
    prefix = os.path.basename(filename)
    load_dir = os.path.join(dir_name, 'cache')
    load_file = os.path.join(load_dir, '%s_%03d.npz' % (prefix, np.random.randint(0, num_aug)))
    if not os.path.isdir(load_dir):
        os.makedirs(load_dir, exist_ok=True)
    return load_file

def from_scratch(file, opt):

    class MeshPrep:
        def __getitem__(self, item):
            return eval('self.' + item)

    mesh_data = MeshPrep()
    mesh_data.vs = mesh_data.edges = None
    mesh_data.gemm_edges = mesh_data.sides = None
    mesh_data.edges_count = None
    mesh_data.ve = None
    mesh_data.v_mask = None
    mesh_data.filename = 'unknown'
    mesh_data.edge_lengths = None
    mesh_data.edge_areas = []
    mesh_data.vs, faces = fill_from_file(mesh_data, file)
    mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)
    faces, face_areas = remove_non_manifolds(mesh_data, faces)
    if opt.num_aug > 1:
        faces = augmentation(mesh_data, opt, faces)
    build_gemm(mesh_data, faces, face_areas)
    if opt.num_aug > 1:
        post_augmentation(mesh_data, opt)
    mesh_data.features = extract_features(mesh_data)
    return mesh_data

def from_scratch_from_vtk(file, opt):

    class MeshPrep:
        def __getitem__(self, item):
            return eval('self.' + item)

    mesh_data = MeshPrep()
    mesh_data.vs = mesh_data.edges = None
    mesh_data.gemm_edges = mesh_data.sides = None
    mesh_data.edges_count = None
    mesh_data.ve = None
    mesh_data.v_mask = None
    mesh_data.sizevalue = None
    mesh_data.filename = 'unknown'
    mesh_data.edge_lengths = None
    mesh_data.edge_areas = []
    mesh_data.edge2face = None    #边对应的另一个点的索引
    mesh_data.faces= None
    mesh_data.face_areas = None
    mesh_data.vs, faces, mesh_data.sizevalue, mesh_data.v2f = fill_from_vtk_file(mesh_data, file)
    mesh_data.faces = faces
    mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)
    faces, face_areas = remove_non_manifolds(mesh_data, faces)
    map_e2f(mesh_data)
    # mesh_data.v_curvature = compute_vertex_gaussian_curvature(mesh=mesh_data, faces=faces)
    if opt.num_aug > 1:
        faces = augmentation(mesh_data, opt, faces)
    build_gemm(mesh_data, faces, face_areas)      #得到网格拓扑
    if opt.num_aug > 1:
        post_augmentation(mesh_data, opt)
    
    mesh_data.features = extract_features(mesh_data)
    return mesh_data

def fill_from_file(mesh, file):
    mesh.filename = ntpath.split(file)[1]
    mesh.fullfilename = file
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces

def fill_from_vtk_file(mesh, file):
    mesh.filename = ntpath.split(file)[1]
    mesh.fullfilename = file
    try:
        vtk_mesh = meshio.read(file)
        vs = vtk_mesh.points

        # 归一化
        min_coords = vs.min(axis=0)
        centered = vs - min_coords
        scale = centered.max()
        vs = centered / scale

        if 'triangle' in vtk_mesh.cells_dict:
            faces = vtk_mesh.cells_dict['triangle']
        else:
            raise ValueError("VTK file does not contain triangle cells.")
        
        if 'sizing_value' in vtk_mesh.point_data:
            size_value = vtk_mesh.point_data['sizing_value']
        elif 'vertex_size' in vtk_mesh.point_data:
            size_value = vtk_mesh.point_data['vertex_size']
        else:
            raise ValueError("VTK file does not contain sizevalue.")
        
        n_vertices = len(vs)
        vertex_faces = [[] for _ in range(n_vertices)]
        for face_idx, face in enumerate(faces):
            for vertex_idx in face:
                vertex_faces[vertex_idx].append(face_idx)
        vertex_faces = np.asarray(vertex_faces, dtype=object)

    except Exception as e:
        print(f"Error reading VTK file: {e}")
        vs = []
        faces = []
        size_value = []

    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces, size_value, vertex_faces

def remove_non_manifolds(mesh, faces):
    mesh.ve = [[] for _ in mesh.vs]
    edges_set = set()
    mask = np.ones(len(faces), dtype=bool)
    _, face_areas = compute_face_normals_and_areas(mesh, faces)
    for face_id, face in enumerate(faces):
        if face_areas[face_id] == 0:
            mask[face_id] = False
            continue
        faces_edges = []
        is_manifold = False
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            if cur_edge in edges_set:
                is_manifold = True
                break
            else:
                faces_edges.append(cur_edge)
        if is_manifold:
            mask[face_id] = False
        else:
            for idx, edge in enumerate(faces_edges):
                edges_set.add(edge)
    return faces[mask], face_areas[mask]


def build_gemm(mesh, faces, face_areas):
    """
    gemm_edges: array (#E x 4) of the 4 one-ring neighbors for each edge
    sides: array (#E x 4) indices (values of: 0,1,2,3) indicating where an edge is in the gemm_edge entry of the 4 neighboring edges
    for example edge i -> gemm_edges[gemm_edges[i], sides[i]] == [i, i, i, i]
    """
    mesh.ve = [[] for _ in mesh.vs]
    edge_nb = []
    sides = []
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                sides.append([-1, -1, -1, -1])
                mesh.ve[edge[0]].append(edges_count)
                mesh.ve[edge[1]].append(edges_count)
                mesh.edge_areas.append(0)
                nb_count.append(0)
                edges_count += 1
            mesh.edge_areas[edge2key[edge]] += face_areas[face_id] / 3
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
            nb_count[edge_key] += 2
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
            sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
    mesh.edges = np.array(edges, dtype=np.int32)
    mesh.gemm_edges = np.array(edge_nb, dtype=np.int64)
    mesh.sides = np.array(sides, dtype=np.int64)
    mesh.edges_count = edges_count
    mesh.edge_areas = np.array(mesh.edge_areas, dtype=np.float32) / np.sum(face_areas) #todo whats the difference between edge_areas and edge_lenghts?


def compute_face_normals_and_areas(mesh, faces):
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_normals /= (face_areas[:, np.newaxis] + 1e-10)
    # assert (not np.any(face_areas[:, np.newaxis] == 0)), 'has zero area face: %s' % mesh.filename
    face_areas *= 0.5
    return face_normals, face_areas

def compute_vertex_gaussian_curvature(mesh, faces):
    vertex_curvatures = np.zeros(len(mesh.vs))
    for v_idx in range(len(mesh.vs)):
        # 获取顶点相邻的面
        adjacent_faces = mesh.v2f[v_idx]
        total_angle = 0.0
        for face_idx in adjacent_faces:
            if face_idx == -1:
                continue  # 跳过无效面索引
            face = faces[face_idx]
            # 找到顶点在面中的位置（0,1,2）
            local_idx = np.where(face == v_idx)[0][0]
            v0 = mesh.vs[face[local_idx]]
            v1 = mesh.vs[face[(local_idx+1)%3]]
            v2 = mesh.vs[face[(local_idx-1)%3]]
            # 计算内角
            vec1 = v1 - v0
            vec2 = v2 - v0
            cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2) + 1e-8)
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            total_angle += theta
        # 高斯曲率
        K = 2 * np.pi - total_angle
        vertex_curvatures[v_idx] = K
    return vertex_curvatures

def map_e2f(mesh):
    mesh.edge2face = defaultdict(list)
    mesh.face2edges = [[] for _ in range(len(mesh.faces))]
    
    for face_idx, face in enumerate(mesh.faces):
        # 获取面的三条边（确保边总是有序存储）
        edges = [
            tuple(sorted((face[i], face[(i+1)%3]))) 
            for i in range(3)
        ]
        mesh.face2edges[face_idx] = edges
        
        # 更新边到面的映射
        for edge in edges:
            mesh.edge2face[edge].append(face_idx)
    
    # 转换为普通字典并去重（如果是非流形网格）
    mesh.edge2face = {k: list(set(v)) for k, v in mesh.edge2face.items()}
    mesh.edges = list(mesh.edge2face.keys())

# Data augmentation methods
def augmentation(mesh, opt, faces=None):
    if hasattr(opt, 'scale_verts') and opt.scale_verts:
        scale_verts(mesh)
    if hasattr(opt, 'flip_edges') and opt.flip_edges:
        faces = flip_edges(mesh, opt.flip_edges, faces)
    return faces


def post_augmentation(mesh, opt):
    if hasattr(opt, 'slide_verts') and opt.slide_verts:
        slide_verts(mesh, opt.slide_verts)


def slide_verts(mesh, prct):
    edge_points = get_edge_points(mesh)
    dihedral = dihedral_angle(mesh, edge_points).squeeze() #todo make fixed_division epsilon=0
    thr = np.mean(dihedral) + np.std(dihedral)
    vids = np.random.permutation(len(mesh.ve))
    target = int(prct * len(vids))
    shifted = 0
    for vi in vids:
        if shifted < target:
            edges = mesh.ve[vi]
            if min(dihedral[edges]) > 2.65:
                edge = mesh.edges[np.random.choice(edges)]
                vi_t = edge[1] if vi == edge[0] else edge[0]
                nv = mesh.vs[vi] + np.random.uniform(0.2, 0.5) * (mesh.vs[vi_t] - mesh.vs[vi])
                mesh.vs[vi] = nv
                shifted += 1
        else:
            break
    mesh.shifted = shifted / len(mesh.ve)


def scale_verts(mesh, mean=1, var=0.1):
    for i in range(mesh.vs.shape[1]):
        mesh.vs[:, i] = mesh.vs[:, i] * np.random.normal(mean, var)


def angles_from_faces(mesh, edge_faces, faces):
    normals = [None, None]
    for i in range(2):
        edge_a = mesh.vs[faces[edge_faces[:, i], 2]] - mesh.vs[faces[edge_faces[:, i], 1]]
        edge_b = mesh.vs[faces[edge_faces[:, i], 1]] - mesh.vs[faces[edge_faces[:, i], 0]]
        normals[i] = np.cross(edge_a, edge_b)
        div = fixed_division(np.linalg.norm(normals[i], ord=2, axis=1), epsilon=0)
        normals[i] /= div[:, np.newaxis]
    dot = np.sum(normals[0] * normals[1], axis=1).clip(-1, 1)
    angles = np.pi - np.arccos(dot)
    return angles


def flip_edges(mesh, prct, faces):
    edge_count, edge_faces, edges_dict = get_edge_faces(faces)
    dihedral = angles_from_faces(mesh, edge_faces[:, 2:], faces)
    edges2flip = np.random.permutation(edge_count)
    # print(dihedral.min())
    # print(dihedral.max())
    target = int(prct * edge_count)
    flipped = 0
    for edge_key in edges2flip:
        if flipped == target:
            break
        if dihedral[edge_key] > 2.7:
            edge_info = edge_faces[edge_key]
            if edge_info[3] == -1:
                continue
            new_edge = tuple(sorted(list(set(faces[edge_info[2]]) ^ set(faces[edge_info[3]]))))
            if new_edge in edges_dict:
                continue
            new_faces = np.array(
                [[edge_info[1], new_edge[0], new_edge[1]], [edge_info[0], new_edge[0], new_edge[1]]])
            if check_area(mesh, new_faces):
                del edges_dict[(edge_info[0], edge_info[1])]
                edge_info[:2] = [new_edge[0], new_edge[1]]
                edges_dict[new_edge] = edge_key
                rebuild_face(faces[edge_info[2]], new_faces[0])
                rebuild_face(faces[edge_info[3]], new_faces[1])
                for i, face_id in enumerate([edge_info[2], edge_info[3]]):
                    cur_face = faces[face_id]
                    for j in range(3):
                        cur_edge = tuple(sorted((cur_face[j], cur_face[(j + 1) % 3])))
                        if cur_edge != new_edge:
                            cur_edge_key = edges_dict[cur_edge]
                            for idx, face_nb in enumerate(
                                    [edge_faces[cur_edge_key, 2], edge_faces[cur_edge_key, 3]]):
                                if face_nb == edge_info[2 + (i + 1) % 2]:
                                    edge_faces[cur_edge_key, 2 + idx] = face_id
                flipped += 1
    # print(flipped)
    return faces


def rebuild_face(face, new_face):
    new_point = list(set(new_face) - set(face))[0]
    for i in range(3):
        if face[i] not in new_face:
            face[i] = new_point
            break
    return face

def check_area(mesh, faces):
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_areas *= 0.5
    return face_areas[0] > 0 and face_areas[1] > 0


def get_edge_faces(faces):
    edge_count = 0
    edge_faces = []
    edge2keys = dict()
    for face_id, face in enumerate(faces):
        for i in range(3):
            cur_edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            if cur_edge not in edge2keys:
                edge2keys[cur_edge] = edge_count
                edge_count += 1
                edge_faces.append(np.array([cur_edge[0], cur_edge[1], -1, -1]))
            edge_key = edge2keys[cur_edge]
            if edge_faces[edge_key][2] == -1:
                edge_faces[edge_key][2] = face_id
            else:
                edge_faces[edge_key][3] = face_id
    return edge_count, np.array(edge_faces), edge2keys


def set_edge_lengths(mesh, edge_points=None):
    if edge_points is not None:
        edge_points = get_edge_points(mesh)
    edge_lengths = np.linalg.norm(mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], ord=2, axis=1)
    mesh.edge_lengths = edge_lengths

def get_edge_simplification_feature(mesh):
    features = []
    edge_points = get_edge_points(mesh)
    set_edge_lengths(mesh, edge_points)
    features.append()

def extract_features(mesh):
    features = []
    edge_points = get_edge_points(mesh)
    set_edge_lengths(mesh, edge_points)
    with np.errstate(divide='raise'):
        try:
            # for extractor in [dihedral_angle, symmetric_opposite_angles, symmetric_ratios, edge_length, size_diff, vertex_gaussian_curvature]:
            # for extractor in [dihedral_angle, symmetric_opposite_angles, symmetric_ratios, edge_length, middle_curvature, element_num, vertex_gaussian_curvature, size_gradinet, sizing]:
            for extractor in [dihedral_angle, symmetric_opposite_angles, symmetric_ratios]:
                feature = extractor(mesh, edge_points)
                features.append(feature)
            return np.concatenate(features, axis=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, 'bad features')

def vertex_gaussian_curvature(mesh, edge_points):
    curvature1 = mesh.v_curvature[edge_points[:, 0]]
    curvature2 = mesh.v_curvature[edge_points[:, 1]]
    curvature = (curvature1 + curvature2) / 2
    return curvature.reshape(1, -1)

def size_gradinet(mesh, edge_points):
    edge_distances = np.linalg.norm(mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], ord=2, axis=1)
    sizing_value_differences = np.abs(mesh.sizevalue[edge_points[:, 0]] - mesh.sizevalue[edge_points[:, 1]]).reshape(-1)
    size_grad = sizing_value_differences / edge_distances
    return size_grad.reshape(1, -1)

def sizing(mesh, edge_points):
    size1 = mesh.sizevalue[edge_points[:, 0]]
    size2 = mesh.sizevalue[edge_points[:, 1]]
    size3 = mesh.sizevalue[edge_points[:, 2]]
    size4 = mesh.sizevalue[edge_points[:, 3]]
    size = np.squeeze(np.stack([size1, size2, size3, size4]))
    return size

def edge_length(mesh, edge_points):
    edge_distances = np.linalg.norm(mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], ord=2, axis=1)
    return edge_distances.reshape(1, -1)

def size_diff(mesh, edge_points):
    sizing_value_differences = np.abs(mesh.sizevalue[edge_points[:, 0]] - mesh.sizevalue[edge_points[:, 1]])
    return sizing_value_differences.reshape(1, -1)

def element_num(mesh, edge_points):
    mean_elements = []
    for edge in edge_points:
        # face1, face2 = mesh.edge2face[(edge[0], edge[1])]
        v0, v1, v2, v3 = edge

        coord_3d_a = torch.stack([torch.tensor(mesh.vs[v0]), torch.tensor(mesh.vs[v1]), torch.tensor(mesh.vs[v2])])
        coord_2d_a = map_triangle_to_2d(coord_3d_a)
        size = np.array([mesh.sizevalue[v0], mesh.sizevalue[v1], mesh.sizevalue[v2]])
        num1 = integrate_over_triangle(coord_2d_a.numpy(), size)

        coord_3d_b = torch.stack([torch.tensor(mesh.vs[v0]), torch.tensor(mesh.vs[v1]), torch.tensor(mesh.vs[v3])])
        coord_2d_b = map_triangle_to_2d(coord_3d_b)
        size = np.array([mesh.sizevalue[v0], mesh.sizevalue[v1], mesh.sizevalue[v3]])
        num2 = integrate_over_triangle(coord_2d_b.numpy(), size)
        mean_elements.append([num1, num2])
    mean_elements = np.array(mean_elements)
    return np.squeeze(mean_elements).T


def extract_features_vtk(mesh):
    # 获取边的信息
    edge_points = get_edge_points(mesh)
    # 计算边的两个点的距离
    edge_distances = np.linalg.norm(mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], ord=2, axis=1)
    # 在第 0 维上增加一个维度
    edge_distances = np.expand_dims(edge_distances, axis=0)
    
    # 计算边的两个点的 sizing_value 差
    sizing_value_differences = np.abs(mesh.sizevalue[edge_points[:, 0]] - mesh.sizevalue[edge_points[:, 1]])
    # 在第 0 维上增加一个维度
    if np.ndim(sizing_value_differences) == 1:
        sizing_value_differences = np.expand_dims(sizing_value_differences, axis=-1)
    sizing_value_difference = np.transpose(sizing_value_differences, (1, 0))
    
    # 将距离和 sizing_value 差作为特征
    features = np.concatenate([edge_distances, sizing_value_difference], axis=0)
    return features


def dihedral_angle(mesh, edge_points):
    normals_a = get_normals(mesh, edge_points, 0)
    normals_b = get_normals(mesh, edge_points, 3)
    dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
    angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
    return angles

def middle_curvature(mesh, edge_points):
    normals_a = get_normals(mesh, edge_points, 0)
    normals_b = get_normals(mesh, edge_points, 3)
    cos_theta = np.einsum('ij,ij->i', normals_a, normals_b)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    curvature = theta / (np.linalg.norm(mesh.vs[edge_points[:, 0]] - mesh.vs[edge_points[:, 1]], ord=2, axis=1) + 1e-8)
    return curvature.reshape(1, -1)


def symmetric_opposite_angles(mesh, edge_points):
    """ computes two angles: one for each face shared between the edge
        the angle is in each face opposite the edge
        sort handles order ambiguity
    """
    angles_a = get_opposite_angles(mesh, edge_points, 0)
    angles_b = get_opposite_angles(mesh, edge_points, 3)
    angles = np.concatenate((np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0)), axis=0)
    angles = np.sort(angles, axis=0)
    return angles


def symmetric_ratios(mesh, edge_points):
    """ computes two ratios: one for each face shared between the edge
        the ratio is between the height / base (edge) of each triangle
        sort handles order ambiguity
    """
    ratios_a = get_ratios(mesh, edge_points, 0)
    ratios_b = get_ratios(mesh, edge_points, 3)
    ratios = np.concatenate((np.expand_dims(ratios_a, 0), np.expand_dims(ratios_b, 0)), axis=0)
    return np.sort(ratios, axis=0)


def get_edge_points(mesh):
    """ returns: edge_points (#E x 4) tensor, with four vertex ids per edge
        for example: edge_points[edge_id, 0] and edge_points[edge_id, 1] are the two vertices which define edge_id 
        each adjacent face to edge_id has another vertex, which is edge_points[edge_id, 2] or edge_points[edge_id, 3]
    """
    edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
    for edge_id, edge in enumerate(mesh.edges):
        edge_points[edge_id] = get_side_points(mesh, edge_id)
        # edge_points[edge_id, 3:] = mesh.get_side_points(edge_id, 2)
    return edge_points


def get_side_points(mesh, edge_id):
    # if mesh.gemm_edges[edge_id, side] == -1:
    #     return mesh.get_side_points(edge_id, ((side + 2) % 4))
    # else:
    edge_a = mesh.edges[edge_id]

    if mesh.gemm_edges[edge_id, 0] == -1:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    else:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    if mesh.gemm_edges[edge_id, 2] == -1:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    else:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    first_vertex = 0
    second_vertex = 0
    third_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1
    return [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]


def get_normals(mesh, edge_points, side):
    edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - mesh.vs[edge_points[:, side // 2]]
    edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2]]
    normals = np.cross(edge_a, edge_b)
    div = fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
    normals /= div[:, np.newaxis]
    return normals

def get_opposite_angles(mesh, edge_points, side):
    edges_a = mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]
    edges_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]

    edges_a /= fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    edges_b /= fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
    return np.arccos(dot)


def get_ratios(mesh, edge_points, side):
    edges_lengths = np.linalg.norm(mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, 1 - side // 2]],
                                   ord=2, axis=1)
    point_o = mesh.vs[edge_points[:, side // 2 + 2]]
    point_a = mesh.vs[edge_points[:, side // 2]]
    point_b = mesh.vs[edge_points[:, 1 - side // 2]]
    line_ab = point_b - point_a
    projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / fixed_division(
        np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
    closest_point = point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
    d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
    return d / edges_lengths

def fixed_division(to_div, epsilon):
    if epsilon == 0:
        to_div[to_div == 0] = 0.1
    else:
        to_div += epsilon
    return to_div
