import numpy as np
def length2(vec):
    return np.dot(vec, vec)


def normalize(vec):
    return vec / np.linalg.norm(vec)


def dot(vec1, vec2):
    return np.dot(vec1, vec2)


def cross(vec1, vec2):
    return np.cross(vec1, vec2)


def distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def sampleTriangle(vs, sampling_dist=None):
    if sampling_dist is None:
        sampling_dist = max(distance(vs[0], vs[1]), distance(vs[0], vs[2]), distance(vs[1], vs[2])) / 10

    sqrt3_2 = np.sqrt(3) / 2
    ls = [length2(vs[i] - vs[(i + 1) % 3]) for i in range(3)]
    min_i = np.argmin(ls)
    max_i = np.argmax(ls)
    N = np.sqrt(ls[max_i]) / sampling_dist
    if N <= 1:
        return [vs[i] for i in range(3)]
    if N == int(N):
        N -= 1

    v0 = vs[max_i]
    v1 = vs[(max_i + 1) % 3]
    v2 = vs[(max_i + 2) % 3]

    n_v0v1 = normalize(v1 - v0)
    ps = [v0 + n_v0v1 * sampling_dist * n for n in range(int(N + 1))]
    ps.append(v1)

    h = distance(dot((v2 - v0), (v1 - v0)) * (v1 - v0) / ls[max_i] + v0, v2)
    M = int(h / (sqrt3_2 * sampling_dist))
    if M < 1:
        ps.append(v2)
        return ps

    n_v0v2 = normalize(v2 - v0)
    n_v1v2 = normalize(v2 - v1)
    sin_v0 = np.linalg.norm(cross((v2 - v0), (v1 - v0))) / (distance(v0, v2) * distance(v0, v1))
    tan_v0 = np.linalg.norm(cross((v2 - v0), (v1 - v0))) / dot((v2 - v0), (v1 - v0))
    tan_v1 = np.linalg.norm(cross((v2 - v1), (v0 - v1))) / dot((v2 - v1), (v0 - v1))
    sin_v1 = np.linalg.norm(cross((v2 - v1), (v0 - v1))) / (distance(v1, v2) * distance(v0, v1))

    for m in range(1, M + 1):
        n = int(sqrt3_2 / tan_v0 * m + 0.5)
        n1 = int(sqrt3_2 / tan_v0 * m)
        if m % 2 == 0 and n == n1:
            n += 1
        v0_m = v0 + m * sqrt3_2 * sampling_dist / sin_v0 * n_v0v2
        v1_m = v1 + m * sqrt3_2 * sampling_dist / sin_v1 * n_v1v2
        if distance(v0_m, v1_m) <= sampling_dist:
            break

        delta_d = ((n + (m % 2) / 2.0) - m * sqrt3_2 / tan_v0) * sampling_dist
        v = v0_m + delta_d * n_v0v1
        N1 = int(distance(v, v1_m) / sampling_dist)
        # ps.append(v0_m)
        ps.extend([v + n_v0v1 * sampling_dist * i for i in range(N1 + 1)])
        # ps.append(v1_m)

    ps.append(v2)

    N = np.sqrt(ls[(max_i + 1) % 3]) / sampling_dist
    if N > 1:
        if N == int(N):
            N -= 1
        n_v1v2 = normalize(v2 - v1)
        ps.extend([v1 + n_v1v2 * sampling_dist * n for n in range(1, int(N + 1))])

    N = np.sqrt(ls[(max_i + 2) % 3]) / sampling_dist
    if N > 1:
        if N == int(N):
            N -= 1
        n_v2v0 = normalize(v0 - v2)
        ps.extend([v2 + n_v2v0 * sampling_dist * n for n in range(1, int(N + 1))])

    return ps