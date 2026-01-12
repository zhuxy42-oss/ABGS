import numpy as np

def split_into_continuous_segments(indices):
    if not indices:
        return []
    segments = []
    current_segment = [indices[0]]
    for idx in indices[1:]:
        if idx == current_segment[-1] + 1:
            current_segment.append(idx)
        else:
            segments.append(current_segment)
            current_segment = [idx]
    segments.append(current_segment)
    return segments

def replace_tensor(tensor, a, b, data):
    tensor = np.array(tensor)  # Ensure it's a numpy array
    a_segments = split_into_continuous_segments(a)
    b_segments = split_into_continuous_segments(b)
    
    if len(a_segments) != len(b_segments):
        raise ValueError("The number of segments in 'a' and 'b' must be the same.")
    
    # Split data into segments corresponding to b_segments
    data = np.array(data)
    data_segments = []
    start = 0
    for segment in b_segments:
        length = len(segment)
        data_segments.append(data[start:start+length])
        start += length
    
    # Perform replacements
    output = []
    begin_index = 0
    for a_seg, b_seg, data_seg in zip(a_segments, b_segments, data_segments):

        output.append(tensor[begin_index:a_seg[0]])
        output.append(data_seg)
        begin_index = a_seg[-1] + 1

    if begin_index < len(tensor):
        output.append(tensor[begin_index:len(tensor)])
    
    return np.concatenate(output)

# # 示例用法
# tensor = np.array([0, 0, 0, 0, 0, 0, 0])
# a = [1, 2, 5, 6]
# b = [1, 2, 3, 6, 7, 8]
# data = [1, 1, 1, 2, 2, 2]
# result = replace_tensor(tensor, a, b, data)
# print(result)  # 输出: [0 1 1 1 0 0 2 2 2]