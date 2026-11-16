import torch
import row_id_map_rt_ext

def kernel_function(routing_map):
    # 与 reference 输出顺序一致：([E,T] trans, [T,E] non-trans)，int64
    return row_id_map_rt_ext.row_id_map(routing_map)
