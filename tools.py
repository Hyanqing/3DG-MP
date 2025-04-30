import numpy as np

import torch

def from_scaler(scaler, list):
    _arr = np.array(list).reshape(-1, 1)
    arr = scaler.inverse_transform(_arr).reshape(-1)
    arr = arr - 273.15
    return arr


def get_complete_covalent_graph(edge_index, n_nodes, edge_attr, device):
    n_list=[]
    for i in range(n_nodes):
        for j in range(n_nodes):
            if(i!=j):
                n_list.append([i, j])
    # n_list.to(device)
    new_edge_index = torch.tensor(np.array(n_list).T)

    combined_edge_index = torch.cat([edge_index, new_edge_index], dim=1)
    combined_edge_index = torch.unique(combined_edge_index, dim=1)  # 去重
    # 为新边分配属性（此处为全零）
    new_edge_attr = torch.zeros(new_edge_index.shape[1], edge_attr[0].shape[0])
    combined_edge_attr = torch.cat([edge_attr, new_edge_attr], dim=0)[:combined_edge_index.shape[1]]

    return combined_edge_index, combined_edge_attr