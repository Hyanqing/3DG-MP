import os

import pandas as pd
import torch
from mol_dataset import MoleculeDatasetComplete
from torch_geometric.data import ( DataLoader)
from models import GNN_graphpredComplete
import egnn_clean as eg
from tools import from_scaler
from train import pred, eval


def use_model():

    n_conf = 3

    dim = 300
    layer_hidden = 7

    batch_train = 64
    batch_test = 64
    lr = 1e-4
    lr_decay = 0.99
    decay = 0
    iteration = 30

    JK = 'last'
    num_workers = 8
    num_tasks = 1

    model_path = rf"./model_dir/model_best_{n_conf}.pth"
    pred_path = 'geometric_data_processed_test.pt'


    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('-' * 100)

    scaler = torch.load(rf"./data/scaler_melt_{n_conf}.pt")

    data_new = MoleculeDatasetComplete(r'./data', dataset="melt", path=pred_path)
    new_loader = DataLoader(data_new, batch_size=batch_test, shuffle=False, num_workers=num_workers)

    molecule_model = eg.EGNN(in_node_nf=9, hidden_nf=dim, out_node_nf=dim, in_edge_nf=3).to(device)
    model = GNN_graphpredComplete(num_layer=layer_hidden, emb_dim=dim, JK=JK, graph_pooling='mean',
                                  num_tasks=num_tasks, molecule_model=molecule_model)

    if os.path.exists(model_path):
        print("loading.........")
        model.load_state_dict(torch.load(model_path), strict=False)
    for _model in model.molecule_model.parameters():
        print(_model)

    model.to(device)

    test_new = pred(model, device, new_loader)
    test_new = from_scaler(scaler, test_new)
    df_new = pd.DataFrame()
    df_new['pred'] = test_new
    _path_new = os.path.join(r"./data", f"pred_{n_conf}_26w_3d.csv")
    df_new.to_csv(_path_new, index=False)

if __name__ == "__main__":
    # mode = "val"
    use_model()
