import os
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import (Data, DataLoader)
import warnings
import egnn_clean as eg
from tqdm import tqdm
from models import GNN_graphpredComplete
from split_set import random_split, scaffold_split
from mol_dataset import MoleculeDatasetnew, MoleculeDatasetComplete
from tools import from_scaler, get_complete_covalent_graph
from train import train, eval, pred
import argparse

def main():
    parser = argparse.ArgumentParser(description='melt')
    parser.add_argument('--dataset', type=str, default='melt_coo')
    parser.add_argument('--model_path', type=str, default=r"./model_dir/conf/model_best.pth")
    parser.add_argument('--train_path', type=str, default='geometric_data_processed_3_23w.pt')
    parser.add_argument('--n_conf', type=int, default=3)
    parser.add_argument('--dim', type=int, default=300)
    parser.add_argument('--layer_hidden', type=str, default=7)
    parser.add_argument('--batch_train', type=str, default=64)
    parser.add_argument('--batch_test', type=str, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=0.99)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--iteration', type=int, default=30)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--eval_train', type=bool, default=True)
    parser.add_argument('--output_model_dir', type=str, default=r'./model_dir')

    args = parser.parse_args()

    dataset = args.dataset
    model_path = args.model_path
    # train_path = args.train_path
    train_path = args.train_path

    n_conf = args.n_conf

    dim = args.dim
    layer_hidden = args.layer_hidden

    batch_train = args.batch_train
    batch_test = args.batch_test
    lr = args.lr
    lr_decay = args.lr_decay
    decay = args.decay
    iteration = args.iteration

    JK = args.JK
    num_workers = args.num_workers
    num_tasks = args.num_tasks

    eval_train = args.eval_train
    output_model_dir = args.output_model_dir


    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('-' * 100)

    print('Preprocessing the', dataset, 'dataset.')
    print('Just a moment......')
    data_all = MoleculeDatasetComplete(fr'./data/MELT_3D_{n_conf}/processed', dataset="melt", path=train_path)
    # pdb.set_trace()
    labels = [data.y.item() for data in data_all]
    if os.path.exists(rf"./data/scaler_melt_{n_conf}.pt"):
        scaler = torch.load(rf"./data/scaler_melt_{n_conf}.pt")
    else:
        scaler = StandardScaler()

        print('Begin fit scaler......')
        scaler.fit(np.array(labels).reshape(-1, 1))  # 计算各列的均值和标准差
        torch.save(scaler, rf"./data/scaler_melt_{n_conf}.pt")

    # 转换数据
    labels = scaler.transform(np.array(labels).reshape(-1, 1))
    data_new_y = []

    if os.path.exists(rf'./data/new_data_set_{n_conf}.pt'):
        data_new_y = torch.load(rf'./data/new_data_set_{n_conf}.pt')
    else:
        for idx, data in enumerate(tqdm(data_all)):
            data.x = data.x + 1
            data.edge_attr = data.edge_attr + 1

            n_nodes = len(data.x)

            data.edge_index, data.edge_attr = get_complete_covalent_graph(data.edge_index, n_nodes, data.edge_attr,
                                                                          device)

            data_new_y.append(Data(id=data.id, y=torch.tensor(labels[idx]), x=data.x, positions=data.positions,
                                   mol_id=data.mol_id, edge_index=data.edge_index, edge_attr=data.edge_attr))

        torch.save(data_new_y, rf'./data/new_data_set_{n_conf}.pt')

    data_all = MoleculeDatasetnew(data_new_y)

    smiles_list = pd.read_csv(r'./data/melt_point.csv')['SMILES'].values.tolist()
    dataset_train, dataset_dev, dataset_test = random_split(
        data_all, smiles_list, null_value=0, frac_train=0.8,
        frac_valid=0.1, frac_test=0.1, conf=n_conf)

    train_loader = DataLoader(dataset_train, batch_size=batch_train, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset_dev, batch_size=batch_test, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset_test, batch_size=batch_test, shuffle=False, num_workers=num_workers)


    print('# of test data samples:', len(data_all))
    print('The preprocess has finished!')
    print('-' * 100)

    print('Creating a model.')
    torch.manual_seed(1234)

    molecule_model = eg.EGNN(in_node_nf=9, hidden_nf=dim, out_node_nf=dim, in_edge_nf=3).to(device)
    model = GNN_graphpredComplete(num_layer=layer_hidden, emb_dim=dim, JK=JK, graph_pooling='mean',
                                  num_tasks=num_tasks, molecule_model=molecule_model)


    if os.path.exists(model_path):
        print("loading.........")
        molecule_model.load_state_dict(saved_model_dict['molecule_model'])
        model.load_state_dict(saved_model_dict['model'])
    for _model in model.molecule_model.parameters():
        print(_model)

    model.to(device)

    model_param_group = [
        {'params': model.molecule_model.parameters()},
        {'params': model.graph_pred_linear.parameters(), 'lr': lr * lr_decay}
    ]
    optimizer = optim.Adam(model_param_group, lr=lr, weight_decay=decay)

    train_result_list, val_result_list, test_result_list = [], [], []
    metric_list = ['R2', 'RMSE', 'MAE']

    best_val_rmse, best_val_idx = 1e10, 0

    for epoch in range(1, iteration + 1):
        loss_acc = train(model, device, train_loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        if eval_train:
            train_result, train_target, train_pred, train_idx = eval(model, device, train_loader, scaler)
        else:
            train_result = {'RMSE': 0, 'MAE': 0, 'R2': 0}
        val_result, val_target, val_pred, val_idx = eval(model, device, val_loader, scaler)
        test_result, test_target, test_pred, test_idx = eval(model, device, test_loader, scaler)
        print(test_target[0], test_pred[0])
        train_result_list.append(train_result)
        val_result_list.append(val_result)
        test_result_list.append(test_result)

        for metric in metric_list:
            print('{} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(metric, train_result[metric], val_result[metric],
                                                                       test_result[metric]))

        if val_result['RMSE'] < best_val_rmse:

            best_val_rmse = val_result['RMSE']
            best_val_idx = epoch - 1
            if not output_model_dir == '':
                output_model_path = os.path.join(output_model_dir, f'model_best_{n_conf}.pth')
                saved_model_dict = {
                    'molecule_model': molecule_model.state_dict(),
                    'model': model.state_dict()
                }
                torch.save(saved_model_dict, output_model_path)


    df_train = pd.DataFrame()
    df_train['idx'] = train_idx
    df_train['label'] = from_scaler(scaler, train_target)
    df_train['pred'] = from_scaler(scaler, train_pred)

    # df2.to_csv(r"./result/train.csv", index=False)

    df_test = pd.DataFrame()
    df_test['idx'] = test_idx
    df_test['label'] = from_scaler(scaler, test_target)
    df_test['pred'] = from_scaler(scaler, test_pred)
    df_all = pd.concat([df_train, df_test])
    # df3.to_csv(r"./result/test.csv", index=False)

    df_val = pd.DataFrame()
    df_val['idx'] = val_idx
    df_val['label'] = from_scaler(scaler, val_target)
    df_val['pred'] = from_scaler(scaler, val_pred)
    # df4.to_csv(r"./result/val.csv", index=False)
    df_all = pd.concat([df_all, df_val])

    df_all = df_all.sort_values(by='idx')
    df_len = df_all.shape[0]
    df_all['idx'] = range(df_len)

    _path = os.path.join(r"./result", f"all_{n_conf}_23w_3d.csv")
    # df_all.to_csv(_path, index=False)
    for metric in metric_list:
        print('Best (RMSE), {} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(
            metric, train_result_list[best_val_idx][metric], val_result_list[best_val_idx][metric],
            test_result_list[best_val_idx][metric]))

    if output_model_dir is not '':
        output_model_path = os.path.join(output_model_dir, f'model_final_{n_conf}.pth')
        saved_model_dict = {
            'molecule_model': molecule_model.state_dict(),
            'model': model.state_dict()
        }
        torch.save(saved_model_dict, output_model_path)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # mode = "val"
    main()

