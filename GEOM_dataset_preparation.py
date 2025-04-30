import argparse
import json
import os
import pickle
import random
from itertools import repeat
from os.path import join

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector


def mol_to_graph_data_obj_simple_3D(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    # todo: more atom/bond features in the future
    # atoms, two features: atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds, two features: bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    conformer = mol.GetConformers()[0]
    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=positions)
    return data


class Molecule3DDataset(InMemoryDataset):

    def __init__(self, root, n_mol, n_conf, transform=None, seed=777,
                 pre_transform=None, pre_filter=None, empty=False, **kwargs):
        os.makedirs(root, exist_ok=True)
        os.makedirs(join(root, 'raw'), exist_ok=True)
        os.makedirs(join(root, 'processed'), exist_ok=True)
        if 'smiles_copy_from_3D_file' in kwargs:  # for 2D Datasets (SMILES)
            self.smiles_copy_from_3D_file = kwargs['smiles_copy_from_3D_file']
        else:
            self.smiles_copy_from_3D_file = None

        self.root, self.seed = root, seed
        self.n_mol, self.n_conf = n_mol, n_conf
        self.pre_transform, self.pre_filter = pre_transform, pre_filter

        super(Molecule3DDataset, self).__init__(
            root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('root: {},\ndata: {},\nn_mol: {},\nn_conf: {}'.format(
            self.root, self.data, self.n_mol, self.n_conf))

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx+1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return f'geometric_data_processed_{self.n_conf}_23w.pt'

    def process(self):
        data_list = []

        whole_SMILES_set = set()
        file_path = "./data/melt_point_test.csv"
        df = pd.read_csv(file_path)
        data_smiles_list, labels = df['SMILES'].values, df['Melt_K'].values
        print("len of downstream SMILES:", len(data_smiles_list))

        if self.smiles_copy_from_3D_file is None:  # 3D datasets
            df = pd.read_csv(file_path)
            data_smiles_list = df['SMILES'].values
            print(f"shape:{df.shape}")
            #data_smiles_list = list(dict.fromkeys(data_smiles_list))
            data_smiles_list = list(data_smiles_list)

            mol_idx, idx, notfound = 0, 0, 0

            for i in range(len(data_smiles_list)):
                for num in range(n_conf):
                    # select the first n_conf conformations
                    filename = os.path.join(r"./sdf",
                                            f"{i}_{num}.xyz")
                    rdkit_mol = Chem.MolFromMolFile(filename)
                    data = mol_to_graph_data_obj_simple_3D(rdkit_mol)
                    data.id = torch.tensor([idx])
                    data.mol_id = torch.tensor([mol_idx])
                    data_list.append(data)
                    data.y = torch.tensor([labels[mol_idx]])
                    idx += 1
                mol_idx += 1

            print('mol id: [0, {}]\tlen of smiles: {}\tlen of set(smiles): {}'.format(
                mol_idx, len(data_smiles_list), len(set(data_smiles_list))))

        else:  # 2D datasets
            df = pd.read_csv(file_path)
            data_smiles_list = df['SMILES'].values
            #data_smiles_list = list(dict.fromkeys(data_smiles_list))
            data_smiles_list = list(data_smiles_list)

            # load 3D structure
            dir_name = '{}/rdkit_folder'.format(data_folder)
            drugs_file = '{}/summary_drugs.json'.format(dir_name)

            mol_idx, idx, notfound = 0, 0, 0

            for i in range(len(data_smiles_list)):
                filename = os.path.join(r"./sdf",
                                        f"{i}_{0}.xyz")

                rdkit_mol = Chem.MolFromMolFile(filename)
                data = mol_to_graph_data_obj_simple_3D(rdkit_mol)
                data.mol_id = torch.tensor([mol_idx])
                data.id = torch.tensor([idx])
                data_list.append(data)
                mol_idx += 1
                idx += 1


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        save_list_to_file(data_smiles_list, r"./data_list.txt")
        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = join(self.processed_dir, 'smiles.csv')
        print('saving to {}'.format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False)

        data, slices = self.collate(data_list)
        print(data)
        torch.save((data, slices), self.processed_paths[0])
        print("%d molecules do not meet the requirements" % notfound)
        print("%d molecules have been processed" % mol_idx)
        print("%d conformers have been processed" % idx)
        return


def save_list_to_file(lst, filename):
    with open(filename, "w") as file:
        for item in lst:
            file.write(str(item) + "\n")


def load_SMILES_from_csv(file_path, need_idx):
    df = pd.read_csv(file_path)
    smis = df['SMILES'].values.tolist()
    print(len(smis))
    smis = [smis[idx] for idx in need_idx['idx'].values]
    return smis, need_idx


def load_SMILES_list(file_path):
    SMILES_list = []
    with open(file_path, 'rb') as f:
        for line in tqdm(f.readlines()):
            SMILES_list.append(line.strip().decode())

    return SMILES_list


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    # parser.add_argument('--n_mol', type=int, help='number of unique smiles/molecules')
    parser.add_argument('--n_conf', type=int, help='number of conformers of each molecule')
    args = parser.parse_args()

    data_folder = r'./data'

    n_mol, n_conf = 237406, args.n_conf
    root_3d = './data/MELT_3D_{}'.format(n_conf)

    # Generate 3D Datasets (2D SMILES + 3D Conformer)
    Molecule3DDataset(root=root_3d, n_mol=n_mol, n_conf=n_conf)

