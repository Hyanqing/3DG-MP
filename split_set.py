import os
import random
import torch
from itertools import compress
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold

def generate_scaffold(smiles, include_chirality=False):
    """ Obtain Bemis-Murcko scaffold from smiles
    :return: smiles of scaffold """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def add_idx(idx, conf):
    _idx = idx
    for num in range(conf - 1):
        _add = [i + num for i in _idx]
        idx = idx + _add
    print(len(idx))
    return idx


def scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1, conf=5,
                   return_smiles=False):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx is not None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_idx = [i * conf for i in train_idx]
    valid_idx = [i * conf for i in valid_idx]
    test_idx = [i * conf for i in test_idx]

    train_dataset = dataset[torch.tensor(add_idx(train_idx, conf))]
    valid_dataset = dataset[torch.tensor(add_idx(valid_idx, conf))]
    test_dataset = dataset[torch.tensor(add_idx(test_idx, conf))]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, \
            (train_smiles, valid_smiles, test_smiles)


def random_split(dataset, smiles_list, task_idx=None, null_value=0,
                 frac_train=0.8, frac_valid=0.1, frac_test=0.1, conf=5,
                 return_smiles=False):
    if conf == 0 or conf == 49 or conf == 2 or conf == -1:
        conf = 1
    num_mols = int(len(dataset) / conf)
    random.seed(1234)
    if os.path.exists(r"./data/test_idx.pt"):
        train_idx = torch.load(r"./data/train_idx.pt")
        valid_idx = torch.load(r"./data/val_idx.pt")
        test_idx = torch.load(r"./data/test_idx.pt")
        print("-----------------index is already load------------------------")
    else:
        all_idx = list(range(num_mols))
        random.shuffle(all_idx)

        train_idx = all_idx[:int(frac_train * num_mols)]
        valid_idx = all_idx[int(frac_train * num_mols):int(frac_valid * num_mols) + int(frac_train * num_mols)]
        test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]

        torch.save(train_idx, r"./data/train_idx.pt")
        torch.save(valid_idx, r"./data/val_idx.pt")
        torch.save(test_idx, r"./data/test_idx.pt")

    train_idx = [i * conf for i in train_idx]
    valid_idx = [i * conf for i in valid_idx]
    test_idx = [i * conf for i in test_idx]

    train_dataset = dataset[torch.tensor(add_idx(train_idx, conf))]
    valid_dataset = dataset[torch.tensor(add_idx(valid_idx, conf))]
    test_dataset = dataset[torch.tensor(add_idx(test_idx, conf))]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, \
            (train_smiles, valid_smiles, test_smiles)

