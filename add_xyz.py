import os

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
from tqdm import tqdm

def add_xyz_sml(df, ls):
    conf_all = 50
    for i in tqdm(ls):
        smile = df.SMILES[i]
        smile = smile.strip()
        mol = Chem.MolFromSmiles(smile)
        mol3d = Chem.AddHs(mol)
        confs = AllChem.EmbedMultipleConfs(mol3d, numConfs=conf_all, numThreads=0, randomSeed=1)

        for conf in confs:
            AllChem.MMFFOptimizeMolecule(mol3d, confId=conf)
        num_conf = mol3d.GetNumConformers()
        _mol3d = mol3d
        
        try:
            res = AllChem.MMFFOptimizeMoleculeConfs(mol3d)
            index = np.argsort([-x[1] for x in res])
        except Exception:
            index = list(range(num_conf))

        if num_conf == 0:
            for conf in range(conf_all):
                w = Chem.SDWriter(os.path.join(r"./sdf", f"{i}_{conf}.xyz"))
                w.write(mol3d, confId=0)
                w.close()
        else:
            #try：
            for conf in range(conf_all):
                w = Chem.SDWriter(os.path.join(r"./sdf", f"{i}_{conf}.xyz"))
                conf = conf % num_conf
                conf_idx = int(index[conf])
                w.write(_mol3d, confId=conf_idx)
                w.close()


if __name__ == "__main__":
    data_path = r"./data/melt_point_test.csv"
    df = pd.read_csv(data_path)
    ls = range(len(df['SMILES']))
    print(ls)
    add_xyz_sml(df, ls)
    # atoms = []
    # for atom in mol3d.GetAtoms():
    #     atoms.append(atom.GetSymbol())
    # return atoms, mol3d.GetConformer().GetPositions().tolist()
