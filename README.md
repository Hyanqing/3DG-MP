
## Environments
Install packages under conda env
```bash
rdkit, pytorch, numpy networkx scikit-learn, ase, ogb, torch_cluster, torch_scatter, torch_sparse, torch-geometric
```

## Dataset Processing
```bash
python add_xyz.py
python GEOM_dataset_preparation.py  --n_conf 3
```

## Training
```bash
python train_egnn_3D.py
```