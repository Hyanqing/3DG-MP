import torch
import torch.nn as nn
from torch_geometric.nn import (MessagePassing, global_add_pool,
                                global_max_pool, global_mean_pool)
class GNN_graphpredComplete(nn.Module):
    def __init__(self, num_layer, emb_dim, JK, graph_pooling, num_tasks, molecule_model=None):
        super(GNN_graphpredComplete, self).__init__()
        if num_layer < 2:
            raise ValueError("# layers must > 1.")

        self.molecule_model = molecule_model
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.JK = JK

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim,
                                               self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.mult * self.emb_dim, self.num_tasks)
        return

    def from_pretrained(self, model_file):
        self.molecule_model.load_state_dict(torch.load(model_file))
        return

    def forward(self, *argv):
        if len(argv) == 5:
            x, edge_index, edge_attr, batch, pos = argv[0], argv[1], argv[2], argv[3], argv[4]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        edge_index = [edge_index[0], edge_index[1]]
        # pdb.set_trace()
        x = x.to(torch.float32)
        node_representation, _ = self.molecule_model(x, pos, edge_index, edge_attr)
        # pdb.set_trace()
        graph_representation = self.pool(node_representation, batch)
        output = self.graph_pred_linear(graph_representation)

        return output