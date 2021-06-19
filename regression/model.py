import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict


'''
MGraphDTA: Deep Multiscale Graph Neural Network for Explainable Drug-target binding affinity Prediction
'''


class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
    
    def forward(self, x):

        return self.inc(x)

class LinearReLU(nn.Module):
    def __init__(self,in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        
        return self.inc(x)

class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):

        return self.inc(x).squeeze(-1)

class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx+1, embedding_num, 96, 3)
            )

        self.linear = nn.Linear(block_num * 96, 96)
        
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x

class NodeLevelBatchNorm(_BatchNorm):
    r"""
    Applies Batch Normalization over a batch of graph data.
    Shape:
        - Input: [batch_nodes_dim, node_feature_dim]
        - Output: [batch_nodes_dim, node_feature_dim]
    batch_nodes_dim: all nodes of a batch graph
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)

class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data.x = F.relu(self.norm(self.conv(x, edge_index)))

        return data

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

    def bn_function(self, data):
        concated_features = torch.cat(data.x, 1)
        data.x = concated_features

        data = self.conv1(data)

        return data
    
    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]

        data = self.bn_function(data)
        data = self.conv2(data)

        return data

class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('layer%d' % (i + 1), layer)


    def forward(self, data):
        features = [data.x]
        for name, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features

        data.x = torch.cat(data.x, 1)

        return data


class GraphDenseNet(nn.Module):
    def __init__(self, num_input_features, out_dim, growth_rate=32, block_config = (3, 3, 3, 3), bn_sizes=[2, 3, 4, 4]):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
        num_input_features = 32

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate=growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i+1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = GraphConvBn(num_input_features, num_input_features // 2)
            self.features.add_module("transition%d" % (i+1), trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

    def forward(self, data):
        data = self.features(data)
        x = gnn.global_mean_pool(data.x, data.batch)
        x = self.classifer(x)

        return x

class MGraphDTA(nn.Module):
    def __init__(self, block_num, vocab_protein_size, embedding_size=128, filter_num=32, out_dim=1):
        super().__init__()
        self.protein_encoder = TargetRepresentation(block_num, vocab_protein_size, embedding_size)
        self.ligand_encoder = GraphDenseNet(num_input_features=22, out_dim=filter_num*3, block_config=[8, 8, 8], bn_sizes=[2, 2, 2])

        self.classifier = nn.Sequential(
            nn.Linear(filter_num * 3 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )

    def forward(self, data):
        target = data.target
        protein_x = self.protein_encoder(target)
        ligand_x = self.ligand_encoder(data)

        x = torch.cat([protein_x, ligand_x], dim=-1)
        x = self.classifier(x)

        return x


