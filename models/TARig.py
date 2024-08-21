"""
jingma
"""
import torch
import torch.nn as nn
from torch.nn import Sequential, Dropout, Linear
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_max, scatter_mean
from models.gcn_basic_modules import MLP, GCU

class BackBone(torch.nn.Module):
    def __init__(self, input_normal, aggr='max'):
        super(BackBone, self).__init__()
        self.input_normal = input_normal
        if self.input_normal:
            self.input_channel = 6
        else:
            self.input_channel = 3
        self.gcu_1 = GCU(in_channels=self.input_channel, out_channels=64, aggr=aggr) 
        self.gcu_2 = GCU(in_channels=64, out_channels=256, aggr=aggr)
        self.gcu_3 = GCU(in_channels=256, out_channels=512, aggr=aggr)
        # feature compression
        self.mlp_glb = MLP([(64 + 256 + 512), 1024]) # mTODO1: 1024, 2048
        self.global_pooling = True # mTODO2: whether enabling global_pooling improves the result or not

    def forward(self, data):
        if self.input_normal:
            x = torch.cat([data.pos, data.x], dim=1)
        else:
            x = data.pos
        geo_edge_index, tpl_edge_index, batch = data.geo_edge_index, data.tpl_edge_index, data.batch

        x_1 = self.gcu_1(x, tpl_edge_index, geo_edge_index)   # mTODO4: effectiveness of tpl_edge_index, geo_edge_index
        x_2 = self.gcu_2(x_1, tpl_edge_index, geo_edge_index)
        x_3 = self.gcu_3(x_2, tpl_edge_index, geo_edge_index)
        x_4 = self.mlp_glb(torch.cat([x_1, x_2, x_3], dim=1))

        if self.global_pooling:
            x_global, _ = scatter_max(x_4, data.batch, dim=0)
            x_global = torch.repeat_interleave(x_global, torch.bincount(data.batch), dim=0)
            x_5 = torch.cat([x_global, x, x_1, x_2, x_3], dim=1)
        else:
            x_5 = torch.cat([x_4, x, x_1, x_2, x_3], dim=1)
        return x_5


class TARigNet(nn.Module):
    """
    Predict joint response, conflow, skinning weights simultaneously
    """
    def __init__(self, n_joints,input_normal=True, dropout=0.0): 
        super().__init__()
        self.n_joints = n_joints
        self.input_normal = input_normal
        if self.input_normal:
            self.input_channel = 6
        else:
            self.input_channel = 3

        # self.pre_norm = nn.BatchNorm1d(self.input_channel) # add pre-normlization
        self.backbone = BackBone(input_normal=self.input_normal)
        self.num_feature_channels = 1024 + self.input_channel + 64 + 256 +512

        # joint heatmap
        self.heatmap_channels = n_joints
        self.heatmap_head = Sequential(MLP([self.num_feature_channels, 1024, 256], batch_norm=False),
                                        Dropout(dropout),
                                        Linear(256, self.heatmap_channels * 2)) # template heatmap + secondary heatmaps

        # skinning weight
        self.skinning_head = Sequential(MLP([self.num_feature_channels, 1024, 256], batch_norm=False),
                                            Linear(256, n_joints))

        # boneflow ToDO: define boneflow on all vertices, connect secondary joints
        self.conflow_head = Sequential(MLP([self.num_feature_channels, 1024, 256], batch_norm=False),
                                            Linear(256, 3))


    def forward(self, data, batch_size):
        # 1. graph convolution: extract features
        vertex_features = self.backbone(data)

        unnormalized_heatmap = self.heatmap_head(vertex_features)
        ### joint positions
        coords = []
        heatmaps = []
        for i in range(batch_size): # for each 3D model
            pred_heatmap = unnormalized_heatmap[data.batch == i, :]

            # 2. Normalize the heatmaps along each joint
            pred_heatmap_sigmoid = torch.sigmoid(pred_heatmap)
            heatmaps.append(pred_heatmap_sigmoid)
            # 3. Calculate the coordinates (n by m) * (n by 3)
            primary_heatmap_sigmoid = pred_heatmap_sigmoid[:, 0:self.heatmap_channels]
            normalized_heatmap = primary_heatmap_sigmoid / (torch.sum(primary_heatmap_sigmoid, dim=0) + 1e-5)

            pos = data.pos[data.batch == i, :]
            keypoint_pos = torch.matmul(normalized_heatmap.transpose(1, 0), pos) # (m by n) * (n by 3) # template joint regression
            coords.append(keypoint_pos)

        ### skinning weights & connection flow
        skinning_weights = self.skinning_head(vertex_features)
        conflow = self.conflow_head(vertex_features)
        conflow = F.normalize(conflow, dim=1, eps=1e-6)

        return coords, heatmaps, skinning_weights, conflow