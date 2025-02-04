import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate


class BatchNorm1d(nn.Module):
    def __init__(self, num_features, alpha, eps=1e-3, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Initialize parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.linear_mean = nn.Linear(num_features, num_features, bias=False)
        # self.linear_var = nn.Linear(num_features, num_features, bias=False)
        nn.init.zeros_(self.linear_mean.weight)
        # nn.init.zeros_(self.linear_var.weight)
        self.alpha = alpha

    def forward(self, x_ins):
        if self.training:
            # Calculate batch statistics
            x = torch.cat(x_ins, dim=0)
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0, unbiased=False)

            # Update running statistics for inference
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            x_norm = []
            for x in x_ins:
                batch_mean_ins = torch.mean(x, dim=0)
                # batch_var_ins = torch.var(x, dim=0, unbiased=False)
                batch_mean_diff = (batch_mean_ins - batch_mean).detach() # (N_p1, C) 
                # batch_var_diff = (batch_var_ins - batch_var).detach() # (N_p1, C)

                x_normalized = (x - batch_mean - self.alpha*self.linear_mean(batch_mean_diff)) / torch.sqrt(batch_var + self.eps)
                x_norm.append(x_normalized)
            x_normalized = torch.cat(x_norm, dim=0)
        else:
            x_norm = []
            for x in x_ins:
                batch_mean_ins = torch.mean(x, dim=0)
                # batch_var_ins = torch.var(x, dim=0, unbiased=False)
                batch_mean_diff = (batch_mean_ins - self.running_mean).detach() # (N_p1, C) 
                # batch_var_diff = (batch_var_ins - self.running_var).detach() # (N_p1, C)

                x_normalized = (x - self.running_mean - self.alpha*self.linear_mean(batch_mean_diff)) / torch.sqrt(self.running_var + self.eps)
                x_norm.append(x_normalized)
            x_normalized = torch.cat(x_norm, dim=0)
        
        # Scale and shift
        scaled = self.weight * x_normalized
        shifted = scaled + self.bias
        return shifted

class BatchNorm1dV2(nn.Module):
    def __init__(self, num_features, alpha, eps=1e-3, momentum=0.1):
        super(BatchNorm1dV2, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Initialize parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        # self.linear_mean = nn.Linear(num_features, num_features, bias=False)
        # self.linear_var = nn.Linear(num_features, num_features, bias=False)
        # nn.init.zeros_(self.linear_mean.weight)
        # nn.init.zeros_(self.linear_var.weight)
        self.alpha = alpha

    def forward(self, x_ins):
        if self.training:
            # Calculate batch statistics
            x = torch.cat(x_ins, dim=0)
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0, unbiased=False)

            # Update running statistics for inference
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            x_norm = []
            for x in x_ins:
                batch_mean_ins = torch.mean(x, dim=0)
                # batch_var_ins = torch.var(x, dim=0, unbiased=False)
                batch_mean_diff = (batch_mean_ins - batch_mean).detach() # (N_p1, C) 
                # batch_var_diff = (batch_var_ins - batch_var).detach() # (N_p1, C)

                x_normalized = (x - batch_mean - self.alpha*batch_mean_diff) / torch.sqrt(batch_var + self.eps)
                x_norm.append(x_normalized)
            x_normalized = torch.cat(x_norm, dim=0)
        else:
            x_norm = []
            for x in x_ins:
                batch_mean_ins = torch.mean(x, dim=0)
                # batch_var_ins = torch.var(x, dim=0, unbiased=False)
                batch_mean_diff = (batch_mean_ins - self.running_mean).detach() # (N_p1, C) 
                # batch_var_diff = (batch_var_ins - self.running_var).detach() # (N_p1, C)

                x_normalized = (x - self.running_mean - self.alpha*batch_mean_diff) / torch.sqrt(self.running_var + self.eps)
                x_norm.append(x_normalized)
            x_normalized = torch.cat(x_norm, dim=0)
        
        # Scale and shift
        scaled = self.weight * x_normalized
        shifted = scaled + self.bias
        return shifted


class PFNLayer(nn.Module): # v0
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):
        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated

class PFNLayerV1(nn.Module): # v1
    def __init__(self,
                 in_channels,
                 out_channels,
                 alpha,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        if self.use_norm:
            self.norm = BatchNorm1d(out_channels, alpha)
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv, points, flag):
        x = self.linear(inputs)
        x_ins = [x[points[:, 0]==i] for i in range(flag)]
        x = self.norm(x_ins) if self.use_norm else x
        x = self.relu(x)

        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated

class PFNLayerV2(nn.Module): # v1
    def __init__(self,
                 in_channels,
                 out_channels,
                 alpha,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        if self.use_norm:
            self.norm = BatchNorm1dV2(out_channels, alpha)
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv, points, flag):
        x = self.linear(inputs)
        x_ins = [x[points[:, 0]==i] for i in range(flag)]
        x = self.norm(x_ins) if self.use_norm else x
        x = self.relu(x)

        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated


class DynamicPillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1,2]] - self.point_cloud_range[[0,1]]) / self.voxel_size[[0,1]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]
        
        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)
        # features = self.linear1(features)
        # features_max = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
        # features = torch.cat([features, features_max[unq_inv, :]], dim=1)
        # features = self.linear2(features)
        # features = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
        
        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                   (unq_coords % self.scale_xy) // self.scale_y,
                                   unq_coords % self.scale_y,
                                   torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                   ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['pillar_features'] = features
        batch_dict['voxel_coords'] = voxel_coords
        return batch_dict
