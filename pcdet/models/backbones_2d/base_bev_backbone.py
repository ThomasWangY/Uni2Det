import numpy as np
import torch
import torch.nn as nn

# from ...utils import uni3d_norm
from ...utils import uni3d_norm as uni3d_norm_used
# from ...utils import uni3d_norm_parallel as uni3d_norm_used
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        if self.model_cfg.get('DUAL_NORM', None):
            self.db_source = int(self.model_cfg.db_source)
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        
        # using the Dual-Norm:
        if self.model_cfg.get('DUAL_NORM', None):
            for idx in range(num_levels):
                cur_layers = [
                    nn.ZeroPad2d(1),
                    nn.Conv2d(
                        c_in_list[idx], num_filters[idx], kernel_size=3,
                        stride=layer_strides[idx], padding=0, bias=False
                    ),
                    uni3d_norm_used.UniNorm2d(num_filters[idx], dataset_from_flag=self.db_source, eps=1e-3, momentum=0.01), # using the dataset-specific norm
                    nn.ReLU()
                ]
                for k in range(layer_nums[idx]):
                    cur_layers.extend([
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        uni3d_norm_used.UniNorm2d(num_filters[idx], dataset_from_flag=self.db_source, eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ])
                self.blocks.append(nn.Sequential(*cur_layers))
                if len(upsample_strides) > 0:
                    stride = upsample_strides[idx]
                    if stride >= 1:
                        self.deblocks.append(nn.Sequential(
                            nn.ConvTranspose2d(
                                num_filters[idx], num_upsample_filters[idx],
                                upsample_strides[idx],
                                stride=upsample_strides[idx], bias=False
                            ),
                            uni3d_norm_used.UniNorm2d(num_upsample_filters[idx], dataset_from_flag=self.db_source, eps=1e-3, momentum=0.01),
                            nn.ReLU()
                        ))
                    else:
                        stride = np.round(1 / stride).astype(np.int)
                        self.deblocks.append(nn.Sequential(
                            nn.Conv2d(
                                num_filters[idx], num_upsample_filters[idx],
                                stride,
                                stride=stride, bias=False
                            ),
                            uni3d_norm_used.UniNorm2d(num_upsample_filters[idx], dataset_from_flag=self.db_source, eps=1e-3, momentum=0.01),
                            nn.ReLU()
                        ))

            c_in = sum(num_upsample_filters)
            if len(upsample_strides) > num_levels:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                    uni3d_norm_used.UniNorm2d(c_in, dataset_from_flag=self.db_source, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ))

            self.num_bev_features = c_in
        else:
            for idx in range(num_levels):
                cur_layers = [
                    nn.ZeroPad2d(1),
                    nn.Conv2d(
                        c_in_list[idx], num_filters[idx], kernel_size=3,
                        stride=layer_strides[idx], padding=0, bias=False
                    ),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ]
                for k in range(layer_nums[idx]):
                    cur_layers.extend([
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ])
                self.blocks.append(nn.Sequential(*cur_layers))
                if len(upsample_strides) > 0:
                    stride = upsample_strides[idx]
                    if stride >= 1:
                        self.deblocks.append(nn.Sequential(
                            nn.ConvTranspose2d(
                                num_filters[idx], num_upsample_filters[idx],
                                upsample_strides[idx],
                                stride=upsample_strides[idx], bias=False
                            ),
                            nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                            nn.ReLU()
                        ))
                    else:
                        stride = np.round(1 / stride).astype(np.int)
                        self.deblocks.append(nn.Sequential(
                            nn.Conv2d(
                                num_filters[idx], num_upsample_filters[idx],
                                stride,
                                stride=stride, bias=False
                            ),
                            nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                            nn.ReLU()
                        ))

            c_in = sum(num_upsample_filters)
            if len(upsample_strides) > num_levels:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                    nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ))

            self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

# TODO: BEV-based range masking
class BaseBEVBackbone_MASK(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx]+1, num_filters[idx], kernel_size=3, # TODO: +1 for mask
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        # TODO: derive mask from point range
        range_ori = self.model_cfg.RANGE_ORI # [-75.2, -75.2, 75.2, 75.2] 
        range_1 = self.model_cfg.RANGE_1 # [0, -40, 70.4, 40]
        range_2 = self.model_cfg.RANGE_2 # [-51.2, -51.2, 51.2, 51.2]
        H, W = self.model_cfg.H, self.model_cfg.W # 376, 376
        
        self.mask_1, self.mask_2 = torch.zeros(H, W).cuda(), torch.zeros(H, W).cuda()
        m_11 = int((range_1[0] - range_ori[0])/(range_ori[2] - range_ori[0]) * H)
        m_12 = int((range_1[2] - range_ori[0])/(range_ori[2] - range_ori[0]) * H)
        m_13 = int((range_1[1] - range_ori[1])/(range_ori[3] - range_ori[1]) * W)
        m_14 = int((range_1[3] - range_ori[1])/(range_ori[3] - range_ori[1]) * W)
        self.mask_1[m_13:m_14, m_11:m_12] = 1
        m_21 = int((range_2[0] - range_ori[0])/(range_ori[2] - range_ori[0]) * H)
        m_22 = int((range_2[2] - range_ori[0])/(range_ori[2] - range_ori[0]) * H)
        m_23 = int((range_2[1] - range_ori[1])/(range_ori[3] - range_ori[1]) * W)
        m_24 = int((range_2[3] - range_ori[1])/(range_ori[3] - range_ori[1]) * W)
        self.mask_2[m_23:m_24, m_21:m_22] = 1
        if self.model_cfg.get('RANGE_3', None):
            self.mask_3 = torch.zeros(H, W).cuda()
            range_3 = self.model_cfg.RANGE_3
            m_31 = int((range_3[0] - range_ori[0])/(range_ori[2] - range_ori[0]) * H)
            m_32 = int((range_3[2] - range_ori[0])/(range_ori[2] - range_ori[0]) * H)
            m_33 = int((range_3[1] - range_ori[1])/(range_ori[3] - range_ori[1]) * W)
            m_34 = int((range_3[3] - range_ori[1])/(range_ori[3] - range_ori[1]) * W)
            self.mask_3[m_33:m_34, m_31:m_32] = 1
        else:
            self.mask_3 = None

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features

        # TODO: generate mask
        if self.training:
            if self.mask_3 is None:
                mask_1 = self.mask_1.unsqueeze(0).unsqueeze(0).expand(len(x)//2, -1, -1, -1)
                mask_2 = self.mask_2.unsqueeze(0).unsqueeze(0).expand(len(x)//2, -1, -1, -1)
                mask = torch.cat((mask_1, mask_2), dim=0)
            else:
                mask_1 = self.mask_1.unsqueeze(0).unsqueeze(0).expand(len(x)//3, -1, -1, -1)
                mask_2 = self.mask_2.unsqueeze(0).unsqueeze(0).expand(len(x)//3, -1, -1, -1)
                mask_3 = self.mask_3.unsqueeze(0).unsqueeze(0).expand(len(x)//3, -1, -1, -1)
                mask = torch.cat((mask_1, mask_2, mask_3), dim=0)
        else:
            flag = self.model_cfg.DATASET_SET.index(data_dict['db_flag'][0])
            if flag == 0:
                mask = self.mask_1.unsqueeze(0).unsqueeze(0).expand(len(x), -1, -1, -1)
            elif flag == 1:
                mask = self.mask_2.unsqueeze(0).unsqueeze(0).expand(len(x), -1, -1, -1)
            elif flag == 2:
                mask = self.mask_3.unsqueeze(0).unsqueeze(0).expand(len(x), -1, -1, -1)
            else:
                raise NotImplementedError

        for i in range(len(self.blocks)):
            x = torch.cat((x, mask), dim=1)
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
