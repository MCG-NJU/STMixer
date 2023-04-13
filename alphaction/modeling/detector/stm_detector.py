from torch import nn

from ..backbone import build_backbone
from ..stm_decoder.stm_decoder import build_stm_decoder
import fvcore.nn.weight_init as weight_init
import torch


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, t, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class STMDetector(nn.Module):
    def __init__(self, cfg):
        super(STMDetector, self).__init__()
        self.backbone = build_backbone(cfg)
        self._construct_space(cfg)
        self.stm_head = build_stm_decoder(cfg)
        self.device = torch.device('cuda')


    def _construct_space(self, cfg):
        out_channel = cfg.MODEL.STM.HIDDEN_DIM
        if 'vit' in cfg.MODEL.BACKBONE.CONV_BODY.lower():
            in_channels = [cfg.ViT.EMBED_DIM]*4
            self.lateral_convs = nn.ModuleList()

            for idx, scale in enumerate([4, 2, 1, 0.5]):
                dim = in_channels[idx]
                if scale == 4.0:
                    layers = [
                        nn.ConvTranspose3d(dim, dim // 2, kernel_size=[1, 2, 2], stride=[1, 2, 2]),
                        LayerNorm(dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=[1, 2, 2], stride=[1, 2, 2]),
                    ]
                    out_dim = dim // 4
                elif scale == 2.0:
                    layers = [nn.ConvTranspose3d(dim, dim // 2, kernel_size=[1, 2, 2], stride=[1, 2, 2])]
                    out_dim = dim // 2
                elif scale == 1.0:
                    layers = []
                    out_dim = dim
                elif scale == 0.5:
                    layers = [nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])]
                    out_dim = dim
                else:
                    raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
                layers.extend(
                    [
                        nn.Conv3d(
                            out_dim,
                            out_channel,
                            kernel_size=1,
                            bias=False,
                        ),
                        LayerNorm(out_channel),
                        nn.Conv3d(
                            out_channel,
                            out_channel,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ]
                )
                layers = nn.Sequential(*layers)

                self.lateral_convs.append(layers)
        else:
            if self.backbone.num_pathways == 1:
                in_channels = [256, 512, 1024, 2048]
            else:
                in_channels = [256+32, 512+64, 1024+128, 2048+256]
            self.lateral_convs = nn.ModuleList()
            for idx, in_channel in enumerate(in_channels):
                lateral_conv = nn.Conv3d(in_channel, out_channel, kernel_size=1)
                weight_init.c2_xavier_fill(lateral_conv)
                self.lateral_convs.append(lateral_conv)


    def space_forward(self, features):
        mapped_features = []
        for i, feature in enumerate(features):
            mapped_features.append(self.lateral_convs[i](feature))
        return mapped_features


    def forward(self, slow_video, fast_video, whwh, boxes, labels, extras={}, part_forward=-1):
        # part_forward is used to split this model into two parts.
        # if part_forward<0, just use it as a single model
        # if part_forward=0, use this model to extract pooled feature(person and object, no memory features).
        # if part_forward=1, use the ia structure to aggregate interactions and give final result.
        # implemented in roi_heads

        if self.backbone.num_pathways == 1:
            features = self.backbone([slow_video])
        else:
            features = self.backbone([slow_video, fast_video])
        mapped_features = self.space_forward(features)

        return self.stm_head(mapped_features, whwh, boxes, labels)


def build_detection_model(cfg):
    return STMDetector(cfg)