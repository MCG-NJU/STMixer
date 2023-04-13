from alphaction.modeling import registry
from . import slowfast, i3d, video_model_builder

@registry.BACKBONES.register("Slowfast-Resnet50")
@registry.BACKBONES.register("Slowfast-Resnet101")
def build_slowfast_resnet_backbone(cfg):
    model = slowfast.SlowFast(cfg)
    return model

@registry.BACKBONES.register("PySlowonly")
def build_pyslowonly_resnet_backbone(cfg):
    model = video_model_builder.ResNet(cfg)
    return model

@registry.BACKBONES.register("PySlowfast-R50")
@registry.BACKBONES.register("PySlowfast-R101")
def build_pyslowfast_resnet_backbone(cfg):
    model = video_model_builder.SlowFast(cfg)
    return model

@registry.BACKBONES.register("MAE-ViT-B")
@registry.BACKBONES.register("MAE-ViT-L")
def build_mae_vit_backbone(cfg):
    model = video_model_builder.ViT(cfg)
    return model

@registry.BACKBONES.register("I3D-Resnet50")
@registry.BACKBONES.register("I3D-Resnet101")
@registry.BACKBONES.register("I3D-Resnet50-Sparse")
@registry.BACKBONES.register("I3D-Resnet101-Sparse")
def build_i3d_resnet_backbone(cfg):
    model = i3d.I3D(cfg)
    return model

def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
