import torch
from torch import nn
from .lr_scheduler import WarmupMultiStepLR, HalfPeriodCosStepLR
import json


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed") or var_name.startswith(
            "encoder.patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks") or var_name.startswith(
            "encoder.blocks"):
        if var_name.startswith("encoder.blocks"):
            var_name = var_name[8:]
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):

    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model,
                         weight_decay=1e-5,
                         skip_list=(),
                         get_num_layer=None,
                         get_layer_scale=None,
                         lr_scale=1.0):
    parameter_group_names = {}
    parameter_group_vars = {}

    # 在这里修改区分scale，encoder一个学习率，其他人一个学习率
    # layer_decay: 需要加上get_num_layer和get_layer_scale
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (len(param.shape) == 1 or name.endswith(".bias")
                or name in skip_list) and name.startswith('encoder.'):
            group_name = "no_decay_encoder"
            this_weight_decay = 0.
            scale = 1.0
        elif len(param.shape) == 1 or name.endswith(
                ".bias") or name in skip_list:
            group_name = "no_decay_others"
            this_weight_decay = 0.
            scale = lr_scale
        elif name.startswith('encoder.'):
            group_name = "decay_encoder"
            this_weight_decay = weight_decay
            scale = 1.0
        else:
            group_name = "decay_others"
            this_weight_decay = weight_decay
            scale = lr_scale

        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id) * scale

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def make_optimizer(cfg, model):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """


    if 'vit' in cfg.MODEL.BACKBONE.CONV_BODY.lower():
        layer_decay = cfg.ViT.LAYER_DECAY < 1.0
        if layer_decay:
            assigner = LayerDecayValueAssigner(
                list(cfg.ViT.LAYER_DECAY ** (cfg.ViT.DEPTH + 1 - i)
                     for i in range(cfg.ViT.DEPTH + 2)))
        else:
            assigner = None

        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        skip_weight_decay_list = set(cfg.ViT.NO_WEIGHT_DECAY)
        print("Skip weight decay list: ", skip_weight_decay_list)
        weight_decay = cfg.ViT.WEIGHT_DECAY
        backbone_parameters = get_parameter_groups(model.backbone,
                                          weight_decay,
                                          skip_weight_decay_list,
                                          get_num_layer=assigner.get_layer_id
                                          if assigner is not None else None,
                                          get_layer_scale=assigner.get_scale
                                          if assigner is not None else None)

        rest_parameters = {'params':[]}
        for name, p in model.named_parameters():
            if "backbone" not in name:
                rest_parameters['params'].append(p)
        optim_params = backbone_parameters + [rest_parameters]
    else:
        bn_parameters = []
        non_bn_parameters = []

        frozn_bn_params = cfg.MODEL.BACKBONE.FROZEN_BN
        # only affine layer frozen, running_mean and var still update.
        if frozn_bn_params:
            for m in model.backbone.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
        for name, p in model.named_parameters():
            if ("backbone" in name) and ('bn' in name):
                if cfg.MODEL.BACKBONE.FROZEN_BN:
                    p.requires_grad = False
                bn_parameters.append(p)
            else:
                non_bn_parameters.append(p)

        optim_params = []
        if (not frozn_bn_params) and len(bn_parameters) > 0:
            optim_params.append({
                "params": bn_parameters,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY_BN,
                "lr": cfg.SOLVER.BASE_LR
            })
        optim_params.append({
                "params": non_bn_parameters,
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                "lr": cfg.SOLVER.BASE_LR
            })

        # Check all parameters will be passed into optimizer.
        assert len(list(model.parameters())) == len(non_bn_parameters) + len(bn_parameters), \
            "parameter size does not match: {} + {} != {}".format(
            len(non_bn_parameters),
            len(bn_parameters),
            len(list(model.parameters())),
        )
        # print(
        #     "bn {}, non bn {}".format(
        #         len(bn_parameters),
        #         len(non_bn_parameters),
        #     )
        # )

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        optimizer = torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        optimizer = torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=cfg.SOLVER.BETAS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    scheduler = cfg.SOLVER.SCHEDULER
    iter_per_epoch = cfg.SOLVER.ITER_PER_EPOCH
    if scheduler not in ("half_period_cosine", "warmup_multi_step"):
        raise ValueError('Scheduler not available')
    if scheduler == 'warmup_multi_step':
        steps = tuple(step*iter_per_epoch for step in cfg.SOLVER.STEPS)
        warmup_iters = cfg.SOLVER.WARMUP_EPOCH*iter_per_epoch
        return WarmupMultiStepLR(
            optimizer,
            steps,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=warmup_iters if cfg.SOLVER.WARMUP_ON else 0,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif scheduler == 'half_period_cosine':
        max_iters = iter_per_epoch*cfg.SOLVER.MAX_EPOCH
        warmup_iters = cfg.SOLVER.WARMUP_EPOCH*iter_per_epoch
        return HalfPeriodCosStepLR(
            optimizer,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=warmup_iters if cfg.SOLVER.WARMUP_ON else 0,
            max_iters=max_iters,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
