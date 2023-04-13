import torch
import torch.nn as nn
import torch.nn.functional as F
from .util.box_ops import bbox_overlaps, box_cxcywh_to_xyxy, clip_boxes_tensor
from .util.msaq import SAMPLE4D
from .util.adaptive_mixing_operator import AdaptiveMixing
from .util.head_utils import _get_activation_layer, bias_init_with_prob, decode_box, position_embedding, make_sample_points
from .util.head_utils import FFN, MultiheadAttention
from .util.loss import SetCriterion, HungarianMatcher


class AdaptiveSTSamplingMixing(nn.Module):

    def __init__(self, spatial_points=32,
                 temporal_points=4,
                 out_multiplier=4,
                 n_groups=4,
                 query_dim=256,
                 feat_channels=None):
        super(AdaptiveSTSamplingMixing, self).__init__()
        self.spatial_points =  spatial_points
        self.temporal_points = temporal_points
        self.out_multiplier = out_multiplier
        self.n_groups = n_groups
        self.query_dim = query_dim
        self.feat_channels = feat_channels if feat_channels is not None else self.query_dim
        self.offset_generator = nn.Sequential(nn.Linear(query_dim, spatial_points * n_groups * 3))

        self.norm_s = nn.LayerNorm(query_dim)
        self.norm_t = nn.LayerNorm(query_dim)

        self.adaptive_mixing_s = AdaptiveMixing(
            self.feat_channels,
            query_dim=self.query_dim,
            in_points=self.spatial_points,
            out_points=self.spatial_points*self.out_multiplier,
            n_groups=self.n_groups,
        )

        self.adaptive_mixing_t = AdaptiveMixing(
            self.feat_channels,
            query_dim=self.query_dim,
            in_points=self.temporal_points,
            out_points=self.temporal_points*self.out_multiplier,
            n_groups=self.n_groups,
        )

        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.offset_generator[-1].weight.data)
        nn.init.zeros_(self.offset_generator[-1].bias.data)

        bias = self.offset_generator[-1].bias.data.view(
            self.n_groups, self.spatial_points, 3)

        if int(self.spatial_points ** 0.5) ** 2 == self.spatial_points:
            h = int(self.in_points ** 0.5)
            y = torch.linspace(-0.5, 0.5, h + 1) + 0.5 / h
            yp = y[:-1]
            y = yp.view(-1, 1).repeat(1, h)
            x = yp.view(1, -1).repeat(h, 1)
            y = y.flatten(0, 1)[None, :, None]
            x = x.flatten(0, 1)[None, :, None]
            bias[:, :, 0:2] = torch.cat([y, x], dim=-1)
            # 格子采样
        else:
            bandwidth = 0.5 * 1.0
            nn.init.uniform_(bias, -bandwidth, bandwidth)
        bias[:, :, 2:3].mul_(0.0)

        self.adaptive_mixing_s._init_weights()
        self.adaptive_mixing_t._init_weights()

    def forward(self, features, proposal_boxes, spatial_queries, temporal_queries, featmap_strides):

        offset = self.offset_generator(spatial_queries)
        sample_points_xy = make_sample_points(offset, self.n_groups * self.spatial_points, proposal_boxes)
        sampled_feature, _ = SAMPLE4D(sample_points_xy, features, featmap_strides=featmap_strides, n_points=self.spatial_points)

        # B, C, n_groups, temporal_points, spatial_points, n_query, _ = sampled_feature.size()
        sampled_feature = sampled_feature.flatten(5, 6)                   # B, n_channels, n_groups, temporal_points, spatial_points, n_query
        sampled_feature = sampled_feature.permute(0, 5, 2, 3, 4, 1)       # B, n_query, n_groups, temporal_points, spatial_points, n_channels

        spatial_feats = torch.mean(sampled_feature, dim=3)                            # out_s has shape [B, n_query, n_groups, spatial_points, n_channels]
        spatial_queries = self.adaptive_mixing_s(spatial_feats, spatial_queries)
        spatial_queries = self.norm_s(spatial_queries)

        temporal_feats = torch.mean(sampled_feature, dim=4)                        # out_t has shape [B, n_query, n_groups, temporal_points, n_channels]
        temporal_queries = self.adaptive_mixing_t(temporal_feats, temporal_queries)
        temporal_queries = self.norm_t(temporal_queries)

        return spatial_queries, temporal_queries


class AMStage(nn.Module):

    def __init__(self, query_dim=256,
                 feat_channels=256,
                 num_heads=8,
                 feedforward_channels=2048,
                 dropout=0.0,
                 num_ffn_fcs=2,
                 ffn_act='RelU',
                 spatial_points=32,
                 temporal_points=4,
                 out_multiplier=4,
                 n_groups=4,
                 num_cls_fcs=1,
                 num_reg_fcs=1,
                 num_action_fcs=1,
                 num_classes_object=1,
                 num_classes_action=80,):


        super(AMStage, self).__init__()

        # MHSA-S
        ffn_act_cfg = dict(type=ffn_act, inplace=True)
        self.attention_s = MultiheadAttention(query_dim, num_heads, dropout)
        self.attention_norm_s = nn.LayerNorm(query_dim, eps=1e-5)
        self.ffn_s = FFN(query_dim, feedforward_channels, num_ffn_fcs, act_cfg=ffn_act_cfg, dropout=dropout)
        self.ffn_norm_s = nn.LayerNorm(query_dim, eps=1e-5)
        self.iof_tau = nn.Parameter(torch.ones(self.attention_s.num_heads,))

        # MHSA-T
        self.attention_t = MultiheadAttention(query_dim, num_heads, dropout)
        self.attention_norm_t = nn.LayerNorm(query_dim, eps=1e-5)
        self.ffn_t = FFN(query_dim, feedforward_channels, num_ffn_fcs, act_cfg=ffn_act_cfg, dropout=dropout)
        self.ffn_norm_t = nn.LayerNorm(query_dim, eps=1e-5)

        self.samplingmixing = AdaptiveSTSamplingMixing(
            spatial_points=spatial_points,
            temporal_points=temporal_points,
            out_multiplier=out_multiplier,
            n_groups=n_groups,
            query_dim=query_dim,
            feat_channels=feat_channels
        )

        cls_feature_dim = query_dim
        reg_feature_dim = query_dim
        action_feat_dim = query_dim * 2

        # human classifier
        self.human_cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.human_cls_fcs.append(
                nn.Linear(cls_feature_dim, cls_feature_dim, bias=True))
            self.human_cls_fcs.append(
                nn.LayerNorm(cls_feature_dim, eps=1e-5))
            self.human_cls_fcs.append(
                _get_activation_layer(ffn_act_cfg))
        self.human_fc_cls = nn.Linear(cls_feature_dim, num_classes_object + 1)

        # human bbox regressor
        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(reg_feature_dim, reg_feature_dim, bias=True))
            self.reg_fcs.append(
                nn.LayerNorm(reg_feature_dim, eps=1e-5))
            self.reg_fcs.append(
                _get_activation_layer(ffn_act_cfg))
        self.fc_reg = nn.Linear(reg_feature_dim, 4)

        # action classifier
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.action_cls_fcs = nn.ModuleList()
        for _ in range(num_action_fcs):
            self.action_cls_fcs.append(
                nn.Linear(action_feat_dim, action_feat_dim, bias=True))
            self.action_cls_fcs.append(
                nn.LayerNorm(action_feat_dim, eps=1e-5))
            self.action_cls_fcs.append(
                _get_activation_layer(ffn_act_cfg))
        self.fc_action = nn.Linear(action_feat_dim, num_classes_action)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                pass
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.fc_action.bias, bias_init)
        nn.init.zeros_(self.fc_reg.weight)
        nn.init.zeros_(self.fc_reg.bias)
        nn.init.uniform_(self.iof_tau, 0.0, 4.0)
        self.samplingmixing.init_weights()

    def refine_xyzr(self, xyzr, xyzr_delta, return_bbox=True):
        z = xyzr[..., 2:3]
        new_xy = xyzr[..., 0:2] + xyzr_delta[..., 0:2] * (2 ** z)
        new_zr = xyzr[..., 2:4] + xyzr_delta[..., 2:4]
        xyzr = torch.cat([new_xy, new_zr], dim=-1)
        if return_bbox:
            return xyzr, decode_box(xyzr)
        else:
            return xyzr

    def forward(self, features, proposal_boxes, spatial_queries, temporal_queries, featmap_strides=[4, 8, 16, 32]):

        N, n_query = spatial_queries.shape[:2]

        with torch.no_grad():
            rois = decode_box(proposal_boxes)
            roi_box_batched = rois.view(N, n_query, 4)
            iof = bbox_overlaps(roi_box_batched, roi_box_batched, mode='iof')[:, None, :, :]
            iof = (iof + 1e-7).log()
            pe = position_embedding(proposal_boxes, spatial_queries.size(-1) // 4)

        # IoF
        attn_bias = (iof * self.iof_tau.view(1, -1, 1, 1)).flatten(0, 1)             # N*num_heads, n_query, n_query
        pe = pe.permute(1, 0, 2)                                                     # n_query, N, content_dim

        # sinusoidal positional embedding
        spatial_queries = spatial_queries.permute(1, 0, 2)  # n_query, N, content_dim
        spatial_queries_attn = spatial_queries + pe
        spatial_queries = self.attention_s(spatial_queries_attn, attn_mask=attn_bias,)
        spatial_queries = self.attention_norm_s(spatial_queries)
        spatial_queries = spatial_queries.permute(1, 0, 2)
        # N, n_query, content_dim

        temporal_queries = temporal_queries.permute(1, 0, 2)
        temporal_queries_attn = temporal_queries + pe
        temporal_queries = self.attention_t(temporal_queries_attn, attn_mask=attn_bias,)
        temporal_queries = self.attention_norm_t(temporal_queries)
        temporal_queries = temporal_queries.permute(1, 0, 2)
        # N, n_query, content_dim

        spatial_queries, temporal_queries = \
            self.samplingmixing(features, proposal_boxes, spatial_queries, temporal_queries, featmap_strides)

        spatial_queries = self.ffn_norm_s(self.ffn_s(spatial_queries))
        temporal_queries = self.ffn_norm_t(self.ffn_t(temporal_queries))

        ################################### heads ###################################
        # objectness head
        cls_feat = spatial_queries
        for cls_layer in self.human_cls_fcs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.human_fc_cls(cls_feat).view(N, n_query, -1)

        # regression head
        reg_feat = spatial_queries
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)
        xyzr_delta = self.fc_reg(reg_feat).view(N, n_query, -1)

        # action head
        action_feat = torch.cat([spatial_queries, temporal_queries], dim=-1)
        for act_layer in self.action_cls_fcs:
            action_feat = act_layer(action_feat)
        action_score = self.fc_action(action_feat).view(N, n_query, -1)

        return cls_score, action_score, xyzr_delta, \
               spatial_queries.view(N, n_query, -1), temporal_queries.view(N, n_query, -1)



class STMDecoder(nn.Module):

    def __init__(self, cfg):

        super(STMDecoder, self).__init__()
        self.device = torch.device('cuda')

        self._generate_queries(cfg)

        self.num_stages = cfg.MODEL.STM.NUM_STAGES
        self.decoder_stages = nn.ModuleList()
        for i in range(self.num_stages):
            decoder_stage = AMStage(
                query_dim=cfg.MODEL.STM.HIDDEN_DIM,
                feat_channels=cfg.MODEL.STM.HIDDEN_DIM,
                num_heads=cfg.MODEL.STM.NUM_HEADS,
                feedforward_channels=cfg.MODEL.STM.DIM_FEEDFORWARD,
                dropout=cfg.MODEL.STM.DROPOUT,
                num_ffn_fcs=cfg.MODEL.STM.NUM_FCS,
                ffn_act=cfg.MODEL.STM.ACTIVATION,
                spatial_points=cfg.MODEL.STM.SPATIAL_POINTS,
                temporal_points=cfg.MODEL.STM.TEMPORAL_POINTS,
                out_multiplier=cfg.MODEL.STM.OUT_MULTIPLIER,
                n_groups=cfg.MODEL.STM.N_GROUPS,
                num_cls_fcs=cfg.MODEL.STM.NUM_CLS,
                num_reg_fcs=cfg.MODEL.STM.NUM_REG,
                num_action_fcs=cfg.MODEL.STM.NUM_ACT,
                num_classes_object=cfg.MODEL.STM.OBJECT_CLASSES,
                num_classes_action=cfg.MODEL.STM.ACTION_CLASSES
                )
            self.decoder_stages.append(decoder_stage)

        object_weight = cfg.MODEL.STM.OBJECT_WEIGHT
        giou_weight = cfg.MODEL.STM.GIOU_WEIGHT
        l1_weight = cfg.MODEL.STM.L1_WEIGHT
        action_weight = cfg.MODEL.STM.ACTION_WEIGHT
        background_weight = cfg.MODEL.STM.BACKGROUND_WEIGHT
        weight_dict = {"loss_ce": object_weight,
                       "loss_bbox": l1_weight,
                       "loss_giou": giou_weight,
                       "loss_bce": action_weight}
        self.person_threshold = cfg.MODEL.STM.PERSON_THRESHOLD


        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=object_weight,
                                   cost_bbox=l1_weight,
                                   cost_giou=giou_weight,
                                   use_focal=False)

        self.intermediate_supervision = cfg.MODEL.STM.INTERMEDIATE_SUPERVISION
        if self.intermediate_supervision:
            for i in range(self.num_stages - 1):
                inter_weight_dict = {k + f"_{i}": v for k, v in weight_dict.items()}
                weight_dict.update(inter_weight_dict)


        losses = ["labels", "boxes"]
        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=cfg.MODEL.STM.OBJECT_CLASSES,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=background_weight,
                                      losses=losses,
                                      use_focal=False)

    def _generate_queries(self, cfg):
        self.num_queries = cfg.MODEL.STM.NUM_QUERIES
        self.hidden_dim = cfg.MODEL.STM.HIDDEN_DIM

        # Build Proposals
        self.init_spatial_queries = nn.Embedding(self.num_queries, self.hidden_dim)
        self.init_temporal_queries = nn.Embedding(self.num_queries, self.hidden_dim)

    def _decode_init_queries(self, whwh):
        num_queries = self.num_queries
        proposals = torch.ones(num_queries, 4, dtype=torch.float, device=self.device, requires_grad=False)
        proposals[:, :2] = 0.5
        proposals = box_cxcywh_to_xyxy(proposals)

        batch_size = len(whwh)
        whwh = whwh[:, None, :] # B, 1, 4
        proposals = proposals[None] * whwh # B, N, 4

        xy = 0.5 * (proposals[..., 0:2] + proposals[..., 2:4])
        wh = proposals[..., 2:4] - proposals[..., 0:2]
        z = (wh).prod(-1, keepdim=True).sqrt().log2()
        r = (wh[..., 1:2] / wh[..., 0:1]).log2()
        xyzr = torch.cat([xy, z, r], dim=-1).detach()

        init_spatial_queries = self.init_spatial_queries.weight.clone()
        init_spatial_queries = init_spatial_queries[None].expand(batch_size, *init_spatial_queries.size())
        init_spatial_queries = torch.layer_norm(init_spatial_queries,
                                                normalized_shape=[init_spatial_queries.size(-1)])

        init_temporal_queries = self.init_temporal_queries.weight.clone()
        init_temporal_queries = init_temporal_queries[None].expand(batch_size, *init_temporal_queries.size())
        init_temporal_queries = torch.layer_norm(init_temporal_queries,
                                                 normalized_shape=[init_temporal_queries.size(-1)])

        return xyzr, init_spatial_queries, init_temporal_queries

    def person_detector_loss(self, outputs_class, outputs_coord, criterion, targets, outputs_actions):
        if self.intermediate_supervision:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_actions':outputs_actions[-1],
                      'aux_outputs': [{'pred_logits': a, 'pred_boxes': b, 'pred_actions':c}
                                      for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_actions[:-1])]}
        else:
            raise NotImplementedError

        loss_dict = criterion(output, targets)
        return loss_dict


    def make_targets(self, gt_boxes, whwh, labels):
        targets = []
        for box_in_clip, frame_size, label in zip(gt_boxes, whwh, labels):
            target = {}
            target['action_labels'] = torch.tensor(label, dtype=torch.float32, device=self.device)
            target['boxes_xyxy'] = torch.tensor(box_in_clip, dtype=torch.float32, device=self.device)
            # num_box, 4 (x1,y1,x2,y2) w.r.t augmented images unnormed
            target['labels'] = torch.zeros(len(target['boxes_xyxy']), dtype=torch.int64, device=self.device)
            target["image_size_xyxy"] = frame_size.to(self.device)
            # (4,) whwh
            image_size_xyxy_tgt = frame_size.unsqueeze(0).repeat(len(target['boxes_xyxy']), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            # (num_box, 4) whwh
            targets.append(target)

        return targets



    def forward(self, features, whwh, gt_boxes, labels, extras={}, part_forward=-1):

        proposal_boxes, spatial_queries, temporal_queries = self._decode_init_queries(whwh)

        inter_class_logits = []
        inter_pred_bboxes = []
        inter_action_logits = []
        B, N, _ = spatial_queries.size()

        for decoder_stage in self.decoder_stages:
            objectness_score, action_score, delta_xyzr, spatial_queries, temporal_queries = \
                decoder_stage(features, proposal_boxes, spatial_queries, temporal_queries)
            proposal_boxes, pred_boxes = decoder_stage.refine_xyzr(proposal_boxes, delta_xyzr)

            inter_class_logits.append(objectness_score)
            inter_pred_bboxes.append(pred_boxes)
            inter_action_logits.append(action_score)


        if not self.training:
            action_scores = torch.sigmoid(inter_action_logits[-1])
            scores = F.softmax(inter_class_logits[-1], dim=-1)[:, :, 0]
            # scores: B*100
            action_score_list = []
            box_list = []
            for i in range(B):
                selected_idx = scores[i] >= self.person_threshold
                if not any(selected_idx):
                    _,selected_idx = torch.topk(scores[i],k=3,dim=-1)

                action_score = action_scores[i][selected_idx]
                box = inter_pred_bboxes[-1][i][selected_idx]
                cur_whwh = whwh[i]
                box = clip_boxes_tensor(box, cur_whwh[1], cur_whwh[0])
                box[:, 0::2] /= cur_whwh[0]
                box[:, 1::2] /= cur_whwh[1]
                action_score_list.append(action_score)
                box_list.append(box)
            return action_score_list, box_list


        targets = self.make_targets(gt_boxes, whwh, labels)
        losses = self.person_detector_loss(inter_class_logits, inter_pred_bboxes, self.criterion, targets, inter_action_logits)
        weight_dict = self.criterion.weight_dict
        for k in losses.keys():
            if k in weight_dict:
                losses[k] *= weight_dict[k]
        return losses

def build_stm_decoder(cfg):
    return STMDecoder(cfg)
