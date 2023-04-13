import torch
import torch.nn.functional as F


def translate_to_linear_weight(ref: torch.Tensor, num_total, tau=2.0):
    # ref: [n, n_query, 1, in_points * n_heads]
    # num_total: feature levels (typically 4)
    grid = torch.arange(num_total, device=ref.device, dtype=ref.dtype).view(
        *[len(ref.shape)*[1, ]+[-1, ]])
    # [1, 1, 1, 1, num_total]

    ref = ref.unsqueeze(-1).clone()
    # [n, n_query, 1, in_points * n_heads, 1]
    l2 = (ref-grid).pow(2.0).div(tau).abs().neg()
    # [n, n_query, 1, in_points * n_heads, num_total]
    weight = torch.softmax(l2, dim=-1)

    return weight


def MHAQ3D(sample_points: torch.Tensor, value: torch.Tensor, weight=None, n_points=1):
    '''
    Args:
        sample_points: [n, n_query, 1, in_points * n_heads, 2]
        value: [n, c, t, h, w]
        weight: [n, n_query, 1, in_points * n_heads]
        n_points: in_points

    Returns:
        [B,c//n_heads,n_heads,t,in_points,n_query,1]
    '''
    B, Hq, Wq, n_heads_points, _ = sample_points.shape
    # print(value.shape)
    B, Ck, Tk, Hk, Wk = value.shape

    n_heads = n_heads_points//n_points

    sample_points = sample_points.view(B, Hq, Wq, n_heads, n_points, 2) \
        .permute(0, 3, 1, 2, 4, 5).contiguous().flatten(0, 1)
    # n*n_heads, n_query, 1, in_points, 2
    sample_points = sample_points.repeat(Tk, 1, 1, 1, 1)
    # n*n_heads*Tk, n_query, 1, in_points, 2
    sample_points = sample_points.flatten(2, 3)
    # n*n_heads*Tk, n_query, in_points, 2
    sample_points = sample_points*2.0-1.0
    value = value.view(B*n_heads, Ck//n_heads, Tk, Hk, Wk).permute(2, 0, 1, 3, 4).flatten(0, 1)
    out = F.grid_sample(
        value, sample_points,
        mode='bilinear', padding_mode='zeros', align_corners=False,
    )
    # n*n_heads*Tk, c//n_heads, n_query, in_points

    if weight is not None:
        weight = weight.view(B, Hq, Wq, n_heads, n_points) \
            .permute(0, 3, 1, 2, 4).flatten(0, 1).flatten(2, 3).unsqueeze(1).repeat(Tk, 1, 1, 1)
        # n*n_heads*Tk, 1, n_query, in_points
        out *= weight

    return out.view(Tk, B, n_heads, Ck//n_heads, Hq, Wq, n_points).permute(1, 3, 2, 0, 6, 4, 5)


def SAMPLE4D(sample_points: torch.Tensor, values: torch.Tensor, featmap_strides, n_points: int = 1, num_levels: int = None, mapping_stride=3.0, tau=2.0, ):
    B, Hq, Wq, n_heads_points, _ = sample_points.shape
    B, C, t, _, _ = values[0].shape

    n_heads = n_heads_points//n_points

    if num_levels is None:
        num_levels = len(values)

    sample_points_xy = sample_points[..., 0:2]
    # print(sample_points_xy.shape) torch.Size([2, 100, 1, 128=32*4, 2])
    # [n, n_query, 1, in_points * n_heads, 2] 
    
    sample_points_lvl = sample_points[..., 2].clone()
    # print(sample_points_lvl.shape) torch.Size([2, 100, 1, 128=32*4])
    # [n, n_query, 1, in_points * n_heads]

    sample_points_lvl_mapped = sample_points_lvl - mapping_stride
    # print(sample_points_lvl_mapped.shape) torch.Size([2, 100, 1, 128=32*4])
    # [n, n_query, 1, in_points * n_heads]

    sample_points_lvl_weight = translate_to_linear_weight(sample_points_lvl_mapped, num_levels, tau=tau)
    # print(sample_points_lvl_weight.shape) torch.Size([2, 100, 1, 128=32*4, 4])
    # [n, n_query, 1, in_points * n_heads, num_levels]

    sample_points_lvl_weight_list = sample_points_lvl_weight.unbind(-1)
    # [[n, n_query, 1, in_points * n_heads],....]

    out = sample_points.new_zeros(B, C//n_heads, n_heads, t, n_points, Hq, Wq)
    # print(out.shape) torch.Size([2, 64=256//4, 4, 4, 32, 100, 1])
    # n, dim//n_heads, n_heads, t, in_points, n_query, 1

    for i in range(num_levels):
        value = values[i]
        # B, C, T, H, W
        lvl_weights = sample_points_lvl_weight_list[i]
        stride = featmap_strides[i]
        
        mapping_size = value.new_tensor([value.size(4), value.size(3)]).view(1, 1, 1, 1, -1) * stride        
        normalized_xy = sample_points_xy / mapping_size
        # [n, n_query, 1, in_points * n_heads, 2]

        out += MHAQ3D(normalized_xy, value, weight=lvl_weights, n_points=n_points)

    return out, None
