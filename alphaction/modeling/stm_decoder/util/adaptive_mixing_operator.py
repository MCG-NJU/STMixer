import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMixing(nn.Module):
    def __init__(self, in_dim, in_points, n_groups, query_dim=None,
                 out_dim=None, out_points=None, sampling_rate=None):
        super(AdaptiveMixing, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim
        sampling_rate = sampling_rate if sampling_rate is not None else 1

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points//sampling_rate
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim//n_groups
        self.eff_out_dim = out_dim//n_groups

        self.pad_bias_dim = 0
        self.pad_bias_points = 0

        self.eff_in_dim = self.eff_in_dim + self.pad_bias_dim
        self.in_points = self.in_points + self.pad_bias_points

        self.REDUCTION = 1

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points

        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Sequential(
            nn.Linear(self.query_dim, self.n_groups*self.total_parameters),
        )

        self.out_proj = nn.Linear(
            self.eff_out_dim*self.out_points*self.n_groups, self.query_dim, bias=True
        )

        self.act = nn.ReLU(inplace=True)

        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        nn.init.zeros_(self.parameter_generator[-1].weight)

    def forward(self, x, query):

        B, N, g, P, C = x.size()
        G = self.n_groups
        assert g == G


        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*N, G, -1)
        out = x.reshape(B*N, G, P, C)

        M, S = params.split(
            [self.m_parameters, self.s_parameters], 2)

        M = M.reshape(B*N, G, self.eff_in_dim, self.eff_in_dim)
        S = S.reshape(B*N, G, self.out_points, self.in_points)


        '''adaptive channel mixing
        the process also can be done with torch.bmm
        but for clarity, we use torch.matmul
        '''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive spatial mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, N, -1)
        out = self.out_proj(out)

        out = query + out

        return out
