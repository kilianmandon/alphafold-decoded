import torch
import math
from torch import nn

from current_implementation.structure_module.geometry import invert_4x4_transform, warp_3d_point

class InvariantPointAttention(nn.Module):
    
    def __init__(self, c_s, c_z, n_query_points=4, n_point_values=8, n_head=12, c=16):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.n_query_points = n_query_points
        self.n_point_values = n_point_values
        self.n_head = n_head
        self.c = c

        # The qkv and qp,kp,vp layers use bias, in contrast to the Supplement,
        # but in accordance with openfold (and they say with alphafold implementation)
        self.linear_q = nn.Linear(c_s, n_head * c, bias=True)
        self.linear_k = nn.Linear(c_s, n_head * c, bias=True)
        self.linear_v = nn.Linear(c_s, n_head * c, bias=True)

        self.linear_q_points = nn.Linear(c_s, n_head*n_query_points*3, bias=True)
        self.linear_k_points = nn.Linear(c_s, n_head*n_query_points*3, bias=True)
        self.linear_v_points = nn.Linear(c_s, n_head*n_point_values*3, bias=True)
        self.linear_b = nn.Linear(c_z, n_head)
        self.linear_out = nn.Linear(n_head*c_z+n_head*c+n_head*4*n_point_values, c_s)

        self.head_weights = nn.Parameter(torch.zeros((n_head,)))
        self.softplus = nn.Softplus()

    def prepare_qkv(self, s):
        c = self.c
        n_head = self.n_head
        n_qp = self.n_query_points
        n_pv = self.n_point_values
        layers = [self.linear_q, self.linear_k, self.linear_v, self.linear_q_points, self.linear_k_points, self.linear_v_points]
        outs = [layer(s) for layer in layers]

        shape_adds = [(n_head, c), (n_head, c), (n_head, c), (3, n_head, n_qp), (3, n_head, n_qp), (3, n_head, n_pv)]
        out_shapes = [out.shape[:-1]+shape_add for out, shape_add in zip(outs, shape_adds)]
        outs = [out.view(out_shape) for out, out_shape in zip(outs, out_shapes)]
        for i in range(3):
            outs[i] = outs[i].movedim(-3, -2)
        for i in range(3, 6):
            outs[i] = outs[i].movedim(-3, -1).movedim(-4, -2)
        return outs

    def forward(self, s, z, T):
        # T has shape [*, N_res, 4, 4]
        # s has shape [*, N_res, c_s]
        # z has shape [*, N_res, N_res, c_z]
        q, k, v, qp, kp, vp = self.prepare_qkv(s)
        # q,k,v have shape [*, n_head, N_res, c]
        # qp,kp,vp have shape [*, n_head, n_qp/n_pv, N_res, 3]
        q *= 1/math.sqrt(self.c)

        wc = math.sqrt(2 / (9*self.n_query_points))
        wl = math.sqrt(1/3)
        gamma = self.softplus(self.head_weights).view((-1, 1, 1))

        bias = self.linear_b(z).movedim(-1, -3)

        qk_term = torch.einsum('...ic,...jc->...ij', q, k)

        T_bc_qkv = T.view(T.shape[:-3] + (1, 1, -1, 4, 4))
        transformed_qp = warp_3d_point(T_bc_qkv, qp).unsqueeze(-2)
        transformed_kp = warp_3d_point(T_bc_qkv, kp).unsqueeze(-3)
        head_weights = gamma * wc / math.sqrt(3)
        sq_dist = torch.sum((transformed_qp - transformed_kp)**2, dim=-1)
        qpkp_term = gamma * wc / 2 * torch.sum(sq_dist, dim=-3)

        # att_score has shape [*, n_head, N_res, N_res]
        att_score = torch.softmax(wl * (qk_term + bias - qpkp_term), dim=-1)

        pairwise_out = torch.einsum('...hij,...ijc->...hic', att_score, z)
        pairwise_out = pairwise_out.movedim(-3, -2).flatten(start_dim=-2)
        v_out = torch.einsum('...hij,...hjc->...hic', att_score, v)
        v_out = v_out.movedim(-3, -2).flatten(start_dim=-2)

        vp_out = torch.einsum('...hij,...hpjc->...hpic', att_score, warp_3d_point(T_bc_qkv, vp))
        T_inv = invert_4x4_transform(T).view(T.shape[:-3] + (1, 1, -1, 4, 4))
        vp_out = warp_3d_point(T_inv, vp_out)
        vp_out = torch.flatten(vp_out, -4, -3)
        vp_out_norm = torch.linalg.vector_norm(vp_out, dim=-1, keepdim=True)
        # Reshaping to [*, N_res, c, num_heads*num_points]
        vp_out = vp_out.movedim(-3, -1).flatten(start_dim=-2)
        vp_out_norm = vp_out_norm.movedim(-3, -2).flatten(start_dim=-2)

        out = torch.cat((v_out, vp_out, vp_out_norm, pairwise_out), dim=-1)

        return self.linear_out(out)
        
