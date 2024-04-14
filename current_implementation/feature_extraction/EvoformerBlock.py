from torch import nn
import math
import torch
from solutions.attention.mha import MultiHeadAttention

class SharedDropout(nn.Module):
    def __init__(self, shared_dim: int, p: float):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.shared_dim = shared_dim

    def forward(self, x):
        mask_shape = list(x.shape)
        mask_shape[self.shared_dim] = 1
        mask = torch.ones(mask_shape, device=x.device)
        mask = self.dropout(mask)
        return x * mask

class DropoutRowwise(SharedDropout):
    def __init__(self, p: float):
        super().__init__(shared_dim=-3, p=p)

class DropoutColumnwise(SharedDropout):
    def __init__(self, p: float):
        super().__init__(shared_dim=-2, p=p)

class OldMultiHeadAttention(nn.Module):

    def __init__(self, c_in, c, N_head, attn_dim, gated=False, average_queries_over=None, N_head_kv=None):
        super().__init__()
        if N_head_kv is None:
            N_head_kv = N_head

        self.c_in = c_in
        self.c = c
        self.N_head = N_head
        self.N_head_kv = N_head_kv
        self.gated = gated
        self.average_queries_over = average_queries_over
        self.attn_dim = attn_dim

        if average_queries_over is None:
            emb_kv_dim = c * N_head_kv
        else:
            emb_kv_dim = c
        
        self.linear_q = nn.Linear(c_in, c*N_head, bias=False)
        self.linear_k = nn.Linear(c_in, emb_kv_dim, bias=False)
        self.linear_v = nn.Linear(c_in, emb_kv_dim, bias=False)
        self.linear_o = nn.Linear(c*N_head, c_in)
        if gated:
            self.linear_g = nn.Linear(c_in, c*N_head)

    def prepare_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if self.average_queries_over is not None:
            q = torch.mean(q, dim=self.average_queries_over, keepdim=True)
        # Start shape: [*, q/k/v, *, N_head*c]
        # Transposing to [*, q/k/v, N_head*c]
        q = q.movedim(self.attn_dim, -2)
        k = k.movedim(self.attn_dim, -2)
        v = v.movedim(self.attn_dim, -2)

        # Unwrapping to [*, q/k/v, N_head, c]
        q_shape = q.shape[:-1] + (self.N_head, -1)
        k_shape = k.shape[:-1] + (self.N_head_kv, -1)
        v_shape = v.shape[:-1] + (self.N_head_kv, -1)

        q = q.view(q_shape)
        k = k.view(k_shape)
        v = v.view(v_shape)

        # Transposing to [*, N_head, q/k/v, c]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        q = q / math.sqrt(self.c)

        return q, k, v

    def forward(self, x, bias=None):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        q, k, v = self.prepare_qkv(q, k, v)
        a = torch.einsum('...qc,...kc->...qk', q, k)
        if bias is not None:
            a = a + bias
        a = torch.softmax(a, dim=-1)
        # o has shape [*, N_head, q, c]
        o = torch.einsum('...qk,...kc->...qc', a, v)
        o = o.transpose(-3, -2)
        o = torch.flatten(o, start_dim=-2)
        o = o.moveaxis(-2, self.attn_dim)
        if self.gated:
            g = torch.sigmoid(self.linear_g(x))
            o = g * o

        m = self.linear_o(o)
        return m
            

        

class MSARowAttentionWithPairBias(nn.Module):
    def __init__(self, c_m, c_z, c=32, N_head=8):
        super().__init__()
        self.c = c
        self.N_head=8
        
        self.layer_norm_m = nn.LayerNorm(c_m)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_z = nn.Linear(c_z, N_head, bias=False)
        self.mha = MultiHeadAttention(c_m, c, N_head, attn_dim=-2, gated=True)

    def forward(self, m, z):
        m = self.layer_norm_m(m)
        b = self.linear_z(self.layer_norm_z(z))
        b = b.moveaxis(-1, -3)

        return self.mha(m, bias=b)

class MSAColumnAttention(nn.Module):
    def __init__(self, c_m, c_z, c=32, N_head=8):
        super().__init__()
        self.c = c
        self.N_head = N_head

        self.layer_norm_m = nn.LayerNorm(c_m)
        self.mha = MultiHeadAttention(c_m, c, N_head, attn_dim=-3, gated=True)

    def forward(self, m):
        m = self.layer_norm_m(m)
        return self.mha(m)



class MSATransition(nn.Module):
    def __init__(self, c_m, n=4):
        super().__init__()
        self.c_m = c_m
        self.n = n
        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, n*c_m)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(n*c_m, c_m)

    def forward(self, m):
        m = self.layer_norm(m)
        a = self.linear_1(m)
        m = self.linear_2(self.relu(a))
        return m

class OuterProductMean(nn.Module):
    def __init__(self, c_m, c_z, c=32):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c = c
        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, c)
        self.linear_2 = nn.Linear(c_m, c)
        self.linear_out  = nn.Linear(c*c, c_z)

    def forward(self, m):
        N_seq = m.shape[-3]
        m = self.layer_norm(m)
        a = self.linear_1(m)
        b = self.linear_2(m)
        o = torch.einsum('...sic,...sjd->...ijcd', a, b) # / N_seq
        o = torch.flatten(o, start_dim=-2)
        z = self.linear_out(o) / N_seq
        return z

class TriangleMultiplication(nn.Module):

    def __init__(self, c_z, mult_type, c=128):
        super().__init__()
        if mult_type not in {'outgoing', 'incoming'}:
            raise ValueError(f'mult_type must be either "outgoing" or "incoming" but is {mult_type}.')

        self.mult_type = mult_type
        self.c_z = c_z
        self.c = c
        self.layer_norm_in = nn.LayerNorm(c_z)
        self.layer_norm_out = nn.LayerNorm(c_z)
        self.linear_a_p = nn.Linear(c_z, c)
        self.linear_a_g = nn.Linear(c_z, c)
        self.linear_b_p = nn.Linear(c_z, c)
        self.linear_b_g = nn.Linear(c_z, c)
        self.linear_g = nn.Linear(c_z, c_z)
        self.linear_z = nn.Linear(c, c_z)

    def forward(self, z):
        z = self.layer_norm_in(z)
        a = torch.sigmoid(self.linear_a_g(z)) * self.linear_a_p(z)
        b = torch.sigmoid(self.linear_b_g(z)) * self.linear_b_p(z)
        g = torch.sigmoid(self.linear_g(z))

        if self.mult_type == 'outgoing':
            z = torch.einsum('...ikc,...jkc->...ijc', a, b)
        else:
            z = torch.einsum('...kic,...kjc->...ijc', a, b)

        z = g * self.linear_z(self.layer_norm_out(z))
        return z

class TriangleAttention(nn.Module):
    def __init__(self, c_z, node_type, c=32, N_head=4):
        super().__init__()
        if node_type not in {'starting_node', 'ending_node'}:
            raise ValueError(f'node_type must be either "starting_node" or "ending_node" but is {node_type}')

        self.node_type = node_type
        self.c_z = c_z
        self.c = c
        self.N_head = N_head

        self.layer_norm = nn.LayerNorm(c_z)
        if node_type == 'starting_node':
            attn_dim = -2
        else:
            attn_dim = -3

        self.mha = MultiHeadAttention(c_z, c, N_head, attn_dim, gated=True)
        self.linear = nn.Linear(c_z, N_head, bias=False)

    def forward(self, z):
        z = self.layer_norm(z)
        bias = self.linear(z)
        bias = bias.moveaxis(-1, -3)
        if self.node_type == 'ending_node':
            bias = bias.transpose(-1, -2)
        return self.mha(z, bias=bias)

class PairTransition(nn.Module):
    
    def __init__(self, c_z, n=4):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_z)
        self.linear_1 = nn.Linear(c_z, n*c_z)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(n*c_z, c_z)

    def forward(self, z):
        z = self.layer_norm(z)
        a = self.linear_1(z)
        z = self.linear_2(self.relu(a))
        return z


class PairStack(nn.Module):

    def __init__(self, c_z):
        super().__init__()
        self.dropout_rowwise = DropoutRowwise(p=0.25)
        self.dropout_columnwise = DropoutColumnwise(p=0.25)
        self.tri_mul_out = TriangleMultiplication(c_z, mult_type='outgoing')
        self.tri_mul_in = TriangleMultiplication(c_z, mult_type='incoming')
        self.tri_att_start = TriangleAttention(c_z, node_type='starting_node')
        self.tri_att_end = TriangleAttention(c_z, node_type='ending_node')
        self.pair_transition = PairTransition(c_z)

    def forward(self, z):
        z = z + self.dropout_rowwise(self.tri_mul_out(z))
        z = z + self.dropout_rowwise(self.tri_mul_in(z))
        z = z + self.dropout_rowwise(self.tri_att_start(z))
        z = z + self.dropout_columnwise(self.tri_att_end(z))
        z = z + self.pair_transition(z)
        return z

    
class EvoformerBlock(nn.Module):
    
    def __init__(self, c_m, c_z):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.dropout_rowwise_m = DropoutRowwise(p=0.15)
        self.msa_att_row = MSARowAttentionWithPairBias(c_m, c_z)
        self.msa_att_col = MSAColumnAttention(c_m, c_z)
        self.msa_transition = MSATransition(c_m, )
        self.outer_product_mean = OuterProductMean(c_m, c_z)
        self.core = PairStack(c_z)
    
    def forward(self, m, z):
        m = m + self.dropout_rowwise_m(self.msa_att_row(m, z))
        m = m + self.msa_att_col(m)
        m = m + self.msa_transition(m)

        z = z + self.outer_product_mean(m)

        z = self.core(z)
        return m, z
