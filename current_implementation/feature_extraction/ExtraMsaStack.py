import torch
from torch import nn
from current_implementation.feature_extraction.EvoformerBlock import MSARowAttentionWithPairBias, MSATransition, OuterProductMean, PairStack, DropoutRowwise
from solutions.attention.mha import MultiHeadAttention

class MSAColumnGlobalAttention(nn.Module):
    
    def __init__(self, c_m, c_z, c=8, N_head=8):
        super().__init__()
        self.layer_norm_m = nn.LayerNorm(c_m)
        self.global_attention = MultiHeadAttention(c_m, c, attn_dim=-3, N_head=N_head, gated=True, is_global=True)

    def forward(self, m):
        m = self.layer_norm_m(m)
        return self.global_attention(m)


class ExtraMsaBlock(nn.Module):
    
    def __init__(self, c_m, c_z):
        super().__init__()
        self.dropout_rowwise = DropoutRowwise(p=0.15)
        self.msa_att_row = MSARowAttentionWithPairBias(c_m, c_z, c=8)
        self.msa_att_col = MSAColumnGlobalAttention(c_m, c_z)
        self.msa_transition = MSATransition(c_m)
        self.outer_product_mean = OuterProductMean(c_m, c_z)
        self.core = PairStack(c_z)

    def forward(self, e, z):
        e = e + self.dropout_rowwise(self.msa_att_row(e, z))
        e = e + self.msa_att_col(e)
        e = e + self.msa_transition(e)

        z = z + self.outer_product_mean(e)
        z = self.core(z)

        return e, z
        


class ExtraMsaStack(nn.Module):
    
    def __init__(self, c_e, c_z, num_blocks):
        super().__init__()
        self.c_e = c_e
        self.c_z = c_z
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([ExtraMsaBlock(c_e, c_z) for _ in range(num_blocks)])

    def forward(self, e, z):
        for i, block in enumerate(self.blocks):
            e, z = block(e, z)
            # torch.save(e, f'kilian/test_outputs/post_extra_{i}_e.pt')
            # torch.save(z, f'kilian/test_outputs/post_extra_{i}_z.pt')

        return z
            
