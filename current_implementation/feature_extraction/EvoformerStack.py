from torch import nn

from current_implementation.feature_extraction.EvoformerBlock import EvoformerBlock

class EvoformerStack(nn.Module):
    
    def __init__(self, c_m, c_z, num_blocks, c_s=384):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_s = c_s
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([EvoformerBlock(c_m, c_z) for _ in range(num_blocks)])
        self.linear = nn.Linear(c_m, c_s)

    def forward(self, m, z):
        for evo_block in self.blocks:
            m, z = evo_block(m, z)
        
        s = self.linear(m[..., 0, :, :])
        return m, z, s
         