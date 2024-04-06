import torch
from torch import nn

class RecyclingEmbedder(nn.Module):
    
    def __init__(self, c_m, c_z):
        super().__init__()
        self.bin_start = 3.25
        self.bin_end = 20.75
        self.bin_count = 15

        self.layer_norm_m = nn.LayerNorm(c_m)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear = nn.Linear(self.bin_count, c_z)
    
    def forward(self, m, z, m_prev, z_prev, x_prev):
        m[..., 0, :, :] += self.layer_norm_m(m_prev[..., 0, :, :])

        d = torch.linalg.vector_norm(x_prev.unsqueeze(-2) - x_prev.unsqueeze(-3), dim=-1)

        bins_lower = torch.linspace(self.bin_start, self.bin_end, self.bin_count, device=x_prev.device)
        bins_upper = torch.cat((bins_lower[1:], torch.tensor([1e8], device=x_prev.device)))

        d = d.unsqueeze(-1)
        d = ((d>bins_lower) * (d<bins_upper)).type(x_prev.dtype)
        d = self.linear(d)



        z += d + self.layer_norm_z(z_prev)

        return m, z