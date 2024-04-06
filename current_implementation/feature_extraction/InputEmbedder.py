import torch
from torch import nn

class InputEmbedder(nn.Module):
    def __init__(self, c_m, c_z, tf_dim, vbins=32):
        super().__init__()
        self.c_z = c_z
        self.c_m = c_m
        self.tf_dim = tf_dim
        self.vbins = vbins

        self.linear_tf_z_i = nn.Linear(tf_dim, c_z)
        self.linear_tf_z_j = nn.Linear(tf_dim, c_z)
        self.linear_tf_m = nn.Linear(tf_dim, c_m)
        self.linear_msa_m = nn.Linear(49, c_m)
        self.linear_relpos = nn.Linear(2*vbins+1, c_z)

    def relpos(self, residue_index):
        residue_index = residue_index.long()
        d = residue_index.reshape(-1, 1) - residue_index.reshape(1, -1)
        d = torch.clamp(d, -self.vbins, self.vbins) + self.vbins
        d_onehot = nn.functional.one_hot(d, num_classes=2*self.vbins+1).float()
        return self.linear_relpos(d_onehot)
        

    def forward(self, batch):
        msa_feat = batch['msa_feat']
        target_feat = batch['target_feat']
        residue_index = batch['residue_index']

        a = self.linear_tf_z_i(target_feat)
        b = self.linear_tf_z_j(target_feat)
        z = a.unsqueeze(1) + b.unsqueeze(0)

        z += self.relpos(residue_index)
        m = self.linear_msa_m(msa_feat) + self.linear_tf_m(target_feat)
        return m, z

class ExtraMsaEmbedder(nn.Module):
    
    def __init__(self, f_e, c_e):
        super().__init__()
        self.linear = nn.Linear(f_e, c_e)

    def forward(self, batch):
        e = batch['extra_msa_feat']
        return self.linear(e)