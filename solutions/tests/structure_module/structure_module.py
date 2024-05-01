import torch
from torch import nn

from tests.structure_module.ipa import InvariantPointAttention
# from tests.structure_module.geometry import compute_all_atom_coordinates
from geometry.geometry import compute_all_atom_coordinates, assemble_4x4_transform, quat_to_3x3_rotation
from tests.structure_module import residue_constants
    

class StructureModuleTransition(nn.Module):
    
    def __init__(self, c_s):
        super().__init__()
        self.c_s = c_s

        self.linear_1 = nn.Linear(c_s, c_s)
        self.linear_2 = nn.Linear(c_s, c_s)
        self.linear_3 = nn.Linear(c_s, c_s)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(c_s)
        self.relu = nn.ReLU()

    def forward(self, s):
        s += self.linear_3(self.relu(self.linear_2(self.relu(self.linear_1(s)))))
        s = self.layer_norm(self.dropout(s))
        return s

class BackboneUpdate(nn.Module):

    def __init__(self, c_s):
        super().__init__()
        self.linear = nn.Linear(c_s, 6)

    def forward(self, s):
        # s has shape [*, N_res, c_s]
        group = self.linear(s)
        quat = torch.cat((torch.ones(group.shape[:-1]+(1,), device=group.device), group[...,:3]), dim=-1)
        quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        t = group[..., 3:]

        # a, b, c, d = torch.unbind(quat, dim=-1)
        # R = [
        #     [a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c],
        #     [2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b],
        #     [2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]
        # ]
        # R = [torch.stack(vals, dim=-1) for vals in R]
        # R = torch.stack(R, dim=-2)
        R = quat_to_3x3_rotation(quat)
        T = assemble_4x4_transform(R,  t)
        return T

class AngleResNetLayer(nn.Module):
    
    def __init__(self, c):
        super().__init__()
        self.linear_1 = nn.Linear(c, c)
        self.linear_2 = nn.Linear(c, c)
        self.relu = nn.ReLU()

    def forward(self, a):
        a += self.linear_2(self.relu(self.linear_1(self.relu(a))))
        return a


class AngleResNet(nn.Module):
    
    def __init__(self, c_s, c, n_torsion_angles=7):
        super().__init__()
        self.n_torsion_angles = n_torsion_angles
        self.linear_in = nn.Linear(c_s, c)
        self.linear_initial = nn.Linear(c_s, c)
        self.layers = nn.ModuleList([AngleResNetLayer(c) for _ in range(2)])
        self.linear_out = nn.Linear(c, 2*n_torsion_angles)
        self.relu = nn.ReLU()

    def forward(self, s, s_initial):
        # ReLUs absent in supplementary methods
        s = self.relu(s)
        s_initial = self.relu(s_initial)
        a = self.linear_in(s) + self.linear_initial(s_initial)
        for layer in self.layers:
            a = layer(a)
        alpha = self.linear_out(self.relu(a))
        alpha_shape = alpha.shape[:-1] + (self.n_torsion_angles, 2)
        alpha = alpha.view(alpha_shape)
        # alpha = alpha / torch.linalg.vector_norm(alpha, dim=-1, keepdim=True)
        return alpha



class StructureModule(nn.Module):
    
    def __init__(self, c_s, c_z, n_layer=8, c=128):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.n_layer = n_layer

        self.layer_norm_s = nn.LayerNorm(c_s)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_in = nn.Linear(c_s, c_s)

        self.layer_norm_ipa = nn.LayerNorm(c_s)
        self.dropout_s = nn.Dropout(0.1)
        self.ipa = InvariantPointAttention(c_s, c_z)
        self.transition = StructureModuleTransition(c_s)
        self.bb_update = BackboneUpdate(c_s)
        self.angle_resnet = AngleResNet(c_s, c)

    def forward(self, s, z, F):
        N_res = z.shape[-2]
        batch_dim = s.shape[:-2]
        T = torch.zeros(N_res, 4, 4)
        s_initial = self.layer_norm_s(s)
        z = self.layer_norm_z(z)

        s = self.linear_in(s_initial)
        T = torch.eye(4, device=s.device).broadcast_to(batch_dim+(N_res, 4, 4))
        outputs = {'angles': [], 'frames': []}

        for i in range(self.n_layer):
            s += self.ipa(s, z, T)
            s = self.layer_norm_ipa(self.dropout_s(s))
            s = self.transition(s)
            T = T @ self.bb_update(s)

            alpha = self.angle_resnet(s, s_initial)
            outputs['angles'].append(alpha)
            outputs['frames'].append(T)

        scaled_T = T.clone()
        scaled_T[..., :3, 3] *= 10
        final_positions, position_mask = compute_all_atom_coordinates(scaled_T, alpha, F)
        outputs['final_positions'] = final_positions
        outputs['position_mask'] = position_mask
        outputs['angles'] = torch.stack(outputs['angles'], dim=0)
        outputs['frames'] = torch.stack(outputs['frames'], dim=0)

        c_beta_ind = residue_constants.atom_types.index('CB')
        c_alpha_ind = residue_constants.atom_types.index('CA')
        glycin_ind = residue_constants.restypes.index('G')
        pseudo_beta_inds  = c_beta_ind * torch.ones(final_positions.shape[:-2]+(1,3), device=final_positions.device, dtype=torch.long)
        pseudo_beta_inds[F == glycin_ind] = c_alpha_ind
        pseudo_beta = torch.gather(final_positions, dim=1, index=pseudo_beta_inds).squeeze()

        outputs['pseudo_beta_positions'] = pseudo_beta

        return outputs
            

