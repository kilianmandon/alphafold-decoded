import torch
from current_implementation.structure_module.residue_constants import rigid_group_atom_positions, chi_angles_mask, atom_types, rigid_group_atom_position_map

chi_angle_chains = {
    'ALA': [],
    'ARG': ['CB', 'CG', 'CD', 'NE'],
    'ASN': ['CB', 'CG'],
    'ASP': ['CB', 'CG'],
    'CYS': ['CB'],
    'GLN': ['CB', 'CG', 'CD'],
    'GLU': ['CB', 'CG', 'CD'],
    'GLY': [],
    'HIS': ['CB', 'CG'],
    'ILE': ['CB', 'CG1'],
    'LEU': ['CB', 'CG'],
    'LYS': ['CB', 'CG', 'CD', 'CE'],
    'MET': ['CB', 'CG', 'SD'],
    'PHE': ['CB', 'CG'],
    'PRO': ['CB', 'CG'],
    'SER': ['CB'],
    'THR': ['CB'],
    'TRP': ['CB', 'CG'],
    'TYR': ['CB', 'CG'],
    'VAL': ['CB'],
}

def create_4x4_transform(ex, ey, translation):
    ex = ex / torch.linalg.norm(ex, dim=-1, keepdim=True)
    ey = ey - torch.einsum('...i,...i->...', ex, ey) * ex
    ey = ey / torch.linalg.norm(ey, dim=-1, keepdim=True)
    ez = torch.linalg.cross(ex, ey, dim=-1)
    base = torch.stack((ex, ey, ez, translation), dim=-1)
    pad = torch.tensor([0,0,0,1], device=base.device).broadcast_to(base.shape[:-2]+(1,4))
    transform = torch.cat((base, pad), dim=-2)
    return transform

def invert_4x4_transform(T):
    # T has shape [..., 4, 4]
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.einsum('...ij,...j->...i', R_inv, t)
    return build_transform(R_inv, t_inv)

def build_transform(R, t):
    # R has shape [*, 3, 3], t has shape [*, 3]
    shape = t.shape[:-1]
    ones_shape = (1,) * len(shape)
    base = torch.tensor([0,0,0,1], device=R.device).view(ones_shape+(1,4)).broadcast_to(shape+(1,4))
    t = t.unsqueeze(-1)
    Rt = torch.cat((R, t), dim=-1)
    T  = torch.cat((Rt, base), dim=-2)
    return T

def warp_3d_point(T, x):
    # T has shape [*, 4, 4]
    # x has shape [*, 3]
    base = x.shape[:-1]
    pad = torch.ones(base+(1,), device=x.device)
    padded = torch.cat((x, pad), dim=-1)
    res = torch.einsum('...ij,...j->...i', T, padded)
    return res[...,:3]

    
def precalculate_rigid_transforms():
    result = torch.zeros((20, 8, 4, 4))
    backbone_group = torch.eye(4)
    pre_omega_group = torch.eye(4)
    result[:, 0, ...] = backbone_group
    result[:, 1, ...] = pre_omega_group

    for i, (aa, atom_positions) in enumerate(rigid_group_atom_position_map.items()):
        ex_phi = atom_positions['N'] - atom_positions['CA']
        ey_phi = torch.tensor([1.0, 0.0, 0.0]).broadcast_to(ex_phi.shape)
        phi_group = create_4x4_transform(
            ex=ex_phi,
            ey=ey_phi,
            translation=atom_positions['N']
        )
        result[i, 2, ...] = phi_group
        psi_group = create_4x4_transform(
            ex = atom_positions['C'] - atom_positions['CA'],
            ey = atom_positions['CA'] - atom_positions['N'],
            translation=atom_positions['C']
        )
        result[i, 3, ...] = psi_group
        for j in range(4):
            if not chi_angles_mask[i][j]:
                chi_group = torch.eye(4)
            elif j==0:
                next_atom = chi_angle_chains[aa][j]
                chi_group = create_4x4_transform(
                    ex = atom_positions[next_atom] - atom_positions['CA'],
                    ey = atom_positions['N'] - atom_positions['CA'],
                    translation= atom_positions[next_atom]
                )
            else:
                next_atom = chi_angle_chains[aa][j]
                ex = atom_positions[next_atom]
                ey = torch.tensor([-1.0, 0.0, 0.0]).broadcast_to(ex.shape)
                chi_group = create_4x4_transform(
                    ex = ex,
                    ey = ey,
                    translation=atom_positions[next_atom]
                )

            result[i, j+4, ...] = chi_group
    return result

def makeRotX(phi):
    phi_shape = phi.shape[:-1]
    rot_mat_shape = phi_shape + (3,3)
    R = torch.eye(3, device=phi.device).broadcast_to(rot_mat_shape)
    R = R.clone()
    phi1, phi2 = torch.unbind(phi, dim=-1)
    # This is following the code rather than the supplement
    R[..., 1, 1] = phi2
    R[..., 1, 2] = -phi1
    R[..., 2, 1] = phi1
    R[..., 2, 2] = phi2
    t = torch.zeros(phi_shape+(3,), device=phi.device)
    T = build_transform(R, t)
    return T
    

def compute_all_atom_coordinates(T, alpha, F):
    # alpha has shape [N_res, N_torsion, 2]
    # T has shape [N_res, 4, 4]
    # F has shape [N_res]
    alpha = alpha / torch.linalg.vector_norm(alpha, dim=-1, keepdim=True)
    omega, phi, psi, chi1, chi2, chi3, chi4 = torch.unbind(alpha, dim=-2)
    # all_rigid_transforms has shape [N_aa, 8, 4, 4]
    all_rigid_transforms = precalculate_rigid_transforms().to(alpha.device)
    # local_rigid_transforms has shape [N_res, 8, 4, 4]
    gather_index = F.view(F.shape+(1,1,1)).broadcast_to(F.shape+all_rigid_transforms.shape[-3:])
    local_rigid_transforms = torch.gather(all_rigid_transforms, dim=0, index=gather_index)
    global_rigid_transforms = torch.zeros_like(local_rigid_transforms)

    global_rigid_transforms[:, 0] = T
    for i, ang in zip(range(1, 5), [omega, phi, psi, chi1]):
        global_rigid_transforms[:, i] = T@local_rigid_transforms[:, i] @ makeRotX(ang)

    for i, ang in zip(range(5, 8), [chi2, chi3, chi4]):
        global_rigid_transforms[:, i] = global_rigid_transforms[:, i-1] @ local_rigid_transforms[:, i] @ makeRotX(ang)

    all_atom_local_positions = torch.zeros((20, 37, 3), device=T.device)
    all_atom_frame_inds = torch.zeros((20, 37), device=T.device, dtype=torch.long)
    atom_position_mask = torch.zeros((20, 37), device=T.device)

    for i, (aa, atoms) in enumerate(rigid_group_atom_positions.items()):
        for atom_name, frame_ind, pos in atoms:
            atom_ind = atom_types.index(atom_name)
            all_atom_frame_inds[i, atom_ind] = frame_ind
            all_atom_local_positions[i, atom_ind, :] = pos
            atom_position_mask[i, atom_ind] = 1

    gather_index = F.view(-1, 1).broadcast_to((-1, 37))
    # all_residue_frame_inds has shape [N_res, 37]
    all_residue_frame_inds = torch.gather(all_atom_frame_inds, dim=0, index=gather_index)
    atom_position_mask = torch.gather(atom_position_mask, dim=0, index=gather_index)
    gather_index = F.view(-1, 1, 1).broadcast_to((-1, 37, 3))
    # all_residue_local_positions has shape [N_res, 37, 3]
    all_residue_local_positions = torch.gather(all_atom_local_positions, dim=0, index=gather_index)
    shape_before_bc = all_residue_frame_inds.shape + (1, 1)
    shape = all_residue_frame_inds.shape + (4, 4)
    all_residue_frame_inds = all_residue_frame_inds.view(shape_before_bc).broadcast_to(shape)
    # all_residue_frames has shape [N_res, 37, 4, 4]
    all_residue_frames = torch.gather(global_rigid_transforms, dim=1, index=all_residue_frame_inds)
    pad = torch.ones(all_residue_local_positions.shape[:-1] + (1,), device=F.device)
    all_residue_homogenous_positions = torch.cat((all_residue_local_positions, pad), dim=-1)
        
    all_atom_coordinates = torch.einsum('...ijk,...ik->...ij', all_residue_frames, all_residue_homogenous_positions)
    all_atom_coordinates = all_atom_coordinates[..., :-1]

    return all_atom_coordinates, atom_position_mask
    



            
  