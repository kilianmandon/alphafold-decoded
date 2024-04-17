import torch
from torch import nn

def create_3x3_rotation(ex, ey):
    """
    Creates a rotation matrix by orthonormalizing ex and ey via Gram-Schmidt.
    Supports batched operation.

    Args:
        ex (torch.tensor): X-axes of the new frames, of shape (*, 3).
        ey (torch.tensor): Y-axes of the new frames, of shape (*, 3).

    Returns:
        torch.tensor: Rotation matrices of shape (*, 3, 3).
    """
    
    R = None
    
    ##########################################################################
    # TODO: Orthonormalize ex and ey, then compute ez as their crossproduct. # 
    #  Use torch.linalg.vector_norm to compute the norms for normalization.  #
    #  Orthogonalize ey against ex by subtracting the non-orthogonal part,   #
    #  ex * <ex, ey> from ey, after normalizing ex.                          #
    #  The keepdim parameter can be helpful for both operations.             #
    #  Stack the vectors as columns to build the rotation matrix.            #
    #  Make your to broadcast correctly, to allow for any number of          #
    #  leading dimensions.                                                   #
    ##########################################################################

    ex = ex / torch.linalg.vector_norm(ex, dim=-1, keepdim=True)
    ey = ey - ex * torch.sum(ex*ey, dim=-1, keepdim=True)
    ez = torch.linalg.cross(ex, ey, dim=-1)
    R = torch.stack((ex, ey, ez), dim=-1)

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################

    return R

def create_quaternion(phi, n):
    a = torch.cos(phi).unsqueeze(-1)
    v = torch.sin(phi) * n
    q = torch.cat((a, v), dim=-1)
    print(torch.linalg.vector_norm(q))

    return q

def quat_mul(q1, q2):
    
    a1 = q1[...,0:1] # a1 has shape (*, 1)
    v1 = q1[..., 1:] # v1 has shape (*, 3)
    a2 = q2[...,0:1] # a2 has shape (*, 1)
    v2 = q2[..., 1:] # v2 has shape (*, 3)

    a_out = a1*a2 - torch.sum(v1*v2, dim=-1, keepdim=True)
    v_out = a1*v2 + a2*v1 + torch.linalg.cross(v1, v2, dim=-1)

    q_out = torch.cat((a_out, v_out), dim=-1)

    return q_out

def conjugate_quat(q):
    q_out = q.clone()
    q_out[..., 1:] = -q_out[..., 1:]

    return q_out

def quat_vector_mul(q, v):
    v_batch_shape = v.shape[:-1]
    zero_pad = torch.zeros(v_batch_shape+(1,), device=v.device, dtype=v.dtype)
    padded_v = torch.cat((zero_pad, v), dim=-1)

    q_pre_bc_shape = (1,) * len(v_batch_shape) + (-1,)
    q_bc_shape = v_batch_shape + (-1,)
    q = q.view(q_pre_bc_shape).broadcast_to(q_bc_shape)

    q_out = quat_mul(q, quat_mul(padded_v, conjugate_quat(q)))
    v_out = q_out[...,1:]
    return v_out

def assemble_4x4_transform(R, t):
    batch_shape = t.shape[:-1]

    Rt = torch.cat((R, t[..., None]), dim=-1)
    pad = torch.zeros(batch_shape+(4,), device=t.device, dtype=t.dtype)
    pad[..., -1] = 1
    T = torch.cat((Rt, pad), dim=-2)
    return T
    

def create_4x4_transform(ex, ey, t):
    R = create_3x3_rotation(ex, ey)
    return assemble_4x4_transform(R, t)
    
def invert_4x4_transform(T):
    R = T[...,:3,:3]
    t = T[...,:3,3]
    inv_R = R.T
    inv_t = - torch.einsum('...ij,...j', R.T, t)
    return assemble_4x4_transform(inv_R, inv_t)







