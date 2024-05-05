import torch
import math
from torch import nn

from geometry.geometry import invert_4x4_transform, warp_3d_point

class InvariantPointAttention(nn.Module):
    """
    Implements invariant point attention, according to Algorithm 22.
    """
    
    def __init__(self, c_s, c_z, n_query_points=4, n_point_values=8, N_head=12, c=16):
        """
        Initializes the invariant point attention module. 

        Args:
            c_s (int): Number of channels for the single representation.
            c_z (int): Number of channels for the pair representation.
            n_query_points (int, optional): Number of query points for point attention. 
                Used for the embedding of q_points and k_points. Defaults to 4.
            n_point_values (int, optional): Number of value points for point attention. 
                Used for the embedding of v_points. Defaults to 8.
            n_head (int, optional): Number of heads for multi-head attention. Defaults to 12.
            c (int, optional): Embedding dimension for each individual head. Defaults to 16.
        """
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.n_query_points = n_query_points
        self.n_point_values = n_point_values
        self.N_head = N_head
        self.c = c


        ##########################################################################
        # TODO: Initialize the layers linear_q, linear_k, linear_v,              #
        #   linear_q_points, linear_k_points, linear_v_points, linear_b, and     # 
        #   linear_out. The embeddings for q, k and v are similar to             #
        #   MultiHeadAttention, except that they use bias (this clashes with the #
        #   supplement, but follows the official implementation).                #
        #   The point embeddings need to create three values per head and point. #
        #   They also use bias.                                                  #
        #   The embedding for the bias computes one bias value per head.         #
        #   For the input dimension of linear_out, count the channels of the     #
        #   various outputs in line 11 from the algorithm. If you have trouble   #
        #   with this, you can look below at the output description of           #
        #   `compute_outputs`. The output dimension of linear_out is c_s.        #
        #                                                                        #
        #   For the weight per head, gamma, initialize head_weights to a         #
        #   zero-tensor wrapped in nn.Parameter. Also, initialize nn.Softplus    #
        #   for the computation of gamma.                                        #
        ##########################################################################

        # The qkv and qp,kp,vp layers use bias, in contrast to the Supplement,
        # but in accordance the official implementation.
        self.linear_q = nn.Linear(c_s, N_head * c, bias=True)
        self.linear_k = nn.Linear(c_s, N_head * c, bias=True)
        self.linear_v = nn.Linear(c_s, N_head * c, bias=True)

        self.linear_q_points = nn.Linear(c_s, N_head*n_query_points*3, bias=True)
        self.linear_k_points = nn.Linear(c_s, N_head*n_query_points*3, bias=True)
        self.linear_v_points = nn.Linear(c_s, N_head*n_point_values*3, bias=True)
        self.linear_b = nn.Linear(c_z, N_head)
        self.linear_out = nn.Linear(N_head*c_z+N_head*c+N_head*4*n_point_values, c_s)

        self.head_weights = nn.Parameter(torch.zeros((N_head,)))
        self.softplus = nn.Softplus()

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def prepare_qkv(self, s):
        """
        Creates the standard attention embeddings q, k, and v, as well as the point 
        embeddings qp, kp, and vp, for invariant point attention.

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).

        Returns:
            tuple: A tuple consisting of the following embeddings:
                q: Tensor of shape (*, N_head, N_res, c)  
                k: Tensor of shape (*, N_head, N_res, c)
                v: Tensor of shape (*, N_head, N_res, c)
                qp: Tensor of shape (*, N_head, N_query_poitns, N_res, 3)
                kp: Tensor of shape (*, N_head, N_query_points, N_res, 3)
                vp: Tensor of shape (*, N_head, N_point_values, N_res, 3)
        """
        c = self.c
        n_head = self.N_head
        n_qp = self.n_query_points
        n_pv = self.n_point_values

        embeddings = None

        ##########################################################################
        # TODO: Implement the embedding preparation in the following steps:      #
        #   - Pass s through all of the embedding layers.                        # 
        #   - Reshape the feature dimension of the embeddings so that q, k and v #
        #     have shape (*, N_head, c), qp and kp have shape                    #
        #     (*, 3, N_head, n_qp) and vp has shape (*, 3, N_head, n_pv).        #
        #   - Move the dimensions to match the shapes in the method description. # 
        ##########################################################################
        layers = [self.linear_q, self.linear_k, self.linear_v, self.linear_q_points, self.linear_k_points, self.linear_v_points]
        embeddings = [layer(s) for layer in layers]

        shape_adds = [(n_head, c), (n_head, c), (n_head, c), (3, n_head, n_qp), (3, n_head, n_qp), (3, n_head, n_pv)]
        out_shapes = [out.shape[:-1]+shape_add for out, shape_add in zip(embeddings, shape_adds)]
        embeddings = [out.view(out_shape) for out, out_shape in zip(embeddings, out_shapes)]
        for i in range(3):
            embeddings[i] = embeddings[i].movedim(-3, -2)
        for i in range(3, 6):
            embeddings[i] = embeddings[i].movedim(-3, -1).movedim(-4, -2)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return embeddings

    def compute_attention_scores(self, q, k, qp, kp, z, T):
        """
        Computes the attention scores for invariant point attention, 
        according to line 7 from Algorithm 22.

        Args:
            q (torch.tensor): Query embeddings of shape (*, N_head, N_res, c).
            k (torch.tensor): Key embeddings of shape (*, N_head, N_res, c).
            qp (torch.tensor): Query point embeddings of shape (*, N_head, N_query_points, N_res, 3).
            kp (torch.tensor): Key point embeddings of shape (*, N_head, N_query_points, N_res, 3).
            T (torch.tensor): Backbone transforms of shape (*, N_res, 4, 4).

        Returns:
            torch.tensor: Attention scores of shape (*, N_head, N_res, N_res).
        """

        att_scores = None

        ##########################################################################
        # TODO: Implement the method in the following steps:                     #
        #   - Compute wc, wl and gamma.                                          # 
        #   - Reshape gamma (formerly shape (N_head,) so that it's broadcastable #
        #     against the attention scores.                                      #
        #   - Scale q and compute the bias. Move the dimension of the bias so    # 
        #     that it matches the attention scores.                              #
        #   - Compute the qk term. You can use torch.einsum for this.            # 
        #   - Reshape the transforms so that they can be used for batched        # 
        #     matrix multiplication against the query and key points.            #
        #   - Use warp_3d_point to warp the query and key points through T.      # 
        #   - Compute the query points / key points term.                        # 
        #   - Compute the full attention scores.                                 # 
        ##########################################################################

        wc = math.sqrt(2 / (9*self.n_query_points))
        wl = math.sqrt(1/3)
        gamma = self.softplus(self.head_weights).view((-1, 1, 1))

        q = q / math.sqrt(self.c)
        bias = self.linear_b(z).movedim(-1, -3)

        qk_term = torch.einsum('...ic,...jc->...ij', q, k)

        T_bc_qkv = T.view(T.shape[:-3] + (1, 1, -1, 4, 4))
        transformed_qp = warp_3d_point(T_bc_qkv, qp).unsqueeze(-2)
        transformed_kp = warp_3d_point(T_bc_qkv, kp).unsqueeze(-3)
        sq_dist = torch.sum((transformed_qp - transformed_kp)**2, dim=-1)
        qpkp_term = gamma * wc / 2 * torch.sum(sq_dist, dim=-3)

        att_scores = torch.softmax(wl * (qk_term + bias - qpkp_term), dim=-1)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return att_scores

    def compute_outputs(self, att_scores, z, v, vp, T):
        """
        Computes the different output vectors for the IPA attention mechanism:
        The pair output, the standard attention output, and the point attention output,
        as well as the norm of the point attention output.

        Args:
            att_scores (torch.tensor): Attention scores of shape (*, N_head, N_res, N_res).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).
            v (torch.tensor): Value vectors of shape (*, N_head, N_res, c).
            vp (torch.tensor): Value points of shape (*, N_head, N_point_values, N_res, 3).
            T (torch.tensor): Backbone transforms of shape (*, N_res, 4, 4).

        Returns:
            tuple: A tuple consisting of the following outputs:
                - output from the value vectors of shape (*, N_res, N_head*c).
                - output from the value points of shape (*, N_res, N_head*3*N_point_values).
                - norm of the output vectors from the value points of shape (*, N_res, N_head*N_point_values)
                - output from the pair representation of shape (*, N_res, N_head*c_z).
        """

        v_out, vp_out, vp_out_norm, pairwise_out = None, None, None, None

        ##########################################################################
        # TODO: Compute the different attention outputs in the following steps:  #
        #   - Compute the pairwise output, move the dimension so that they       # 
        #     are (**, N_head, c), then flatten the heads and channels.          #
        #   - Compute the value vector output, move the dimensions so that they  #
        #     are (**, N_head, c), then flatten the heads and channels.          #
        #   - Reshape the transforms so that they can be used for batched        # 
        #     matrix multiplication against the value points.                    #
        #   - Warp the value points, compute the point attention values, compute # 
        #     the inverse of the transforms with invert_4x4_transform            #
        #     and warp the value points back through them.                       #
        #   - Transpose the axes of the value points from ...hpic to ...ichp     # 
        #     (the letters mean N_head, point_values, N_res, c). You can use     #
        #     torch.einsum for this.                                             #
        #   - Compute the vector norms of the point values.                      # 
        #   - Flatten the trailing channel, N_head and N_point_value dims for    # 
        #     the value points and their norm.                                   #
        ##########################################################################

        pairwise_out = torch.einsum('...hij,...ijc->...hic', att_scores, z)
        pairwise_out = pairwise_out.movedim(-3, -2).flatten(start_dim=-2)
        v_out = torch.einsum('...hij,...hjc->...hic', att_scores, v)
        v_out = v_out.movedim(-3, -2).flatten(start_dim=-2)

        T_bc_qkv = T.view(T.shape[:-3] + (1, 1, -1, 4, 4))
        vp_out = torch.einsum('...hij,...hpjc->...hpic', att_scores, warp_3d_point(T_bc_qkv, vp))
        T_inv = invert_4x4_transform(T_bc_qkv)
        vp_out = warp_3d_point(T_inv, vp_out)
        vp_out = torch.einsum('...hpic->...ichp', vp_out)

        vp_out_norm = torch.linalg.vector_norm(vp_out, dim=-3, keepdim=True)
        vp_out = vp_out.flatten(start_dim=-3)
        vp_out_norm = vp_out_norm.flatten(start_dim=-3)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return v_out, vp_out, vp_out_norm, pairwise_out
        
        

    def forward(self, s, z, T):
        """
        Implements the forward pass for InvariantPointAttention, as specified in Algorithm 22.

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).
            T (torch.tensor): Backbone transforms of shape (*, N_res, 4, 4).

        Returns:
            torch.tensor: Output tensor of shape (*, N_res, c_s).
        """

        out = None
        
        ##########################################################################
        # TODO: Implement the forward pass by combining all the methods above.   #
        ##########################################################################
        q, k, v, qp, kp, vp = self.prepare_qkv(s)

        att_scores = self.compute_attention_scores(q, k, qp, kp, z, T)
        v_out, vp_out, vp_out_norm, pairwise_out = self.compute_outputs(att_scores, z, v, vp, T)
        

        out = torch.cat((v_out, vp_out, vp_out_norm, pairwise_out), dim=-1)
        out = self.linear_out(out)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out
        
