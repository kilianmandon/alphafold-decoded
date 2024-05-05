import torch
from torch import nn

from structure_module.ipa import InvariantPointAttention
from geometry.geometry import compute_all_atom_coordinates, assemble_4x4_transform, quat_to_3x3_rotation
from geometry.geometry import residue_constants
    

class StructureModuleTransition(nn.Module):
    """
    Implements the transition in the Structure Module (lines 8 and 9 from Algorithm 20).
    """
    
    def __init__(self, c_s):
        """
        Initializes StructureModuleTransition.

        Args:
            c_s (int): Number of channels for the single representation.
        """
        super().__init__()
        self.c_s = c_s

        ##########################################################################
        # TODO: Initialize the layers linear_1, linear_2, linear_3 and           #
        #   and layer_norm, as well as a ReLU module. Optionally, you can also   #
        #   initialize dropout, but this is not relevant for evaluation.         #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, s):
        """
        Implements the forward pass for the transition as
        s -> linear -> relu -> linear -> relu -> linear + s -> layer_norm

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).

        Returns:
            torch.tensor: Output single representation of shape (*, N_res, c_s).
        """

        ##########################################################################
        # TODO: Implement the forward pass.                                      #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return s

class BackboneUpdate(nn.Module):
    """
    Implements the backbone update, according to Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Initializes BackboneUpdate.

        Args:
            c_s (int): Number of channels for the single representation.
        """
        super().__init__()

        ##########################################################################
        # TODO: Initialize the module 'linear' for use in BackboneUpdate.        #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, s):
        """
        Computes the forward pass for Algorithm 23.

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).

        Returns:
            torch.tensor: Backbone transforms of shape (*, N_res, 4, 4).
        """

        T = None
        
        ##########################################################################
        # TODO: Implement the forward pass with the following steps:             #
        #   - Pass s through the linear layer.                                   #
        #   - Construct a (*, N_res, 4) quaternion from 1s and the first three   #
        #       elements of the output from the linear layer.                    #
        #   - Select the last three elements as the translation.                 #
        #   - Normalize the quaternion to get a rotational quaternion.           #
        #   - Construct the rotation matrix with the explicit formula from       #
        #       Algorithm 22, or by calling quat_to_3x3_rotation.                #
        #   - Assemble the 4x4 transforms from R and t.                          #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return T

class AngleResNetLayer(nn.Module):
    """
    Implements a layer of the AngleResNet for the Structure Module, 
    which is line 12 or line 13 from Algorithm 20.
    """
    
    def __init__(self, c):
        """
        Initializes AngleResNetLayer.

        Args:
            c (int): Embedding dimension for the AngleResNet.
        """
        super().__init__()

        ##########################################################################
        # TODO: Initialize the modules linear_1, linear_2 and relu according     #
        #   Algorithm 20.                                                        #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, a):
        """
        Computes the forward pass as 
        a -> relu -> linear -> relu -> linear + a

        Args:
            a (torch.tensor): Embedding of shape (*, N_res, c).

        Returns:
            torch.tensor: Output embedding of shape (*, N_res, c).
        """

        ##########################################################################
        # TODO: Implement the forward pass for AngleResNetLayer.                 #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
         
        return a


class AngleResNet(nn.Module):
    """
    Implements the AngleResNet from the Structure Module (lines 11-14 in Algorithm 20).
    """
    
    def __init__(self, c_s, c, n_torsion_angles=7):
        """
        Initializes the AngleResNet.

        Args:
            c_s (int): Number of channels for the single representation.
            c (int): Embedding dimension of the AngleResNet.
            n_torsion_angles (int, optional): Number of torsion angles to be predicted. Defaults to 7.
        """
        super().__init__()
        self.n_torsion_angles = n_torsion_angles

        ##########################################################################
        # TODO: Initialize the modules linear_in, linear_initial, layers,        #
        #   linear_out, and a ReLU module. layers is an nn.ModuleList of two     #
        #   AngleResNet layers. Remember that the torsion angles are predicted   #
        #   as unnormalized (cos(phi), sin(phi)) pairs, meaning you need         #
        #   two outputs per torsion angle.                                       #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, s, s_initial):
        """
        Implements the forward pass through the AngleResNet according to Algorithm 20.
        In contrast to the supplement, s and s_initial are passed through a ReLU
        function before the first linear layers.

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).
            s_initial (torch.tensor): Initial single representation of shape (*, N_res, c_s).

        Returns:
            torch.tensor: Torsion angles of shape (*, N_res, 2*n_torsion_angles).
        """
        
        alpha = None

        ##########################################################################
        # TODO: Implement the forward pass for the module.                       #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        
        return alpha



class StructureModule(nn.Module):
    """
    Implements the Structure Module according to Algorithm 20.
    """
    
    def __init__(self, c_s, c_z, n_layer=8, c=128):
        """
        Initializes the Structure Module.

        Args:
            c_s (int): Number of channels for the single representation.
            c_z (int): Number of channels for the pair representation.
            n_layer (int, optional): Number of layers for the whole module. Defaults to 8.
            c (int, optional): Embedding dimension for the AngleResNet. Defaults to 128.
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.n_layer = n_layer


        ##########################################################################
        # TODO: Initialize the modules layer_norm_s, layer_norm_z, linear_in,    #
        #   layer_norm_ipa, ipa, transition, bb_update, and angle_resnet.        #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def process_outputs(self, T, alpha, F):
        """
        Computes the final atom positions, the atom mask and the pseudo beta positions
        from the backbone transforms, torsion angles and amino acid labels.

        Args:
            T (torch.tensor): Backbone transforms of shape (*, N_res, 4, 4). Units 
                are measured in nanometers (this affects only the translation). 
            alpha (torch.tensor): Torsion angles of shape (*, N_res, n_torsion_angles, 2).
            F (torch.tensor): Labels for the amino acids of shape (*, N_res). Labels are encoded
                as 0 -> Alanine, 1 -> Arginine, ..., 19 -> Valine. 

        Returns:
            tuple: A tuple consisting of the following values:
                - final_positions: Tensor of shape (*, N_res, 37, 3). The 3D positions of 
                    all atoms, measured in Angstrom.
                - position_mask: Boolean tensor of shape (*, N_res, 37). Masks the side-chain 
                    atoms that aren't present in the amino acids.
                - pseudo_beta_positions: Tensor of shape (*, N_res, 3). 3D positions in Angstrom
                    of C-beta (for all amino acids except glycine) or C-alpha (for glycine).
        """

        final_positions, position_mask, pseudo_beta_positions = None, None, None

        ##########################################################################
        # TODO: Implement the processing of outputs with the following steps:    #
        #   - Clone T and scale the translation vector by 10, to switch from     #
        #       nanometers to Angstrom.                                          #
        #   - Use compute_all_atom_coordinates to compute the final positions    #
        #       and the position mask.                                           #
        #   - Clone T and scale the translation vector by 10, to switch from     #
        #   - Use residue_constants.atom_types to compute the indices for the    #
        #       c_alpha atoms and the c_beta atoms.                              #
        #   - Use residue_constants.restypes to compute the index for glycine.   #
        #   - Use the atom indices to select the alpha and beta positions for    #
        #       each residue from final_atom_positions.                          #
        #   - Use boolean indexing to overwrite the beta positions with the      #
        #       alpha positions where F is equal to the index of glycine.        #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return final_positions, position_mask, pseudo_beta_positions
        

    def forward(self, s, z, F):
        """
        Forward pass for the Structure Module.

        Args:
            s (torch.tensor): Single representation of shape (*, N_res, c_s).
            z (torch.tensor): Pair representation of shape (*, N_res, c_z).
            F (torch.tensor): Labels for the amino acids of shape (*, N_res).

        Returns:
            dict: Output dictionary with the following entries:
                - angles: Torsion angles of shape (*, N_layers, N_res, n_torsion_angles, 2). 
                - frames: Backbone frames of shape (*, N_layers, N_res, 4, 4).  
                - final_positions: Heavy atom positions in Angstrom of shape (*, N_res, 37, 3).
                - position_mask: Boolean tensor of shape (*, N_res, 37), masking atoms that are
                    not present in the amino acids.
                - pseudo_beta_positions: C-beta-positions (non-glycine) or C-alpha-positions
                    (glycine) for each residue, of shape (*, N_res, 3).
        """
        N_res = z.shape[-2]
        batch_dim = s.shape[:-2]
        outputs = {'angles': [], 'frames': []}
        device = s.device
        dtype = s.dtype

        ##########################################################################
        # TODO: Implement the forward pass with the following steps:             #
        #   - Implement lines 1-4. Initialize T as the 4x4 identity matrices,    #
        #       broadcasted to shape (*, N_res, 4, 4).                           #
        #   - Implement the for-loop in lines 5-22. You can skip the loss        #
        #       computation and the gradient stop, as this is only relevant      #
        #       for training. The compisition of the 4x4 transforms is simply    #
        #       their batched matrix product (you can use torch.einsum or @).    #
        #       Append the angles and frames to the output dict.                 #
        #   - Implement the for-loop in lines 5-22. You can skip the loss        #
        #   - Stack the angles and frames along the correct axes.                #
        #   - Call process_outputs and add the results to the output dict.       #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return outputs
            

