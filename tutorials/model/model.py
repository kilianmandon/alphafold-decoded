import torch
from torch import nn

from feature_embedding.input_embedder import InputEmbedder
from feature_embedding.recycling_embedder import RecyclingEmbedder
from feature_embedding.extra_msa_stack import ExtraMsaStack, ExtraMsaEmbedder
from evoformer.evoformer import EvoformerStack
from structure_module.structure_module import StructureModule

class Model(nn.Module):
    """
    Implements the Alphafold model according to Algorithm 2.
    """
    
    def __init__(self, c_m=256, c_z=128, c_e=64, f_e = 25, tf_dim=21, c_s=384, num_blocks_extra_msa=4, num_blocks_evoformer=48):
        """
        Initializes the Alphafold model.

        Args:
            c_m (int, optional): Number of channels for the MSA representation. Defaults to 256.
            c_z (int, optional): Number of channels for the pair representation. Defaults to 128.
            c_e (int, optional): Number of channels for the extra MSA representation. Defaults to 64.
            f_e (int, optional): Number of channels of the extra MSA feature. Defaults to 25.
            tf_dim (int, optional): Number of channels of the target feature. Defaults to 22.
            c_s (int, optional): Number of channels for the single representation. Defaults to 384.
            num_blocks_extra_msa (int, optional): Number of blocks for the extra MSA stack. Defaults to 4.
            num_blocks_evoformer (int, optional): Number of blocks for the Evoformer. Defaults to 48.
        """
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_e = c_e
        self.c_s = c_s

        ##########################################################################
        # TODO: Initialize the modules input_embedder, extra_msa_embedder,       #
        #   recycling_embedder, extra_msa_stack, evoformer and structure_module. #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, batch):
        """
        Forward pass for the Alphafold model.

        Args:
            batch (dict): A dictionary containing the following features:
                * msa_feat:  Tensor of shape (*, N_seq, N_res, msa_feat_dim, N_cycle).
                * extra_msa_feat: Tensor of shape (*, N_extra, N_res, f_e, N_cycle).
                * target_feat: Tensor of shape (*, N_res, tf_dim, N_cycle). One-hot encoding of the target sequence.
                * residue_index: Tensor of shape (*, N_res, N_cycle). The index of each residue, which is [0,...,N_res-1].

        Returns:
            dict: A dictionary with the following entries:
                * final_positions: Heavy-atom positions in Angstrom of shape (*, N_res, 37, 3, N_cycle).
                * position_mask: Boolean tensor of shape (*, N_res, 37, N_cycle), masking atoms that
                    aren't present in the amino acids.
                * angles: Torsion angles of shape (*, N_layers, N_res, n_torsion_angles, 2, N_cycle) for 
                    every iteration of the Structure Module in every cycle.
                * frames: Backbone frames of shape (*, N_layers, N_res, 4, 4, N_cycle) for every iteration
                    of the Structure Module in every cycle.
        """
        N_cycle = batch['msa_feat'].shape[-1]
        N_seq, N_res = batch['msa_feat'].shape[-4:-2]
        batch_shape = batch['msa_feat'].shape[:-4]
        device = batch['msa_feat'].device
        dtype = batch['msa_feat'].dtype

        c_m = self.c_m
        c_z = self.c_z

        outputs = {}

        
        ##########################################################################
        # TODO: Implement the forward pass of Algorithm 2:                       #
        #   - Create the initial prev_m, prev_z, and prev_pseudo_beta_x          #
        #       as zeros of shape (*, N_seq, N_res, c_m), (*, N_res, N_res, c_z) #
        #       and (*, N_res, 3).                                               #
        #   - Loop for N_cycle times. At the start of the loop, you can print    #
        #       'Starting iteration {i}'. Select current_batch from batch by     #
        #       selecting the i-th element for each tensor in batch.             #
        #       Implement the main loop according to Algorithm 2. The labels F   #
        #       for the Structure Module can be computed from target_feat via    #
        #       `argmax`, as target_feat is one-hot encoded.                     #
        #       At the end of each loop, append the outputs from the structure   #
        #       module to outputs (if they are already present) or create a new  #
        #       list for them in outputs (in the first iteration).               #
        #   - Stack the outputs along the last dimension.                        #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return outputs