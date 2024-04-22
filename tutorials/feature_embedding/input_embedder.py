import torch
from torch import nn

class InputEmbedder(nn.Module):
    """
    Implements Algorithm 3 and Algorithm 4.
    """

    def __init__(self, c_m, c_z, tf_dim, msa_feat_dim=49, vbins=32):
        """
        Initializes the InputEmbedder.

        Args:
            c_m (int): Embedding dimension of the MSA feature.
            c_z (int): Embedding dimension of the pair feature.
            tf_dim (int): Embedding dimension of target_feat.
            msa_feat_dim (int, optional): Dimension of the initial msa feature. 
                Defaults to 49.
            vbins (int, optional): Determines the bins for relpos as 
                (-vbins, -vbins+1,...,vbins). Defaults to 32.
        """
        super().__init__()
        self.tf_dim = tf_dim
        self.vbins = vbins

        ##########################################################################
        # TODO: Initialize the modules linear_tf_z_i, linear_tf_z_j,             #
        #   linear_tf_m, linear_msa_m and linear_rel_pos (from Algorithm 4).     #
        #   Note the difference between the initial MSA feature                  #
        #   (as created during feature extraction) and the MSA feature m that    #
        #   is used throughout the Evoformer.                                    #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def relpos(self, residue_index):
        """
        Implements Algorithm 4.

        Args:
            residue_index (torch.tensor): Index of the residue in the original amino
                acid sequence. In this context, this is simply [0,... N_res-1].

        Returns:
            tuple: Tuple consisting of the embedded MSA feature m and pair feature z.
        """

        out = None

        ##########################################################################
        # TODO: Implement Algorithm 4. Since the residue index is just a number, #
        #   we can directly use the shifted d_ij as class labels.                #
        #   You can follow these steps:                                          #
        #   * Cast residue_index to long.                                        #
        #   * unsqueeze residue_index accordingly to calculate the outer         #
        #      difference d_ij.                                                  #
        #   * use torch.clamp to clamp d_ij between -self.vbins and self.vbins.  #
        #   * offset the clamped d_ij by self.vbins, so that it is in the range  #
        #      [0, 2*vbins] instead of [-vbins, vbins].                          #
        #   * use nn.functional.one_hot to convert the class labels into         #
        #      one-hot encodings.                                                #
        #   * use the linear module to create the output embedding.              #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out
        

    def forward(self, batch):
        """
        Implements the forward pass for Algorithm 3.

        Args:
            batch (dict): Feature dictionary with the following entries:
                * msa_feat: Initial MSA feature of shape (*, N_seq, N_res, msa_feat_dim).
                * target_feat: Target feature of shape (*, N_res, tf_dim).
                * residue_index: Residue index of shape (*, N_res)

        Returns:
            tuple: Tuple consisting of the MSA feature m and the pair feature z.
        """

        m = None
        z = None

        msa_feat = batch['msa_feat']
        target_feat = batch['target_feat']
        residue_index = batch['residue_index']

        ##########################################################################
        # TODO: Implement the forward pass for Algorithm 4. For the calculation  #
        #   of the outer sum in line 2, the embeddings a and b must be           #
        #   unsqueezed correctly to allow for broadcasting along the N_res dim.  #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return m, z

class ExtraMsaEmbedder(nn.Module):
    """
    Creates the embeddings of extra_msa_feat for the Extra MSA Stack.
    """
    
    def __init__(self, f_e, c_e):
        """
        Initializes the ExtraMSAEmbedder.

        Args:
            f_e (int): Initial dimension of the extra_msa_feat.
            c_e (int): Embedding dimension of the extra_msa_feat.
        """
        super().__init__()
        
        ##########################################################################
        # TODO: Initialize the module self.linear for the extra MSA embedding.   #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################


    def forward(self, batch):
        """
        Passes extra_msa_feat through a linear embedder.

        Args:
            batch (dict): Feature dictionary with the following entries:
                * extra_msa_feat: Extra MSA feature of shape (*, N_extra, N_res, f_e).

        Returns:
            torch.tensor: Output tensor of shape (*, N_extra, N_res, c_e):
        """

        e = batch['extra_msa_feat']
        out = None

        ##########################################################################
        # TODO: Pass extra_msa_feat through the linear layer defined in init.    #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out