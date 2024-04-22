from torch import nn
from attention.mha import MultiHeadAttention
from evoformer.dropout import DropoutRowwise
from evoformer.msa_stack import MSARowAttentionWithPairBias, MSATransition, OuterProductMean
from evoformer.pair_stack import PairStack

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

class MSAColumnGlobalAttention(nn.Module):
    """
    Implements Algorithm 19.
    """
    
    def __init__(self, c_m, c_z, c=8, N_head=8):
        """
        Initializes MSAColumnGlobalAttention.

        Args:
            c_m (int): Embedding dimension of the MSA feature.
            c_z (int): Embedding dimension of the pair feature.
            c (int, optional): Embedding dimension for MultiHeadAttention. Defaults to 8.
            N_head (int, optional): Number of heads for MultiHeadAttention. Defaults to 8.
        """

        super().__init__()

        ##########################################################################
        # TODO: Initialize the modules layer_norm_m and global_attention.        #
        #   Set the parameters for MultiHeadAttention correctly to use global    #
        #   attention.                                                           #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, m):
        """
        Implements the forward pass for Algorithm 19.

        Args:
            m (torch.tensor): MSA feature of shape (*, N_seq, N_res, c_m).

        Returns:
            torch.tensor: Output tensor of the same shape as m.
        """

        out = None
        
        ##########################################################################
        # TODO: Implement the forward pass for Algorithm 19.                     #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out


class ExtraMsaBlock(nn.Module):
    """
    Implements one block for Algorithm 18.
    """
    
    def __init__(self, c_m, c_z):
        """
        Initializes ExtraMSABlock.

        Args:
            c_m (int): Embedding dimension of the MSA feature.
            c_z (int): Embedding dimension of the pair feature.
        """
        super().__init__()

        ##########################################################################
        # TODO: Initialize the modules msa_att_row, msa_att_col, msa_transition, #
        #   outer_product_mean, core (the PairStack), and (optionally for        #
        #   inference) dropout_rowwise. Your implementation should be looking    #
        #   very similar to the one for the EvoformerBlock, but using global     #
        #   column attention.                                                    #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, e, z):
        """
        Forward pass for Algorithm 18.

        Args:
            e (torch.tensor): Extra MSA feature of shape (*, N_extra, N_res, c_e).
            z (torch.tensor): Pair feature of shape (*, N_res, N_res, c_z).

        Returns:
            tuple: Tuple consisting of the transformed features e and z.
        """

        ##########################################################################
        # TODO: Implement one block of Algorithm 18. This should look very       #
        #   similar to your implementation in EvoformerBlock.                    #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return e, z
        


class ExtraMsaStack(nn.Module):
    """
    Implements Algorithm 18.
    """
    
    def __init__(self, c_e, c_z, num_blocks):
        """
        Initializes the ExtraMSAStack.

        Args:
            c_e (int): Embedding dimension of the extra MSA feature.
            c_z (int): Embedding dimension of the pair feature.
            num_blocks (int): Number of blocks in the ExtraMSAStack.
        """
        super().__init__()

        ##########################################################################
        # TODO: Initialize self.blocks as a ModuleList of ExtraMSABlocks.        #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, e, z):
        """
        Implements the forward pass for Algorithm 18.

        Args:
            e (torch.tensor): Extra MSA feature of shape (*, N_extra, N_res, c_e).
            z (torch.tensor): Pair feature of shape (*, N_res, N_res, c_z).

        Returns:
            torch.tensor: Output tensor of the same shape as z.
        """

        ##########################################################################
        # TODO: Implement the forward pass for Algorithm 18.                     #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return z