import torch
from torch import nn
from attention.mha import MultiHeadAttention

class MSARowAttentionWithPairBias(nn.Module):
    """
    Implements Algorithm 7.
    """
    def __init__(self, c_m, c_z, c=32, N_head=8):
        """
        Initializes MSARowAttentionWithPairBias.

        Args:
            c_m (int): Embedding dimension of the msa feature.
            c_z (int): Embedding dimension of the pair feature.
            c (int, optional): Embedding dimension for multi-head attention. Defaults to 32.
            N_head (int, optional): Number of heads for multi-head attention. Defaults to 8.
        """
        super().__init__()

        ##########################################################################
        # TODO: Initialize the modules layer_norm_m, layer_norm_z, linear_z      #
        #        and mha for Algorithm 7.                                        #
        #        linear_z is used to embed the pair bias and needs to create     #
        #        one value per head, therefore c_out is N_head.                  #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, m, z):
        """
        Implements the forward pass according to Algorithm 7.

        Args:
            m (torch.tensor): MSA feature of shape (*, N_seq, N_res, c_m).
            z (torch.tensor): Pair feature of shape (*, N_res, N_res, c_z).

        Returns:
            torch.tensor: Output tensor of the same shape as m.
        """
        out = None
        ##########################################################################
        # TODO: Implement the forward pass for Algorithm 7. Note that the bias   #
        #        is embedded as (*, z, z, N_head) but needs to have shape        #
        #       (*, N_head, z, z) for MultiHeadAttention.                        #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out

class MSAColumnAttention(nn.Module):
    """
    Implements Algorithm 8.
    """
    def __init__(self, c_m, c=32, N_head=8):
        """
        Initializes MSAColumnAttention.

        Args:
            c_m (int): Embedding dimension of the msa feature.
            c (int, optional): Embedding dimension for multi-head attention. Defaults to 32.
            N_head (int, optional): Number of heads for multi-head attention. Defaults to 8.
        """
        super().__init__()

        ##########################################################################
        # TODO: Initialize the modules layer_norm_m and mha for Algorithm 8.     #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, m):
        """
        Implements the forward pass according to algorithm Algorithm 8.

        Args:
            m (torch.tensor): MSA feature of shape (N_seq, N_res, c_m).

        Returns:
            torch.tensor: Output tensor of the same shape as m.
        """


        ##########################################################################
        # TODO: Implement the forward pass for Algorithm 8.                      #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################


class MSATransition(nn.Module):
    """
    Implements Algorithm 9.
    """
    def __init__(self, c_m, n=4):
        """
        Initializes MSATransition.

        Args:
            c_m (int): Embedding dimension of the MSA feature.
            n (int, optional): Factor for the number of channels in the intermediate dimension. 
             Defaults to 4.
        """
        super().__init__()

        ##########################################################################
        # TODO: Initialize the modules layer_norm, linear_1, relu and linear_2   #
        #   for Algorithm 9.
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, m):
        """
        Implements the forward pass for Algorithm 9.

        Args:
            m (torch.tensor): MSA feat of shape (*, N_seq, N_seq, c_m).

        Returns:
            torch.tensor: Output tensor of the same shape as m.
        """
        out = None

        ##########################################################################
        # TODO: Implement the forward pass for Algorithm 9.                      #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out

class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.
    """
    def __init__(self, c_m, c_z, c=32):
        """
        Initializes OuterProductMean.

        Args:
            c_m (int): Embedding dimension of the MSA feature.
            c_z (int): Embedding dimension of the pair feature. 
            c (int, optional): Embedding dimension of a and b from Algorithm 10. 
                Defaults to 32.
        """
        super().__init__()
        
        ##########################################################################
        # TODO: Initialize the modules layer_norm, linear_1, linear_2 and        #
        #   linear_out for Algorithm 10. linear_1 creates the embdding for a,    #
        #   while linear_2 creates the embedding for b.                          #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, m):
        """
        Forward pass for Algorithm 10.

        Args:
            m (torch.tensor): MSA feat of shape (*, N_seq, N_res, c_m).

        Returns:
            torch.tensor: Output tensor of shape (*, N_res, N_res, c_z).
        """
        N_seq = m.shape[-3]
        z = None

        ##########################################################################
        # TODO: Implement the forward pass for Algorithm 10. In contrast to the  #
        #   supplement, the AlphaFold implementation doesn't compute the mean    #
        #   in line 3, but the sum overr all sequences. Instead, the output z    #
        #   is divided by N_seq after line 4. This changes the results, as the   #
        #   biases of the affine linear output layer are affected by the         #
        #   scaling as well. We follow the implementation.                       #
        #                                                                        #
        #   After summation over the sequences, the intermediate o has shape     #
        #   (*, N_res, N_res, c, c) before flattening to (*, N_res, N_res, c*c). #
        #                                                                        #
        #   The outer product and the summation over the sequences in line 4     #
        #   can be computed efficiently using torch.einsum.                      #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return z