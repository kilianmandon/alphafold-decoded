import torch
import math
from torch import nn

class MultiHeadAttention(nn.Module):
    """
    A MultiHeadAttention module with optional bias and optional gating.
    """

    def __init__(self, c_in, c, N_head, attn_dim, gated=False, is_global=False, use_bias_for_embeddings=False):
        """
        Initializes the module. MultiHeadAttention theoretically consists of 
        N_head separate linear layers for the query, key and value embeddings.
        However, the embeddings can be computed jointly and split afterwards,
        so we only need one query, key and value layer with larger c_out.

        Args:
            c_in (int): Input dimension for the embeddings.
            c (int): Embedding dimension for each individual head.
            N_head (int): Number of heads.
            attn_dim (int): The dimension in the input tensor along which
                the attention mechanism is performed.
            gated (bool, optional): If True, an additional sigmoid-activated 
                linear layer will be multiplicated against the weighted 
                value vectors before feeding them through the output layer. 
                Defaults to False.
            is_global (bool, optional): If True, global calculation will be performed.
                For global calculation, key and value embeddings will only use one head,
                and the q query vectors will be averaged to one query vector.
                Defaults to False.
            use_bias_for_embeddings (bool, optional): If True, query, 
                key, and value embeddings will use bias, otherwise not. 
                Defaults to False.
        """
        super().__init__()

        self.c_in = c_in
        self.c = c
        self.N_head = N_head
        self.gated = gated
        self.attn_dim = attn_dim
        self.is_global = is_global

        ##########################################################################
        # TODO: Initialize the query, key, value and output layers.              #
        #   Whether or not query, key, and value layers use bias is determined   #
        #   by `use_bias` (False for AlphaFold). The output layer should always  #
        #   use a bias. If gated is true, initialize another linear with bias.   #
        #   For compatibility use the names linear_q, linear_k, linear_v,        #
        #   linear_o and linear_g.                                               #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def prepare_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Splits the embeddings into individual heads and transforms the input
        shapes of form (*, q/k/v, *, N_head*c) into the shape 
        (*, N_head, q/k/v, c). The position of the q/k/v dimension 
        in the original tensors is given by attn_dim.

        Args:
            q (torch.Tensor): Query embedding of shape (*, q, *, N_head*c).
            k (torch.Tensor): Key embedding of shape (*, k, *, N_head*c).
            v (torch.Tensor): Value embedding of shape (*, v, *, N_head*c).

        Returns:
            tuple: The rearranged embeddings q, k, and v of 
                shape (*, N_head, q/k/v, c) respectively.
        """

        ##########################################################################
        # TODO: Rearrange the tensors with the following changes:                #
        #   - (*, q/k/v, *, N_head*c) -> (*, q/k/v, N_head*c) with movedim       # 
        #   - (*, q/k/v, N_head*c) -> (*, q/k/v, N_head, c)                      #
        #   - (*, q/k/v, N_head, c) -> (*, N_head, q/k/v, c)                     #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return q, k, v

    def prepare_qkv_global(self, q, k, v):
        """
        Prepares the query, key and value embeddings with the following 
        differences to the non-global version:
            - key and value embeddings use only one head.
            - the query vectors are contracted into one, average query vector.
        

        Args:
            q (torch.tensor): Query embeddings of shape (*, q, *, N_head*c).
            k (torch.tensor): Key embeddings of shape (*, k, *, c).
            v (torch.tensor): Value embeddings of shape (*, v, *, c).

        Returns:
            tuple: The rearranged embeddings q, k, and v of
                shape (*, N_head, 1, c) for q and shape (*, 1, k, c) for k and v. 
        """

        ##########################################################################
        # TODO: Rearrange the tensors to match the output dimensions. Use        #
        #   torch.mean for the contraction of q at the end of this function.     #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return q, k, v

    def forward(self, x, bias=None, attention_mask=None):
        """
        Forward pass through the MultiHeadAttention module.

        Args:
            x (torch.tensor): Input tensor of shape (*, q/k/v, *, c_in).
            bias (torch.tensor, optional): Optional bias tensor of shape
                (*, N_head, q, k) that will be added to the attention weights. 
                Defaults to None.
            attention_mask (torch.tensor, optional): Optional attention mask
                of shape (*, k). If set, the keys with value 0 in the mask will
                not be attended to.

        Returns:
            torch.tensor: Output tensor of shape (*, q/k/v, *, c_in)
        """

        out = None

        ##########################################################################
        # TODO: Implement the forward pass consisting of the following steps:    #
        #   - Create query, key and value embeddings.                            #
        #   - Rearrange the embeddings with prepare_qkv.                         #
        #   - Scale the queries by 1/sqrt(c).                                    #
        #   - Calculate the attention weights of shape (*, N_head, q, k)         #
        #       from q and k. You can use torch.einsum for this.                 #
        #   - If a bias was given:                                               #
        #       - extract the bias batch shape by omitting the last 3 dims       #
        #         from bias.                                                     #
        #       - construct a broadcastable bias shape, by concatenating         #
        #           bias_batch_shape, (1,) * n, and the last three dims of bias. #
        #           Choose n such that the broadcastable shape has as many dims  #
        #           as the attention scores.                                     #
        #       - add the bias to the attention scores.                          #
        #   - If an attention mask was given (not needed for AlphaFold):         #
        #       - unsqueeze the mask to make it broadcastable against the        #
        #         attention scores of shape (*, N_head, q, k).                   #
        #       - create a tensor `offset`` of the same shape as the mask with   #
        #         the value -1e8 where the mask is 0 and zero elsewhere.         #
        #       - add the offset to the raw attention scores.                    #
        #   - Use softmax to convert the attention scores into a                 #
        #       probability distribution.                                        #
        #   - Weight the value vectors by the attention weights and sum          #
        #       them up along the key dimension. You can use torch.einsum        #
        #       to do this in one line. The result should be                     #
        #       of shape (*, N_head, q, c).                                      #
        #   - Rearrange the intermediate output in the following way:            #
        #       * (*, N_head, q, c) -> (*, q, N_head, c)                         #
        #       * (*, q, N_head, c) -> (*, q, N_head * c)                        #
        #       * (*, q, N_head * c) -> (*, q, *, N_head * c)                    #
        #       The order of these transformations is crucial, as moving q       #
        #       to attn_dim before flattening the heads will result in an        #
        #       incorrect positioning if attn_dim uses negative indexing.        #
        #   - if gated, calculate the gating with linear_g and sigmoid and       #
        #       multiply it against the output.                                  #
        #   - apply linear_o to calculate the final output.                        #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out
