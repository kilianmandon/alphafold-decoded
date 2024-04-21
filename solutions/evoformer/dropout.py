import torch
from torch import nn

class SharedDropout(nn.Module):
    """
    A module for dropout, that is shared along one dimension,
    i.e. for dropping out whole rows or columns.
    """
    def __init__(self, shared_dim: int, p: float):
        super().__init__()

        ##########################################################################
        # TODO: Store shared_dim for later use and initialize an                 #
        #        nn.Dropout module for the forward pass.                         #
        ##########################################################################

        self.dropout = nn.Dropout(p)
        self.shared_dim = shared_dim

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, x: torch.tensor):
        """
        Forward pass for shared dropout. The dropout mask is broadcasted along
        the shared dimension.

        Args:
            x (torch.tensor): Input tensor of arbitrary shape.

        Returns:
            torch.tensor: Output tensor of the same shape as x.
        """

        out = None

        ##########################################################################
        # TODO: Apply shared dropout by implementing the following steps:        #
        #        * Create a mask of ones with the same shape as x, but with      #
        #           dim 1 at the shared dimension.                               #
        #        * Apply dropout to the mask and multiply it against x to mask   #
        #           out the values. The mask is implicitly broadcasted to the    #
        #           shape of x.                                                  #
        ##########################################################################

        mask_shape = list(x.shape)
        mask_shape[self.shared_dim] = 1
        mask = torch.ones(mask_shape, device=x.device)
        mask = self.dropout(mask)

        out = x * mask

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return out

class DropoutRowwise(SharedDropout):
    def __init__(self, p: float):
        ##########################################################################
        # TODO: Initialize the super class by choosing the right shared          #
        #        dimension for row-wise dropout.                                 #
        ##########################################################################
        super().__init__(shared_dim=-2, p=p)
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

class DropoutColumnwise(SharedDropout):
    def __init__(self, p: float):
        ##########################################################################
        # TODO: Initialize the super class by choosing the right shared          #
        #        dimension for column-wise dropout.                              #
        ##########################################################################
        super().__init__(shared_dim=-3, p=p)
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################