"""
##################################################
##################################################
## TODO                                         ##
##################################################
##################################################
"""

###################
##### IMPORTS #####
###################

# *** General modules imports. *** #

import torch


# *** Own modules imports. *** #

# Import here!


##########################
##### MASKED SOFTMAX #####
##########################

class MaskedSoftmaxNormalisation(torch.autograd.Function):
    """
    Performs the normalisation step of the softmax, but accounting for a masked tensor, whose normaliser
    might have 0 values. This way we avoid the operation yielding 'NaN' values in the forward and backward passes.
    """

    @staticmethod
    def forward(ctx, masked_exp_tensor, normaliser, mask):
        """
        Performs the forward pass of the masked softmax normalisation. Avoids 'NaN' values, caused by masking, in
        the result of the softmax.

        :param ctx              : Pytorch's autograd context variable.
        :param masked_exp_tensor: The (masked) tensor that is to be normalised.
        :param normaliser       : The normalising factors tensor that will be used to normalise masked_exp_tensor.
        :param mask             : The mask corresponding of masked_exp_tensor.

        :return: The 'NaN' free normalised masked_exp_tensor (the result of the entire softmax function).
        """

        # Due to the masking some normaliser values might be 0. Where the normaliser is 0, we replace it by some other
        # value so that there are no 'NaN' values after the normalisation.
        normaliser[normaliser <= 0.] = 1

        # Saves in context the tensors required for the backward pass.
        ctx.save_for_backward(masked_exp_tensor, normaliser, ~mask)

        # Compute the masked softmax tensor. All entries where the corresponding normaliser values have been changed
        # to 1 will themselves be 0, so the result will be correctly masked.
        result = torch.div(masked_exp_tensor, normaliser)

        return result

    @staticmethod
    def backward(ctx, grad_result):
        """
        Performs the backward pass of the masked softmax normalisation. Avoids 'NaN' values, caused by masking, in
        the gradients of the softmax's input tensors.

        :param ctx        : Pytorch's autograd context variable.
        :param grad_result: The computed gradient of the result of the forward pass.

        :return: The 'NaN' free (when caused by masking) gradients of the inputs to the softmax normalisation step.
        """

        # Retrieve the necessary elements from 'context'.
        masked_exp_tensor, normaliser, not_mask = ctx.saved_tensors

        # Compute and mask correctly the gradient w.r.t. the masked_exp_tensor.
        grad_masked_exp_tensor = torch.div(grad_result, normaliser)
        grad_masked_exp_tensor[not_mask] = 0.

        # Compute and mask correctly the gradient w.r.t. the normaliser.
        grad_normaliser = grad_result * (-torch.div(masked_exp_tensor, normaliser ** 2))
        grad_normaliser[not_mask] = 0.

        return grad_masked_exp_tensor, grad_normaliser, None




def masked_softmax(tensor, mask, dim=-1):
    """
    Computes the softmax of a tensor, along a given dimension, 'dim', and taking into account elements that are meant
    to be masked, according to a specific mask. This avoids mask caused 'NaN' values in both the result of the softmax
    and its input tensors' gradients.

    :param tensor: The tensor on which the softmax function is to be applied.
    :param mask  : The mask that is associated with the input tensor.
    :param dim   : the dimension over which the softmax will be computed.

    :return: The correctly softmaxed tensor, accounting for masked elements.
    """

    # Compute the exponential of the tensor.
    exp_tensor = torch.exp(tensor)
    # print("\nexp_tensor:")
    # print(exp_tensor)

    # Mask the exponentiated tensor, so that entries that are meant to be masked do not contribute to the normaliser.
    masked_exp_tensor = exp_tensor * mask.float()
    # print("\nmasked_exp_tensor:")
    # print(masked_exp_tensor)

    # Compute the normaliser.
    normaliser = torch.sum(masked_exp_tensor, dim=dim).unsqueeze(dim)
    # print("\nnormaliser:")
    # print(normaliser)

    # Compute the normalised softmax values, accounting for possible '0' values in the normaliser.
    return MaskedSoftmaxNormalisation.apply(masked_exp_tensor, normaliser, mask)