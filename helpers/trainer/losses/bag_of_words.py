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
from   torch.nn.modules.loss import _Loss
from   torch._jit_internal import weak_module, weak_script_method
from   torch.nn import _reduction as _Reduction

# *** Own modules imports. *** #

# Import here!





#################
##### CLASS #####
#################

@weak_script_method
def bag_of_words_log_loss(input, target, weight=None, size_average=None, ignore_index=0,
                      reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor
    r"""This criterion combines `log` and word-wise (at sentence level) likelihood in a single
    function.

    See :class:`~torch.nn.CrossEntropyLoss` for details.

    # TODO: This needs to be re-written.
    Args:
        input (Tensor) : :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K > 1`
            in the case of K-dimensional loss.
        target (Tensor) : :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
            K-dimensional loss.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

    Examples::

        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randint(5, (3,), dtype=torch.int64)
        >>> loss = bag_of_words_log_loss(input, target)
        >>> loss.backward()
    """

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    word_probs = input[0]
    word_labels = target[0]

    # Creates a masked target tensor, where it is assumed that the labels indices will start at 1.
    mask = (word_labels != ignore_index).long()
    masked_word_labels = word_labels * mask

    # Adds a column of ones to the beginning of the input. This will allow to batch the computation ignoring the
    # 'ignore_idx'.
    input_accounting_for_masking = torch.cat((torch.ones(word_probs.shape[0], 1, device=mask.device), word_probs),
                                             dim=1)

    # Gets the probabilities of the words in each sentence (including the ignore_index ones, but those won't contribute
    # to the result).
    word_log_probs = torch.log(torch.gather(input_accounting_for_masking, 1, masked_word_labels))

    # Computes the probability of each sequence.
    sequence_log_prob = torch.sum(word_log_probs, dim=1)

    # Compute whatever reduction is necessary.
    if (reduction == 'mean' or reduction == 'sum'):
        batch_log_prob = torch.sum(sequence_log_prob)
        if (reduction == 'mean'):
            batch_log_prob /= sequence_log_prob.shape[0]
        return -batch_log_prob

    return -sequence_log_prob


@weak_module
class BagOfWordsLogLoss(_Loss):
    """docstring here"""

    def __init__(self, size_average=None, reduce=None, reduction='sum', ignore_index=0, **params_dict):
        super(BagOfWordsLogLoss, self).__init__(size_average, reduce, reduction)

        self._ignore_index = ignore_index

    @weak_script_method
    def forward(self, input, target):
        return (bag_of_words_log_loss(input, target, reduction=self.reduction, ignore_index=self._ignore_index), )


    def annealing_step(self):
        pass