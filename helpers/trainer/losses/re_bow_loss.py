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
from   torch.distributions.categorical import Categorical
from   torch.distributions.dirichlet import Dirichlet
from   torch.distributions.kl import kl_divergence
from   torch._jit_internal import weak_module, weak_script_method
from   torch.nn import _reduction as _Reduction

# *** Own modules imports. *** #

# Import here!





#################
##### CLASS #####
#################

@weak_script_method
def re_bow_loss(input, target, prior, kl_annealing=1., weight=None, size_average=None, ignore_index=0,
                      reduce=None, reduction='mean', _DEBUG=False):
    # type: (Tensor, Tensor, Distribution, float, Optional[Tensor], Optional[bool], int, Optional[bool], str) -> Tensor
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


    relation_probs = input[0]
    word_probs     = input[1]
    word_labels    = target[1]

    # Creates a masked target tensor, where it is assumed that the labels indices will start at 1.
    mask = (word_labels!=ignore_index).long()
    masked_word_labels = word_labels*mask


    # Here we compute the logarithm of the estimated word probabilities.
    log_word_probs = torch.log(word_probs)

    # Adds a column of zeros to the beginning of the log_word_probs. This will allow to batch the computation while
    # ignoring the 'ignore_idx'.
    masked_log_word_probs = torch.cat((torch.zeros(word_probs.shape[0], 1, device=mask.device), log_word_probs), dim=1)

    # Here we compute the expectation, with respect to the predicted relation probabilities, of the logarithm of the
    # predicted word probabilities.
    masked_expected_log_word_probs = torch.matmul(relation_probs, masked_log_word_probs)

    # Gets the probabilities of the words in each sentence (including the ignore_index ones, but those won't contribute
    # to the result).
    expected_log_words_in_sentences_probs = torch.gather(masked_expected_log_word_probs, 1, masked_word_labels)

    # Computes the probability of each sequence.
    expected_sequence_log_prob = torch.sum(expected_log_words_in_sentences_probs, dim=1)

    # Here we compute the KL divergence of the predicted relation probabilities against the specified prior.
    kls = kl_divergence(Categorical(probs=relation_probs), prior)


    # If we are 'DEBUGGING', i.e. tracking KL and P(X) values, we compute those values here.
    if (_DEBUG):
        # We compute P(x) = sum_{r \in \mathcal{R}} P(r)*P(x|r) = \frac{1}{|\mathcal{R}|} sum_{r \in \mathcal{R}} P(x|r)
        # We actually compute the log of that quantity.

        # First we expand the word probabilities to allow for each training instance to select from each relation.
        ex_word_probs = masked_log_word_probs.unsqueeze(0).expand(masked_word_labels.shape[0], -1, -1)
        ex_word_probs = ex_word_probs.reshape(ex_word_probs.shape[0] * ex_word_probs.shape[1], ex_word_probs.shape[2])

        # Then we expand the labels, for the same reason.
        ex_labels = masked_word_labels.unsqueeze(1).expand(-1, word_probs.shape[0], -1)
        ex_labels = ex_labels.reshape(ex_labels.shape[0] * ex_labels.shape[1], ex_labels.shape[2])

        # We gather the relevant probabilities.
        log_p_x = torch.gather(ex_word_probs, 1, ex_labels)

        # We reshape log(P(X)).
        log_p_x  = log_p_x.reshape(int(log_p_x.shape[0] / word_probs.shape[0]), word_probs.shape[0], log_p_x.shape[1])
        # We compute the log probability of each sentence, per relation.
        summed_log_p_x = log_p_x.sum(dim=-1)
        # We compute the per instance final value of log(P(x)), by employing the log-sum-exp trick.
        log_p_x = torch.logsumexp(summed_log_p_x, dim=-1) - torch.log(torch.tensor(word_probs.shape[0]).float())

        # Compute whatever reduction is necessary.
        if (reduction == 'mean' or reduction == 'sum'):
            batch_log_p_x = torch.sum(log_p_x)
            batch_kls  = torch.sum(kls)
            if (reduction == 'mean'):
                batch_log_p_x /= log_p_x.shape[0]
                batch_kls     /= kls.shape[0]


    # Compute the instance-wise loss.
    instance_wise_loss = expected_sequence_log_prob - kl_annealing*kls

    # Compute whatever reduction is necessary.
    if (reduction == 'mean' or reduction == 'sum'):
        batch_log_prob = torch.sum(instance_wise_loss)
        if (reduction == 'mean'):
            batch_log_prob /= instance_wise_loss.shape[0]
        if (_DEBUG):
            return (-batch_log_prob, batch_kls, -batch_log_p_x)
        else:
            return (-batch_log_prob, )

    if (_DEBUG):
        return (-instance_wise_loss, kls, -log_p_x)
    else:
        return (-instance_wise_loss, )


@weak_module
class RE_BoW_LOSS(_Loss):
    """docstring here"""

    def __init__(self, size_average=None, reduce=None, reduction='sum', ignore_index=0,
                 full_kl_step=None, alpha_prior_val=None, instance_prior=False, _DEBUG=False, **params_dict):
        super(RE_BoW_LOSS, self).__init__(size_average, reduce, reduction)

        self._ignore_index    = ignore_index
        self._current_step    = 1
        self._full_kl_step    = 1 if full_kl_step is None else full_kl_step
        self._alpha_prior_val = alpha_prior_val
        self._instance_prior  = instance_prior

        self._DEBUG = _DEBUG


    @weak_script_method
    def forward(self, input, target):
        # Compute the KL Annealing factor.
        kl_annealing = self.annealing_factor()

        # Get the prior of the relation probabilities, for this batch.
        if (self._alpha_prior_val is None):
            # Pytorch's Categorical distribution normalises input probs, yielding a proper probability distribution.
            probs = torch.ones(input[0].shape[1], device=input[0].device).unsqueeze(0).expand(input[0].shape[0], -1)
        else:
            alpha = torch.tensor(self._alpha_prior_val, device=input[0].device).expand(input[0].shape[1])
            if (self._instance_prior):
                alpha = alpha.unsqueeze(0).expand(input[0].shape[0], -1)
                probs = Dirichlet(alpha).sample()
            else:
                probs = Dirichlet(alpha).sample().unsqueeze(0).expand(input[0].shape[0], -1)


        # Compute the loss.
        loss = re_bow_loss(input, target, prior=Categorical(probs=probs),
                           reduction=self.reduction, ignore_index=self._ignore_index,
                           kl_annealing=kl_annealing, _DEBUG=self._DEBUG)
        return loss


    def annealing_step(self):
        self._current_step += 1

    def annealing_factor(self):
        return 1. if self._current_step >= self._full_kl_step else (self._current_step - 1)/self._full_kl_step




    @property
    def ignore_index(self):
        return self._ignore_index

    @property
    def current_step(self):
        return self._current_step

    @property
    def full_kl_step(self):
        return self._full_kl_step

    @property
    def alpha_prior_val(self):
        return self._alpha_prior_val

    @property
    def instance_prior(self):
        return self._instance_prior

    @property
    def DEBUG(self):
        return self._DEBUG