import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch import Tensor
from typing import Callable, Optional


class SampleCrossEntropyLoss(CrossEntropyLoss):
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)

        nll_loss.masked_fill_(padding_mask, 0.0)
        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.shape[1] - padding_mask.long().sum(1)
        # import pdb
        # pdb.set_trace()
        nll_loss = nll_loss.sum(1) / num_active_elements
        if self.label_smoothing == 0:
            return nll_loss

        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        smoothed_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss = smoothed_loss.sum(1) / (num_active_elements * log_probs.shape[-1])
        return (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smoothed_loss