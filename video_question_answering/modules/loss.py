import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossEn(nn.Module):
    def __init__(self, config=None):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = th.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss
    
    
class InfoNceLoss(nn.Module):
  """Implementation of the noise-constrastive estimation loss."""

  def __init__(self):
    super().__init__()
    self.loss = th.nn.CrossEntropyLoss(reduction='mean')

  def forward(self, x):
    n = x.size()[0]
    target = th.arange(n)
    if x.is_cuda:
      target = target.cuda()

    return self.loss(x, target) + self.loss(th.transpose(x, 0, 1), target)


class MaxMarginRankingLoss(nn.Module):
    """Implementation of the Max-margin ranking loss."""
    def __init__(self, margin=1, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = th.nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x):
        n = x.size()[0]

        x1 = th.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = th.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = th.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
          # remove the elements from the diagonal
          keep = th.ones(x.shape) - th.eye(x.shape[0])
          keep1 = keep.view(-1, 1)
          keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
          keep_idx = th.nonzero(th.cat((keep1, keep2), 0).flatten()).flatten()
          if x1.is_cuda:
            keep_idx = keep_idx.cuda()
          x1_ = th.index_select(x1, dim=0, index=keep_idx)
          x2_ = th.index_select(x2, dim=0, index=keep_idx)
          max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()