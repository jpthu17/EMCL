import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math


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


class TripletLoss(object):
    def __init__(self, margin=None, mining_type="hard", topk=1):
        self.margin = margin
        if self.margin is not None and self.margin > 0:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()
        self.mining_type = mining_type
        self.type = type
        self.topk = topk

    def __call__(self, mat_dist):
        if self.mining_type == "hard":
            dist_ap, dist_an = hard_example_mining(mat_dist)
        elif self.mining_type == "topk":
            dist_ap, dist_an = topk_example_mining(mat_dist, self.topk)
        elif self.mining_type == "weighted":
            dist_ap, dist_an = batch_weight(mat_dist)
        elif self.mining_type == "topk2":
            dist_ap, dist_an = topk_example_mining2(mat_dist, self.topk)
        elif self.mining_type == "topk3":
            _dist_ap, _dist_an = topk_example_mining(mat_dist, self.topk)
            dist_ap = F.softmax(_dist_ap, dim=1)
            dist_an = F.softmax(_dist_an, dim=1)
        y = dist_ap.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None and self.margin > 0:
            loss = self.ranking_loss(dist_ap, dist_an, y)
        else:
            loss = self.ranking_loss(dist_ap - dist_an, y)
        return loss


def hard_example_mining(dist_mat):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    is_pos = th.eye(N)
    is_neg = th.ones(dist_mat.shape) - th.eye(N)

    is_pos = is_pos.cuda()
    is_neg = is_neg.cuda()

    dist_ap = th.mul(dist_mat, is_pos)
    dist_ap[dist_ap == 0.] = 100000000.
    dist_ap, relative_p_inds = th.min(dist_ap, dim=1, keepdim=True)
    dist_ap2 = th.mul(dist_mat.t(), is_pos)
    dist_ap2[dist_ap2 == 0.] = 100000000.
    dist_ap2, relative_p_inds = th.min(dist_ap2, dim=1, keepdim=True)
    dist_ap = th.cat((dist_ap, dist_ap2), dim=0)

    dist_an = th.mul(dist_mat, is_neg)
    dist_an[dist_an == 0.] = -100000000.
    dist_an, relative_n_inds = th.max(dist_an, dim=1, keepdim=True)
    dist_an2 = th.mul(dist_mat.t(), is_neg)
    dist_an2[dist_an2 == 0.] = -100000000.
    dist_an2, relative_n_inds = th.max(dist_an2, dim=1, keepdim=True)
    dist_an = th.cat((dist_an, dist_an2), dim=0)

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an


def topk_example_mining(dist_mat, topk):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    is_pos = th.eye(N)
    is_neg = th.ones(dist_mat.shape) - th.eye(N)

    is_pos = is_pos.cuda()
    is_neg = is_neg.cuda()

    dist_ap = th.mul(dist_mat, is_pos)
    dist_ap[dist_ap == 0.] = 100000000.
    dist_ap, relative_p_inds = th.topk(dist_ap, k=1, dim=1, largest=False)
    dist_ap2 = th.mul(dist_mat.t(), is_pos)
    dist_ap2[dist_ap2 == 0.] = 100000000.
    dist_ap2, relative_p_inds = th.topk(dist_ap2, k=1, dim=1, largest=False)
    dist_ap = th.cat((dist_ap, dist_ap2), dim=0)
    temp = dist_ap
    for i in range(topk-1):
        dist_ap = th.cat((dist_ap, temp), dim=1)

    dist_an = th.mul(dist_mat, is_neg)
    dist_an[dist_an == 0.] = -100000000.
    dist_an, relative_n_inds = th.topk(dist_an, k=topk, dim=1, largest=True)
    dist_an2 = th.mul(dist_mat.t(), is_neg)
    dist_an2[dist_an2 == 0.] = -100000000.
    dist_an2, relative_n_inds = th.topk(dist_an2, k=topk, dim=1, largest=True)
    dist_an = th.cat((dist_an, dist_an2), dim=0)

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an


def topk_example_mining2(dist_mat, topk):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    is_pos = th.eye(N)
    is_neg = th.ones(dist_mat.shape) - th.eye(N)

    _dist_mat = F.softmax(dist_mat, dim=1) * dist_mat
    _dist_mat_t = F.softmax(dist_mat, dim=0) * dist_mat

    is_pos = is_pos.cuda()
    is_neg = is_neg.cuda()

    dist_ap = th.mul(_dist_mat, is_pos)
    dist_ap[dist_ap == 0.] = 100000000.
    dist_ap, relative_p_inds = th.topk(dist_ap, k=1, dim=1, largest=False)
    dist_ap2 = th.mul(_dist_mat_t.t(), is_pos)
    dist_ap2[dist_ap2 == 0.] = 100000000.
    dist_ap2, relative_p_inds = th.topk(dist_ap2, k=1, dim=1, largest=False)
    dist_ap = th.cat((dist_ap, dist_ap2), dim=0)
    temp = dist_ap
    for i in range(topk-1):
        dist_ap = th.cat((dist_ap, temp), dim=1)

    dist_an = th.mul(_dist_mat, is_neg)
    dist_an[dist_an == 0.] = -100000000.
    dist_an, relative_n_inds = th.topk(dist_an, k=topk, dim=1, largest=True)
    dist_an2 = th.mul(_dist_mat_t.t(), is_neg)
    dist_an2[dist_an2 == 0.] = -100000000.
    dist_an2, relative_n_inds = th.topk(dist_an2, k=topk, dim=1, largest=True)
    dist_an = th.cat((dist_an, dist_an2), dim=0)

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an


def batch_all(dist_mat):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    is_pos = th.eye(N)
    is_neg = th.ones(dist_mat.shape) - th.eye(N)

    is_pos = is_pos.cuda()
    is_neg = is_neg.cuda()

    dist_ap = th.mul(dist_mat, is_pos)
    dist_an = th.mul(dist_mat, is_neg)

    dist_ap_pos = dist_ap[dist_ap > 0]
    dist_an_pos = dist_an[dist_an > 0]

    ap_num = int(dist_ap_pos.size()[0] / N)
    an_num = int(dist_an_pos.size()[0] / N)
    all_num = N * ap_num * an_num

    dist_ap_re = dist_ap_pos.reshape(N, ap_num, 1)
    dist_ap_re = dist_ap_re.expand(N, ap_num, an_num)
    dist_ap_re = dist_ap_re.reshape(all_num)

    dist_an_re = dist_an_pos.reshape(N, an_num, 1)
    dist_an_re = dist_an_re.expand(N, an_num, ap_num)
    dist_an_re = th.transpose(dist_an_re, 1, 2)
    dist_an_re = dist_an_re.reshape(all_num)

    return dist_ap_re, dist_an_re


def batch_weight(dist_mat):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    is_pos = th.eye(N)
    is_neg = th.ones(dist_mat.shape) - th.eye(N)

    is_pos = is_pos.cuda()
    is_neg = is_neg.cuda()

    dist_ap = th.mul(dist_mat, is_pos)
    dist_an = th.mul(dist_mat, is_neg)

    dist_ap_weighted = F.softmax(dist_ap, dim=1)
    dist_an_weighted = F.softmax(-dist_an, dim=1)

    dist_ap_w = dist_ap * dist_ap_weighted
    dist_an_w = dist_an * dist_an_weighted

    dist_ap_pos = dist_ap_w[dist_ap_w > 0]
    dist_an_pos = dist_an_w[dist_an_w > 0]

    ap_num = int(dist_ap_pos.size()[0] / N)
    an_num = int(dist_an_pos.size()[0] / N)
    all_num = N * ap_num * an_num

    dist_ap_re = dist_ap_pos.reshape(N, ap_num, 1)
    dist_ap_re = dist_ap_re.expand(N, ap_num, an_num)
    dist_ap_re = dist_ap_re.reshape(all_num)

    dist_an_re = dist_an_pos.reshape(N, an_num, 1)
    dist_an_re = dist_an_re.expand(N, an_num, ap_num)
    dist_an_re = th.transpose(dist_an_re, 1, 2)
    dist_an_re = dist_an_re.reshape(all_num)

    return dist_ap_re, dist_an_re
