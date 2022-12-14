# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

import logging
import torch
from torch import nn
import torch.nn.functional as F
import math

logger = logging.getLogger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


##################################
###### LOSS FUNCTION #############
##################################
class CrossEn(nn.Module):
    def __init__(self, config=None):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss
    

class Emcl(object):
    def __init__(self, k=32, stage_num=9, momentum=0.9, lamd=1, beta=3):
        self.k = k
        self.lamd = lamd
        self.stage_num = stage_num
        self.beta = beta
        self.momentum = momentum
        self.mu = torch.Tensor(1, self.k)
        self.mu.normal_(0, math.sqrt(2. / self.k))
        self.mu = self.mu / (1e-6 + self.mu.norm(dim=0, keepdim=True))

    def __call__(self, embds, if_train=True):
        b, n = embds.size()
        mu = self.mu.repeat(b, 1).cuda(embds.device)
        _embds = embds
        with torch.no_grad():
            for i in range(self.stage_num):
                _embds_t = _embds.permute(1, 0)  # n * b
                z = torch.mm(_embds_t, mu)  # n * k
                z = z / self.lamd
                z = F.softmax(z, dim=1)
                z = z / (1e-6 + z.sum(dim=0, keepdim=True))
                mu = torch.mm(_embds, z)  # b * k
                mu = mu / (1e-6 + mu.norm(dim=0, keepdim=True))
        z_t = z.permute(1, 0)  # k * n
        _embds = torch.mm(mu, z_t)  # b * n

        if if_train:
            mu = mu.cpu()
            self.mu = self.momentum * self.mu + (1 - self.momentum) * mu.mean(dim=0, keepdim=True)
        return self.beta * _embds + embds


def _batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1,
                                                       descending=False)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1,
                                                       descending=True)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if (indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


class SoftTripletLoss(nn.Module):
    def __init__(self, config=None):
        super(SoftTripletLoss, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        N = sim_matrix0.size(0)
        mat_sim = torch.eye(N).float().to(sim_matrix0.device)
        dist_ap, dist_an, ap_idx, an_idx = _batch_hard(sim_matrix0, mat_sim, indice=True)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = F.log_softmax(triple_dist, dim=1)
        dist_ap_ref = torch.gather(sim_matrix1, 1, ap_idx.view(N, 1).expand(N, N))[:, 0]
        dist_an_ref = torch.gather(sim_matrix1, 1, an_idx.view(N, 1).expand(N, N))[:, 0]
        triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
        triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()
        loss = (- triple_dist_ref * triple_dist).mean(0).sum()
        return loss


class MSE(nn.Module):
    def __init__(self, config=None):
        super(MSE, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        logpt = (sim_matrix0 - sim_matrix1)
        loss = logpt * logpt
        return loss.mean()


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def uniformity_loss(x, y):
    input = torch.cat((x, y), dim=0)
    m = input.size(0)
    dist = euclidean_dist(input, input)
    return torch.logsumexp(torch.logsumexp(dist, dim=-1), dim=-1) - torch.log(torch.tensor(m * m - m))


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        if args.world_size == 1:
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return tensor
        else:
            output = [torch.empty_like(tensor) for _ in range(args.world_size)]
            torch.distributed.all_gather(output, tensor)
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
        )


class AllGather2(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    # https://github.com/PyTorchLightning/lightning-bolts/blob/8d3fbf7782e3d3937ab8a1775a7092d7567f2933/pl_bolts/models/self_supervised/simclr/simclr_module.py#L20
    @staticmethod
    def forward(ctx, tensor, args):
        if args.world_size == 1:
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return tensor
        else:
            output = [torch.empty_like(tensor) for _ in range(args.world_size)]
            torch.distributed.all_gather(output, tensor)
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)
        return (grad_input[ctx.rank * ctx.batch_size:(ctx.rank + 1) * ctx.batch_size], None)