from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from .file_utils import cached_path
from .until_config import PretrainedConfig
from .until_module import PreTrainedModel, LayerNorm, ACT2FN
from collections import OrderedDict

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'


class CrossConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `CrossModel`.
    """
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    config_name = CONFIG_NAME
    weights_name = WEIGHTS_NAME
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs CrossConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `CrossModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `CrossModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        # attn_mask_ = attn_mask.repeat(self.n_head, 1, 1)
        if attn_mask is not None:
            attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        x, attn_mask = para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))[0]

class CrossEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(CrossEmbeddings, self).__init__()

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, concat_embeddings, concat_type=None):

        batch_size, seq_length = concat_embeddings.size(0), concat_embeddings.size(1)
        # if concat_type is None:
        #     concat_type = torch.zeros(batch_size, concat_type).to(concat_embeddings.device)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=concat_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(concat_embeddings.size(0), -1)

        # token_type_embeddings = self.token_type_embeddings(concat_type)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = concat_embeddings + position_embeddings # + token_type_embeddings
        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class CrossPooler(nn.Module):
    def __init__(self, config):
        super(CrossPooler, self).__init__()
        self.ln_pool = LayerNorm(config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = QuickGELU()

    def forward(self, hidden_states, hidden_mask):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = self.ln_pool(hidden_states)
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CrossModel(PreTrainedModel):

    def initialize_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def __init__(self, config):
        super(CrossModel, self).__init__(config)

        self.embeddings = CrossEmbeddings(config)

        transformer_width = config.hidden_size
        transformer_layers = config.num_hidden_layers
        transformer_heads = config.num_attention_heads
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads,)
        self.pooler = CrossPooler(config)
        self.apply(self.init_weights)

    def build_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0
        extended_attention_mask = extended_attention_mask.expand(-1, attention_mask.size(1), -1)
        return extended_attention_mask

    def forward(self, concat_input, concat_type=None, attention_mask=None, output_all_encoded_layers=True):

        if attention_mask is None:
            attention_mask = torch.ones(concat_input.size(0), concat_input.size(1))
        if concat_type is None:
            concat_type = torch.zeros_like(attention_mask)

        extended_attention_mask = self.build_attention_mask(attention_mask)

        embedding_output = self.embeddings(concat_input, concat_type)
        embedding_output = embedding_output.permute(1, 0, 2)  # NLD -> LND
        embedding_output = self.transformer(embedding_output, extended_attention_mask)
        embedding_output = embedding_output.permute(1, 0, 2)  # LND -> NLD

        pooled_output = self.pooler(embedding_output, hidden_mask=attention_mask)

        return embedding_output, pooled_output


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
    is_pos = torch.eye(N)
    is_neg = torch.ones(dist_mat.shape) - torch.eye(N)

    is_pos = is_pos.cuda()
    is_neg = is_neg.cuda()

    dist_ap = torch.mul(dist_mat, is_pos)
    dist_ap[dist_ap == 0.] = 100000000.
    dist_ap, relative_p_inds = torch.min(dist_ap, dim=1, keepdim=True)
    dist_ap2 = torch.mul(dist_mat.t(), is_pos)
    dist_ap2[dist_ap2 == 0.] = 100000000.
    dist_ap2, relative_p_inds = torch.min(dist_ap2, dim=1, keepdim=True)
    dist_ap = torch.cat((dist_ap, dist_ap2), dim=0)

    dist_an = torch.mul(dist_mat, is_neg)
    dist_an[dist_an == 0.] = -100000000.
    dist_an, relative_n_inds = torch.max(dist_an, dim=1, keepdim=True)
    dist_an2 = torch.mul(dist_mat.t(), is_neg)
    dist_an2[dist_an2 == 0.] = -100000000.
    dist_an2, relative_n_inds = torch.max(dist_an2, dim=1, keepdim=True)
    dist_an = torch.cat((dist_an, dist_an2), dim=0)

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an


def topk_example_mining(dist_mat, topk):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    is_pos = torch.eye(N)
    is_neg = torch.ones(dist_mat.shape) - torch.eye(N)

    is_pos = is_pos.cuda()
    is_neg = is_neg.cuda()

    dist_ap = torch.mul(dist_mat, is_pos)
    dist_ap[dist_ap == 0.] = 100000000.
    dist_ap, relative_p_inds = torch.topk(dist_ap, k=1, dim=1, largest=False)
    dist_ap2 = torch.mul(dist_mat.t(), is_pos)
    dist_ap2[dist_ap2 == 0.] = 100000000.
    dist_ap2, relative_p_inds = torch.topk(dist_ap2, k=1, dim=1, largest=False)
    dist_ap = torch.cat((dist_ap, dist_ap2), dim=0)
    temp = dist_ap
    for i in range(topk-1):
        dist_ap = torch.cat((dist_ap, temp), dim=1)

    dist_an = torch.mul(dist_mat, is_neg)
    dist_an[dist_an == 0.] = -100000000.
    dist_an, relative_n_inds = torch.topk(dist_an, k=topk, dim=1, largest=True)
    dist_an2 = torch.mul(dist_mat.t(), is_neg)
    dist_an2[dist_an2 == 0.] = -100000000.
    dist_an2, relative_n_inds = torch.topk(dist_an2, k=topk, dim=1, largest=True)
    dist_an = torch.cat((dist_an, dist_an2), dim=0)

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an
