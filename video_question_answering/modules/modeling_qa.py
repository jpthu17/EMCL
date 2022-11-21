from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from modules.until_module import PreTrainedModel, AllGather, CrossEn, Emcl
from modules.module_cross import CrossConfig, Transformer as TransformerClip
from modules.module_clip import CLIP, convert_weights
from modules.loss import CrossEn
logger = logging.getLogger(__name__)
allgather = AllGather.apply


class EMCL4QAPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, cross_config, *inputs, **kwargs):
        super(EMCL4QAPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        clip_state_dict = CLIP.get_config(pretrained_clip_name="ViT-B/16")
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None,
                                                 task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        contain_frame_position = False
        for key in state_dict.keys():
            if key.find("frame_position_embeddings") > -1:
                contain_frame_position = True
                break
        if contain_frame_position is False:
            for key, val in clip_state_dict.items():
                if key == "positional_embedding":
                    state_dict["frame_position_embeddings.weight"] = val.clone()
                    continue
                if key.find("transformer.resblocks") == 0:
                    num_layer = int(key.split(".")[2])
                    # cut from beginning
                    if num_layer < task_config.cross_num_hidden_layers:
                        state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                        continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model


def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)


def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config


def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class EMCL4QA(EMCL4QAPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(EMCL4QA, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1
        self.dropout = nn.Dropout(0.1)

        self.emcl = Emcl(k=task_config.K,
                         stage_num=task_config.stage_num,
                         momentum=task_config.momentum,
                         lamd=task_config.lamd,
                         beta=task_config.beta)

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b
                            in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(
            set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers - cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers - cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        cross_config.max_position_embeddings = context_length
        self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
        self.transformerClip = TransformerClip(width=transformer_width,
                                                   layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )

        hidden_size = transformer_width * 8
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size,
                      hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(hidden_size * 2, task_config.num_labels)
        )

        self.v_proj = nn.Linear(transformer_width, 4 * transformer_width)
        self.t_proj = nn.Linear(transformer_width, 4 * transformer_width)
        self.loss_fct = CrossEn()

        if vit:
            self.mean_proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.mean_proj = nn.Linear(embed_dim, embed_dim)

        self.v_w = nn.Sequential(
            nn.Linear(transformer_width, transformer_width),
            nn.ReLU(True),
            nn.Linear(embed_dim, 1)
        )

        self.t_w = nn.Sequential(
            nn.Linear(transformer_width, transformer_width),
            nn.ReLU(True),
            nn.Linear(embed_dim, 1)
        )

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,
                labels=None):

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts


        sequence_output, visual_output = self.get_sequence_visual_output(input_ids,
                                                                                                         token_type_ids,
                                                                                                         attention_mask,
                                                                         video, video_mask, shaped=True, video_frame=video_frame)


        visual_output = self.v_proj(visual_output)
        sequence_output = self.t_proj(sequence_output)
        input = torch.cat((visual_output, sequence_output), dim=1)
        pooled_output = self.dropout(input)
        logits = self.classifier(pooled_output)

        if self.training:
            labels = allgather(labels, self.task_config)
            logits, loss = self.calc_loss(logits, labels)
            
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            attention_mask = allgather(attention_mask, self.task_config)
            torch.distributed.barrier()

            b0, n = visual_output.size()
            b1, n = sequence_output.size()
            all_embedding = torch.cat((visual_output, sequence_output), dim=0)
            all_embedding = self.emcl(all_embedding, if_train=self.training)
            visual_output = all_embedding[:b0, :].view(b0, n)
            sequence_output = all_embedding[b0:, :].view(b1, n)
            visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
            sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

            logit_scale = self.clip.logit_scale.exp()
            retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
            sim_loss = (self.loss_fct(retrieve_logits) + self.loss_fct(retrieve_logits.T)) / 2
            loss = loss + sim_loss * 0.5
            
            return loss
        else:
            return logits

    def calc_loss(self, logits, labels):
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="mean")
            loss = loss_fct(
                        logits.view(-1, self.task_config.num_labels),
                        labels.view(-1))
        else:
            loss = 0
        return logits, loss

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_output = self.clip.encode_text(input_ids, return_hidden=True)[0].float()
        sequence_output = sequence_output.view(bs_pair, -1, sequence_output.size(-1))
        return sequence_output

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_output = self.clip.encode_image(video, video_frame=video_frame, return_hidden=True)[0].float()
        visual_output = visual_output.view(bs_pair, -1, visual_output.size(-1))
        
        visual_output_original = visual_output
        seq_length = visual_output.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        visual_output = visual_output + frame_position_embeddings

        extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
        visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
        visual_output = self.transformerClip(visual_output, extended_video_mask)
        visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
        visual_output = self.mean_proj(visual_output)
        visual_output = visual_output + visual_output_original

        return visual_output

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            attention_mask = allgather(attention_mask, self.task_config)
            torch.distributed.barrier()

        text_weight = torch.softmax(self.t_w(sequence_output).squeeze(2), dim=-1)  # BxN_t
        video_weight = torch.softmax(self.v_w(visual_output).squeeze(2), dim=-1)  # BxN_t
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        sequence_output = sequence_output * attention_mask_un
        sequence_output = torch.einsum(" atc,at->ac ", [sequence_output, text_weight])
        visual_output = torch.einsum(" atc,at->ac ", [visual_output, video_weight])
        return sequence_output, visual_output