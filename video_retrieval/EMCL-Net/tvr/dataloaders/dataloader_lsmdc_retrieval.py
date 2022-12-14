from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import tempfile
import pandas as pd
from os.path import join, splitext, exists
from collections import OrderedDict
from .dataloader_retrieval import RetrievalDataset
import os


class LsmdcDataset(RetrievalDataset):
    """LSMDC dataset."""

    def __init__(self, subset, anno_path, video_path, tokenizer, max_words=32,
                 max_frames=12, video_framerate=1, image_resolution=224, mode='all', config=None):
        super(LsmdcDataset, self).__init__(subset, anno_path, video_path, tokenizer, max_words,
                                            max_frames, video_framerate, image_resolution, mode, config=config)
        pass

    def _get_anns(self, subset='train'):
        """
        video_dict: dict: video_id -> video_path
        sentences_dict: list: [(video_id, caption)] , caption (list: [text:, start, end])
        """
        video_json_path_dict = {}
        video_json_path_dict["train"] = os.path.join(self.anno_path, "LSMDC16_annos_training.csv")
        video_json_path_dict["val"] = os.path.join(self.anno_path, "LSMDC16_annos_val.csv")
        video_json_path_dict["test"] = os.path.join(self.anno_path, "LSMDC16_challenge_1000_publictect.csv")

        # <CLIP_ID>\t<START_ALIGNED>\t<END_ALIGNED>\t<START_EXTRACTED>\t<END_EXTRACTED>\t<SENTENCE>
        # <CLIP_ID> is not a unique identifier, i.e. the same <CLIP_ID> can be associated with multiple sentences.
        # However, LSMDC16_challenge_1000_publictect.csv has no repeat instances
        video_id_list = []
        caption_dict = {}
        with open(video_json_path_dict[self.subset], 'r') as fp:
            for line in fp:
                line = line.strip()
                line_split = line.split("\t")
                assert len(line_split) == 6
                clip_id, start_aligned, end_aligned, start_extracted, end_extracted, sentence = line_split
                if clip_id not in ["0017_Pianist_00.23.28.872-00.23.34.843", "0017_Pianist_00.30.36.767-00.30.38.009",
                                   "3064_SPARKLE_2012_01.41.07.000-01.41.11.793"]:
                    caption_dict[len(caption_dict)] = (clip_id, (sentence, None, None))
                    if clip_id not in video_id_list: video_id_list.append(clip_id)

        video_dict = OrderedDict()
        sentences_dict = OrderedDict()

        for root, dub_dir, video_files in os.walk(self.video_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_id_list:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_

        # Get all captions
        for clip_id, sentence in caption_dict.values():
            if clip_id not in video_dict:
                continue
            sentences_dict[len(sentences_dict)] = (clip_id, sentence)

        unique_sentence = set([v[1][0] for v in sentences_dict.values()])
        print('[{}] Unique sentence is {} , all num is {}'.format(subset, len(unique_sentence), len(sentences_dict)))

        return video_dict, sentences_dict