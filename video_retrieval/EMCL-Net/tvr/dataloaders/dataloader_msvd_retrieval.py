from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import tempfile
import os
import pickle
import pandas as pd
from os.path import join, splitext, exists
from collections import OrderedDict
from .dataloader_retrieval import RetrievalDataset


class MsvdDataset(RetrievalDataset):
    """MSVD dataset loader."""

    def __init__(self, subset, anno_path, video_path, tokenizer, max_words=32,
                 max_frames=12, video_framerate=1, image_resolution=224, mode='all', config=None):
        super(MsvdDataset, self).__init__(subset, anno_path, video_path, tokenizer, max_words,
                                          max_frames, video_framerate, image_resolution, mode, config=config)
        pass

    def _get_anns(self, subset='train'):
        self.sample_len = 0
        self.cut_off_points = []
        self.multi_sentence_per_video = True  # !!! important tag for eval

        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.anno_path, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.anno_path, "val_list.txt")
        video_id_path_dict["test"] = os.path.join(self.anno_path, "test_list.txt")
        caption_file = os.path.join(self.anno_path, "raw-captions.pkl")

        with open(video_id_path_dict[subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]

        with open(caption_file, 'rb') as f:
            captions = pickle.load(f)

        video_dict = OrderedDict()
        sentences_dict = OrderedDict()

        for root, dub_dir, video_files in os.walk(self.video_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_

        for video_id in video_ids:
            assert video_id in captions
            for cap in captions[video_id]:
                cap_txt = " ".join(cap)
                sentences_dict[len(sentences_dict)] = (video_id, (cap_txt, None, None))
            self.cut_off_points.append(len(sentences_dict) - 1)

        if subset == "val" or subset == "test":
            self.sentence_num = len(sentences_dict)
            self.video_num = len(video_ids)
            assert len(self.cut_off_points) == self.video_num
            print("For {}, sentence number: {}".format(subset, self.sentence_num))
            print("For {}, video number: {}".format(subset, self.video_num))

        print("Video number: {}".format(len(video_dict)))
        print("Total Paire: {}".format(len(sentences_dict)))

        self.sample_len = len(sentences_dict)

        return video_dict, sentences_dict