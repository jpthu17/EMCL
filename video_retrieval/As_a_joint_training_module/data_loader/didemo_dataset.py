import os

from base.base_dataset import BaseDataset
import numpy as np
from utils.util import get_expert_paths


class DiDeMo(BaseDataset):
  """DiDeMo dataset."""

  def configure_train_test_splits(self, cut_name, split_name):
    if cut_name in ["full"]:
      if split_name in ["train", "trn"]:
        list_path = "train_list.txt"
      elif split_name in ["val"]:
        list_path = "val_list.txt"
      elif split_name in ["test"]:
        list_path = "test_list.txt"
      else:
        raise ValueError(f"unrecognised DiDeMo split: {split_name}")

      list_path = os.path.join(self.root_feat, list_path)
      with open(list_path) as f:
        self.vid_list = f.readlines()
      self.vid_list = [x.strip() for x in self.vid_list]

      # We want the trn split to be the same size as the val set
      if split_name in ["trn"]:
        rng = np.random.RandomState(0)
        rng.shuffle(self.vid_list)
        if cut_name in ["c"]:
          self.vid_list = self.vid_list[:840]
        else:
          self.vid_list = self.vid_list[:1065]

    elif cut_name in ["c"]:
      self.expert_paths = get_expert_paths(self.data_dir)
      if split_name in ["train", "trn", "val", "trainval"]:
        train_list_path = "train_list.txt"
        train_list_path = os.path.join(self.data_dir, train_list_path)
        with open(train_list_path) as f:
          train_vid_list = f.readlines()
        nb_train_samples = len(train_vid_list)

        val_list_path = "val_list.txt"
        val_list_path = os.path.join(self.data_dir, val_list_path)
        with open(val_list_path) as f:
          val_vid_list = f.readlines()
        nb_val_samples = len(val_vid_list)

        cross_vid_list = train_vid_list + val_vid_list
        cross_vid_list = [x.strip() for x in cross_vid_list]

        if self.cross_seed != 0:
          # The cross seed is used to split training videos into different
          # cross validation splits.
          rng = np.random.RandomState(self.cross_seed)
          rng.shuffle(cross_vid_list)

        if split_name in ["train", "trn", "trainval"]:
          if split_name in ["trainval"]:
            self.vid_list = cross_vid_list
          elif split_name in ["train", "trn"]:
            self.vid_list = cross_vid_list[:nb_train_samples]
          if split_name in ["trn"]:
            # In order to monitor performance on the training set, we sample
            # from it as many samples as there are validation samples.
            rng = np.random.RandomState(0)
            rng.shuffle(self.vid_list)
            self.vid_list = self.vid_list[:nb_val_samples]

        elif split_name in ["val"]:
          self.vid_list = cross_vid_list[nb_train_samples:]

      else:
        if split_name == "test1":
          list_path = "public_server_val.txt"
        elif split_name == "test2":
          list_path = "public_server_test.txt"
        list_path = os.path.join(self.data_dir, list_path)
        with open(list_path) as f:
          self.vid_list = f.readlines()
        self.vid_list = [x.strip() for x in self.vid_list]

    else:
      msg = "unrecognised cut: {}"
      raise ValueError(msg.format(cut_name))

    self.split_name = split_name
    self.dataset_name = f"DiDeMo_{cut_name}_{split_name}"

    self.expert_timings = {}
