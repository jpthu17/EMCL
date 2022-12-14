## Video Question answering
### Requirement
```sh
# From CLIP
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install opencv-python boto3 requests pandas
```

### Data Preparing

For MSRVTT, the official data and video links can be found in [link](http://ms-multimedia-challenge.com/2017/dataset).

For the convenience, the splits and captions can be found in sharing from [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip/),

```shell
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
```

Besides, the raw videos can be found in sharing from [Frozen in Time](https://github.com/m-bain/frozen-in-time), i.e.,

```shell
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
```

### Compress Video for Speed-up (optional)
```sh
python preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]
```
This script will compress the video to *3fps* with width *224* (or height *224*). Modify the variables for your customization.

### MSRVTT-QA

```sh
DATA_PATH=[Your MSRVTT data and videos path]

python -m torch.distributed.launch \
--nproc_per_node=2 \
main.py \
--do_train \
--num_thread_reader=0 \
--epochs=5 \
--batch_size=32 \
--n_display=50 \
--train_csv data/MSRVTT/train.jsonl \
--val_csv data/MSRVTT/test.jsonl \
--data_path data/MSRVTT/train_ans2label.json \
--features_path ${DATA_PATH} \
--output_dir ckpts/msrvtt_qa \
--lr 1e-4 --max_words 32 \
--max_frames 12 \
--batch_size_val 16 \
--datatype msrvtt \
--expand_msrvtt_sentences \
--feature_framerate 1 \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--slice_framepos 2 \
--loose_type \
--linear_patch 2d
```

## Acknowledge
* This code implementation are adopted from [CLIP](https://github.com/openai/CLIP) and [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip/).
We sincerely appreciate for their contributions.