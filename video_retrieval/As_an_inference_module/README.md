## EMCL as an inference module with no extra training

EMCL can be incorporated into trained baselines as an out-of-the-box inference module with no extra training.

### Requirement
```
pip install -r requirements.txt 
```

### Download data and Pre-trained Model

**Download official video:**
Official videos of different data can be found as follows:

* MSRVTT: [link](http://ms-multimedia-challenge.com/2017/dataset).

**Pre-process**

To train and test the above datasets: you should use `sample_frame.py` to transform video into frames.
~~~
python sample_frame.py --input_path [raw video path] --output_path [frame path]
~~~

(*Optional*) The splits and captions can be found in the links of used dataset. For the convenience, you can also use the split in ` data/` directly.

**Download CLIP model**

To train and test the above datasets based on pre-trained CLIP model, you should visit [CLIP](https://github.com/openai/CLIP) and download [ViT-B/32](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt).



### Test Model

The models trained on MSR-VTT can be found in sharing from [CLIP2Video](https://github.com/CryhanFang/CLIP2Video), i.e.,

|    Model Name         |   checkpoint|
| :-----------:  | :-----------: |
|CLIP2Video_MSRVTT9k |	[link](https://drive.google.com/drive/folders/1a5Dcg8wNh88Z-bxb0ZMV3IJFjtSe7X2A?usp=sharing)	|


To test the trained model, please refer  `test/`.

(*Optional*) If the path of trained model(`--checkpoint`) doesn't exist, the parameters of basic CLIP (`--clip_path`) will be loaded.

### Testing

####CLIP2Video
```sh
DATA_PATH=${VERSION}/data/msrvtt_data/
CHECKPOINT=[downloaded trained model path]
MODEL_NUM=2

python ${VERSION}/infer_retrieval.py \
--num_thread_reader=2 \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path [frame path] \
--output_dir ${CHECKPOINT}/test_${MODEL_NUM}.txt \
--max_words 32 \
--max_frames 12 \
--batch_size_val 64 \
--datatype msrvtt \
--feature_framerate 2 \
--sim_type seqTransf \
--checkpoint ${CHECKPOINT} \
--do_eval \
--model_num ${MODEL_NUM} \
--temporal_type TDB \
--temporal_proj sigmoid_selfA \
--center_type TAB \
--centerK 5 \
--center_weight 0.5 \
--center_proj TAB_TDB \
--clip_path ${VERSION}/ViT-B-32.pt
```
####CLIP2Video + EMCL (EMCL as an inference module with no extra training)
```sh
DATA_PATH=${VERSION}/data/msrvtt_data/
CHECKPOINT=[downloaded trained model path]
MODEL_NUM=2

python ${VERSION}/infer_retrieval.py \
--num_thread_reader=2 \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path [frame path] \
--output_dir ${CHECKPOINT}/test_${MODEL_NUM}.txt \
--max_words 32 \
--max_frames 12 \
--batch_size_val 64 \
--datatype msrvtt \
--feature_framerate 2 \
--sim_type seqTransf \
--checkpoint ${CHECKPOINT} \
--do_eval \
--model_num ${MODEL_NUM} \
--temporal_type TDB \
--temporal_proj sigmoid_selfA \
--center_type TAB \
--centerK 5 \
--center_weight 0.5 \
--center_proj TAB_TDB \
--clip_path ${VERSION}/ViT-B-32.pt \
--EMCL 1 \
--K 7 \
--stage_num 10 \
--momentum 0.9 \
--lamd 0.1 \
--beta 4
```


|    Protocol         |   T2V R@1     |   T2V R@5     |   V2T R@1    | V2T R@5 |
| :-----------:  | :-----------: | ---------- | :-----------:  | :-----------: |
|CLIP2Video |	45.6	|   72.6	|   43.5	|   72.3   |
|+ EMCL  |	46.6	|   72.7	|   44.0	|   72.3	|



## Acknowledge
This code implementation are adopted from [CLIP](https://github.com/openai/CLIP) , [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip/) and [CLIP2Video](https://github.com/CryhanFang/CLIP2Video).
We sincerely appreciate for their contributions.




