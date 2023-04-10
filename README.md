<div align="center">
  
# 【NeurIPS'2022 🔥】Expectation-Maximization Contrastive Learning for Compact Video-and-Language Representations
  
[![Conference](http://img.shields.io/badge/NeurIPS-2022-FFD93D.svg)](https://neurips.cc/Conferences/2022)
[![Paper](http://img.shields.io/badge/Paper-arxiv.2211.11427-FF6B6B.svg)](https://arxiv.org/abs/2211.11427)
</div>

The implementation of NeurIPS 2022 paper [Expectation-Maximization Contrastive Learning for Compact Video-and-Language Representations](https://arxiv.org/pdf/2211.11427.pdf).

### 📣 Updates
* Apr 10 2023: Add MSVD, LSMDC, ActivityNet Captions, and DiDeMo datasets (See [EMCL-Net](video_retrieval/EMCL-Net)).
* Jan 12 2023: Our approach achieves better performance (46.8 -> 48.2 on MSR-VTT dataset) when training with more GPUs (2 -> 8). So we recommend using more GPUs for better performance.

![results](pic/results.png)
* Dec 14 2022: Add the code of [EMCL-Net](video_retrieval/EMCL-Net).
* Nov 21 2022: Release code for reimplementing the experiments in the paper.

## 🚀 Quick Start
### Text-video Retrieval
* The implementation of EMCL-Net ([video_retrieval/EMCL-Net](https://github.com/jpthu17/EMCL/tree/main/video_retrieval/EMCL-Net)).

* An example of using EMCL as a joint training module ([video_retrieval/as_a_joint_training_module](https://github.com/jpthu17/EMCL/tree/main/video_retrieval/As_a_joint_training_module)).

* An example of using EMCL as an inference module with no extra training ([video_retrieval/as_an_inference_module](https://github.com/jpthu17/EMCL/tree/main/video_retrieval/As_an_inference_module)).

### Video-question Answering
* The implementation of EMCL-QA ([video_question_answering](https://github.com/jpthu17/EMCL/tree/main/video_question_answering)).

## 📕 Overview
Most video-and-language representation learning approaches employ contrastive learning, e.g., CLIP, to project the video and text features into a common latent space according to the semantic similarities of text-video pairs. However, such learned shared latent spaces are not often optimal, and the modality gap between visual and textual representation can not be fully eliminated. In this paper, we propose Expectation-Maximization Contrastive Learning (EMCL) to learn compact video-and-language representations.

![motivation](pic/Modality_gap.png)

## 📚 Method
![EMCL](pic/EMCL.png)


## 📌 Citation
If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:
```
@inproceedings{
jin2022expectationmaximization,
title={Expectation-Maximization Contrastive Learning for Compact Video-and-Language Representations},
author={Peng Jin and JinFa Huang and Fenglin Liu and Xian Wu and Shen Ge and Guoli Song and David A. Clifton and Jie Chen},
booktitle={Advances in Neural Information Processing Systems},
volume={35},
pages={30291--30306},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022}
}
```

## 🎗️ Acknowledgments
Our code is based on [MMT](https://github.com/gabeur/mmt), [CLIP](https://github.com/openai/CLIP), [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip/), [DRL](https://github.com/foolwood/DRL) and [CLIP2Video](https://github.com/CryhanFang/CLIP2Video). We sincerely appreciate for their contributions.

[def]: motivation.pdf
