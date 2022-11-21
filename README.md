# EMCL
The implementation of paper [Expectation-Maximization Contrastive Learning for Compact Video-and-Language Representations]().

## Intro
Most video-and-language representation learning approaches employ contrastive learning, e.g., CLIP, to project the video and text features into a common latent space according to the semantic similarities of text-video pairs. However, such learned shared latent spaces are not often optimal, and the modality gap between visual and textual representation can not be fully eliminated. In this paper, we propose Expectation-Maximization Contrastive Learning (EMCL) to learn compact video-and-language representations.

![EMCL](EMCL.png)

# Citation
If you find our code useful in your work, you can cite the following paper:
```
@inproceedings{
jin2022expectationmaximization,
title={Expectation-Maximization Contrastive Learning for Compact Video-and-Language Representations},
author={Peng Jin and JinFa Huang and Fenglin Liu and Xian Wu and Shen Ge and Guoli Song and David A. Clifton and Jie Chen},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022}
}
```