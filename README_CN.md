#  基于国产深度框架Jittor计图的训练与部署解决方案

<p align="center">
    <br>
    <img src="assets/logo.png"/>
    <br>
<p>
<p align="center">
<a href="">项目主页</a>
<br>
        中文&nbsp ｜ &nbsp<a href="README.md">English</a>&nbsp
</p>
<p align="center">
<img src="https://img.shields.io/badge/python-3.8-5be.svg">
<img src="https://img.shields.io/badge/jittor-1.3.9-orange.svg">
<a href="https://github.com/zhongqihebut/Affective_Computing/blob/master/LICENSE"><img src="https://img.shields.io/github/license/zhongqihebut/Affective_Computing"></a>
<a href="https://github.com/zhongqihebut/Affective_Computing/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
</p>

<p align="center">
        <a href="./docs/en/papers.md">相关论文</a> &nbsp ｜ <a href="./docs/en">English Documentation</a> &nbsp ｜ &nbsp <a href="./docs/cn">中文文档</a> &nbsp
</p>

## 📖 目录
- [简介](#-简介)
- [新闻](#-新闻)
- [安装](#%EF%B8%8F-安装)
- [使用](#-使用)
- [License](#-License)
- [Citation](#-citation)

  



## 📝 简介

***情智兼备*** 是新一代人工智能的重要发展方向，是迈向通用人工智能的关键一步。在人机交互场景中，具备情智的数字人与机器人需要精准解译多模态交互信息，深度挖掘人类内在情感状态，从而实现更具真实感与自然性的人机对话。然而，面对多模态情感数据语义的高度复杂性，如何有效建模跨模态关联关系仍是领域内亟待突破的核心挑战。

JACK（基于Jittor的情感计算模型训练与部署框架）是由南开大学计算机视觉团队提供的官方框架，基于国产化高性能深度学习框架计图（Jittor）进行情感计算方法的训练与部署。目前，该框架支持先进的视频情感分析方法以及步态视频情感分析方法。基于Jittor国产框架，情感计算方法的部署速度相比PyTorch可提升1.1至1.6倍，从而支持下游应用如游客情感检测、对话分析、舆情监控等。



该项目目前支持基于Jittor深度学习框架的情感计算领域中的四项工作：

| **工作**| **训练** | **测试** |
|-----------------------------------------------------------------------------------------------------------|-----------|----------|
| [[CTEN]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Weakly_Supervised_Video_Emotion_Detection_and_Prediction_via_Cross-Modal_Temporal_CVPR_2023_paper.pdf) | [[Train]]() | [[Test]]() |
| [[TSL_Net]](https://github.com/nku-zhichengzhang/TSL300/blob/main/assests/acm22_zzc_videosenti_official.pdf) | [[Train]]() | [[Test]]() |
| [[VAANet]](https://arxiv.org/abs/2003.00832)                                                              | [[Train]]() | [[Test]]() |
| [[Gait]](https://ieeexplore.ieee.org/document/10433680)                                                   | [[Train]]() | [[Test]]() |


在情感计算领域，Jittor高性能深度学习框架与PyTorch框架的性能对比：

| **工作**| **训练** | **测试** |
|-----------------------------------------------------------------------------------------------------------|-----------|----------|
| [[CTEN]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Weakly_Supervised_Video_Emotion_Detection_and_Prediction_via_Cross-Modal_Temporal_CVPR_2023_paper.pdf) | [[Train]]() | [[Test]]() |
| [[TSL_Net]](https://github.com/nku-zhichengzhang/TSL300/blob/main/assests/acm22_zzc_videosenti_official.pdf) | [[Train]]() | [[Test]]() |
| [[VAANet]](https://arxiv.org/abs/2003.00832)                                                              | [[Train]]() | [[Test]]() |
| [[Gait]](https://ieeexplore.ieee.org/document/10433680)                                                   | [[Train]]() | [[Test]]() |



## 🎉 新闻
- 🎁 2025.07.16: Project initialized. This project supports four video emotion analysis tasks, including [CTEN](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Weakly_Supervised_Video_Emotion_Detection_and_Prediction_via_Cross-Modal_Temporal_CVPR_2023_paper.pdf), [TSL_Net](https://github.com/nku-zhichengzhang/TSL300/blob/main/assests/acm22_zzc_videosenti_official.pdf), [VAANet](https://arxiv.org/abs/2003.00832), and [Gait](https://ieeexplore.ieee.org/document/10433680). Training and testing scripts are provided for all methods.


## 🛠️ 安装
To install using pip:
```shell
pip install -r requirements.txt
```

Running Environment:

|              | Range        | Recommended | Notes                                     |
| ------------ |--------------| ----------- | ----------------------------------------- |
| python       | >=3.8        | 3.8        |                                           |
| cuda         |              | cuda11.3   | No need to install if using CPU, NPU, MPS |
| jittor       |        |  1.3.9.14     |                                           |
| transformers | >=4.30       | 4.31.0      |                                           |


For more optional dependencies, you can refer to [here](./docs/en/env.md).




## ✨ 使用
Here is a minimal example of training and deployment using JACK.

- If you want to use other models or datasets (including multimodal models and datasets), you only need to modify `--model` to specify the corresponding model's ID or path, and modify `--dataset` to specify the corresponding dataset's ID or path.

|   Useful Links |
| ------ |
|   [🔥Supported Methods](./docs/en/papers.md)   |
|   [Train](./docs/en/train.md)   |
|   [Test](./docs/en/test.md) |
|   [Datasets](./docs/en/dataset.md)   |
|   [Torch2Jittor FAQ](./docs/en/FAQ.md)   |



## 🏛 License

本框架使用[Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE)进行许可。模型和数据集请查看原资源页面并遵守对应License。


## 📎 引用

```bibtex
@inproceedings{zhang2025moda,
  author = {Zhang, Zhicheng and Xia, Wuyou and Zhao, Chenxi and Yan, Zhou and Liu, Xiaoqiang and Zhu, Yongjie and Qin, Wenyu and Wan, Pengfei and Zhang, Di and Yang, Jufeng},
  title = {MODA: MOdular Duplex Attention for Multimodal Perception, Cognition, and Emotion Understanding},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year = {2025},
}
```
```bibtex
@inproceedings{zhang2024masked,
  author = {Zhang, Zhicheng and Zhao, Pancheng and Park, Eunil and Yang, Jufeng},
  title = {MART: Masked Affective RepresenTation Learning via Masked Temporal Distribution Distillation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024},
}
```

```bibtex
@inproceedings{Zhang_2023_CVPR,
  author = {Zhang, Zhicheng and Wang, Lijuan and Yang, Jufeng},
  title = {Weakly Supervised Video Emotion Detection and Prediction via Cross-Modal Temporal Erasing Network},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023},
}
```
```bibtex
@inproceedings{10.1145/3503161.3548007,
  author = {Zhang, Zhicheng and Yang, Jufeng},
  title = {Temporal Sentiment Localization: Listen and Look in Untrimmed Videos},
  year = {2022},
  booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
}
```
```bibtex
@inproceedings{zhao2020end,
  title={An end-to-end visual-audio attention network for emotion recognition in user-generated videos},
  author={Zhao, Sicheng and Ma, Yunsheng and Gu, Yang and Yang, Jufeng and Xing, Tengfei and Xu, Pengfei and Hu, Runbo and Chai, Hua and Keutzer, Kurt},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  year={2020}
}
```
```bibtex
@article{zhai2024looking,
  title={Looking into gait for perceiving emotions via bilateral posture and movement graph convolutional networks},
  author={Zhai, Yingjie and Jia, Guoli and Lai, Yu-Kun and Zhang, Jing and Yang, Jufeng and Tao, Dacheng},
  journal={IEEE Transactions on Affective Computing},
  volume={15},
  number={3},
  pages={1634--1648},
  year={2024},
}
```