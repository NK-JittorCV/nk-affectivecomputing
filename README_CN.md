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
- [引用](#-引用)

  



## 📝 简介

***情智兼备*** 是新一代人工智能的重要发展方向，是迈向通用人工智能的关键一步。在人机交互场景中，具备情智的数字人与机器人需要精准解译多模态交互信息，深度挖掘人类内在情感状态，从而实现更具真实感与自然性的人机对话。然而，面对多模态情感数据语义的高度复杂性，如何有效建模跨模态关联关系仍是领域内亟待突破的核心挑战。

JACK（基于Jittor的情感计算模型训练与部署框架）是由南开大学计算机视觉团队提供的官方框架，基于国产化高性能深度学习框架计图（Jittor）进行情感计算方法的训练与部署。目前，该框架支持先进的视频情感分析方法以及步态视频情感分析方法。基于Jittor国产框架，情感计算方法的部署速度相比PyTorch可提升1.1至1.6倍，从而支持下游应用如游客情感检测、对话分析、舆情监控等。



该项目目前支持基于Jittor深度学习框架的情感计算领域中的四项工作：

| **工作**| **训练** | **测试** |
|-----------------------------------------------------------------------------------------------------------|-----------|----------|
| [[CVPR'23] CTEN](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Weakly_Supervised_Video_Emotion_Detection_and_Prediction_via_Cross-Modal_Temporal_CVPR_2023_paper.pdf) | [[训练脚本]](./docs/cn/训练.md#-CTEN) | [[测试脚本]](./docs/cn/测试.md#-CTEN) |
| [[MM'22] TSL-Net](https://github.com/nku-zhichengzhang/TSL300/blob/main/assests/acm22_zzc_videosenti_official.pdf) | [[训练脚本]](./docs/cn/训练.md#-TSL-Net) | [[测试脚本]](./docs/cn/测试.md#-TSL-Net) |
| [[AAAI'20] VAANet](https://arxiv.org/abs/2003.00832)                                                              | [[训练脚本]](./docs/cn/训练.md#-TSL-Net) | [[测试脚本]](./docs/cn/测试.md#-TSL-Net) |
| [[TAC'24] Gait](https://ieeexplore.ieee.org/document/10433680)                                                   | [[训练脚本]](./docs/cn/训练.md#-Gait) | [[测试脚本]](./docs/cn/测试.md#-Gait) |


在情感计算领域，Jittor高性能深度学习框架与PyTorch框架的性能对比：


| TSL-Net指标                        | PyTorch  | Jittor  |
|-----------------------------|----------|---------|
| 平均前向时间（训练）         | 0.324s   | 0.068s  |
| 平均内存使用量（训练）       | 11319MB  | 16132MB |
| 单次迭代时间（训练）         | 1.162s   | 0.981s  |
| 平均_mAP[0.1:0.3]（测试）    | 0.1985   | 0.1949  |
| 平均_pAP[0.1:0.3]（测试）    | 0.2106   | 0.2095  |
| 平均_nAP[0.1:0.3]（测试）    | 0.1865   | 0.1803  |
| F2@AVG（测试）               | 0.3369   | 0.3577  |



## 🎉 新闻
- 🎁 2025.07.16: 项目初始化。本项目支持四项视频情感分析任务，包括 [CTEN](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Weakly_Supervised_Video_Emotion_Detection_and_Prediction_via_Cross-Modal_Temporal_CVPR_2023_paper.pdf), [TSL_Net](https://github.com/nku-zhichengzhang/TSL300/blob/main/assests/acm22_zzc_videosenti_official.pdf), [VAANet](https://arxiv.org/abs/2003.00832), 和 [Gait](https://ieeexplore.ieee.org/document/10433680)。所有方法均提供了训练和测试脚本。


## 🛠️ 安装
使用 pip 安装：

```shell
pip install -r requirements.txt
```

依赖的环境:

| 包            | 范围         | 推荐        | 注意                                       |
| ------------ |--------------| ---------- | ----------------------------------------- |
| python       | >=3.8        | 3.8        |                                           |
| cuda         |              | cuda11.3   | 如果使用 CPU、NPU 或 MPS，则无需安装          |
| jittor       |              | 1.3.9.14   |                                           |
| transformers | >=4.30       | 4.31.0     |                                           |

有关更多可选依赖项，请参考 [这里](./docs/en/env.md)。




## ✨ 使用
以下是使用 JACK 进行训练和部署的简要示例：

- 如果您希望使用其他模型或数据集（包括多模态模型和数据集），只需修改 `--model` 来指定对应模型的名字，并修改 `--dataset` 来指定对应数据集的路径。

|  实用链接 |
| ------ |
|   [🔥支持的方法](./docs/en/papers.md)   |
|   [训练](./docs/en/train.md)   |
|   [测试](./docs/en/test.md) |
|   [数据集](./docs/en/dataset.md)   |
|   [Torch转Jittor常见问题](./docs/en/FAQ.md)   |



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