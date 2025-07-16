# JACK (Jittor-based Affective Computing model training and deployment frameworK)

<p align="center">
    <br>
    <img src="assets/logo_en.png"/>
    <br>
<p>
<p align="center">
<a href="">Project Homepage</a>
<br>
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish&nbsp
</p>
<p align="center">
<img src="https://img.shields.io/badge/python-3.8-5be.svg">
<img src="https://img.shields.io/badge/jittor-1.3.9-orange.svg">
<a href="https://github.com/zhongqihebut/Affective_Computing/blob/master/LICENSE"><img src="https://img.shields.io/github/license/zhongqihebut/Affective_Computing"></a>
<a href="https://github.com/zhongqihebut/Affective_Computing/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
</p>

<p align="center">
        <a href="./docs/en/papers.md">Paper Collections</a> &nbsp ｜ <a href="./docs/en">English Documentation</a> &nbsp ｜ &nbsp <a href="./docs/cn">中文文档</a> &nbsp
</p>

## 📖 Table of Contents
- [Introduction](#-introduction)
- [News](#-news)
- [Installation](#%EF%B8%8F-installation)
- [Usage](#-Usage)
- [License](#-License)
- [Citation](#-citation)


## 📝 Introduction

***Emotionally and Intellectually Integrated Intelligent*** is an important development direction for next-generation artificial intelligence and a key step toward achieving general artificial intelligence. In human-computer interaction scenarios, emotionally intelligent digital humans and robots need to accurately interpret multimodal interaction information and deeply explore human internal emotional states to enable more realistic and natural human-computer dialogues. However, given the high complexity of multimodal emotional data semantics, effectively modeling cross-modal associative relationships remains a critical challenge that urgently needs to be addressed in the field.

JACK (Jittor-based affective computing model training and deployment framework) is an official framework provided by Nankai University CV group for training and deploying affective computing methods based on Jittor, a Chinese high-performance deep learning framework. It currently support cutting-edge video emotion analysis methods and gait video emotion analysis method. Built on the Jittor framework, the deployment speed of affective computing methods can be improved by 1.1 to 1.6 times compared to PyTorch, supporting downstream applications such as tourist emotion detection, dialogue analysis, and public opinion monitoring.


The project now support four works in the field of affective computing based on the Jittor deep learning framework:

| **Work**| **Train** | **Test** |
|---------|-----------|----------|
| [[CVPR'23] CTEN](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Weakly_Supervised_Video_Emotion_Detection_and_Prediction_via_Cross-Modal_Temporal_CVPR_2023_paper.pdf) | [[train script]](./docs/en/train.md#-CTEN) | [[test script]](./docs/en/test.md#-CTEN) |
| [[MM'22] TSL-Net](https://github.com/nku-zhichengzhang/TSL300/blob/main/assests/acm22_zzc_videosenti_official.pdf) | [[train script]](./docs/en/train.md#-TSL-Net) | [[test script]](./docs/en/test.md#-TSL-Net) |
| [[AAAI'20] VAANet](https://arxiv.org/abs/2003.00832)                                                              | [[train script]](./docs/en/train.md#-TSL-Net) | [[test script]](./docs/en/test.md#-TSL-Net) |
| [[TAC'24] Gait](https://ieeexplore.ieee.org/document/10433680)                                                   | [[train script]](./docs/en/train.md#-Gait) | [[test script]](./docs/en/test.md#-Gait) |


Performance comparison between the Jittor high-performance deep learning framework and the PyTorch framework in the field of affective computing:


| Metrics computed by TSL-Net                 | PyTorch  | Jittor  |
|-------------------------|----------|---------|
| Average Forward Time (train)    | 0.324s   | 0.068s  |
| Average Memory Usage (train)    | 11319MB  | 16132MB |
| Single Iteration Time(train)   | 1.162s   | 0.981s  |
| Average_mAP[0.1:0.3] (test)    | 0.1985   | 0.1949  |
| Average_pAP[0.1:0.3] (test)    | 0.2106   | 0.2095  |
| Average_nAP[0.1:0.3] (test)   | 0.1865   | 0.1803  |
| F2@AVG (test)                  | 0.3369   | 0.3577  |

## 🎉 News
- 🎁 2025.07.16: Project initialized. This project supports four video emotion analysis tasks, including [CTEN](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Weakly_Supervised_Video_Emotion_Detection_and_Prediction_via_Cross-Modal_Temporal_CVPR_2023_paper.pdf), [TSL_Net](https://github.com/nku-zhichengzhang/TSL300/blob/main/assests/acm22_zzc_videosenti_official.pdf), [VAANet](https://arxiv.org/abs/2003.00832), and [Gait](https://ieeexplore.ieee.org/document/10433680). Training and testing scripts are provided for all methods.


## 🛠️ Installation
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


## ✨ Usage
Here is a minimal example of training and deployment using JACK.

- If you want to use other models or datasets (including multimodal models and datasets), you only need to modify `--model` to specify the corresponding model's name, and modify `--dataset` to specify the corresponding dataset's path.

|   Useful Links |
| ------ |
|   [🔥Supported Methods](./docs/en/papers.md)   |
|   [Train](./docs/en/train.md)   |
|   [Test](./docs/en/test.md) |
|   [Datasets](./docs/en/dataset.md)   |
|   [Torch2Jittor FAQ](./docs/en/FAQ.md)   |





## 🏛 License

This framework is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE). For models and datasets, please refer to the original resource page and follow the corresponding License.


## 📎 Citation

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