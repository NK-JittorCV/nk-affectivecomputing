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

***Emotionally and Intellectually Integrated Intelligent*** is an important development direction for next-generation artificial intelligence and a key step toward achieving artificial general intelligence (AGI). In human-computer interaction scenarios, emotionally intelligent digital humans and robots need to accurately interpret multimodal interaction information and deeply explore human internal emotional states to enable more realistic and natural human-computer dialogues. However, given the high complexity of multimodal emotional data semantics, effectively modeling cross-modal associative relationships remains a critical challenge that urgently needs to be addressed in the field.

JACK (Jittor-based affective computing model training and deployment framework) is an official framework provided by Nankai University CV group for training and deploying affective computing methods based on Jittor, a Chinese high-performance deep learning framework. It currently supports cutting-edge video emotion analysis methods and gait video emotion analysis method. Built on the Jittor framework, the deployment speed of affective computing methods can be improved by 1.1 to 1.6 times compared to PyTorch, supporting downstream applications such as tourist emotion detection, dialogue analysis, and public opinion monitoring.

The high-performance deep learning framework Jittor can seamlessly integrate with the mainstream PyTorch framework. Taking the [`TSL-Net`](https://zzcheng.top/assets/pdf/2022_ACMMM_TSL300.pdf) network architecture as an example, when adapting to JACK, only about 10 modifications are required in the model code to complete the conversion, significantly reducing migration costs. Additionally, we provide detailed conversion experiences and guidance in [`Torch to Jittor FAQ.md`](./docs/en/Torch-to-Jittor-FAQ.md) to help developers get started quickly. We warmly invite more researchers to join us in advancing the development of affective computing! Let’s work together to build a stronger Jittor-based AI system!

With the support of the high-performance deep learning framework Jittor, this project now incorporates the cutting-edge methods in the field of affective computing.

| **Work**| **Train** | **Test** |
|---------|-----------|----------|
| [`[CVPR'23] CTEN`](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Weakly_Supervised_Video_Emotion_Detection_and_Prediction_via_Cross-Modal_Temporal_CVPR_2023_paper.pdf) | [`[train script]`](./docs/en/train.md#-CTEN) | [`[test script]`](./docs/en/test.md#-CTEN) |
| [`[MM'22] TSL-Net`](https://zzcheng.top/assets/pdf/2022_ACMMM_TSL300.pdf) | [`[train script]`](./docs/en/train.md#-TSL-Net) | [`[test script]`](./docs/en/test.md#-TSL-Net) |
| [`[AAAI'20] VAANet`](https://arxiv.org/abs/2003.00832)                                                              | [`[train script]`](./docs/en/train.md#-TSL-Net) | [`[test script]`](./docs/en/test.md#-TSL-Net) |
| [`[TAC'24] Gait`](https://ieeexplore.ieee.org/document/10433680)                                                   | [`[train script]`](./docs/en/train.md#-Gait) | [`[test script]`](./docs/en/test.md#-Gait) |

For the affective computing method [`TSL-Net`](https://zzcheng.top/assets/pdf/2022_ACMMM_TSL300.pdf), the high-performance deep learning framework Jittor outperforms the PyTorch framework with faster inference speed, shorter training duration, and consistent output results:

| Metrics computed by TSL-Net | PyTorch | Jittor |
|-----------------------------|---------|--------|
| Average Forward Time (train)| 0.324s  | 0.068s |
| Average Memory Usage (train)| 11319MB | 16132MB|
| Single Iteration Time(train)| 1.162s  | 0.981s |
| Average_mAP[0.1:0.3] (test) | 0.1985  | 0.1949 |
| Average_pAP[0.1:0.3] (test) | 0.2106  | 0.2095 |
| Average_nAP[0.1:0.3] (test) | 0.1865  | 0.1803 |
| F2@AVG (test)               | 0.3369  | 0.3577 |

## 🎉 News
- 🎁 2025.07.16: Project initialized. This project supports four video emotion analysis methods, including [CTEN](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Weakly_Supervised_Video_Emotion_Detection_and_Prediction_via_Cross-Modal_Temporal_CVPR_2023_paper.pdf), [TSL_Net](https://zzcheng.top/assets/pdf/2022_ACMMM_TSL300.pdf), [VAANet](https://arxiv.org/abs/2003.00832), and [Gait](https://ieeexplore.ieee.org/document/10433680). Training and testing scripts are provided for all methods.


## 🛠️ Installation
To install using pip:
```shell
pip install -r requirements.txt
```

Running Environment:

|  Dependence  | Range        | Recommended | Notes                                     |
| ------------ |--------------| ----------- | ----------------------------------------- |
| python | >=3.8 | 3.8        | |
| cuda   |              | cuda11.3   | No need to install if using CPU, NPU, MPS |
| jittor |              | 1.3.9.14   |                                           |
| transformers | >=4.30       | 4.31.0     |                                           |


For more optional dependencies, you can refer to [`env.md`](./docs/en/env.md).


## ✨ Usage
Here is a minimal example of training and deployment using JACK.

Training
```
bash script/run.sh TSL main
```

Testing
```
bash script/run.sh TSL test
```

- If you want to use other affective computing methods, you only need to modify the first argument to specify the corresponding model's name, and modify the second argument to specify the corresponding function such as train or test.


### ✨ Advanced Usage with Customized Arguments

Before further exploration, it is recommended to read our documentation in advance to gain a better understanding of the features supported by this project.

|   Useful Links |
| ------ |
|   [🔥Supported Methods](./docs/en/papers.md)   |
|   [Train](./docs/en/train.md)   |
|   [Test](./docs/en/test.md) |
|   [Datasets](./docs/en/dataset.md)   |
|   [Torch2Jittor FAQ](./docs/en/FAQ.md)   |

Taking [CTEN](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Weakly_Supervised_Video_Emotion_Detection_and_Prediction_via_Cross-Modal_Temporal_CVPR_2023_paper.pdf) as an example, JACK provides a solution from training to deployment.


#### Training
```
bash script/run.sh CTEN main \
--dataset ve8 \
--resnet101_pretrained your_path \
--video_path your_path \
--annotation_path src/CTEN/data/ve8_04.json \
--audio_path your_path \
--result_path your_path \
--batch_size 4 \
--n_epochs 100 \
--sample_size 112 \
--fps 30 \
--snippet_duration 16 \
--audio_embed_size 2048 \
--audio_n_segments 16 \
--audio_time 100
```

Before running the script, make sure to replace the placeholders for the video path, audio path, pretrained model path, and result output path with your actual local paths. Once properly configured, you can train the model on the VE8 dataset to obtain the final trained weight file. For a detailed explanation of each parameter, please refer to the [`train.md`](./docs/en/train.md)   documentation.During training or evaluation, validation results will also be printed alongside the emotion classification outputs.



#### Testing
```
bash script/run.sh CTEN test \
--dataset ve8 \
--video_path your_path \
--audio_path your_path \
--result_path your_path \
--checkpoint_path your_path
```
⚠️ Note: If random data augmentation was used during training, the predictions may vary slightly each time the script is run. For more information, see the  [`FAQ.md`](./docs/en/FAQ.md).

If your goal is only to obtain the emotion classification results for each video segment in the dataset, execute the script from the command line to generate the classification results.


#### Inference a single video
```
bash script/run.sh CTEN test_singlevideo \
--video_path your_path \
--output_frame_dir  your_path  \
--output_audio_dir  your_path  \
--checkpoint_path  your_path
```


#### Arguments from CTEN:

- `dataset`: the name of the dataset to be used. Please refer to [`dataset.md`](./docs/cn/dataset.md) for relevant instructions.
- `resnet101_pretrained`: pre-trained image model weight path (such as ResNet-101); used for video frame feature extraction.
- `result_path`: inference or training result save path.
- `video_path`: video frame sequence or video file path.
- `audio_path`: corresponding video and audio file (such as .mp3) path.
- `annotation_path`: annotation file path, annotation file for training or testing and corresponding sentiment classification.
- `batch_size`: batch size: the number of samples fed into the model each time.
- `n_epochs`: total number of training rounds.
- `sample_size`: input size (width and height) of video frame image.
- `fps`: video frame rate.
- `snippet_duration`: the number of frames per clip.
- `audio_embed_size`: audio feature dimension size.
- `checkpoint_path`: Path to the pretrained model weights.
- `audio_n_segments`: divide the entire audio into 16 segments, extract an embedding for each segment; corresponding frame alignment.
- `audio_time`: The duration of each audio sample.



### 🔧 Script Parameter Explanation

Each parameter in this script is **required**:

- **First parameter**: Specifies the name of the method to be used.
- **Second parameter**: Indicates whether the script is used for single-video inference.
- `video_path`: Path to the input video to be analyzed.

Before inference begins, the script will automatically extract both **video frames** and **audio** from the input video. These will be saved into two separate folders:

- `output_frame_dir`: Directory where the extracted video frames will be stored.
- `output_audio_dir`: Directory where the extracted audio file (in `.mp3` format) will be saved.
- `checkpoint_path`: Path to the pretrained model weights.
- `annotation_path`: Path to the annotation file associated with the video.

> ⚠️ **Note**: Make sure all paths are correctly set before running the script. The script handles the frame and audio extraction automatically.For annotation files, please refer to the following format.

```
{
    "labels": ["Anger", "Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust"],
    "database": {
      "1": {
        "subset": "validation",
        "annotations": {
          "label": "Anger"
        }
      }
    }
  }
```

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
