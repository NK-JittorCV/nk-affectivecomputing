<div align="center">

# Looking into Gait for Perceiving Emotions via Bilateral Posture and Movement Graph Convolutional Networks [TAFFC 2024]


<i>Yingjie Zhai*, Guoli Jia*, Yu-Kun Lai, Jing Zhang, Jufeng Yang, and Dacheng Tao</i>


[![Conference](https://img.shields.io/badge/TAFFC-2024-green)](https://cis.ieee.org/publications/t-neural-networks-and-learning-systems)
[![License](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

</div>

This is the official implementation with **Jittor** of our **TAFFC 2024** paper.  </br>

## Publication

>**Looking into Gait for Perceiving Emotions via Bilateral Posture and Movement Graph Convolutional Networks**<br>
Yingjie Zhai*, Guoli Jia*, Yu-Kun Lai, Jing Zhang, Jufeng Yang, and Dacheng Tao<br>
<i>IEEE Transactions on Affective Computing (TAFFC)</i>.</br>
[[PDF]](https://exped1230.github.io/BPM-GCN/GaitEmotion-BPM-GCN/static/pdfs/TAFFC_BPM_GCN.pdf)
[[Project Page]](https://exped1230.github.io/BPM-GCN/GaitEmotion-BPM-GCN/index.html)</br>

<img src="./imgs/fig1.png" width="60%" align=center>


## Abstract

Emotions can be perceived from a person's gait, i.e., their walking style. Existing methods on gait emotion recognition mainly leverage the posture information as input, but ignore the body movement, which contains complementary information for recognizing emotions evoked in the gait. In this paper, we propose a Bilateral Posture and Movement Graph Convolutional Network (BPM-GCN) that consists of two parallel streams, namely posture stream and movement stream, to recognize emotions from two views. The posture stream aims to explicitly analyse the emotional state of the person. Specifically, we design a novel regression constraint based on the hand-engineered features to distill the prior affective knowledge into the network and boost the representation learning. The movement stream is designed to describe the intensity of the emotion, which is an implicitly cue for recognizing emotions. To achieve this goal, we employ a higher-order velocity-acceleration pair to construct graphs, in which the informative movement features are utilized. Besides, we design a PM-Interacted feature fusion mechanism to adaptively integrate the features from the two streams. Therefore, the two streams collaboratively contribute to the performance from two complementary views. Extensive experiments on the largest benchmark dataset Emotion-Gait show that BPM-GCN performs favorably against the state-of-the-art approaches (with at least 4.59% performance improvement).

<img src="./imgs/fig2.png" width="80%" align=center>

## Framework
This project is built on the Jittor deep learning framework.

Jittor: A deep learning framework based entirely on Just-in-time compilation, using innovative meta-operators and unified computing graphs internally.


## Performance

| Metric             | PyTorch | Jittor  |
|--------------------|---------|---------|
| Average Forward Time (train) | 0.218s  | 0.054s  |
| Average Memory Usage (train) | 3599MB  | 6899MB  |
| Single epoch Time(train) | 17.318s | 15.790s |
| Average_Acc(test)  | 0.8899  | 0.9115  |
| Happy_Acc(test)  | 0.9683  | 0.9820  |
| Sad_Acc(test)  | 0.8049  | 0.9756  |
| Angry_Acc(test)  | 0.7879  | 0.7778  |
| Neutral_Acc(test)  | 0.7222  | 0.3846  |


## Dependencies

#### Recommended Environment

* Python 3.8.0
* jittor 1.3.9.14
* CUDA 11.3

## Requirements
* jittor==1.3.9.14
* PyYAML==6.0.2
* GPUtil==1.4.0
* fsspec==2023.6.0
* idna==3.4
* joblib==1.3.2
* numpy==1.22.0
* pandas==2.0.3
* Pillow==10.0.0
* protobuf==4.24.1
* regex==2023.8.8
* scikit-learn==1.3.0
* scipy==1.10.1
* tokenizers==0.13.3
* tqdm==4.66.1
* transformers==4.31.0
* triton==2.0.0
* networkx==3.1

## Running

You can easily train and evaluate the model by running the script below.


***Installation***: Please clone the repository, prepare enviroment, and compile corresponding packages. 

1. Clone repository
```
git clone https://github.com/exped1230/BPM-GCN.git
```
2. Create an Anaconda environment and install the dependencies

```
conda create --name BPM-GCN-jittor
conda activate BPM-GCN-jittor
pip install -r requirements.txt
```

***Datasets***: The used datasets are provided in the homepage of [Emotion-Gait](https://gamma.umd.edu/software). 

Note that since the dataset is changed on the official website, we provide the original dataset. The code and dataset are provided for research only.
[Baidu Drive](https://pan.baidu.com/s/1rNC7SQrwNnZBVMRzPaifZA) 
(acil)


***Hyperparameter***: You can adjust more details such as epoch, batch size, etc. Please refer to config directory.


***begin***

The pre-trained model can be found in [pretrained model](https://pan.baidu.com/s/1Rzc-j16BNOsbh9V1FiZdcg?pwd=btef).

You can train and eval the model by running the command below.

~~~~
$ python main_diff_combine_double_fagg.py
~~~~
## Citation
If you find this repo useful in your project or research, please consider citing the relevant publication.

**Bibtex Citation**
````
@article{zhai2024Looking,
  author={Zhai, Yingjie and Jia, Guoli and Lai, Yu-Kun and Zhang, Jing and Yang, Jufeng and Tao, Dacheng}
  journal={IEEE Transactions on Affective Computing}, 
  title={Looking into Gait for Perceiving Emotions via Bilateral Posture and Movement Graph Convolutional Networks}, 
  year={2024}
}

````