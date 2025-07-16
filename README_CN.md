<div align="center">

# 基于 Jittor 的情感计算模型

</div>

## 简介

这是一个 GitHub 仓库，包含四个基于 Jittor 深度学习框架的情感计算方向的工作：

- [TSL_Net](./TSL_Net/README.md)
- [CTEN](./CTEN/README.md)
- [VAANet](./VAANet/README.md)
- [Gait](./Gait/README.md)

## 环境依赖

#### 推荐环境

* Python 3.8.0
* jittor 1.3.9.14
* CUDA 11.3

## 所需依赖

* jittor==1.3.9.14
* numpy==1.22.0
* pandas==2.0.3
* Pillow==10.0.0
* protobuf==4.24.1
* scikit-learn==1.3.0
* scipy==1.10.1
* tokenizers==0.13.3
* tqdm==4.66.1
* transformers==4.31.0

## 运行方式

你可以通过运行以下脚本轻松地训练和评估模型。其中，方法可以选择"BPM_GCN"、"TSL"、"CTEN"或者"VAANet"。
```text
bash script/run.py {方法} {main/test} {batch_size}
```

例如：
```text
bash script/run.py CTEN main 16
```


