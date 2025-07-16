<div align="center">

# 基于 Jittor 的情感计算模型

</div>

## 简介

这是一个 GitHub 仓库，包含四个基于 Jittor 深度学习框架的情感计算方向的工作：

- [TSL_Net](./TSL_Net/README.md)
- [CTEN](./CTEN/README.md)
- [VAANet](./VAANet/README.md)
- [Gait](./Gait/README.md)



## 运行方式

你可以通过运行以下脚本轻松地训练和评估模型。其中，方法可以选择"BPM_GCN"、"TSL"、"CTEN"或者"VAANet"。
```text
bash script/run.py {方法} {main/test} {batch_size}
```

例如：
```text
bash script/run.py CTEN main 16
```


