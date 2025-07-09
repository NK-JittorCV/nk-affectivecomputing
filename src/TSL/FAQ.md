# 常见问题整理

下面是jittor版本代码使用过程中遇到的一些常见问题。

## 训练

### Q1: 使用test.py前后两次生成的视频数据集情感分类类别不一致是为什么？
总结有两条原因：一是对每条视频的随机取帧，或者是对取到的帧采用不同的预处理方法。

首先介绍取帧，下面这行代码用于视频时序分段采样，seq_len参数的作用是将视频均匀分成 seq_len 个时间段（segments），每个段内再采样帧。snippet_duration参数的作用是从每个 segment 中连续采样 snippet_duration 帧，形成一个小片段（snippet）。center参数的作用是控制从每个 segment 中如何选取 snippet_duration 帧。当center=True时，取每段的中间连续帧（确定性采样，适用于测试/验证）；center=False时，随机位置采样（数据增强，适用于训练）。

```text
temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=True)
```


接下来介绍视频帧空间预处理变换方法。下面这段代码位于core/utils.py中，其中，is_aug代表是否使用数据增强，center代表是否进行中心化处理。一般情况下，不同模式下的处理策略如下：

|处理方式  |训练模式 |验证模式|验证模式|
|---------|---------|---------|---------|
|数据增强(is_aug)| 通常启用(True) |禁用(False)|禁用(False)|
|中心化(center) |通常启用(True) |启用(True)|禁用(False)|
|尺寸变换| 启用 |启用|启用|
||||

```text
def get_spatial_transform(opt, mode):
    if mode == "train":
        return Preprocessing(size=opt.sample_size, is_aug=False, center=True)
    elif mode == "val":
        return Preprocessing(size=opt.sample_size, is_aug=False, center=True)
    elif mode == "test":
        return Preprocessing(size=opt.sample_size, is_aug=False, center=True)
    else:
        raise Exception
```

### Q2: 如何评估修改后的Jittor版本的代码与Torch版本的代码的一致性和性能表现？
一致性使用对数据集的情感分类结果评估；性能表现通过平均前向传播时间、平均占用内存、单个训练epoch时间（不包括加载数据）三个指标比较。具体指标可以参考readme.md文档。

### Q3:修改jittor版本的代码之后，运行的时候发现有很多参数加载失败是什么原因？
主要考虑两个方面的原因。一方面是保存的参数名称与当前模型期望的名称不一致；另一方面可能是当前代码中的模型结构与保存的模型参数结构不一致。

例如，在 PyTorch 中使用 DataParallel 或 DistributedDataParallel（DDP） 进行多卡训练时，模型参数名称会自动添加 module. 前缀（例如 ta_net.conv.weight → module.ta_net.conv.weight）。而 Jittor 的模型没有这个前缀，导致加载失败。
