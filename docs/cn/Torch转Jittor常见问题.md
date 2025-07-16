## 常见问题整理

下面是jittor版本代码使用过程中遇到的一些常见问题。

## 训练

### Q1: 在VAANet和CTEN项目中使用test.py前后两次生成的视频数据集情感分类类别不一致是为什么？
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
为了全面评估 Jittor 版本与 Torch 版本代码的一致性和性能表现，可从两个方面进行分析：在一致性方面，应对比相同输入下两者输出的 logits 或分类结果。若使用 logits 进行评估，可计算其 L2 距离或最大绝对误差；若采用分类结果进行评估，则可统计两者分类结果不一致的样本数量。此外，对于训练过程的对比，可以通过观察 loss 曲线的趋势进行判断。


### Q3:修改jittor版本的代码之后，运行的时候发现有很多参数加载失败是什么原因？
主要考虑两个方面的原因。一方面是保存的参数名称与当前模型期望的名称不一致；另一方面可能是当前代码中的模型结构与保存的模型参数结构不一致。

例如，在 PyTorch 中使用 DataParallel 或 DistributedDataParallel（DDP） 进行多卡训练时，模型参数名称会自动添加 module. 前缀（例如 ta_net.conv.weight → module.ta_net.conv.weight）。而 Jittor 的模型没有这个前缀，导致加载失败。

## PyTorch 转 Jittor 常见问题与解答（FAQ）

### Q1. Jittor 支持 PyTorch 的哪些基本操作？

**答：** Jittor 支持大多数常见的张量操作，如 `matmul`, `conv2d`, `relu`, `batchnorm`, `dropout`, `softmax`, `cross_entropy`, `reshape`, `permute`, `view` 等。可通过查阅 [Jittor API 文档](https://cg.cs.tsinghua.edu.cn/jittor/api/) 获取对应函数的替代实现。

---

### Q2. 如何将 `torch.nn.Module` 转换为 Jittor 模型？

**答：** 继承的基类把 `torch.nn.Module` 替换成 `jittor.nn.Module` ，并使用jittor.nn中的函数重写 `__init__` 和 `execute`  `(即forward方法)` 方法，替换掉对应的torch.nn中的函数。例如：

```python
# PyTorch 写法
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
    def forward(self, x):
        return self.linear(x)

# Jittor 写法
class Net(jittor.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = jittor.nn.Linear(10, 5)
    def execute(self, x):
        return self.linear(x)
```

---

### Q3. Jittor 的张量与 PyTorch 的 `tensor` 有什么区别？

**答：**
- PyTorch 使用 `torch.tensor`，Jittor 使用 `jt.array` 。
- Jittor 的基础类型是 `Var`，对应 Torch 里边的 `Tensor`。
- 两者的数据结构类似，但 Jittor 默认支持动态图与静态图切换，所有计算图操作默认记录。
- Jittor 默认使用 `float32` 类型，可通过 `dtype=jt.float64` 等指定精度。
- Jittor 可以使用 `jittor.array(data)` 从np类型转成 `Var` 类型。

---

### Q4. 如何进行梯度反向传播和参数优化？

**答：**
Jittor中反向传播和参数优化的方法略有区别，主要是在loss的使用上，示例程序如下所示：

```python
# PyTorch
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# 反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()


# Jittor
# 定义优化器
optimizer = jt.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), weight_decay=0.0005)
# 反向传播
optimizer.zero_grad()
optimizer.backward(loss)
optimizer.step()
```
---
### Q5. 使用Jittor如何加载数据集？

**答：**
Jittor中使用jittor.dataset.Dataset作为基类定义数据集，并重写__len__(),__init()__,__getitem()__ 等方法，示例代码如下所示：

```python
import jittor.dataset as data
class SentiFeature(data.Dataset):
    def __init__(self, data_path, mode, modal, feature_fps, num_segments, sampling, seed=-1, supervision='point'):
        super().__init__()
    def __len__(self):
        return len(self.vid_list)
    def __getitem__(self, index):
        return data[index]
```
jittor的数据加载器DataLoder位于 jittor.dataset 中 使用 from jittor.dataset import DataLoader导入，使用方式和torch类似。


---

### Q6. 使用Jittor如何加载预训练模型或保存模型？

**答：**
jittor中保存和加载预训练模型的方法和torch中类似，jittor中具有同名的函数，只需要把torch替换成jittor即可。
```python
# 保存模型参数
jt.save(model.state_dict(), "model.pth")

# 加载参数
model.load_parameters(jt.load("model.pth"))
```
Jittor 支持加载 `.pth` 格式的预训练模型，也支持使用torch训练的预训练模型参数的加载`(.pth格式)`。

---

### Q7. PyTorch 的 `with torch.no_grad()` 在 Jittor 中如何处理？

**答：** Jittor 通过 `jt.no_grad()` 实现类似功能，用于测试阶段关闭梯度：

```python
with jt.no_grad():
    output = model(x)
```

---

### Q8. Jittor中如何使用 GPU 运算？是否需要 `.cuda()`？

**答：**
Jittor 自动检测 GPU 并优先使用。Jittor会进行统一内存管理，不需要在手动调用 `.cuda()`,`.cpu()`,`.to(device)` 等函数，如果需要明确设置设备，可以使用：

```python
jt.flags.use_cuda = 1  # 开启 GPU
jt.flags.use_cuda = 0  # 使用 CPU
```

---


### Q9. 迁移时遇到 PyTorch 特有的函数怎么办？

**答：**
建议采取以下方案：
- 查找 Jittor 是否有对应 API；
- 手动实现替代逻辑；
- 查阅社区资源或 GitHub 仓库寻找迁移方案；
- 如功能缺失，可通过自定义 `jt.Function` 编写 C++/CUDA 插件。

---


