## FAQ Collection

Below are some common issues encountered when using the Jittor version of the code.

## Training

### Q1: In the VAANet and CTEN projects, why are the emotion classification results inconsistent between two runs of `test.py` on the same video dataset?

There are two main reasons:

1. **Random frame sampling** for each video.
2. **Different preprocessing methods** applied to the sampled frames.

#### Frame Sampling

The following line of code is used for temporal segmentation sampling. The parameter `seq_len` divides the video into `seq_len` equal temporal segments. Within each segment, frames are sampled.

The parameter `snippet_duration` determines how many consecutive frames are sampled from each segment to form a snippet.

The parameter `center` controls how the frames are sampled from each segment:
- If `center=True`, the center snippet of each segment is used (deterministic sampling, suitable for validation/testing).
- If `center=False`, the snippet is randomly selected (data augmentation, suitable for training).

```text
temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=True)
```
Next, we introduce the spatial preprocessing transformation methods for video frames. The following code is located in `core/utils.py`, where `is_aug` indicates whether to use data augmentation, and `center` indicates whether to perform center cropping. Generally, the processing strategies under different modes are as follows:

| Processing Method | Training Mode | Validation Mode | Test Mode |
|------------------|--------------|-----------------|-----------|
| Data Augmentation (is_aug) | Typically Enabled (True) | Disabled (False) | Disabled (False) |
| Center Cropping (center) | Typically Enabled (True) | Enabled (True) | Disabled (False) |
| Size Transformation | Enabled | Enabled | Enabled |

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

### Q2: How to evaluate the consistency and performance between the modified Jittor version and the original Torch version?

Consistency is evaluated using sentiment classification results on the dataset. Performance is compared through three metrics:
1. Average forward propagation time
2. Average memory usage
3. Single training epoch time (excluding data loading)

For specific metrics, please refer to the readme.md document.

### Q3: After modifying the Jittor version code, many parameters fail to load during execution. What could be the reasons?

There are two main potential causes:
1. The saved parameter names don't match the names expected by the current model
2. The model structure in the current code doesn't match the structure of the saved model parameters

For example:
- When using PyTorch's DataParallel or DistributedDataParallel (DDP) for multi-GPU training, parameter names automatically get a `module.` prefix (e.g., `ta_net.conv.weight` becomes `module.ta_net.conv.weight`)
- Jittor's model doesn't have this prefix, which causes loading failures

## PyTorch to Jittor Conversion: Frequently Asked Questions (FAQ)

### Q1. Which basic PyTorch operations does Jittor support?

**Answer:** Jittor supports most common tensor operations such as `matmul`, `conv2d`, `relu`, `batchnorm`, `dropout`, `softmax`, `cross_entropy`, `reshape`, `permute`, `view`, etc. You can find alternative implementations in the [Jittor API documentation](https://cg.cs.tsinghua.edu.cn/jittor/api/).

---

### Q2. How to convert `torch.nn.Module` to a Jittor model?

**Answer:** Replace the base class `torch.nn.Module` with `jittor.nn.Module` and rewrite the `__init__` and `execute` (equivalent to PyTorch's `forward`) methods using Jittor's functions. For example:

```python
# PyTorch implementation
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
    def forward(self, x):
        return self.linear(x)

# Jittor implementation
class Net(jittor.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = jittor.nn.Linear(10, 5)
    def execute(self, x):
        return self.linear(x)
```

---
### Q3. What's the difference between Jittor's tensors and PyTorch's `tensor`?

**Answer:**
- PyTorch uses `torch.tensor`, Jittor uses `jt.array`.
- Jittor's basic type is `Var`, corresponding to Torch's `Tensor`.
- The data structures are similar, but Jittor supports dynamic/static graph switching by default and records all computation graph operations.
- Jittor uses `float32` by default, precision can be specified with `dtype=jt.float64`.
- Jittor can use `jittor.array(data)` to convert from numpy type to `Var` type.

---
### Q4. How to perform gradient backpropagation and parameter optimization?

**Answer:**
The methods for backpropagation and optimization in Jittor are slightly different, mainly in loss usage. Example code:

```python
# PyTorch
# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# Backpropagation
optimizer.zero_grad()
loss.backward()
optimizer.step()


# Jittor
# Define optimizer
optimizer = jt.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), weight_decay=0.0005)
# Backpropagation
optimizer.zero_grad()
optimizer.backward(loss)
optimizer.step()
```

### Q5. How to load datasets in Jittor?

**Answer:**
In Jittor, use jittor.dataset.Dataset as base class to define datasets, and override __len__(), __init()__, __getitem()__ methods. Example code:

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
Jittor's data loader DataLoader is in jittor.dataset, import with from jittor.dataset import DataLoader, usage is similar to torch.

### Q6. How to load pretrained models or save models in Jittor?

**Answer:**
Saving and loading pretrained models in Jittor is similar to torch, with same-named functions (just replace torch with jittor):

```python
# Save model parameters
jt.save(model.state_dict(), "model.pth")

# Load parameters
model.load_parameters(jt.load("model.pth"))
```

Jittor supports loading .pth format pretrained models, including those trained with torch.

### Q7. How to handle PyTorch's with torch.no_grad() in Jittor?
**Answer:**: Jittor implements similar functionality with jt.no_grad(), used to disable gradients during testing:

```python
with jt.no_grad():
    output = model(x)
```


### Q8. How to use GPU computation in Jittor? Is .cuda() needed?
**Answer:**
Jittor automatically detects and prioritizes GPU usage. It performs unified memory management, no need to manually call .cuda(), .cpu(), .to(device) etc. To explicitly set device:

```python
jt.flags.use_cuda = 1  # Enable GPU
jt.flags.use_cuda = 0  # Use CPU
```

### Q9. What to do when encountering PyTorch-specific functions during migration?
**Answer:**
Recommended solutions:
```python
Check if Jittor has corresponding API

Manually implement alternative logic

Check community resources or GitHub repositories for migration solutions

For missing functionality, write C++/CUDA plugins using custom jt.Function
```
