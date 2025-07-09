## Framework
This project is built on the Jittor deep learning framework.

Jittor: A deep learning framework based entirely on Just-in-time compilation, using innovative meta-operators and unified computing graphs internally.

You can set up the environments by using `pip3 install -r requirements.txt`.

## Performance

| Metric             | PyTorch | Jittor  |
|--------------------|---------|---------|
| Average Forward Time (train) | 0.22s  | 0.13s  |
| Average Memory Usage (train) | 12380MB  | 15673MB  |
| Single epoch Time(train) | 172s | 468s |
| Average_Acc(test)  | 49.1%  | 26.9%  |

The benchmark data presented above was acquired under single-GPU testing conditions using an NVIDIA GeForce RTX 3090.

## Dependencies

#### Recommended Environment

* Python 3.8.0
* jittor 1.3.9.14
* CUDA 11.3

## Requirements_jittor
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



## begin

The pre-trained model can be found in [pretrained model]( https://pan.baidu.com/s/1RcLG4CJNgkAs9-MEoxtgnQ?pwd=7gst).

You can train and eval the model by running the command below.

~~~~
$ python main_erase.py
~~~~
