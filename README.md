<div align="center">

# Affective Computing Models based on Jittor

</div>

## Introduction

This is a GitHub repository containing four works in the field of affective computing based on the Jittor deep learning framework:

- [TSL_Net](./TSL_Net/README.md)
- [CTEN](./CTEN/README.md)
- [VAANet](./VAANet/README.md)
- [Gait](./Gait/README.md)





## Running

You can easily train and evaluate the model by running the script below.Below are some common issues encountered during the use of the Jittor version code. Among them, the methods can be "BPM_GCN", "TSL", "CTEN", or "VAANet".

```text
bash script/run.py {method} {main/test} {batch_size}
```
For example:
```text
bash script/run.py CTEN main 16
```
