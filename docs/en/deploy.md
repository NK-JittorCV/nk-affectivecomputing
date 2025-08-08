# Inference a single video
## TSL-Net
To perform inference on a single video using **TSL-Net**, you must first preprocess the video, which includes both video and audio feature extraction:

- **Video Feature Extraction**:  
  RGB features are extracted using the I3D model. We use the pre-trained model and implementation from [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d).

- **Audio Feature Extraction**:  
  We use the most well-known descriptor Mel-Frequency cepstral coefficients (MFCC).

The extracted **RGB** and **MFCC** features are saved in `.npy` format.

For detailed preprocessing steps, please refer to the [inference.sh](../../src/TSL/inference.sh).

Once the features are prepared, run the inference using the provided inference code.

```
bash script/run.sh TSL main_inference \
--rgb_path ./src/TSL/tmp/rgb/40_VideoEmotion8_Anger_88.npy \
--mfcc_path ./src/TSL/tmp/mfcc/40_VideoEmotion8_Anger_88.npy \
--output_path ./src/TSL/outputs/inference \
--log_path ./src/TSL/logs/inference \
--modal all \
--seed 123 \
--model_file ./src/TSL/models/train/model_seed_123.pth
```

### Arguments
- rgb_path: Video features in .npy format extracted using I3D.
- mfcc_path: Audio features in .npy format extracted using MFCC.
- output_path: the path to store output files such as prediction results or visualizations.
- log_path: the path to store log files that record testing details.
- modal: the input modality option. Choices are 'rgb' (video only), 'logmfcc' (audio only), or 'all' (video + audio).
- seed: the random seed for reproducibility. If set to -1, no manual seed will be applied.
- model_file: the path to the pre-trained model file.
