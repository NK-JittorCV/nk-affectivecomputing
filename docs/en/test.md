# Testing instructions

## CTEN
```
bash script/run.sh CTEN test \
--dataset ve8 \
--resnet101_pretrained your_path \
--video_path your_path your_path \
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

### Arguments
- dataset: Specify the name of the dataset to be used. For details, refer to docs/cn/dataset.md.
- resnet101_pretrained: pre-trained image model weight path (such as ResNet-101); used for video frame feature extraction.
- result_path: inference or training result save path.
- video_path: video frame sequence or video file path.
- audio_path: corresponding video and audio file (such as .mp3) path.
- annotation_path: annotation file path, annotation file for training or testing and corresponding sentiment classification.
- batch_size: batch size: the number of samples fed into the model each time.
- n_epochs: total number of training rounds.
- sample_size: input size (width and height) of video frame image.
- fps: video frame rate.
- snippet_duration: the number of frames per clip.
- audio_embed_size: audio feature dimension size.
- audio_n_segments: divide the entire audio into 16 segments, extract an embedding for each segment; corresponding frame alignment.
- audio_time: The duration of each audio sample.

The first two parameters are mandatory: CTEN specifies the method, and test indicates the testing mode. The remaining parameters are optional and can be adjusted according to actual requirements or conditions.


## VAANet

```
bash script/run.sh VAANet test \
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
--snippet_duration 16 
```

### Arguments
- dataset: Specify the name of the dataset to be used. For details, refer to docs/cn/dataset.md.
- resnet101_pretrained: pre-trained image model weight path (such as ResNet-101); used for video frame feature extraction.
- result_path: inference or training result save path.
- video_path: video frame sequence or video file path.
audio_path: corresponding video and audio file (such as .mp3) path.
- annotation_path: annotation file path, annotation file for training or testing and corresponding sentiment classification.
- batch_size: batch size: the number of samples fed into the model each time.
- n_epochs: total number of training rounds.
- sample_size: input size (width and height) of video frame image.
- fps: video frame rate.
- snippet_duration: the number of frames per clip.

The first two parameters are mandatory: VAANet specifies the method, and test indicates the testing mode. The remaining parameters are optional and can be adjusted based on actual conditions.


## TSL-Net
```
bash script/run.sh TSL main_eval \
--data_path ./src/TSL/dataset/VideoSenti \
--output_path ./src/TSL/outputs/test \
--log_path ./src/TSL/logs/test \
--modal all \
--r_act 8 \
--class_th 0.5 \
--batch_size 16 \
--num_workers 8 \
--seed 123 \
--model_file ./src/TSL/models/train/model_seed_123.pth
```

### Arguments
- data_path: the path to the dataset used for evaluation.
- output_path: the path to store output files such as prediction results or visualizations.
- log_path: the path to store log files that record testing details.
- modal: the input modality option. Choices are 'rgb' (video only), 'logmfcc' (audio only), or 'all' (video + audio).
- r_act: the radius parameter for activation regions, typically used for Gaussian kernels or region-based activations.
- class_th: the classification threshold for converting predicted probabilities to labels.
- batch_size: the batch size used during testing.
- num_workers: the number of subprocesses used by the DataLoader to load data in parallel.
- seed: the random seed for reproducibility. If set to -1, no manual seed will be applied.
- model_file: the path to the pre-trained model file.

## Gait
```
bash script/run.sh Gait main \
--work-dir ./work_dir/temp \
--config ./src/Gait/config/EGait_journal/train_diff_combine_double_score_fagg.yaml \
--phase test \
--seed 1 \
--print-log True \
--num-worker 20 \
--test_ratio 0.1 \
--device 0 \
--batch_size 16 \
--weights ./src/Gait/egait_runs/runs_diff_combine_fagg/fagg_att_cascade_st/model_epoch_best.pth
```
### Arguments

- work-dir: the working directory used to store logs, models, and output results.
- config: the path to the configuration file containing model and training settings.
- phase: the execution phase, must be 'train' (training) or 'test' (testing).
- seed: the random seed used by PyTorch for reproducibility.
- print-log: whether to print logs to the console (True/False).
- num-worker: the number of subprocesses used for data loading.
- test_ratio: the proportion of data used for testing.
- device: the GPU indexes used for training or testing (e.g., 0 for single GPU, 0 1 for multi-GPU).
- batch_size: the number of samples per training batch.
- weights: the path to the pre-trained model file to load.

