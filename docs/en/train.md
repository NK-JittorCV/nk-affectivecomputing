# Training instructions

## CTEN

```
bash script/run.sh CTEN main \
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
- dataset: is the name of the dataset to be used. Please refer to docs/cn/dataset preparation.md for relevant instructions.
- resnet101_pretrained: pre-trained image model weight path (such as ResNet-101); used for video frame feature extraction.
- result_path: inference or training result save path.
video_path: video frame sequence or video file path.
audio_path: corresponding video and audio file (such as .mp3) path.
- annotation_path: annotation file path, annotation file for training or testing and corresponding sentiment classification.
- batch_size: batch size: the number of samples fed into the model each time.
- n_epochs: total number of training rounds.
sample_size: input size (width and height) of video frame image.
- fps: video frame rate.
- snippet_duration: the number of frames per clip.
- audio_embed_size: audio feature dimension size.
- audio_n_segments: divide the entire audio into 16 segments, extract an embedding for each segment; corresponding frame alignment.
- audio_time: The duration of each audio sample.


The first two parameters are mandatory: CTEN specifies the method, and main indicates the training mode. All subsequent parameters are optional and can be adjusted according to actual requirements or conditions.


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
- dataset: is the name of the dataset to be used. Please refer to docs/cn/dataset preparation.md for relevant instructions.
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

The first two parameters are mandatory: VAANet specifies the method, and main indicates the training mode. The remaining parameters are optional and can be adjusted based on actual conditions.

## TSL-Net

```
bash script/run.sh TSL main \
--data_path ./src/TSL/dataset/VideoSenti \
--model_path ./src/TSL/models/train \
--output_path ./src/TSL/outputs/train \
--log_path ./src/TSL/logs/train \
--modal all \
--lambdas "[2,1,0.5,1]" \
--r_act 8 \
--class_th 0.5 \
--lr "[0.00001]*2000" \
--batch_size 16 \
--num_workers 8 \
--seed 123 \
--model_file ./src/TSL/models/train/model_seed_123.pth
```
### Arguments
- data_path: the path to the dataset used for training or evaluation.
- model_path: the directory where model checkpoints will be saved during training.
- output_path: the path to store output files such as prediction results or visualizations.
- log_path: the path to store log files that record training or testing details.
- modal: the input modality to be used. Options are 'rgb' (video only), 'logmfcc' (audio only), or 'all' (video + audio).
- lambdas: the weights for different parts of the loss function, specified as a list. For example, [2, 1, 0.5, 1].
- r_act: the radius parameter for activation regions, typically used for Gaussian kernels or region-based activation.
- class_th: the classification threshold for converting predicted probabilities to labels.
- lr: the learning rate schedule, specified as a list. For example, [0.00001]*2000.
- batch_size: the batch size used during training.
- num_workers: the number of subprocesses used by the DataLoader to load data in parallel.
- seed: the random seed for reproducibility. If set to -1, no manual seed will be applied.
- model_file: the path to a pre-trained model file. If None, no pre-trained weights will be loaded.

## Gait

```
bash script/run.sh Gait main \
--work-dir ./work_dir/temp \
--model_saved_name 'model_best' \
--config ./src/Gait/config/EGait_journal/train_diff_combine_double_score_fagg.yaml \
--phase train \
--seed 1 \
--log-interval 100 \
--save-interval 2 \
--eval-interval 5 \
--print-log True \
--num-worker 20 \
--train_ratio 0.9 \
--test_ratio 0.1 \
--base-lr 0.01 \
--step 20 40 60 \
--device 0 \
--optimizer SGD \
--batch_size 16 \
--start-epoch 0 \
--num-epoch 80 \
--weight_decay 0.0005
```
### Arguments
- work-dir: the working directory to store results such as logs, models, and outputs.
- model_saved_name: the name used to save the model file.
- config: the path to the configuration file containing model and training settings.
- phase: the execution phase, must be 'train' or 'test'.
- seed: the random seed used for PyTorch to ensure reproducibility.
- log-interval: the number of iterations between logging training messages.
- save-interval: the number of epochs between saving model checkpoints.
- eval-interval: the number of epochs between evaluations on the validation or test set.
- print-log: whether to print logs to the console (True/False).
- num-worker: the number of worker processes for data loading.
- train_ratio: the ratio of data used for training.
- test_ratio: the ratio of data used for testing.
- base-lr: the initial learning rate.
- step: a list of epochs at which the learning rate is decayed.
- device: GPU indexes used for training or testing (e.g., 0 or 0 1 for multi-GPU).
- optimizer: the optimizer type (e.g., SGD, Adam).
- batch_size: the number of samples per training batch.
- start-epoch: the epoch number to start training from.
- num-epoch: the total number of epochs for training.
- weight_decay: the weight decay (L2 penalty) used by the optimizer.
