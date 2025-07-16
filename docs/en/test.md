# Testing instructions

## CTEN
```
bash script/run.sh 
CTEN \
test \
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
bash script/run.sh 
VAANet \
test \
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
## Gait
