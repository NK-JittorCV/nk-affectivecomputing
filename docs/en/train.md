# Training instructions

## CTEN

```
bash script/run.sh CTEN main --dataset ve8 --resnet101_pretrained your_path --video_path your_path --annotation_path src/CTEN/data/ve8_04.json --audio_path your_path --result_path your_path --batch_size 4 --n_epochs 100 --sample_size 112 --fps 30 --snippet_duration 16 --audio_embed_size 2048 --audio_n_segments 16 --audio_time 100 --learning_rate 1e-4

```
At least the first two parameters are required, where CTEN represents the method and main represents the train. The following parameters are optional and can be modified based on actual conditions.
## VAANet

```
bash script/run.sh VAANet main --dataset ve8 --resnet101_pretrained your_path --video_path your_path --annotation_path src/CTEN/data/ve8_04.json --audio_path your_path --result_path your_path --batch_size 4 --n_epochs 100 --sample_size 112 --fps 30 --snippet_duration 16 --learning_rate 8e-4

```
At least the first two parameters are required, where VAANet represents the method and main represents the train. The following parameters are optional and can be modified based on actual conditions.
