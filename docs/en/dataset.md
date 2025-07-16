# 数据集准备

## VideoEmotion-8 

The VideoEmotion-8 dataset is a benchmark dataset designed for video-level emotion recognition, containing 1,101 user-generated videos collected from online video platforms. Each video is annotated with one of eight discrete emotion categories: Anger, Anticipation, Disgust, Fear, Joy, Sadness, Surprise, and Trust, based on established psychological emotion models. The dataset captures a wide range of real-world scenarios and emotional expressions, providing a challenging and diverse resource for research in affective computing, video understanding, and multimodal sentiment analysis. VideoEmotion-8 has been widely used to evaluate models that integrate visual, audio, and textual features for holistic emotion classification in videos.

References：[Predicting Emotions in User-Generated Videos](https://cdn.aaai.org/ojs/8724/8724-13-12252-1-2-20201228.pdf )

The dataset structure is as follows：
~~~~
├── VideoEmotion-8 
   └── img
       ├── Anger
            ├── 1
                ├── 000001.jpg
                ├── 000002.jpg
                ├── ...
            ├── 2
            ├── 3
            ├── ...
       ├── Anticipation
       ├── Disgust
       ├── Fear
       ├── Joy
       ├── Sadness
       ├── Surprise
       ├── Trust
   └── mp3
       ├── Anger
            ├── 1.mp3
            ├── 2.mp3
            ├── 3.mp3
            ├── ...
       ├── Anticipation
       ├── Disgust
       ├── Fear
       ├── Joy
       ├── Sadness
       ├── Surprise
       ├── Trust



~~~~
