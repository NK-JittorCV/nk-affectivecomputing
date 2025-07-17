# Dataset preparation

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
   └── srt (optional, only for MART)
       ├── Anger
            ├── 1.srt
            ├── 2.srt
            ├── 3.srt
            ├── ...
       ├── Anticipation
       ├── Disgust
       ├── Fear
       ├── Joy
       ├── Sadness
       ├── Surprise
       ├── Trust
~~~~

## Ekman6

The Ekman-6 dataset is a benchmark emotion classification dataset based on Paul Ekman’s theory of six basic emotions. It is widely used in affective computing and emotion recognition research. The dataset typically includes a large number of labeled samples—images, audio, or videos—annotated with one of six universal emotional categories: Anger, Disgust, Fear, Happiness, Sadness, and Surprise. These emotions are considered to be universally recognized across different cultures and contexts. Ekman-6 is commonly applied in tasks such as facial expression recognition, speech emotion recognition, and multimodal emotion analysis, enabling the development and evaluation of emotion-aware models and systems.

References：[ Heterogeneous Knowledge Transfer in Video Emotion Recognition, Attribution and Summarization](https://ieeexplore.ieee.org/document/7723914 )

The dataset structure is as follows：
~~~~
├── Ekman6
   └── img
       ├── Anger
            ├── 1
                ├── 000001.jpg
                ├── 000002.jpg
                ├── ...
            ├── 2
            ├── 3
            ├── ...
       ├── Disgust
       ├── Fear
       ├── Joy
       ├── Sadness
       ├── Surprise
   └── mp3
       ├── Anger
            ├── 1.mp3
            ├── 2.mp3
            ├── 3.mp3
            ├── ...
       ├── Disgust
       ├── Fear
       ├── Joy
       ├── Sadness
       ├── Surprise
   └── srt (optional, only for MART)
       ├── Anger
            ├── 1.srt
            ├── 2.srt
            ├── 3.srt
            ├── ...
       ├── Disgust
       ├── Fear
       ├── Joy
       ├── Sadness
       ├── Surprise
~~~~

## ERATO

ERATO (Emotional RelAtionship of inTeractiOn) is a large-scale, multimodal video dataset specifically designed for the Pairwise Emotional Relationship Recognition (PERR) task. Collected from drama and movie clips, ERATO comprises 31,182 video segments totaling approximately 203 hours, covering visual, audio, and textual modalities (dialogue/subtitles). Each clip focuses on the emotional relationship between two characters and is annotated with both fine-grained emotional labels (e.g., hostile, tense, intimate, neutral) and coarse-grained categories (positive, neutral, negative). With its diverse scenarios and rich annotations, ERATO offers a challenging benchmark for advancing multimodal emotion understanding and relationship inference in video analysis.

References：[ Pairwise Emotional Relationship Recognition in Drama Videos: Dataset and Benchmark](https://arxiv.org/pdf/2109.11243)

~~~~
├── ERATO
   └── img
       ├── Hostile
            ├── 1
                ├── 000001.jpg
                ├── 000002.jpg
                ├── ...
            ├── 2
            ├── 3
            ├── ...
       ├── Intimate
       ├── Mild
       ├── Neutral
       ├── tense
   └── mp3
       ├── Hostile
            ├── 1.mp3
            ├── 2.mp3
            ├── 3.mp3
            ├── ...
       ├── Intimate
       ├── Mild
       ├── Neutral
       ├── tense
   └── srt (optional, only for MART)
       ├── Hostile
            ├── 1.srt
            ├── 2.srt
            ├── 3.srt
            ├── ...
       ├── Intimate
       ├── Mild
       ├── Neutral
       ├── tense
~~~~

## Affwild2

Aff-Wild2 is one of the most comprehensive and large-scale "in-the-wild" datasets for affect recognition, serving as an extension to the original Aff-Wild dataset. While Aff-Wild initially comprised approximately 1.2 million video frames, Aff-Wild2 adds 260 new subjects and over 1.4 million additional frames, bringing the total to nearly 2.8 million frames. All videos are collected from YouTube, exhibiting extensive variability in pose, age, illumination, ethnicity, and profession, thereby ensuring high realism and diversity.

References：[ Aff-Wild2: Extending the Aff-Wild Database for Affect Recognition](https://arxiv.org/pdf/1811.07770)

~~~~
├──Affwild2
   └── img
       ├── Anger
            ├── 7-60-1920x1080_Train_46to175
                ├── 000001.jpg
                ├── 000002.jpg
                ├── ...
            ├── 7-60-1920x1080_Train_46to175
            ├── 7-60-1920x1080_Train_2337to2480
            ├── ...
       ├── Disgust
       ├── Fear
       ├── Happiness
       ├── Neutral
       ├── Sadness
       ├── Surprise
   └── mp3
       ├── Anger
            ├── 1.mp3
            ├── 2.mp3
            ├── 3.mp3
            ├── ...
       ├── Disgust
       ├── Fear
       ├── Happiness
       ├── Neutral
       ├── Sadness
       ├── Surprise
   └── srt (optional, only for MART)
       ├── Anger
            ├── 1.srt
            ├── 2.srt
            ├── 3.srt
            ├── ...
       ├── Disgust
       ├── Fear
       ├── Happiness
       ├── Neutral
       ├── Sadness
       ├── Surprise
~~~~

## IEMOCAP

The IEMOCAP dataset was designed to study expressive human communication. It includes recordings from multiple subjects performing scripted and spontaneous interactions, with the goal of capturing genuine emotional expressions.The main emotion categories annotated in the dataset are: happiness, anger, sadness, neutral, and frustration , which were later expanded to include disgust, fear, excitement, and surprise for more detailed emotion characterization, especially in spontaneous scenarios.Subjects performed in a controlled environment equipped with lighting and visual design, and their interactions were captured using multimodal sensors, including video, audio, text transcription, and motion capture.This dataset supports research in emotion recognition, speech-driven gesture synthesis, and expressive behavior modeling in virtual agents.

References：[IEMOCAP: Interactive emotional dyadic motion capture
database](https://sail.usc.edu/publications/files/bussolre2008.pdf)

~~~~
├──IEMOCAP
   └── img
       ├── Anger
            ├── Ses01F_impro01_F012
                ├── 000001.jpg
                ├── 000002.jpg
                ├── ...
            ├── Ses01F_impro01_M011
            ├── Ses01F_impro01_M013
            ├── ...
       ├── Happy
       ├── Neutral
       ├── Sadness
   └── mp3
       ├── Anger
            ├── 1.mp3
            ├── 2.mp3
            ├── 3.mp3
            ├── ...
       ├── Happy
       ├── Neutral
       ├── Sadness
   └── srt (optional, only for MART)
       ├── Anger
            ├── 1.srt
            ├── 2.srt
            ├── 3.srt
            ├── ...
       ├── Happy
       ├── Neutral
       ├── Sadness
~~~~


## TSL-300 dataset
TSL-300 dataset consists of 300 untrimmed videos with an average video time of 4.3 minutes. We filter out three types of abnormal videos to provide a high-quality and diverse dataset. The selected videos are annotated by four well-trained annotators from different backgrounds. In order to alleviate the burden of dense annotation, we study the weakly-supervised setting and the fully-supervised setting used for training a model. Therefore, the dataset contains two types of annotations: frame-by-frame annotations and single-frame annotations.
If you need the TSL-300 dataset for academic purposes, please download the [application form](./assests/TSL-300_Data_Access_Form.docx) and fill out the request information, then send it to ***gloryzzc6@sina.com***.
We will process your application as soon as possible.
Please make sure that the email used comes from your educational institution.

### Data Preparation
1. Prepare [TSL-300](./assests/TSL-300_Data_Access_Form.docx) dataset.
    - We have provided constructed dataset and pre-extracted features.

2. Extract features with two-stream I3D networks
    - We recommend extracting features using [this repo](https://github.com/piergiaj/pytorch-i3d).
    - For convenience, we provide the features we used which is available at [Baidu Drive](https://pan.baidu.com/s/1RHsm3d-ixMhmqmAJsTf3sQ?pwd=jj9w).
    - Link the features folder by using `sudo ln -s path-to-feature ./dataset/VideoSenti/`.
    
3. Place the features inside the `dataset` folder.
    - Please ensure the data structure is as below.

~~~~
├── dataset
   └── VideoSenti
       ├── gt.json
       ├── split_train.txt
       ├── split_test.txt
       ├── fps_dict.json
       ├── time.json
       ├── videosenti_gt.json
       ├── point_gaussian
           └── point_labels.csv
           ├── train
       └── features
           ├── train
               ├── rgb
                   ├── 1_Ekman6_disgust_3.npy
                   ├── 2_Ekman6_joy_1308.npy
                   └── ...
               └── logmfcc
                   ├── 1_Ekman6_disgust_3.npy
                   ├── 2_Ekman6_joy_1308.npy
                   └── ...
           └── test
               ├── rgb
                   ├── 9_CMU_MOSEI_lzVA--tIse0.npy
                   ├── 17_CMU_MOSEI_CbRexsp1HKw.npy
                   └── ...
               └── logmfcc
                   ├── 9_CMU_MOSEI_lzVA--tIse0.npy
                   ├── 17_CMU_MOSEI_CbRexsp1HKw.npy
                   └── ...
~~~~

## Emotion-Gait dataset

The used datasets are provided in the homepage of [Emotion-Gait](https://gamma.umd.edu/software). 

Emotion Gait dataset contains 2,177 real gaits and 1,000 synthetic gaits with four emotion classes: happiness, sadness, anger, and neutral. The real gaits consist of 1,835 samples collected from an existing motion capture database and 342 gaits collected by the authors. Each gait in the 1,835 samples has 240 frames, while the 342 samples have flexible numbers of frames ranging from 27 to 75. The 3D skeletal data of the gaits from videos are extracted by a representative pose estimation method. All real gaits are labeled by domain experts. The synthetic gaits are generated from a trained auto-encoder that has been fed with emotion labels. In this work, we only use the 2,177 real gaits.

Note that since the dataset is changed on the official website, we provide the original dataset. The code and dataset are provided for research only.
[Baidu Drive](https://pan.baidu.com/s/1rNC7SQrwNnZBVMRzPaifZA) (acil)

Please ensure the data structure is as below.

~~~~
BPM_GCN/
├── test_affective.npy
├── test_joint.npy
├── test_label.pkl
├── test_movement.npy
├── train_affective.npy
├── train_joint.npy
├── train_label.pkl
└── train_movement.npy
~~~~
