# VRDL-HW1
Code for Selected Topics in Visual Recognition using Deep Learning Homework 1

## Hardware
The following specs were used to create the original solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
- NVIDIA GTX 1080ti

## Reproducing Submission
To reproduct my submission, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
2. [Training](#training)
3. [Inference](#inference)
4. [Make Submission](#make-submission)

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n hpa python=3.6
source activate hpa
pip install -r requirements.txt
```

## Dataset Preparation
All required files except images are already in data directory.
If you generate CSV files (duplicate image list, split, leak.. ), original files are overwritten. The contents will be changed, but It's not a problem.

### Prepare Images
After downloading and converting images, the data directory is structured as:
```
data
  +- raw
  |  +- train
  |  +- test
  |  +- external
  +- rgby
  |  +- train
  |  +- test
  |  +- external
```

#### Download Official Image
Download and extract *train.zip* and *test.zip* to *data/raw* directory.
If the Kaggle API is installed, run following command.
```
$ kaggle competitions download -c human-protein-atlas-image-classification -f train.zip
$ kaggle competitions download -c human-protein-atlas-image-classification -f test.zip
$ mkdir -p data/raw
$ unzip train.zip -d data/raw/train
$ unzip test.zip -d data/raw/test
```

## Training
In configs directory, you can find configurations I used train my final models. My final submission is ensemble of resnet34 x 5, inception-v3 and se-resnext50, but ensemble of inception-v3 and se-resnext50's performance is better.

### Search augmentation
To find suitable augmentation, 256x256 image and resnet18 are used.
It takes about 2 days on TitanX. The result(best_policy.data) will be located in *results/search* directory.
The policy that I used is located in *data* directory.
```
$ python train.py --config=configs/search.yml
```

### Train models
To train models, run following commands.
```
$ python train.py --config={config_path}
```
To train all models, run `sh train.sh`

The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time
------------ | ------------- | ------------- | ------------- | -------------
resnet34 | 1x TitanX | 512 | 40 | 16 hours
inception-v3 | 3x TitanX | 1024 | 27 | 1day 15 hours
se-resnext50 | 2x TitanX | 1024 | 22 | 2days 15 hours

### Average weights
To average weights, run following commands.
```
$ python swa.py --config={config_path}
```
To average weights of all models, simply run `sh swa.sh`
The averages weights will be located in *results/{train_dir}/checkpoint*.

## Inference
If trained weights are prepared, you can create files that contains class probabilities of images.
```
$ python inference.py \
  --config={config_filepath} \
  --num_tta={number_of_tta_images, 4 or 8} \
  --output={output_filepath} \
  --split={test or test_val}
```
To make submission, you must inference test and test_val splits. For example:
```
$ python inference.py --config=configs/resnet34.0.policy.yml --num_tta=8 --output=inferences/resnet34.0.test_val.csv --split=test_val
$ python inference.py --config=configs/resnet34.0.policy.yml --num_tta=8 --output=inferences/resnet34.0.test.csv --split=test
```
To inference all models, simply run `sh inference.sh`

## Make Submission
Following command will ensemble of all models and make submissions.
```
$ python make_submission.py
```
If you don't want to use, modify *make_submission.py*.
For example, if you want to use inception-v3 and se-resnext50 then modify *test_val_filenames, test_filenames and weights* in *make_submission.py*.
```
test_val_filenames = ['inferences/inceptionv3.0.test_val.csv',
                      'inferences/se_resnext50.0.test_val.csv']
                      
test_filenames = ['inferences/inceptionv3.0.test.csv',
                  'inferences/se_resnext50.0.test.csv']
                  
weights = [1.0, 1.0]
```
The command generate two files. One for original submission and the other is modified using data leak.
- submissions/submission.csv
- submissions/submission.csv.leak.csv
