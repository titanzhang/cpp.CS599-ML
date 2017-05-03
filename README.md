# cpp.CS599-ML

## Python libs
* librosa
* numpy, scipy, sklearn, matplotlib
* tensorflow
* keras

## Feature extraction
```
python audio_feature.py [data_path] [label]
```
Extract audio features of *.wav in folder [data_path] and tag them with [label].   
The features are stored in [data_path]/features.csv file. File formatted as:
* Last column is category name
* The rest columns are features
Also a header file is saved in [data_path]/header.csv to explain what features are there
in features.csv

## Training/Testing
```
python train.py [data_file]
```
The [data_file] has to be formatted as
* Last column is category name
* Other columns are features