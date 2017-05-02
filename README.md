# cpp.CS599-ML

## Python libs
* librosa

## Feature extraction
```
python audio_feature.py [data_path] [label]
```
Extract audio features of *.wav in folder [data_path] and tag them with [label].   
The features are stored in [data_path]/features.csv file. File formatted as:
* First row is header
* First column is category name
* The rest columns are features