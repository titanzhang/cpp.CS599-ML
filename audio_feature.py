import glob
import os
import sys
import librosa
import numpy as np
import csv

def extract_feature(file_name, seconds=None):
  mfccs = []
  chroma = []
  mel = []
  contrast = []
  tonnetz = []

  X, sample_rate = librosa.load(file_name)
  start = 0
  if seconds is None:
    duration = len(X)
  else:
    duration = sample_rate * seconds
  while start < len(X):
    end = min(start + duration, len(X))

    stft = np.abs(librosa.stft(X[start:end]))
    mfccs.append(np.mean(librosa.feature.mfcc(y=X[start:end], sr=sample_rate, n_mfcc=40).T,axis=0))
    chroma.append(np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0))
    mel.append(np.mean(librosa.feature.melspectrogram(X[start:end], sr=sample_rate).T,axis=0))
    contrast.append(np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0))
    tonnetz.append(np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X[start:end]), sr=sample_rate).T,axis=0))

    start = end

  return mfccs,chroma,mel,contrast,tonnetz

def merge_features(mfccs, chroma, mel, contrast, tonnetz):
  features = []
  for i in range(len(mfccs)):
    ext_features = np.hstack([mfccs[i],chroma[i],mel[i],contrast[i],tonnetz[i]])
    features.append(ext_features)
  return np.array(features)


def label_audio_files(dir_name, label, file_ext='*.wav'):
  features, labels = np.empty((0,193)), np.empty(0)
  for fn in glob.glob(os.path.join(dir_name, file_ext)):
    #label = os.path.basename(os.path.dirname(fn))
    print("Processing %s" % fn)
    mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
    # mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn, 10)
    for i in range(len(mfccs)):
      ext_features = np.hstack([mfccs[i],chroma[i],mel[i],contrast[i],tonnetz[i]])
      features = np.vstack([features, ext_features])
      labels = np.hstack([labels, label])

  return np.array(features), np.array(labels)

def write_features(features, labels, dir_name, file_name='features.csv'):
  fn = os.path.join(dir_name, file_name)
  with open(fn, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(features.shape[0]):
      writer.writerow(np.hstack([features[i], labels[i]]))
  print("Features saved in %s" % fn)

def load_features(dir_name, file_name='features.csv'):
  features, labels = [], [];
  with open(os.path.join(dir_name, file_name), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
      features.append(row[:-1])
      labels.append(row[-1])
  return np.array(features, dtype=np.float64), np.array(labels)

def save_label(label_list, dir_name, file_name='labels.csv'):
  with open(os.path.join(dir_name, file_name), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(label_list)

def load_label(dir_name, file_name='labels.csv'):
  label_list = []
  with open(os.path.join(dir_name, file_name), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
      label_list = np.hstack([label_list, row])
  label_numbers = [n for n in range(len(label_list))]
  label_dict = dict(zip(label_list, label_numbers))
  return label_list, label_dict

def one_hot_encode(labels, num_classes=None):
  if num_classes is None:
    num_classes = np.max(labels) + 1

  one_hot_list = np.zeros((len(labels), num_classes), dtype=np.int)
  one_hot_list[np.arange(len(labels)), labels] = 1
  return np.array(one_hot_list)

def one_hot_decode(one_hot_list):
  return np.array([n.argmax() for n in one_hot_list])


if __name__ == "__main__":
  if (len(sys.argv) < 3):
    print("Please specify [directory] and [label]")
    exit()

  dir_name = sys.argv[1]
  label = sys.argv[2]

  features,labels = label_audio_files(dir_name, label)
  write_features(features, labels, dir_name)

