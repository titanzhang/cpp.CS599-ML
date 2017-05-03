import sys, os
import numpy as np
import matplotlib.pyplot as plt
import audio_feature as af
from sklearn import model_selection, preprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation

def label_mapping(labels):
  label_list = np.unique(labels)
  label_numbers = [n for n in range(len(label_list))]
  label_dict = dict(zip(label_list, label_numbers))
  num_labels = [label_dict[n] for n in labels]
  return np.array(num_labels), label_list, label_dict

def load_data(file_name):
  features, labels = af.load_features(os.path.dirname(file_name), os.path.basename(file_name))
  labels, label_list, label_dict = label_mapping(labels)
  return features, labels, label_list, label_dict

# File with features first and labels in the last column
file_name = sys.argv[1]

# load pre-extracted acoustic features
X, y, label_list, label_dict = load_data(file_name)
print("X: {}".format(X.shape))
print("y: {}".format(y.shape))
print("label_dict: {}".format(label_dict))
y = af.one_hot_encode(y)

# split data for train/test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=33)

# data scaling
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

n_features = X_train.shape[1]
n_classes = label_list.shape[0]
training_epochs = 20
batch_size = 2

# Model design
model = Sequential()
# input layer
model.add(Dense(units=200, input_dim=n_features))
model.add(Activation('relu'))
# Hidden layer2
# model.add(Dense(units=400))
# model.add(Activation('relu'))
# Output layer
model.add(Dense(units=n_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=training_epochs, verbose=0)
print('Train accuracy: ', history.history['acc'][-1])
# print(history.history)

# Test
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: ', loss)
print('Test accuracy: ', accuracy)

# plot
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
