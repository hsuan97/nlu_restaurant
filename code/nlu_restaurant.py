# -*- coding: utf-8 -*-
"""
nlu_restaurant.py
"""
from __future__ import print_function, division
from builtins import range

import os
#import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.metrics import roc_auc_score

# some configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 50

# Download the word vectors:
# http://nlp.stanford.edu/data/glove.6B.zip

# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {} 
with open(os.path.join('C:/Users/e211/Desktop/NLU/glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM),"r",encoding="utf-8") as f:
  #/content/drive/MyDrive/glove.6B
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
      values = line.split()
      word = values[0] 
      vec = np.asarray(values[1:], dtype='float32')
      word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

train = pd.read_csv("C:/Users/e211/Desktop/NLU/nlu_dct_150.csv")
sentences = train["comment_text"].fillna("DUMMY_VALUE").values
possible_labels = ["locate_restaurant", "restaurant_type", "table_reservation", "restaurant_review"]
targets = train[possible_labels].values
target = train[possible_labels].values.tolist()

# convert the sentences (strings) into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
#print("sequences:", sequences); exit()

print("max sequence length:", max(len(s) for s in sequences))
print("min sequence length:", min(len(s) for s in sequences))
s = sorted(len(s) for s in sequences)
print("median sequence length:", s[len(s) // 2])

print("max word index:", max(max(seq) for seq in sequences if len(seq) > 0))

word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
          embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False
)


print('Building model...')
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(possible_labels), activation='sigmoid')(x)

model = Model(input_, output)
model.compile(
  loss='binary_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)
model.summary()

print('Training model...')
r = model.fit(
  data,
  targets,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=VALIDATION_SPLIT
)

def save_model(model):
    model_name = 'NLP_Model.h5'
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('\nSaved trained model at %s ' % model_path)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

save_model(model)        
'''
json_string = model.to_json() 
with open("model.config", "w") as text_file:    
    text_file.write(json_string)

model.save_weights("model.weight")
'''
#model.save('nlu_model.h5')

# plot some data
plt.plot(r.history['loss'], label='train')
plt.plot(r.history['val_loss'], label='validation')
plt.title('Loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='train')
plt.plot(r.history['val_accuracy'], label='validation')
plt.title('Accuracy')
plt.legend()
plt.show()

# plot the mean AUC over each label
p = model.predict(data)
aucs = []
for j in range(4):
    auc = roc_auc_score(targets[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))

predicts = p.tolist()
label = []

for i in predicts:
  for j in range(4):
    #d = i[j]
    if i[j] == max(i): 
      label.append(j)

t = []
for i in range(len(target)):
  if target[i] == [1, 0, 0, 0]:
    t.append(0)
    #t.append('locate_restaurant')
  elif target[i] == [0, 1, 0, 0]:
    t.append(1) 
    #t.append('restaurant_type')
  elif target[i] == [0, 0, 1, 0]:
    t.append(2)
    #t.append('table_reservation')
  else:
    t.append(3)
    #t.append('restaurant_review')

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm,target_names,
                          title='Confusion matrix',
                          cmap=None,normalize=True):
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

y_true = t
y_pred = label

plot_confusion_matrix(cm = confusion_matrix(y_true, y_pred), 
                      normalize = False,
                      target_names = possible_labels,
                      title= "Confusion Matrix")
