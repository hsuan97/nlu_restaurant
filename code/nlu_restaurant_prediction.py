import os
import numpy as np 
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import model_from_json
#from sklearn.preprocessing import LabelEncoder

MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100

# Download the word vectors:
# http://nlp.stanford.edu/data/glove.6B.zip

# load in pre-trained word vectors
#print('Loading word vectors...')
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
#print('Found %s word vectors.' % len(word2vec))

#sentence = 'Great service and food.'
#sentence = ['Book a table for 2 people']

#s.append(sentence)

test = pd.read_csv("C:/Users/e211/Desktop/NLU/test.csv")
sentences = test["comment_text"].fillna("DUMMY_VALUE").values
possible_labels = ["locate_restaurant", "restaurant_type", "table_reservation", "restaurant_review"]
targets = test[possible_labels].values
#target = test[possible_labels].values.tolist()

# convert the sentences (strings) into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
#print("sequences:", sequences); exit()

word2idx = tokenizer.word_index
#print('Found %s unique tokens.' % len(word2idx))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#print('Shape of data tensor:', data.shape)

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

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/NLP_Model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])    

#evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy',
                     optimizer='rmsprop',
                     metrics=['accuracy'])

predictions = loaded_model.predict(data, batch_size=16, verbose = 1)
#print(predictions)

p = predictions.tolist()
label = []

for i in p:
  for j in range(4):
    if i[j] == max(i): 
      if j == 0:
          label.append('locate_restaurant')  
      elif j == 1:
          label.append('restaurant_type')
      elif j == 2:
          label.append('table_reservation')
      elif j == 3:
          label.append('restaurant_review')
          
test['target'] = label
result = test[['comment_text' , 'target']]
print(result)



'''
preds= predictions.argmax(axis=1) #返回最大值，橫著比較
liveabc = preds.astype(int).flatten()
test_labels = ['locate_restaurant', 'restaurant_type', 'table_reservation', 'restaurant_review']
lb = LabelEncoder()
result = lb.fit_transform(test_labels)
livepredictions = (lb.inverse_transform((liveabc)))
print(livepredictions)
'''
