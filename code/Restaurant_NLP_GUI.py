import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import ttk, Scrollbar, VERTICAL
import os
import numpy as np 
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import model_from_json

def import_csv_data():
    global v
    csv_file_path = askopenfilename()
    #print(csv_file_path)
    v.set(csv_file_path)
    #df = pd.read_csv(csv_file_path)
    
def Start():
    filepath = entry.get()
    #print(filepath)
    return filepath

def Recongition():
    filepath = Start()
    #print(filepath)
    
    MAX_SEQUENCE_LENGTH = 100
    MAX_VOCAB_SIZE = 20000
    EMBEDDING_DIM = 100
    
    word2vec = {}
    with open(os.path.join('C:/Users/e211/Desktop/NLU/glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM),"r",encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0] 
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec

    test = pd.read_csv(filepath)
    sentences = test["comment_text"].fillna("DUMMY_VALUE").values
    possible_labels = ["locate_restaurant", "restaurant_type", "table_reservation", "restaurant_review"]
    targets = test[possible_labels].values

    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    word2idx = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # prepare embedding matrix
    num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word2idx.items():
        if i < MAX_VOCAB_SIZE:
            embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
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
    loaded_model.load_weights("saved_models/NLP_Model.h5")
    #print("Loaded model from disk")
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
    s = sentences.tolist()
    
    for i in range(min(len(s),len(label))): # 寫入數據
        table = treeview.insert('', i, values=(s[i], label[i]))
    
root = tk.Tk()
root.title('Restaurant NLP')
root.geometry('2100x900')

lab1 = tk.Label(root, text ='Restaurant NLP ',bg = 'lightblue', width = 20, 
                font = ('Arial', 30, 'bold'))
lab1.pack()
#lab1.grid(row = 0, column = 1)

lab2 = tk.Label(root, text='load data file',font = ('Arial', 20))
lab2.pack(pady = 10)
#lab2.grid(row = 2, column = 0)

v = tk.StringVar()
entry = tk.Entry(root, textvariable=v, width = 60, font = ('Arial', 15))
entry.pack(pady = 5)
#entry.grid(row = 2, column = 2)

btn1 = tk.Button(root, text='Browse Data Set',command = import_csv_data,
                 font = ('Arial', 15))
btn1.pack(pady = 5)
#btn1.grid(row = 2, column = 1)

btn2 = tk.Button(root, text='Close',command  =root.destroy, font = ('Arial', 15))
btn2.pack(anchor = tk.S, side = tk.RIGHT, padx = 15, pady = 15)
#btn2.grid(row = 4, column = 3)

btn3 = tk.Button(root, text = 'Start', command = Recongition,
                 font = ('Arial', 15))
btn3.pack(pady = 7)
#btn3.grid(row = 2, column = 3)


columns = ("Comment_text", "Recongition")
treeview = ttk.Treeview(root, height=350, show="headings", 
                        columns=columns, selectmode='browse')  # 表格

vbar = Scrollbar(root, orient = VERTICAL, 
                 command=treeview.yview)
vbar.pack(side = tk.RIGHT, fill = tk.Y)

style = ttk.Style()
style.configure("Treeview.Heading", font=('Arial', 20, 'bold'), rowheight=40)
style.configure("Treeview", font=('Arial', 15), rowheight=40)

treeview.column("Comment_text", width=1395, anchor='center') # 表示列
treeview.column("Recongition", width=400, anchor='center')

treeview.configure(yscrollcommand=vbar.set)
vbar.config(command = treeview.set)


treeview.heading("Comment_text", text="Comment_text") # 顯示表頭
treeview.heading("Recongition", text="Recongition")


treeview.pack(side= tk.BOTTOM, pady = 15, fill = tk.Y)
#treeview.grid(row = 3, column = 1)




root.mainloop()