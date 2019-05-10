import csv
from gensim.models import word2vec
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, GRU, LeakyReLU, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras import losses, regularizers
import numpy as np
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys

input_path = 'segment_train.txt'
input_path2 = sys.argv[1]       #./data/train_y.csv'
#jieba.load_userdict("dict.txt.big")
model = word2vec.Word2Vec.load("word2vec.model")


data_size = 120000
time_step = 35
train_seg = []
train_vec = []
y_train = []
y_label = []

# get label
with open(file = input_path2, mode = 'r') as csvfile:
    rows = csv.reader(csvfile)
    first = True
    for row in rows:
        if(first == True):
            first = False
            continue
        y_train.append(int(row[1]))
    csvfile.close()


# get segment train word
fp = open(input_path, "r")
for i in range(data_size):
    temp = []
    line = fp.readline()
    line = line.split(' ')
    for j in range(len(line)):
        #if( line[j]!='' and (line[j][0]=='b' or line[j][0]=='B') ) :
        #    continue
        if(line[j]!=''):
            if(line[j][-1]=='\n'):
                temp.append(line[j][:-1])
            else:
                temp.append(line[j])
    train_seg.append( temp )
fp.close()

"""
embedding_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
word2idx = {}
encoded_docs = []

vocab_list = [(word, model.wv[word]) for word, _ in model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1

for i in range(data_size):
    temp = []
    for j in train_seg[i]:
        if j in word2idx:
            temp.append(word2idx[j])
        else:
            temp.append(0)
    encoded_docs.append(temp)

"""
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(train_seg)
vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(train_seg)

embedding_matrix = [[0 for i in range(250)]] * vocab_size
check = []
encoder = {}
for i in range(data_size):
    for j in range(len(train_seg[i])):
        if(train_seg[i][j] in model.wv):
            vector = model.wv[train_seg[i][j]]
            embedding_matrix[encoded_docs[i][j]] = vector
            check.append(encoded_docs[i][j])
        if(train_seg[i][j] not in encoder):
            encoder[train_seg[i][j]] = encoded_docs[i][j]




embedding_matrix = np.array(embedding_matrix)
embedding_matrix = np.reshape(embedding_matrix, (embedding_matrix.shape[0], 250))

padded_docs = pad_sequences(encoded_docs, maxlen=time_step, padding='pre')


# get segment test word
test_seg = []
test_encode = []
fp = open("segment_test.txt", "r")
for i in range(20000):
    temp = []
    line = fp.readline()
    line = line.split(' ')
    for j in range(len(line)):
        if(line[j]!=''):
            if(line[j][-1]=='\n'):
                temp.append(line[j][:-1])
            else:
                temp.append(line[j])
    test_seg.append( temp )
fp.close()


# construct model
model = Sequential()
model.add( Embedding(vocab_size, 250, weights=[embedding_matrix], input_length=time_step, trainable=True ))

model.add(GRU(units = 64, return_sequences = False, activation='sigmoid' ))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(16, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

# Compiling
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
# training
print(model.summary())
model.fit(padded_docs, y_train, epochs = 2, batch_size = 100)
model.save('RNN.h5')


