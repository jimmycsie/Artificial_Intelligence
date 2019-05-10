import csv
from gensim.models import word2vec
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, GRU, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras import losses, regularizers
import numpy as np
from keras.layers import Embedding
import sys

input_path = 'segment_train.txt'
input_path2 = sys.argv[1]                          #'./data/train_y.csv'
model_name = sys.argv[2]                     
time_step = int(sys.argv[3])
#jieba.load_userdict("dict.txt.big")
model = word2vec.Word2Vec.load("word2vec.model")

"""
si = model.most_similar(positive=['學校'], topn = 20)
print(si[0][0])

for x in si:
    print(x[0],x[1])
"""

#vector = model.wv['>']

data_size = 120000
train_seg = []
train_vec = []
second_index = []
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


# get segment word
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
print(train_seg[0])

# make train_vec = size * train_step * 250
zero_vector = [0]*250
for i in range(data_size):       #second_index:      #  
    temp = []
    count = 0
    for j in range(len(train_seg[i])):
        if(j<len(train_seg[i])-1 and train_seg[i][j]==train_seg[i][j+1]):
            continue
        if train_seg[i][j] in model.wv:
            vector = model.wv[train_seg[i][j]]
            temp.append(vector)
            count += 1

        if(count == time_step):
            train_vec.append(temp)
            y_label.append(y_train[i])
            temp = []
            count = 0
            break
    if(count != 0):
        j = 0
        while(count != time_step):
            #temp.append(zero_vector)
            if train_seg[i][j] in model.wv:
                vector = model.wv[train_seg[i][j]]
                temp.append(vector)
                count += 1
            j = (j+1)%len(train_seg[i])

        y_label.append(y_train[i])
        train_vec.append(temp)


train_vec = np.array(train_vec, dtype = float)
train_vec = np.reshape(train_vec, (train_vec.shape[0], time_step, 250))
print(train_vec.shape)
print("finished splitting and label")
#----------------------------------------------------------------
# Construct the model
model = Sequential()
if(model_name == "RNN128_50.h5"):
    print("GRU = 128")
    model.add(GRU(units = 128, input_shape = (time_step, 250), return_sequences = False, activation='sigmoid' ))
else:
    print("GRU = 64")
    model.add(GRU(units = 64, input_shape = (time_step, 250), return_sequences = False, activation='sigmoid' ))

model.add(BatchNormalization())
model.add(Dropout(0.2))



model.add(Dense(128))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(Dropout(0.2))


model.add(Dense(256))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(Dropout(0.2))


model.add(Dense(512))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(Dropout(0.2))


model.add(Dense(1, activation='sigmoid'))

# Compiling
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
# training
#print(model.summary())
model.fit(train_vec, y_label, epochs = 9, batch_size = 100)
model.save(model_name)

