import sys, os
import csv
import numpy as np
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt

import sys, os
import csv
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt


model = load_model('CNN.h5')
input_path = "train.csv"
isget = [0]*7
sample_data = []
count_data = []
class_data = []
with open(file = input_path, mode = 'r', encoding = "GB18030") as csvfile:
    rows = csv.reader(csvfile)
    count = 0
    for row in rows:
        temp = []
        if(count<23000):
            count += 1
            continue
        if(isget[int(row[0])] == False):
            isget[int(row[0])] = True
            class_data.append(int(row[0]))
            row = row[1].split(' ')
            for i in range(len(row)):
                row[i] = float(row[i])
            sample_data.append(row)  
            count_data.append(row)    
    csvfile.close()

sample_data = np.array(sample_data, dtype = float)
sample_data = np.reshape(sample_data, (7, 48, 48, 1))
count_data = np.array(count_data, dtype = float)
count_data = np.reshape(count_data, (7, 48, 48, 1))
explain = np.zeros([7, 48, 48], dtype = float)
average = []
for i in range(7):
    average.append(np.mean(sample_data[i]))

size = 5
for i in range(47+size):
    from_i = max(i+1-size, 0)
    to_i = min(i, 47)+1
    for j in range(47+size):
        from_j = max(j+1-size, 0)
        to_j = min(j, 47)+1
        #--1--
        for k in range(7):  
            count_data[k][from_i:to_i,from_j:to_j] = 255#average[k] 
        result = model.predict(count_data)
        for k in range(7):
            explain[k][from_i:to_i,from_j:to_j] += (1-result[k][class_data[k]])
        #--2--
        for k in range(7):  
            count_data[k][from_i:to_i,from_j:to_j] = 0#average[k] 
        result = model.predict(count_data)
        for k in range(7):
            explain[k][from_i:to_i,from_j:to_j] += (1-result[k][class_data[k]])
        
        for k in range(7):
            for l in range(size):
                if(i-l<0 or i-l>47):
                    continue
                for m in range(size):
                    if(j-m<0 or j-m>47):
                        continue
                    count_data[k][i-l][j-m] = sample_data[k][i-l][j-m]
    print(i)

"""
ans = np.zeros([7, 50-2*size, 50-2*size], dtype = float)
for i in range(7):
    for j in range(size-1, 49-size):
        ans[i][j-size+1] = explain[i][j][(size-1):(49-size)]
"""
ans = explain

for i in range(7):
    ans[i] /= (np.sqrt(np.mean(np.square(ans[i]))) + 1e-5)


for i in range(7):
    plt.figure()
    plt.imshow(ans[i], cmap='gray')
    plt.colorbar()
    name = "./picture/square_" + str(class_data[i]) + ".png"
    plt.savefig(name)
    plt.clf()

result = model.predict(sample_data)
for i in range(7):
    print(result[i][class_data[i]])
