# conda env create --file=hw7_env.yml

from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import math
import csv
from keras.models import load_model
import keras.backend as K
#from MulticoreTSNE import MulticoreTSNE as TSNE
import sys


# const variable-----------------------------------------------
data_size = 40000
epochs = 100
np.random.seed(7)
model = load_model('auto_encoder2.h5')
#------------------------------------------------------
print(model.summary())
def data_reduction(model):
    input_img = model.layers[0].input
    layer_output = model.layers[3].output
    reconstruct = model.output
    return K.function([input_img], [layer_output, reconstruct])

image_path = sys.argv[1]                        # images/
test_path = sys.argv[2]                         # "test_case.csv"
ans_path = sys.argv[3]                          # "submission.csv"
number = []
ans = []
pos = 0
neg = 0
# create name format ----------------------------------
for i in range(1, 10):
    number.append("00000"+str(i))
for i in range(10, 100):
    number.append("0000"+str(i))
for i in range(100, 1000):
    number.append("000"+str(i))
for i in range(1000, 10000):
    number.append("00"+str(i)) 
for i in range(10000, 40001):
    number.append("0"+str(i))
# get testing index----------------------------------------------
index = []
with open(file = test_path, mode = 'r') as csvfile:
    rows = csv.reader(csvfile)
    count = 0
    for row in rows:  
        if(count == 0):
            count = 1
            continue
        for i in range(len(row)):
            row[i] = int(row[i])   
        index.append(row[1:3])
    csvfile.close()
#----------------------------------------------------------------

pic = []
origin_pic = []

# get by original picture
for i in range(data_size):
    name = image_path + number[i] + '.jpg'
    temp = Image.open(name)
    temp = np.array(temp)
    origin_pic.append(temp)

origin_pic = np.array(origin_pic, dtype=float)
origin_pic = origin_pic / 255
#-----------------------------------------------------------
pic = data_reduction(model)([origin_pic])[0]
pic = np.reshape(pic, (data_size, 8*8*8))

"""
# get picture reducted by auto_encoder
with open(file = "data_reduction2.csv", mode = 'r') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:  
        for i in range(len(row)):
            row[i] = float(row[i])   
        pic.append(row)
    csvfile.close()
pic = np.array(pic)
pic = np.array(pic, dtype=float)
pic = np.reshape(pic, (data_size, 8*8*8))
#-----------------------------------------------------------
"""

for i in range(data_size):
    pic[i] -= np.mean(pic[i])
    pic[i] /= np.sqrt( np.sum( np.square(pic[i]) ) / data_size )




"""
tsne = TSNE(n_jobs=6, n_components=2)
reducted = tsne.fit_transform(pic)
print("finished TSNE reduction")
print(reducted)
print(reducted.shape)
"""
pca = PCA(copy=True, iterated_power='auto', n_components=300, random_state=None,
          svd_solver='auto', tol=0.0, whiten=True)
reducted = pca.fit_transform(pic)
print("finished PCA reduction1")




# K-means cluster method------------------------------------------------------------
two_means = []
two_means.append(reducted[20950])
two_means.append(reducted[10319])

# k-means cluster
two_means = np.array(two_means)
last = np.zeros((two_means.shape[1]))
classification = np.zeros((2, data_size))
isconverge = 1

count = 0
while(isconverge > 1e-8):
    for j in range(data_size):
        first = np.sum(np.square(reducted[j]-two_means[0]))
        second = np.sum(np.square(reducted[j]-two_means[1]))
        if(first < second):
            classification[0][j] = 1
            classification[1][j] = 0
        else:
            classification[0][j] = 0
            classification[1][j] = 1
    
    two_means[0] =  np.dot(classification[0], reducted) / np.sum(classification[0])
    two_means[1] =  np.dot(classification[1], reducted) / np.sum(classification[1])
    isconverge = np.sum(np.square(last-two_means[0])) 
    last = two_means[0]*1
    count += 1
    #if(count%5==0):
    #    print(two_means[0][:3])
    #    print(two_means[1][:3])
    #    print()

for i in range(len(index)):
    first_1 = np.sum(np.square( reducted[index[i][0]-1]-two_means[0] ))
    second_1 = np.sum(np.square( reducted[index[i][0]-1]-two_means[1] ))
    first_2 = np.sum(np.square( reducted[index[i][1]-1]-two_means[0] ))
    second_2 = np.sum(np.square( reducted[index[i][1]-1]-two_means[1] ))
    #if(i<10):
        #print(index[i][0], index[i][1], first_1, second_1, first_2, second_2)
    if(first_1 < second_1):
        label_1 = 1
    else:
        label_1 = 0
    if(first_2 < second_2):
        label_2 = 1
    else:
        label_2 = 0

    if(label_1 == label_2):
        ans.append(1)
        pos += 1
    else:
        ans.append(0)
        neg += 1
#--------------------------------------------------------------------------------


"""  
# distance method ---------------------------------------------------------------
distance = []
for i in range(len(index)):
    distance.append( np.sum(np.square( reducted[index[i][0]-1]-reducted[index[i][1]-1] )) )

distance = np.array(distance)
threshold = np.sum(distance) / len(index) - 20
print(threshold)
for i in range(len(index)):
    if(distance[i]<threshold):
        ans.append(1)
        pos += 1
    else:
        ans.append(0)
        neg += 1
#--------------------------------------------------------------------------------
"""



print(pos, neg)
with open(ans_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "label"])
    for i in range(len(ans)):
        writer.writerow([str(i), ans[i]])


print(ans[:10])
