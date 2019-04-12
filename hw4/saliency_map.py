import sys, os
import csv
import numpy as np
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt
import matplotlib

model = load_model('CNN.h5')
input_path = sys.argv[1]        # "train.csv"
directory = sys.argv[2]            # "picture/"
isget = [0]*7
sample_data = []
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
    csvfile.close()

sample_data = np.array(sample_data, dtype = float)
sample_data = np.reshape(sample_data, (7, 48, 48, 1))
target = np.zeros([7, 7], dtype = float)
for i in range(7):
    target[i][class_data[i]] = 1

def compile_saliency_function(model):
    inp = model.input
    outp = model.output
    saliency = K.gradients(K.categorical_crossentropy(target,outp), inp)[0]
    
    return K.function([inp], [saliency])

sal = compile_saliency_function(model)([sample_data])

sal = np.reshape(sal[0], (7, 48, 48))
sample_data = np.reshape(sample_data, (7, 48, 48))




sal = np.abs(sal)
for i in range(7):
    sal[i] = (sal[i] - np.mean(sal[i])) / (np.std(sal[i]) + 1e-30)
sal *= 0.1
sal = np.clip(sal, 0, 1)

for index in range(7):
    #img = plt.imshow(sample_data[index])
    #img.set_cmap('gray')
    name = './' + directory +'original_' + str(class_data[index]) + '.jpg'
    #plt.savefig(name)
    plt.imsave(name, sample_data[index], cmap="gray")
    #plt.clf()

    heatmap = sal[index]
    thres = 0.001
    mask = sample_data[index]
    mask[np.where(heatmap <= thres)] = np.mean(mask)
    #plt.figure()
    #plt.imshow(heatmap, cmap=plt.cm.jet)
    #plt.colorbar()
    #plt.tight_layout()
    #fig = plt.gcf()
    name = './' + directory + 'fig1_' + str(class_data[index]) + '.jpg'          # saliency_map
    #plt.savefig(name)
    plt.imsave(name, heatmap, cmap=plt.cm.jet)
    #plt.clf()

    #plt.figure()
    #plt.imshow(mask, cmap='gray')
    #plt.colorbar()
    #plt.tight_layout()
    #fig = plt.gcf()
    name = './' + directory + 'mask_original_' + str(class_data[index]) + '.jpg'
    #plt.savefig(name)
    #plt.imsave(name, mask)
    plt.imsave(name, mask, cmap="gray")
    #plt.clf()

#plt.show()
