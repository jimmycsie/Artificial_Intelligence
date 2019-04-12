import sys, os
import csv
import numpy as np
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt

model = load_model('CNN.h5')
img_width = 48
img_height = 48 
input_path = sys.argv[1]
directory = sys.argv[2]
filter_index = [0,6,12,13, 16,17,18,19, 21,22,23,26, 29,30,31,34]

picture = []
with open(file = input_path, mode = 'r', encoding = "GB18030") as csvfile:
    rows = csv.reader(csvfile)
    count = 0
    for row in rows:
        temp = []
        if(count<27499):
            count += 1
            continue            
        row = row[1].split(' ')
        for i in range(len(row)):
            row[i] = float(row[i])
        picture.append(row)             # 27500
        break
    csvfile.close()
picture = np.reshape(picture, (1, 48, 48, 1))

def compile_gradient_function(model, filter_index):
    input_img = model.layers[0].input
    layer_output = model.layers[1].output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    return K.function([input_img], [grads, layer_output, loss])

# fig2_1------------------------------------------------------------------------------------
fg = plt.figure()    
plt.axis('off')
plt.title("Filters of layer conv2d_1")
# run gradient ascent for 100 steps
lr = 1

count = 1
for j in filter_index:
    # we start from a gray image with some noise
    input_img_data = np.zeros((1, 48, 48, 1), dtype=float) #np.random.random((1, img_width, img_height, 1))
    for i in range(30):
        grads_value = compile_gradient_function(model, j)([input_img_data])
        input_img_data += grads_value[0] * lr
        if(i%10==0):
            print("i=", i)

    input_img_data = np.reshape(input_img_data, (48, 48))        #80
    #input_img_data = input_img_data.transpose([2, 0, 1])
    fg.add_subplot(4, 4, count)
    plt.imshow(input_img_data, cmap='Blues')
    plt.axis('off')
    count += 1
    print(j)
    
plt.axis('off')
name = './' + directory + 'fig2_1.jpg'
plt.savefig(name)
plt.clf()

#-------------------------------------------------------------------------------------------
# fig2_2------------------------------------------------------------------------------------
fg = plt.figure()    
plt.axis('off')
plt.title("Output of layer conv2D_1 (Given image 27500)")
change = compile_gradient_function(model, 0)([picture])
change = np.reshape(change[1], (48, 48, 80))
change = change.transpose((2,0,1))

count = 1
for i in filter_index:
    fg.add_subplot(4, 4, count)
    plt.imshow(change[i], cmap='Blues')
    plt.axis('off')
    count += 1

plt.axis('off')
name = './' + directory + 'fig2_2.jpg'
plt.savefig(name)
plt.clf()
