#from keras.applications import resnet50 #import ResNet50, preprocess_input, decode_predictions
#from keras_applications import resnet
#from keras.applications.densenet import DenseNet121, DenseNet169, preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import keras
import keras.backend as K
from PIL import Image
import numpy as np
import sys

# const variable----------------------------------
class_num = 1000
pic_num = 200
infinity_norm = 22
iteration = 1
epsilon = 19#infinity_norm / iteration
print("epsilon =", epsilon)
# ------------------------------------------------
def find_outp(model):
    inp = model.input
    outp = model.output
    return K.function([inp], [outp])
# create name format ----------------------------------
number = []
for i in range(10):
    number.append("00"+str(i))
for i in range(10, 100):
    number.append("0"+str(i))
for i in range(100, 200):
    number.append(str(i))
#------------------------------------------------------
model1 = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
# find label---------------------------------------------
pic = []
original_pic = []
original_ans = []
onehot_label = []

for i in range(pic_num):
    img_path = sys.argv[1] + '/' + number[i] + ".png"
    pic = Image.open(img_path)
    pic =  pic.convert('RGB')           # change to jpg format
    pic = np.array(pic) 
    original_pic.append(pic)  

original_pic = np.reshape(original_pic, (pic_num, 224, 224, 3)) 
original_ans = model1.predict(original_pic)
original_ans = np.reshape(original_ans, (pic_num, class_num))
for i in range(pic_num):
    max = 0
    second = 0
    label1 = 0
    label2 = 0
    for j in range(class_num):
        if(original_ans[i][j] > max):
            second = max
            max = original_ans[i][j]
            label2 = label1
            label1 = j
        elif(original_ans[i][j] > second):
            second = original_ans[i][j]
            label2 = j
    temp = [0]*class_num
    target = label1
    while(target==label1 or target==label2):
        target = (target+66)%class_num
    #temp[target] = 1
    temp[label1] = -1
    #temp[label2] = -1
    onehot_label.append(temp)
onehot_label = np.reshape(onehot_label, (pic_num, class_num))
#-------------------------------------------------------


def compile_gradient_function(model):
    inp = model.input
    outp = model.output
    loss = K.categorical_crossentropy(onehot_label, outp)
    grad = K.gradients(loss, inp)[0]   
    return K.function([inp], [grad])


# print original label----------------------------------------
pic_noise = original_pic

momentum = np.zeros((pic_num, 224, 224, 3))
grad = np.zeros((pic_num, 224, 224, 3))
print("prepare to calculate gradient")
for i in range(iteration):
    pre = np.copy(pic_noise)
    momentum += compile_gradient_function(model1)([pre])[0]
    grad[np.where(momentum < 0)] = -1
    grad[np.where(momentum > 0)] = 1
    momentum *= 0
    pic_noise = pic_noise - epsilon * grad
    

#pic_noise = pic_noise/255
pic_noise = pic_noise.clip(0, 255)

for i in range(pic_num):
    name = sys.argv[2] + '/' + number[i] + ".png"
    #plt.imsave(name, pic_noise[i])
    pic = Image.fromarray(pic_noise[i].astype("uint8"))
    pic.save(name)
