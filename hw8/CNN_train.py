import csv
import numpy as np
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, Flatten, Dropout, LeakyReLU, AveragePooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras import losses, regularizers
from keras.models import load_model
import sys


input_path = sys.argv[1]    #"train.csv"
model_teacher = load_model('./CNN_teacher.h5')


# parameters
num_classes = 7
epochs = 15
epochs_teacher = 8
epochs_last = 12
isvalid = 0
train_size = 25000
batch_size = 200
zoom_range = 0.2
#-----------------------------------------


train_x = []
data_y = []
validation_x = []
validation_y = []
vdata_y = []
test_x = []
count = 0

with open(file = input_path, mode = 'r', encoding = "GB18030") as csvfile:
    rows = csv.reader(csvfile)
    first = True
    for row in rows:
        temp = []
        temp2 = []
        if(first == True):
            first = False
            continue
        if(isvalid==0 or count<train_size):
            data_y.append(int(row[0]))
            row = row[1].split(' ')
            for i in range(len(row)):
                row[i] = float(row[i])
            train_x.append(row)  
        else:
            vdata_y.append(int(row[0]))
            row = row[1].split(' ')
            for i in range(len(row)):
                row[i] = float(row[i])
            validation_x.append(row)
        
        count += 1
    csvfile.close()



data_dim = len(train_x[0])
classify_num = 7
train_x = np.array(train_x, dtype = float)
train_x = np.reshape(train_x, (len(data_y), 48, 48, 1))
train_y = np.zeros([len(data_y), classify_num], dtype = float)

test_x = np.array(test_x, dtype = float)
test_x = np.reshape(test_x, (len(test_x), 48, 48, 1))

validation_y = np.zeros([len(validation_x), classify_num], dtype = float)
validation_x = np.array(validation_x, dtype = float)
validation_x = np.reshape(validation_x, (len(validation_x), 48, 48, 1))


for i in range(train_y.shape[0]):
    train_y[i][data_y[i]] = 1
for i in range(validation_y.shape[0]):
    validation_y[i][vdata_y[i]] = 1

print(train_x.shape)
print(train_y.shape)
print(validation_x.shape)
print(validation_y.shape)
print("finished extracting data")


# Construct the model
model = Sequential()

model.add(Conv2D(16, (5, 5), input_shape=(48,48,1) ))
model.add(LeakyReLU(alpha=0.03))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))

model.add(SeparableConv2D(16, (5, 5)))
model.add(LeakyReLU(alpha=0.03))
model.add(Conv2D(70, (1, 1)))
model.add(LeakyReLU(alpha=0.03))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(SeparableConv2D(70, (4, 4)))
model.add(LeakyReLU(alpha=0.03))
model.add(Conv2D(25, (1, 1)))
model.add(LeakyReLU(alpha=0.03))
#model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.3))


model.add(SeparableConv2D(25, (3, 3)))
model.add(LeakyReLU(alpha=0.03))
model.add(Conv2D(7, (1, 1)))
model.add(LeakyReLU(alpha=0.03))
#model.add(BatchNormalization())
#model.add(Dropout(0.4))


model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# image data generator
train_gen = ImageDataGenerator(rotation_range=25,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.1,
                              zoom_range=[1-zoom_range, 1+zoom_range],
                              horizontal_flip=True)

train_gen.fit(train_x)


# fit the model
if(isvalid==True):
	model.fit_generator(train_gen.flow(train_x, train_y, batch_size=batch_size),
                    steps_per_epoch=10*train_x.shape[0]//batch_size,
                    epochs=epochs,
                    validation_data=(validation_x, validation_y))
else:
	model.fit_generator(train_gen.flow(train_x, train_y, batch_size=batch_size),
			steps_per_epoch=10*train_x.shape[0]//batch_size,
			epochs=epochs)

teacher_ans = model_teacher.predict(train_x)
train_y = 0.6*train_y + 0.4*teacher_ans
model.fit_generator(train_gen.flow(train_x, train_y, batch_size=batch_size),
			steps_per_epoch=10*train_x.shape[0]//batch_size,
			epochs=epochs_teacher)

train_x = np.load("aug.npy")
train_y = model_teacher.predict(train_x)
model.fit(train_x, train_y, epochs=epochs_last, batch_size=batch_size, validation_data=(validation_x, validation_y))

# save model
model.save('strong.h5')

