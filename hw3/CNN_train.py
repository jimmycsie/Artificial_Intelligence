import csv
import numpy as np
import sys
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras import losses, regularizers
import matplotlib.pyplot as plt

input_path = sys.argv[1]
# parameters
num_classes = 7
epochs = 30
isvalid = 0
train_size = 23000
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

validation_y = np.zeros([len(validation_x), classify_num], dtype = float)
validation_x = np.array(validation_x, dtype = float)
validation_x = np.reshape(validation_x, (len(validation_x), 48, 48, 1))


for i in range(train_y.shape[0]):
    train_y[i][data_y[i]] = 1
for i in range(validation_y.shape[0]):
    validation_y[i][vdata_y[i]] = 1
print("finished extracting data")

# Construct the model
model = Sequential()

model.add(Conv2D(64, (5, 5), padding='same', input_shape=(48,48,1) ))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
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

# show accuracy of validation data and plot losses
if(isvalid==True):
	score = model.evaluate(validation_x, validation_y)
	print('Total loss', score[0])
	print('Accuracy', score[1])
	# plot loss
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Training Process accuracy CNN')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig('CNN_acc.png')
	plt.clf()

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Training Process loss CNN')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig('CNN_loss.png')


# save model
model.save('CNN.h5')


