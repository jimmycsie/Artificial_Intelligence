import csv
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix

input_path = "train.csv"
model = load_model('CNN.h5')
train_size = 23000
classify_num = 7

validation_x = []
validation_y = []
vdata_y = []
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
        if(count>=train_size):
            vdata_y.append(int(row[0]))
            row = row[1].split(' ')
            for i in range(len(row)):
                row[i] = float(row[i])
            validation_x.append(row)      
        count += 1
    csvfile.close()

validation_y = np.array(vdata_y, dtype = float)
validation_x = np.array(validation_x, dtype = float)
validation_x = np.reshape(validation_x, (len(validation_x), 48, 48, 1))
print(validation_y.shape)
y_pred = []
result = model.predict(validation_x)
for i in range(result.shape[0]):
    max = 0
    index = 0
    for j in range(result.shape[1]):
        if(result[i][j]>max):
            max = result[i][j]
            index = j
    y_pred.append(index)

y_pred = np.array(y_pred, dtype = float)

labels = [0, 1, 2, 3, 4, 5, 6]
cm = confusion_matrix(validation_y, y_pred, labels)

# accuracy
cm = cm.astype(float)
for i in range(cm.shape[0]):
    number = np.sum(cm[i])
    cm[i] = cm[i] / number
"""
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix')
fig.colorbar(cax)
ax.set_xticklabels([''] + ['Angry ', 'Disgust ', 'Fear ', 'Happy ', 'Sad ', 'Surprise ', 'Neutral '])
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
ax.set_yticklabels([''] + ['Angry ', 'Disgust ', 'Fear ', 'Happy ', 'Sad ', 'Surprise ', 'Neutral '])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
"""
label = ['Angry ', 'Disgust ', 'Fear ', 'Happy ', 'Sad ', 'Surprise ', 'Neutral ']
df_cm = pd.DataFrame(cm, index = [i for i in label],
                  columns = [i for i in label])
plt.figure(figsize = (10,7))
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
sn.heatmap(df_cm, annot=True, cmap=sn.diverging_palette(220, 20, as_cmap=True))
plt.savefig('confusion_matrix.png')