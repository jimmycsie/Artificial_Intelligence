import csv
import numpy as np
import math
import sys

test_path = sys.argv[5]
test_path2 = sys.argv[2]
ans_path = sys.argv[6]
weight_path = "./logistic_weight.csv"

income_GNI = [
1230, 42870, 8690, 5890, 7150, 6630, 5920, 3560, 40530, 37970,
43490, 18090, 4060, 760, 46180, 2250, 46310, 12870, 1800, 5430,
55290, 31020, 4760, 38550, 2270, 8610, 2130, 16000, 5960, 3660,
12730, 19820, 19820, 40530, 18167, 24000, 5950, 15340, 58270,
2160, 18167, 18167
]
temp = []
mean = []
var = []
w = []
highdimension = [0, 1, 3, 4, 5, 48]             # 0:age, 1:fnlwgt, 3:capital_gain, 4:capital_loss, 5:hour_per_week, 48(position-1):income_GNI
with open(file = weight_path, mode = 'r', encoding = "GB18030") as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        temp.append(row)

mean = temp[0]
var = temp[1]
w = temp[2]
for i in range(len(mean)):
    mean[i] = float(mean[i])
    var[i] = float(var[i])
    w[i] = float(w[i])
w = np.array(w, dtype = float)

# testing data
count = 0
test_x = []
with open(file = test_path, mode = 'r', encoding = "GB18030") as csvfile:
    rows = csv.reader(csvfile)
    first = True
    for row in rows:
        if(first == True):
            first = False
            continue

        test_x.append(row[:15])               # 15
        for i in range(31,64):
            test_x[count].append(row[i])      # 33

        for i in range(len(test_x[count])):
            test_x[count][i] = int(test_x[count][i])

        for j in range(64, 106):
            if(row[j] == '1'):
                test_x[count].append(income_GNI[j-64]) # 1
                break
        
        #--------------------------------
        for i in highdimension:
            for j in highdimension:
                test_x[count].append(test_x[count][i] * test_x[count][j])
                for k in highdimension:
                    test_x[count].append(test_x[count][i] * test_x[count][j] * test_x[count][k])
        count += 1
    csvfile.close()


with open(file = test_path2, mode = 'r', encoding = "GB18030") as csvfile:
    rows = csv.reader(csvfile)
    first = True
    count = 0
    for row in rows:
        if(first == True):
            first = False
            continue
        temp = int(row[4].strip())
        test_x[count].append(temp)
        test_x[count].append(math.log2(temp))
        test_x[count].append(math.exp(temp))
        for i in highdimension:
            test_x[count].append(test_x[count][i] * temp)
        count += 1


for i in range(len(test_x)):
    test_x[i].append(1)


test_x = np.array(test_x, dtype = float)
for i in range(test_x.shape[1]-1):
    for j in range(test_x.shape[0]):
        test_x[j][i] = (test_x[j][i] - mean[i]) / var[i]
test_ans = test_x.dot(w)

pos = 0
neg = 0
ans = []
for i in range(test_ans.shape[0]):
    if(test_ans[i]>=0):
        ans.append(1)
        pos += 1
    else:
        ans.append(0)
        neg += 1

print(pos, neg)


with open(ans_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "label"])
    for i in range(1, test_ans.shape[0]+1):
        writer.writerow([str(i), ans[i-1]])