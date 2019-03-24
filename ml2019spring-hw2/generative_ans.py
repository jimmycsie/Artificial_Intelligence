import csv
import numpy as np
import math
import sys

test_path = sys.argv[5]
ans_path = sys.argv[6]
weight_path = "./generative_weight.csv"

temp = []
mean1 = []
mean2 = []
inv_covariance = []
with open(file = weight_path, mode = 'r', encoding = "GB18030") as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        temp.append(row)

mean1 = temp[0]
mean2 = temp[1]
for i in range(2, len(temp)):
    inv_covariance.append(temp[i])
mean1 = np.array(mean1, dtype = float)
mean2 = np.array(mean2, dtype = float)
inv_covariance = np.array(inv_covariance, dtype = float)
class1_count = 7841
class2_count = 24720

count = 0
test_x = []
with open(file = test_path, mode = 'r', encoding = "GB18030") as csvfile:
    rows = csv.reader(csvfile)
    first = True
    for row in rows:
        if(first == True):
            first = False
            continue

        test_x.append(row)               # 15
        for i in range(len(test_x[count])):
            test_x[count][i] = int(test_x[count][i])

        count += 1
    csvfile.close()

for i in range(len(test_x)):
    test_x[i].append(1)

test_x = np.array(test_x, dtype = float)
print(test_x.shape)
ans = []
correct = 0
high = 0
low = 0
for i in range(test_x.shape[0]):
    class1_prob = -0.5 * (test_x[i]-mean1).dot(inv_covariance).dot( (test_x[i]-mean1).transpose() )
    class2_prob = -0.5 * (test_x[i]-mean2).dot(inv_covariance).dot( (test_x[i]-mean2).transpose() ) 
    class1_prob = math.exp(class1_prob) * class1_count
    class2_prob = math.exp(class2_prob) * class2_count

    if(class1_prob > class2_prob):
        ans.append(1)
        high += 1
    else:
        ans.append(0)
        low += 1

print(high, low)
with open(ans_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "label"])
    for i in range(1, len(ans)+1):
        writer.writerow([str(i), ans[i-1]])
