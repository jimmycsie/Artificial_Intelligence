import csv
import numpy as np
import math
import sys

#test_path = "/Users/jimmy/desktop/ML2019SPRING/ml2019spring-hw1/test.csv"
#ans_path = "/Users/jimmy/desktop/ML2019SPRING/ml2019spring-hw1/sampleSubmission.csv"
test_path = sys.argv[1]
ans_path = sys.argv[2]
weight_path = "./weight.csv"

w = []
# extracting weight
with open(file = weight_path, mode = 'r', encoding = "GB18030") as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        w.append(row)
    csvfile.close()

w = np.array(w, dtype = float) 

# extracting test data
data = []
testing_feature = []
with open(file = test_path, mode = 'r', encoding = "GB18030") as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        data.append(row[2:])
    csvfile.close()

for i in range(240):
    temp = []
    temp.append(1)
    for j in range(18):
        for k in range(9):
            if(data[i*18+j][k] == "NR"):
                temp.append(0.)
            else:
                temp.append(float(data[i*18+j][k]))
    testing_feature.append(temp)

testing_feature = np.array(testing_feature, dtype = float)             # 239 * 162
ans = testing_feature.dot(w)
for i in range(ans.shape[0]):
    if(ans[i][0]<0):
        ans[i][0] = 0



# writing ans to csv

with open(ans_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "value"])
    for i in range(testing_feature.shape[0]):
        writer.writerow(["id_"+str(i), ans[i][0]])
