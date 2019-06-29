import json as js
import jieba
import numpy as np
import csv
from gensim.models import word2vec



ans1 = []
count = 0
with open(file = "./strong/ans.csv", mode = 'r') as csvfile:
    rows = csv.reader(csvfile)
    first = True
    for row in rows:
        if(first == True):
            first = False
            continue
        if(count<15):
            ans1.append(row[1:])
        else:
            ans1.append(row)
        count += 1
    csvfile.close()

temp = []
ans2 = []
count = 0
with open(file = "test_data.csv", mode = 'r') as csvfile:
    rows = csv.reader(csvfile)
    first = True
    for row in rows:
        if(first == True):
            first = False
            continue
        temp.append(row[2])
        count += 1
        if(count==500):
            ans2.append(temp)
            temp = []
            count = 0
    csvfile.close()


q_idx = []
for i in range(1, 10):
    q_idx.append('q_0'+str(i))
for i in range(10, 21):
    q_idx.append('q_'+str(i))

header = []
header.append("Query_Index")
for i in range(1, 10):
    header.append("Rank_00"+str(i))
for i in range(10, 100):
    header.append("Rank_0"+str(i))
for i in range(100, 301):
    header.append("Rank_"+str(i))


ans = []
ans.append(header)
for i in range(15):
    check = {}
    temp = []
    temp.append(q_idx[i])
    for j in range(600):
        if(j%2==0):
            if(ans1[i][j//2] not in check):
                temp.append(ans1[i][j//2])
                check[ans1[i][j//2]] = 1
        else:
            if(ans2[i][j//2] not in check):
                temp.append(ans2[i][j//2])
                check[ans2[i][j//2]] = 1
        if(len(temp)==301):
            break
    ans.append(temp)


for i in range(15, 20):
    ans.append(ans1[i])


for j in range(1, 21):
    check = {}
    for i in range(300):
        if(ans[j][i] not in check):
            check[ans[j][i]] = 1
        else:
            print(j, i)

with open("ans2.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(ans)):
        writer.writerow(ans[i])

