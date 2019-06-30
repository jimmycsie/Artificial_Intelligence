import csv
import numpy as np

relevance_path = "TD.csv"
stop = [230, 460, 500, 595, 734]
# 支持前總統保外就醫 (15, q16)
# 年金改革         (16, q17)
# 同意動物實驗      (17, q18)
# 油價應該凍漲      (18, q19)
# 反對旺旺併購      (19, q20)
peek = []
exclude = []

count = 0 #734
with open(file = relevance_path, mode = 'r', encoding = "utf-8") as csvfile:
    rows = csv.reader(csvfile)
    first = True
    temp = []
    temp2 = []
    for row in rows:
        if(first == True):
            first = False
            continue
        if(int(row[2])>0):
            temp.append(row[1])
        else:
            temp2.append(row[1])
        count += 1
        for i in stop:
            if(i == count):
                peek.append(temp)
                exclude.append(temp2)
                temp = []
                temp2 = []
                break
        if(count == 734):
            break
    csvfile.close()

peek = np.array(peek)
exclude = np.array(exclude)


np.save("peek.npy", peek)
np.save("exclude.npy", exclude)



