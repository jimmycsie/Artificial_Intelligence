import csv
import sys
import jieba
from gensim.models import word2vec

#input_path1 = "./data/train_x.csv"
input_path2 =  sys.argv[1]                 #   "./data/test_x.csv"
jieba.load_userdict(sys.argv[2])           #   "dict.txt.big"
#seg_list = jieba.cut("韓國瑜只會在高雄賣水果，還想競選總統哈哈", cut_all=False)   # 精確模式
#print("/".join(seg_list))



data= []
train_seg=[]
"""
with open(file = input_path1, mode = 'r') as csvfile:
    rows = csv.reader(csvfile)
    first = True
    for row in rows:
        if(first == True):
            first = False
            continue
        data.append(row[1])
    csvfile.close()
"""

with open(file = input_path2, mode = 'r', encoding="utf-8") as csvfile:
    rows = csv.reader(csvfile)
    first = True
    for row in rows:
        if(first == True):
            first = False
            continue
        data.append(row[1])
    csvfile.close()


for i in range(len(data)):
    train_seg.append([' '.join(list(jieba.cut(data[i],cut_all=False)))])

print(train_seg[0])
print(len(train_seg))
print(len(train_seg[0]))

# 將 jieba 的斷詞產出存檔
segment_path ='segment_test.txt'
with open(segment_path,'w', encoding="utf-8") as fd:
    for i in range(len(train_seg)):
        fd.write(train_seg[i][0])
        fd.write('\n')

