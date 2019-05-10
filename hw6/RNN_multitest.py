from gensim.models import word2vec
import csv
import numpy as np
import sys
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

model = word2vec.Word2Vec.load("word2vec.model")
model_num = 4
RNN_model = [0] * model_num
RNN_model[0] = load_model('RNN128_50.h5')
RNN_model[1] = load_model('RNN64_45.h5')
#RNN_model[2] = load_model('RNN64_30.h5')
RNN_model[2] = load_model('RNN64_40.h5')
RNN_model[3] = load_model('RNN64_35.h5')
model_time_step = [50, 45, 40, 35]
#model_second = load_model('RNN.h5')

input_path = "segment_test.txt"
ans_path = sys.argv[1]                     #   "submission.csv"
data_size = 20000



test_seg = []
rude_word = []
"""
# get rude word but not in word2vec
fp = open("word_expand.txt", "r", encoding="utf-8")
line = fp.readline()
rude_word = line.split(' ')
fp.close()
"""
rude_word = "始作俑者 幹文 肥宅社 肥宅座 賤女 豬之 小鬼子 早死早好 練肖話 紅你媽 Egg Itrash xxxxshit 魯蛇癌 Shit 臭甲文 死腦粉 吸洋腸 哈龜 李宗瑞"
rude_word = rude_word.split(' ')

# get segment data
fp = open(input_path, "r", encoding="utf-8")
for i in range(data_size):
    temp = []
    line = fp.readline()
    line = line.split(' ')
    for j in range(len(line)):
        #if( line[j]!='' and (line[j][0]=='b' or line[j][0]=='B') ) :
        #    continue
        if(line[j]!=''):
            if(line[j][-1]=='\n'):
                temp.append(line[j][:-1])
            else:
                temp.append(line[j])
    test_seg.append( temp )
fp.close()


# transfer segment data to vector
def seg_to_vec(time_step, test_seg):
    zero_vector = [0]*250
    y_label = []
    test_vec = []
    for i in range(data_size):
        temp = []
        count = 0
        y_count = 0
        for j in range(len(test_seg[i])):
            if(j<len(test_seg[i])-1 and test_seg[i][j]==test_seg[i][j+1]):
                continue
            if( test_seg[i][j] in model.wv ):
                vector = model.wv[test_seg[i][j]]
                temp.append(vector)
                count += 1
            elif( test_seg[i][j] in rude_word ):
                vector = model.wv["幹"]
                temp.append(vector)
                count += 1
            if(count == time_step):
                test_vec.append(temp)
                y_count += 1
                temp = []
                count = 0
        if(count != 0):
            j = 0
            while(count != time_step):
                if( test_seg[i][j] in model.wv ):
                    vector = model.wv[test_seg[i][j]]
                    temp.append(vector)
                    count += 1
                elif( test_seg[i][j] in rude_word ):
                    vector = model.wv["幹"]
                    temp.append(vector)
                    count += 1
                j = (j+1)%len(test_seg[i])
            y_count += 1
            test_vec.append(temp)
        y_label.append(y_count)
    return test_vec, y_label

vote = [0]*data_size
second_vote = [0]*data_size

# store first five model voting answer
for i in range(model_num):
    test_vec, y_label = seg_to_vec(model_time_step[i], test_seg) 
    test_vec = np.array(test_vec, dtype = float)
    test_vec = np.reshape(test_vec, (test_vec.shape[0], model_time_step[i], 250))
    result = RNN_model[i].predict(test_vec)     # get result
    print(result[0], result[1])

    if(i >= 2):
        weight = 2
    else:
        weight = 1
    index = 0
    pos = 0
    neg = 0
    for i in range(data_size):
        count = 0
        sum = 0
        while(count < y_label[i]):
            sum += result[index][0]
            count += 1
            index += 1
        if(sum > 0.5*y_label[i]):         # average score
            vote[i] += weight
            pos += 1
        else:
            vote[i] -= weight
            neg += 1
    print(pos, neg)


# embedding test --------------------------------------------------------
RNN_model = load_model('RNN_1.h5')
time_step = 35
test_encoder = []
pos = 0
neg = 0

#test_encoder = np.load( "encoder_test.npy" )

# get test segment encoder
fp = open("encoder.txt", "r", encoding="utf-8")
for i in range(data_size):
    temp = []
    line = fp.readline()
    line = line.split(' ')
    for j in range(len(line)):
        if(line[j]!=''):
            if(line[j][-1]=='\n'):
                temp.append(line[j][:-1])
            else:
                temp.append(line[j])
    test_encoder.append( temp )
fp.close()


padded_docs = pad_sequences(test_encoder, maxlen=time_step, padding='pre')
result = RNN_model.predict(padded_docs)
for i in range(data_size):
    if(result[i]>0.5):
        vote[i] += 1
        pos += 1
    else:
        vote[i] -= 1
        neg += 1

print(pos, neg)
#-------------------------------------------------------------------------

print(result[0], test_seg[0])
print(result[1], test_seg[1])

pos = 0
neg = 0
ans = []
for i in range(data_size):
    if(vote[i]>0):
        ans.append(1)
        pos += 1
    else:
        ans.append(0)
        neg += 1
ans[1039] = 1
ans[1012] = 1
print("average :", pos, neg)



with open(ans_path, 'w', encoding="utf-8", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "label"])
    for i in range(len(ans)):
        writer.writerow([str(i), ans[i]])

