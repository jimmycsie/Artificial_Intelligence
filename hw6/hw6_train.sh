touch $4
python3 segmentation_train.py $1 $3 $4
python3 embedding_train.py $2
python3 RNN_train.py $2 RNN128_50.h5 50
python3 RNN_train.py $2 RNN64_35.h5 35
python3 RNN_train.py $2 RNN64_40.h5 40
python3 RNN_train.py $2 RNN64_45.h5 45
