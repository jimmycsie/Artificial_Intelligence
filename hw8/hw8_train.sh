touch $1
wget -O CNN_teacher.h5 'https://www.dropbox.com/s/1jo0izflznwhhef/CNN_teacher.h5?dl=1'
wget -O aug.npy 'https://www.dropbox.com/s/bj1ghj6mcz10m88/aug.npy?dl=1'
python3 CNN_train.py $1