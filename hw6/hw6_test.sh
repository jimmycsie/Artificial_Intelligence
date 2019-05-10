touch $3
wget -O RNN_1.h5 'https://www.dropbox.com/s/rpxnr0ik5k4l69v/RNN_1.h5?dl=1'
wget -O RNN64_45.h5 'https://www.dropbox.com/s/8qwc9klh3l04u1o/RNN64_45.h5?dl=1"'
wget -O RNN64_40.h5 'https://www.dropbox.com/s/fzop00i6r6ch2dl/RNN64_40.h5?dl=1"'
wget -O RNN64_35.h5 'https://www.dropbox.com/s/edf4zhmeao8rdbt/RNN64_35.h5?dl=1"'
wget -O RNN128_50.h5 'https://www.dropbox.com/s/yit3sp4bbvvp3vm/RNN128_50.h5?dl=1"'
python3 segmentation.py $1 $2
python3 RNN_multitest.py $3
