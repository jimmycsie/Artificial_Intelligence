touch $2
mkdir $2
python3 saliency_map.py $1 $2
python3 filter.py $1 $2
python3 localexplain.py $1 $2