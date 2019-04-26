touch $2
mkdir -p $2
python3 attack.py $1 $2
python3 labor_intelligent.py $1 $2