export TRAINING_DATA=input/train_folds.csv
export TESTING_DATA=input/test.csv
export TRAINING_DATA_HISTORY=input/
export MODEL=$1

source C:/Users/kachowdh/Software/anaconda/Scripts/activate base
# FOLD=0 python -m src.train
# FOLD=1 python -m src.train
# FOLD=2 python -m src.train
# FOLD=3 python -m src.train
# FOLD=4 python -m src.train
# FOLD=5 python -m src.train
# FOLD=6 python -m src.train
# FOLD=7 python -m src.train
# FOLD=8 python -m src.train
# FOLD=9 python -m src.train

# python -m src.predict
FOLD=8 python -m src.feature_selection