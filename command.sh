#!/bin/bash
echo "begin"
source activate nsxenv
python dataset_make_2.py --env prod
nohup python train.py > logs/train_log.txt 2>&1 &
python evaluation.py
