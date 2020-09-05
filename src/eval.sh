#!/usr/bin/env bash


train_data=../data/VAST/vast_train.csv
dev_data=../data/VAST/vast_dev.csv

if [ $1 == 'eval' ]
then
    echo "Evaluating a model"
    python eval_model.py -m "eval" -k BEST -s $2 -i ${train_data} -d ${dev_data}

elif [ $1 == 'predict' ]
then
    echo "Saving predictions from a model to $3"
    python eval_model.py -m "predict" -k BEST -s $2 -i ${train_data} -d ${dev_data} -o $3

else
    echo "Doing nothing"
fi
