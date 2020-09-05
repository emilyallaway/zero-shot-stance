#!/usr/bin/env bash

train_data=../../data/VAST/vast_train.csv
dev_data=../../data/VAST/vast_dev.csv
test_data=../../data/VAST/vast_test.csv

echo "Saving document and topic vectors from BERT"
python stance_clustering.py -m 1 -i ${train_data} -d ${dev_data} -e ${test_data}

echo "Generating generalized topic representations through clustering"
python stance_clustering.py -m 2 -i ${train_data} -d ${dev_data} -p ../../resources/topicreps/ -k 197

echo "Getting cluster assignments"
python stance_clustering.py -m 3 -i ${train_data} -d ${dev_data} -e ${test_data} -p ../../resources/topicreps/