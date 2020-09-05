#Software for 'Zero-Shot Stance Detection: A Dataset and Model Using Generalized Topic Representations'
Submission to EMNLP 2020

## Directory Structure
Download and unzip the data and place it inside the software directory.
Then the structure should be:  
.  
+-- data   
|  +-- mpqa  
|  |  +-- subjclueslen1-HLTEMNLP05.README  
|  |  +-- subjclueslen1-HLTEMNLP05.tff  
|  +-- VAST  
|  |  +-- vast_train.csv  
|  |  +-- vast_dev.csv  
|  |  +-- vast_test.csv 
|  |  +-- vast_test-sentswap.csv   
+-- config  
|  |  +-- config-bert-joint.txt  
|  |  +-- config-bert-sep.txt  
|  |  +-- config-bicond.txt  
|  |  +-- config-crossnet.txt  
|  |  +-- config-cffnn.txt  
|  |  +-- config-tganet.txt  
|  |  +-- hyperparam-bert-joint.txt  
|  |  +-- hyperparam-bert-sep.txt  
|  |  +-- hyperparam-bicond.txt  
|  |  +-- hyperparam-crossnet.txt  
|  |  +-- hyperparam-cffnn.txt  
|  |  +-- hyperparam-tganet.txt  
+-- checkpoints   
+-- resources  
|  +-- topicreps  
|  |  +-- bert_tfidfW_ward_euclidean_197.centroids.npy  
|  |  +-- bert_tfidfW_ward_euclidean_197-dev.labels.pkl  
|  |  +-- bert_tfidfW_ward_euclidean_197-train.labels.pkl  
|  |  +-- bert_tfidfW_ward_euclidean_197-test.labels.pkl  
|  +-- glove.6B.100d.vectors.npy  
|  +-- glove.6B.100d.vocab.pkl  
|  +-- text_vocab_top10000.txt  
|  +-- topic_vocab.txt     
+-- src  
|  +--clustering  
|  |  +-- stance_clustering.py   
|  |  +-- gen_reps.sh  
|  +-- modeling    
|  |  +-- models.py   
|  |  +-- model_layers.py  
|  |  +-- data_utils.py    
|  |  +-- datasets.py  
|  |  +-- input_models.py  
|  |  +-- model_utils.py  
|  |  +-- eval_model.py   
|  |  +-- baselines.py   
|  +-- train_model.py  
|  +-- hyperparam_selection.py  
|  +-- train.sh  
|  +-- README.md


## Requirements
python                    3.7.6  
scikit-learn              0.22.1  
transformers              2.3.0  
pytorch                   1.5.0 (Cuda 9.2)
numpy                     1.18.1  
pandas                    0.25.3  
matplotlib                3.1.3
scipy                     1.4.1  

## Training a model
Run
```angular2html
cd src/
./train.sh <config_name> <num_warmup_epochs> <optimization_key>
```
For example:
```angular2html
cd src/
./train.sh ../config/config-tganet.txt 0 "f-0_macro"
```
Opimization key is for early stopping. Potential values are
- "f_macro": macro F1 on all examples in the dev set
- "f-0_macro": macro F1 on zero-shot examples in the dev set
- "f-1_macro": macro F1 on few-shot examples in the dev set

## Evaluating a model
Run
```angular2html
cd src/
./eval.sh "eval" <config_name> 
```

For example
```
cd src/
./eval.sh "eval" ../config/config-tganet.txt
```

## Saving predictions from a model
Run
```angular2html
cd src/
./eval.sh "predict" <config_name> <output_file_name>
```

For example
```angular2html
cd src/
./eval.sh "predict" ../config/config-tganet.txt ../tganet_predictions.csv
```


## Evaluating baselines
For BoWV model, run
```angular2html
cd src/modeling/
python baselines.py -m "eval" -i vast_train.csv -d vast_test.csv -t "bowv"
```

For CMaj model, run
```angular2html
cd src/modeling/
python baselines.py -m "eval" -i vast_train.csv -d vast_test.csv -t "cmaj"
```

## Hyperparameter search
Run
```angular2html
cd src/
python hyperparameter_selection.py -m 1 -k <optimization_key> -s <config_name>
```

For example
```angular2html
python hyperparameter_selection.py -m 1 -k "f-0_macro" -s ../config/hyperparam-tganet.txt
```

## Generate Generalized Topic Representations
Run
```angular2html
cd src/clustering/
./gen_reps.sh
```