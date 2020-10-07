# Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representations

Paper link: [TODO]  
Please cite:
```angular2html
@inproceedings{Allaway2020Zero,
  title={Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representations},
  author={Emily Allaway and Kathleen McKeown},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  year={2020}
}
```

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
python hyperparam_selection.py -m 1 -k <optimization_key> -s <config_name>
```

For example
```angular2html
python hyperparam_selection.py -m 1 -k "f-0_macro" -s ../config/hyperparam-tganet.txt
```

## Generate Generalized Topic Representations
Run
```angular2html
cd src/clustering/
./gen_reps.sh
```

## Contact Info
Please contact [Emily Allaway](http://www.cs.columbia.edu/~eallaway/) at [eallaway@cs.columbia.edu](eallaway@cs.columbia.edu) with an questions.
