#Corpus-Level End-to-End Exploration for Interactive Systems

### Dependencies:
```
baselines==0.1.6
tensorflow==1.12
gym==0.10.9
nltk==3.4
scikit-learn==0.20.2
```

###Run the code

##### 0 Extract documents and segment text
```shell script
python texttiling.py --corpus_direc /path/to/nyt/corpus 
```

##### 1 Preprocess and build the corpus embedding
```shell script
python preprocessing.py
```

##### 2 Install the customized environment
```shell script
pip install -e ds_gym/
```

##### 3 Run the RL agent
```shell script
./main.sh
```
The running log will be saved in the directory `ds_log`, which will be used for evaluation.

##### 4 Evaluate
Generate run file that is compatible with TREC DD evaluation scripts
```shell script
python eval.py generate --run /your/run/file
```
Evaluate with TREC DD evaluation script
```shell script
python eval.py metrics --cutoff 10
```

### Citation
If this repo is useful for you, please cite the following paper:

    @inproceedings{ce3_2019,
        title={Corpus-Level End-to-End Exploration for Interactive Systems},
        author={Tang, Zhiwen and Yang, Grace Hui},
        journal={AAAI 2020},
        year={2020}
    }
