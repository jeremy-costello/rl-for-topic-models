# rl-for-topic-models

## Reinforcement Learning for Topic Models

Code accompanying the paper [Reinforcement Learning for Topic Models](https://aclanthology.org/2023.findings-acl.265/) in *Findings of the Association for Computational Linguistics: ACL 2023*.

## Table of Contents

- [Install](#install)
- [Training](#training)
- [Data](#data)
- [Experiments](#experiments)
- [Evaluations](#evaluations)

## INSTALL
```python
conda create -n rl-for-topic-models python=3.9
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## TRAINING

- update the configs in ```model/decoder_network.py``` and ```trainer/config.py``` with your training settings.
- run ```python train.py /path/to/data/pickle```
  - add ```--test``` if your data has a test subset

## DATA

### Pre-training is a Hot Topic

#### Tweets 2011
- get raw data from https://trec.nist.gov/data/tweets
- put processed ```Tweet.txt``` in the ```data/raw/texts``` folder
- run ```python data/dataset.py tweets2011```

#### Stack Overflow
- get data from https://github.com/qiang2100/STTM
- put ```StackOverflow.txt``` in the ```data/raw/texts``` folder
- run ```python data/dataset.py stackoverflow```

#### Google News
- get data from https://github.com/qiang2100/STTM
- put ```GoogleNews.txt``` in the ```data/raw/texts``` folder
- run ```python data/dataset.py googlenews```

#### Wiki 20k
- get data from https://github.com/vinid/data
- put ```dbpedia_sample_abstract_20k_unprep.txt``` in the ```data/raw/texts folder```
- run ```python data/dataset.py wiki20k```

#### 20 Newsgroups
- run ```python data/dataset.py 20ng```

### Other Comparison Papers

#### New York Times
- get raw data from https://catalog.ldc.upenn.edu/LDC2008T19
- put tarball in ```data/raw/nyt```
- run ```pip install beautifulsoup4 lxml```
- run ```python data/raw/nyt/nyt_untar.py```
- run ```python data/dataset.py nytcorpus```

#### Contrastive Learning for Neural Topic Model
- get raw data from https://github.com/nguyentthong/CLNTM
- put tarball in ```data/raw/clntm```
- run ```python data/raw/clntm/scholar_untar.py```
- run ```python data/dataset.py contrastive```

#### Benchmarking Neural Topic Models
- get data from https://github.com/smutahoang/ntm
- put ```*_preprocessed_data.txt``` in ```data/raw/texts```

### Our Experiments

#### 20 Newsgroups
- run ```python data/dataset.py 20ng --mwl 3```

### Custom Data

- run ```python data/dataset.py custom --train_file /path/to/train/file --save_name /path/to/save/name```
  - other arguments can be found at the bottom of ```data/dataset.py```

## EXPERIMENTS

- update search_dict in ```run_experiments.py``` with your hyperparameter search values.
- run ```python run_experiments.py experiment_name num_seeds --meta_seed meta_seed```
  - meta seed will be randomly chosen from ```random.randint(0, 2 ** 32)``` if not included as argument

## EVALUATIONS

### Compute Metrics
- run ```python -m evals.compute_metrics topk num_experiments num_seeds /path/to/data/pickle /path/to/experiment```

### Dataset Statistics
- run ```python -m evals.dataset_stats /path/to/data/pickle```

### Dropout Sweep
- run experiment with hyperparameters from paper
- run ```python -m evals.dropout_sweep /path/to/dropout/experiment```

### Benchmarking Neural Topic Models Plot
- run ```empirical_studies/examine_models.py``` from https://github.com/smutahoang/ntm with ```top_ks = [10]```
- move the ```run.*.pkl``` files into the ```ntm_runs``` folder
- run experiments with hyperparameters from paper
  - call them ```ntm_20news_sweep```, ```ntm_snippets_sweep```, ```ntm_w2e_sweep```, and ```ntm_w2e_text_sweep```
- run ```python -m evals.plot_ntm_results```

### Saving Topic Words
- run ```python -m evals.save_topic_words topk /path/to/data/pickle /path/to/experiment/experiment_num/seeds/seed_num/seed_num_plotting_arrays.pkl```
