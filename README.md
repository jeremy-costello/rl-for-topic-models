# rl-for-topic-models

## Reinforcement Learning for Topic Models

Code accompanying the forthcoming paper "Reinforcement Learning for Topic Models" in the Findings of the Association for Computational Linguistics: ACL 2023.

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

## EXPERIMENTS

## EVALUATIONS

