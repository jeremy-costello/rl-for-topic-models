# rl-for-topic-models
Reinforcement Learning for Topic Models

***temp***

INSTALL \
python 3.9 \
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch \
pip install -r requirements.txt

DATASETS

Pre-training is a Hot Topic:

Tweets 2011 \
from https://trec.nist.gov/data/tweets \
put Tweet.txt in the data/raw/texts folder \
python data/dataset.py tweets2011 \
if you need help, email me

Stack Overflow \
from https://github.com/qiang2100/STTM \
put StackOverflow.txt in the data/raw/texts folder \
python data/dataset.py stackoverflow

Google News \
from https://github.com/qiang2100/STTM \
put GoogleNews.txt in the data/raw/texts folder \
python data/dataset.py googlenews

Wiki 20k \
from https://github.com/vinid/data \
put dbpedia_sample_abstract_20k_unprep.txt in the data/raw/texts folder \
python data/dataset.py wiki20k

20 Newsgroups \
python data/dataset.py 20ng

Other:

NYT \
from https://catalog.ldc.upenn.edu/LDC2008T19 \
put tarball in data/raw/nyt \
python data/raw/nyt/nyt_untar.py \
python data/dataset.py nytcorpus

CLNTM \
from https://github.com/nguyentthong/CLNTM \
put tarball in data/raw/clntm \
python data/raw/clntm/scholar_untar.py \
python data/dataset.py contrastive

NTM \
from https://github.com/smutahoang/ntm \
put \*_preprocessed_data.txt in data/raw/texts

Our Paper:
20 Newsgroups
python data/dataset.py 20ng --mwl 3
