import json
import nltk
import pickle
import random
import string
import argparse

from gensim.utils import deaccent
from transformers import AutoTokenizer
from nltk.corpus import stopwords as stop_words
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from torch import FloatTensor
from torch.utils.data.dataset import MapDataPipe


def main(dataset, function_dict, min_word_length, custom_data_dict):
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    if dataset == '20ng':
        function_dict[dataset](min_word_length)
    elif dataset == 'custom':
        function_dict[dataset](min_word_length, custom_data_dict)
    else:
        function_dict[dataset]()


def custom_data(min_word_length, custom_data_dict):
    create_dataset = CreateDataset(data_type=custom_data_dict['data_type'],
                                   train_file=custom_data_dict['train_file'],
                                   test_file=custom_data_dict['test_file'],
                                   save_name=custom_data_dict['save_name'],
                                   vocabulary_size=custom_data_dict['vocabulary_size'],
                                   sbert_models=custom_data_dict['sbert_models'],
                                   do_preprocess=custom_data_dict['do_preprocess'],
                                   test_ratio=custom_data_dict['test_ratio'],
                                   min_word_length=min_word_length,
                                   stopwords=custom_data_dict['stopwords'],
                                   max_df=custom_data_dict['max_df'],
                                   min_df=custom_data_dict['min_df'],
                                   remove_numbers=custom_data_dict['remove_numbers'],
                                   batch_size=custom_data_dict['batch_size'])
    create_dataset.save_data_file()


def tweets2011():
    filename = 'data/raw/texts/Tweet.txt'
    save_name = 'data/pickles/tweets2011'
    create_dataset = CreateDataset(data_type='text',
                                   train_file=filename,
                                   test_file=None,
                                   save_name=save_name,
                                   vocabulary_size=999999,
                                   sbert_models=['all-MiniLM-L6-v2'],
                                   do_preprocess=False,
                                   test_ratio=None,
                                   stopwords=None,
                                   batch_size=32)
    create_dataset.save_data_file()


def stackoverflow():
    filename = 'data/raw/texts/StackOverflow.txt'
    save_name = 'data/pickles/stackoverflow'
    create_dataset = CreateDataset(data_type='text',
                                   train_file=filename,
                                   test_file=None,
                                   save_name=save_name,
                                   vocabulary_size=999999,
                                   sbert_models=['all-MiniLM-L6-v2'],
                                   do_preprocess=False,
                                   test_ratio=None,
                                   stopwords=None,
                                   batch_size=32)
    create_dataset.save_data_file()


def googlenews():
    filename = 'data/raw/texts/GoogleNews.txt'
    save_name = 'data/pickles/googlenews'
    create_dataset = CreateDataset(data_type='text',
                                   train_file=filename,
                                   test_file=None,
                                   save_name=save_name,
                                   vocabulary_size=999999,
                                   sbert_models=['all-MiniLM-L6-v2'],
                                   do_preprocess=False,
                                   test_ratio=None,
                                   stopwords=None,
                                   batch_size=32)
    create_dataset.save_data_file()


def nytcorpus():
    filename = 'data/raw/nyt/nyt_output.txt'
    save_name = 'data/pickles/nyt_corpus_no_stopwords'
    create_dataset = CreateDataset(data_type='text',
                                   train_file=filename,
                                   test_file=None,
                                   save_name=save_name,
                                   vocabulary_size=10283,
                                   sbert_models=['all-MiniLM-L6-v2'],
                                   do_preprocess=True,
                                   test_ratio=None,
                                   stopwords=None,
                                   batch_size=32)
    create_dataset.save_data_file()


def contrastive_data():
    datasets = ['20ng', 'wiki', 'imdb']
    vocab_sizes = [2000, 20000, 5000]
    for dataset, vocab_size in zip(datasets, vocab_sizes):
        save_name = f'data/pickles/contrastive_{dataset}'
        create_dataset = CreateDataset(data_type='contrastive',
                                       train_file=dataset,
                                       test_file=dataset,
                                       save_name=save_name,
                                       vocabulary_size=vocab_size,
                                       sbert_models=['all-MiniLM-L6-v2'],
                                       do_preprocess=True,
                                       test_ratio=None,
                                       min_word_length=2,
                                       stopwords='english',
                                       min_df=1,
                                       batch_size=32)
        create_dataset.save_data_file()


def ntm_data():
    for dataset in ['20news', 'snippets', 'w2e', 'w2e_text']:
        filename = f'data/raw/texts/{dataset}_preprocessed_data.txt'
        save_name = f'data/pickles/ntm_{dataset}_fullvocab'
        create_dataset = CreateDataset(data_type='ntm',
                                       train_file=filename,
                                       test_file=None,
                                       save_name=save_name,
                                       vocabulary_size=999999,
                                       sbert_models=['all-MiniLM-L6-v2'],
                                       do_preprocess=False,
                                       test_ratio=None,
                                       stopwords='english',
                                       batch_size=32)
        create_dataset.save_data_file()


def dbpediasample():
    filename = 'data/raw/texts/dbpedia_sample_abstract_20k_unprep.txt'
    save_name = 'data/pickles/wiki20k'
    create_dataset = CreateDataset(data_type='text',
                                   train_file=filename,
                                   test_file=None,
                                   save_name=save_name,
                                   vocabulary_size=2000,
                                   sbert_models=['all-MiniLM-L6-v2'],
                                   do_preprocess=True,
                                   test_ratio=None,
                                   stopwords='english',
                                   batch_size=32)
    create_dataset.save_data_file()


def twentynewsgroups(min_word_length):
    if min_word_length == 1:
        save_name = 'data/pickles/20newsgroups'
    else:
        save_name = f'data/pickles/20newsgroups_mwl{min_word_length}'

    create_dataset = CreateDataset(data_type='sklearn',
                                   train_file='20newsgroups',
                                   test_file='20newsgroups',
                                   save_name=save_name,
                                   vocabulary_size=2000,
                                   sbert_models=['all-MiniLM-L6-v2'],
                                   do_preprocess=True,
                                   test_ratio=None,
                                   min_word_length=min_word_length,
                                   stopwords='english',
                                   batch_size=32)
    create_dataset.save_data_file()


class CreateDataset:
    def __init__(self, data_type, train_file, test_file, save_name, vocabulary_size,
                 sbert_models, do_preprocess, test_ratio=None, min_word_length=1,
                 stopwords=None, max_df=1.0, min_df=1, min_words=1, remove_numbers=True, batch_size=32):

        assert isinstance(sbert_models, list)
        self.sbert_models = sbert_models

        if stopwords is None:
            self.stopwords = set()
        elif isinstance(stopwords, str):
            self.stopwords = set(stop_words.words(stopwords))
        elif isinstance(stopwords, list):
            self.stopwords = set(stopwords)
        else:
            raise TypeError('stopwords should be a string, a list, or None.')

        self.data_type = data_type
        self.train_file = train_file
        self.test_file = test_file
        self.save_name = save_name
        self.vocabulary_size = vocabulary_size
        self.do_preprocess = do_preprocess

        self.test_ratio = test_ratio
        self.min_word_length = min_word_length
        self.max_df = max_df
        self.min_df = min_df
        self.min_words = min_words
        self.remove_numbers = remove_numbers
        self.batch_size = batch_size

    def get_documents(self):
        # text data sets
        if self.data_type == 'text':
            train_documents = [line.strip() for line in open(self.train_file, encoding="utf-8").readlines()]
            test_documents = None
            if self.test_file is not None:
                test_documents = [line.strip() for line in open(self.test_file, encoding="utf-8").readlines()]

        # sklearn data sets (20newsgroups)
        elif self.data_type == 'sklearn':
            assert self.train_file == self.test_file or self.test_file is None
            if self.train_file == '20newsgroups':
                train_documents = fetch_20newsgroups(subset='train').data
                test_documents = None
                if self.test_file is not None:
                    test_documents = fetch_20newsgroups(subset='test').data
            else:
                raise ValueError('Invalid sklearn dataset.')
            
        # ntm data sets
        elif self.data_type == 'ntm':
            train_documents = []
            test_documents = None
            with open(self.train_file) as f:
                file_len = 0
                for line in f:
                    file_len += 1
                    line = line.replace('{', '').replace('}', '')
                    splits = line.split(',')
                    for split in splits:
                        split_len = len(split.split(':'))
                        if split_len == 2:
                            s1, s2 = split.split(':')
                            if s1.strip() == '"bow"':
                                train_documents.append(s2.strip().replace('"', ''))

        # clntm data sets
        elif self.data_type == 'contrastive':
            assert self.train_file in ['20ng', 'imdb', 'wiki']
            assert self.train_file == self.test_file or self.test_file is None

            if self.train_file == '20ng':
                subfolder = f'{self.train_file}/{self.train_file}_all'
            else:
                subfolder = self.train_file

            with open(f'data/raw/clntm/scholar_data/data/{subfolder}/train.jsonlist') as f:
                train_list = f.readlines()
            train_documents = [json.loads(doc)['text'] for doc in train_list]

            if self.test_file is not None:
                with open(f'data/raw/clntm/scholar_data/data/{subfolder}/test.jsonlist') as f:
                    test_list = f.readlines()
                test_documents = [json.loads(doc)['text'] for doc in test_list]
            else:
                test_documents = None
        else:
            raise ValueError('Invalid data type.')
        return train_documents, test_documents

    def preprocessing(self, documents):
        if self.do_preprocess:
            # remove accents
            preprocessed_docs_tmp = [deaccent(doc.lower()) for doc in documents]
            # remove punctuation
            preprocessed_docs_tmp = [doc.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                                     for doc in preprocessed_docs_tmp]
            if self.remove_numbers:
                preprocessed_docs_tmp = [doc.translate(str.maketrans("0123456789", ' ' * len("0123456789")))
                                         for doc in preprocessed_docs_tmp]
            # remove stopwords
            preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) >= self.min_word_length
                                     and w not in self.stopwords]) for doc in preprocessed_docs_tmp]
        else:
            preprocessed_docs_tmp = [' '.join([w for w in doc.split()]) for doc in documents]
        return preprocessed_docs_tmp

    def get_temp_vocabulary(self, documents):
        preprocessed_docs_tmp = self.preprocessing(documents)
        # bag of words
        vectorizer = CountVectorizer(max_features=self.vocabulary_size, max_df=self.max_df, min_df=self.min_df)
        vectorizer.fit(preprocessed_docs_tmp)
        temp_vocabulary = set(vectorizer.get_feature_names_out())
        return temp_vocabulary

    def preprocess_documents(self, documents, temp_vocabulary):
        # preprocessed documents without removing non-vocabulary words
        preprocessed_docs_full = self.preprocessing(documents)
        # remove words not in bag of words
        preprocessed_docs_vocab = [' '.join([w for w in doc.split() if w in temp_vocabulary])
                                   for doc in preprocessed_docs_full]

        # document lists, and which documents are retained (len >= min_words)
        preprocessed_docs, unpreprocessed_docs, retained_indices = [], [], []
        for i, doc in enumerate(preprocessed_docs_vocab):
            if len(doc) > 0 and len(doc) >= self.min_words:
                preprocessed_docs.append(doc)
                unpreprocessed_docs.append(documents[i])
                retained_indices.append(i)

        return preprocessed_docs, unpreprocessed_docs, retained_indices, preprocessed_docs_full

    def save_data_file(self):
        train_documents, test_documents = self.get_documents()

        assert isinstance(train_documents, list)
        if test_documents is not None:
            assert isinstance(test_documents, list)
            full_documents = train_documents.copy() + test_documents.copy()
        else:
            full_documents = train_documents.copy()
        temp_vocabulary = self.get_temp_vocabulary(full_documents)

        if test_documents is None:
            if self.test_ratio is not None:
                full_documents = train_documents.copy()
                random.shuffle(full_documents)
                num_test_examples = int(self.test_ratio * len(full_documents))
                test_documents = full_documents[:num_test_examples]
                train_documents = full_documents[num_test_examples:]
        train_preproc_docs, train_unpreproc_docs, train_indices, train_preproc_docs_full = \
            self.preprocess_documents(train_documents, temp_vocabulary)
        if test_documents is not None:
            test_preproc_docs, test_unpreproc_docs, test_indices, test_preproc_docs_full = \
                self.preprocess_documents(test_documents, temp_vocabulary)
            preproc_docs = train_preproc_docs + test_preproc_docs
        else:
            test_preproc_docs = None
            test_unpreproc_docs = None
            test_indices = None
            test_preproc_docs_full = None
            preproc_docs = train_preproc_docs.copy()

        # bag of words embeddings
        vectorizer = CountVectorizer()
        bow_embeddings_sparse = vectorizer.fit_transform(preproc_docs)

        # bag of words vocabulary
        real_vocabulary = {
            'id2token': dict(),
            'token2id': dict()
        }

        for k, v in vectorizer.vocabulary_.items():
            real_vocabulary['id2token'][v] = k
            real_vocabulary['token2id'][k] = v

        vocabulary = list(set([item for doc in preproc_docs for item in doc.split()]))
        assert sorted(list(real_vocabulary['id2token'].values())) == sorted(vocabulary)
        assert sorted(list(real_vocabulary['token2id'].keys())) == sorted(vocabulary)

        train_bow_embeddings_sparse = vectorizer.transform(train_preproc_docs)
        train_bow_embeddings = FloatTensor(train_bow_embeddings_sparse.todense())
        if test_documents is not None:
            test_bow_embeddings_sparse = vectorizer.transform(test_preproc_docs)
            test_bow_embeddings = FloatTensor(test_bow_embeddings_sparse.todense())
        else:
            test_bow_embeddings = None

        dataset_save_dict = {
            'vocabulary': real_vocabulary,
            'sparse_corpus_bow': bow_embeddings_sparse,
            'train': {
                'preprocessed_docs': train_preproc_docs,
                'unpreprocessed_docs': train_unpreproc_docs,
                'indices': train_indices,
                'preprocessed_docs_full': train_preproc_docs_full,
                'bow_embeddings': train_bow_embeddings,
                'bert_embeddings': dict()
            },
            'test': {
                'preprocessed_docs': test_preproc_docs,
                'unpreprocessed_docs': test_unpreproc_docs,
                'indices': test_indices,
                'preprocessed_docs_full': test_preproc_docs_full,
                'bow_embeddings': test_bow_embeddings,
                'bert_embeddings': dict()
            }
        }

        # sbert embeddings
        for sbert_model in self.sbert_models:
            model = SentenceTransformer(sbert_model)
            train_bert_embeddings = model.encode(train_unpreproc_docs,
                                                 batch_size=self.batch_size,
                                                 show_progress_bar=True,
                                                 convert_to_tensor=True)
            train_bert_embeddings = train_bert_embeddings.detach().cpu()
            assert len(train_bow_embeddings) == len(train_bert_embeddings)

            if test_documents is not None:
                test_bert_embeddings = model.encode(test_unpreproc_docs,
                                                    batch_size=self.batch_size,
                                                    show_progress_bar=True,
                                                    convert_to_tensor=True)
                test_bert_embeddings = test_bert_embeddings.detach().cpu()
                assert len(test_bow_embeddings) == len(test_bert_embeddings)
            else:
                test_bert_embeddings = None

            dataset_save_dict['train']['bert_embeddings'][sbert_model] = train_bert_embeddings
            dataset_save_dict['test']['bert_embeddings'][sbert_model] = test_bert_embeddings

        with open(f'{self.save_name}.pkl', 'wb') as handle:
            pickle.dump(dataset_save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_dataset(pickle_name):
    with open(f'{pickle_name}.pkl', 'rb') as handle:
        dataset_save_dict = pickle.load(handle)

    return dataset_save_dict


def get_datapipe(pickle_name, split, frozen_embeddings=True,
                 sbert_model=None, hugface_model=None, max_length=None):

    assert split in ['train', 'test']

    with open(f'{pickle_name}.pkl', 'rb') as handle:
        dataset_save_dict = pickle.load(handle)

    if dataset_save_dict[split]['bow_embeddings'] is None:
        return None

    if frozen_embeddings:
        if sbert_model is None:
            return FrozenMapDataPipeBow(x_bow=dataset_save_dict[split]['bow_embeddings'])
        else:
            return FrozenMapDataPipeBoth(x_bow=dataset_save_dict[split]['bow_embeddings'],
                                         x_bert=dataset_save_dict[split]['bert_embeddings'][sbert_model])
    else:
        assert hugface_model is not None and max_length is not None
        tokenizer = AutoTokenizer.from_pretrained(hugface_model)
        return ThawedMapDataPipe(x_bow=dataset_save_dict[split]['bow_embeddings'],
                                 sentences=dataset_save_dict['unpreprocessed_docs'],
                                 tokenizer=tokenizer,
                                 max_length=max_length)


class FrozenMapDataPipeBoth(MapDataPipe):
    def __init__(self, x_bow, x_bert):
        self.x_bow = x_bow
        self.x_bert = x_bert

    def __len__(self):
        assert len(self.x_bow) == len(self.x_bert)
        return len(self.x_bow)

    def __getitem__(self, index):
        item = {
            'x_bow': self.x_bow[index],
            'x_bert': self.x_bert[index]
        }
        return item


class FrozenMapDataPipeBow(MapDataPipe):
    def __init__(self, x_bow):
        self.x_bow = x_bow

    def __len__(self):
        return len(self.x_bow)

    def __getitem__(self, index):
        item = {
            'x_bow': self.x_bow[index],
        }
        return item


class ThawedMapDataPipe(MapDataPipe):
    def __init__(self, x_bow, sentences, tokenizer, max_length):
        self.x_bow = x_bow
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        assert len(self.x_bow) == len(self.sentences)
        return len(self.x_bow)

    def __getitem__(self, index):
        x_bert = self.sentences[index]
        x_bert = self.tokenizer.encode_plus(x_bert,
                                            add_special_tokens=False,
                                            max_length=self.max_length,
                                            padding='max_length',
                                            truncation=True,
                                            return_attention_mask=True,
                                            return_tensors='pt')
        item = {
            'x_bow': self.x_bow[index],
            'x_bert': x_bert
        }
        return item


def type_int_or_float(value):
    if not isinstance(value, int) and not isinstance(value, float):
        raise argparse.ArgumentTypeError('Input must be an int or a float.')
    return value


if __name__ == '__main__':

    function_dict = {
        'tweets2011': tweets2011,
        'stackoverflow': stackoverflow,
        'googlenews': googlenews,
        'wiki20k': dbpediasample,
        '20ng': twentynewsgroups,
        'nytcorpus': nytcorpus,
        'contrastive': contrastive_data,
        'ntm': ntm_data,
        'custom': custom_data
    }
    
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('dataset', type=str, choices=list(function_dict.keys()), help='Data set to create pickle for.')
    my_parser.add_argument('--mwl', type=int, default=1, help='Minimum word length to keep for bag-of-words embeddings.')
    my_parser.add_argument('--train_file', type=str, default=None, help='Training file.')
    my_parser.add_argument('--test_file', type=str, default=None, help='Test file.')
    my_parser.add_argument('--save_name', type=str, default=None, help='Name for saving the data pickle.')
    my_parser.add_argument('--vocab_size', type=int, default=999999, help='Vocabulary size.')
    my_parser.add_argument('--sbert_model', type=str, default=None, help='SBERT model.')
    my_parser.add_argument('--preprocess', type=bool, action='store_true', help='Preprocess the data.')
    my_parser.add_argument('--test_ratio', type=str, default=None, help='Test set ratio.')
    my_parser.add_argument('--stopwords', type=str, default=None, help='Stopwords to use (e.g. "english").')
    my_parser.add_argument('--max_df', type=type_int_or_float, default=1.0, help='Ignore terms with higher document frequency.')
    my_parser.add_argument('--min_df', type=type_int_or_float, default=1, help='Ignore terms with lower document frequency.')
    my_parser.add_argument('--keep_numbers', type=bool, action='store_true', help='Keep numbers in the data')
    my_parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generating SBERT embeddings.')


    args = my_parser.parse_args()

    if args.dataset == 'custom':
        if args.train_file is None and args.save_name is None:
            raise ValueError('Please provide train_file and save_name in your custom arguments.')
        elif args.train_file is None:
            raise ValueError('Please provide train_file in your custom arguments.')
        elif args.save_name is None:
            raise ValueError('Please provide save_name in your custom arguments.')
        
        sbert_models = [] if args.sbert_model is None else [args.sbert_model]

        custom_data_dict = {
            'train_file': args.train_file,
            'test_file': args.test_file,
            'save_name': args.save_name,
            'vocabulary_size': args.vocab_size,
            'sbert_models': sbert_models,
            'do_preprocess': args.preprocess,
            'test_ratio': args.test_ratio,
            'stopwords': args.stopwords,
            'max_df': args.max_df,
            'min_df': args.min_df,
            'remove_numbers': not args.keep_numbers,
            'batch_size': args.batch_sze
        }
    else:
        custom_data_dict = None

    main(args.dataset, function_dict, args.mwl, custom_data_dict)
