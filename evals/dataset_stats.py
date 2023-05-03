import pickle
import argparse
import numpy as np


decimals = 1

my_parser = argparse.ArgumentParser()
my_parser.add_argument('dataset', type=str, help='Location of the dataset folder to get stats for')
my_parser.add_argument('--decimals', type=int, default=1, help='How many decimal places to report results to.')

args = my_parser.parse_args()

dataset_name = args.dataset

with open(f'data/{dataset_name}.pkl', 'rb') as handle:
    dataset_save_dict = pickle.load(handle)

vocab_length = len(dataset_save_dict['vocabulary']['id2token'])
print(f'Vocab Size: {vocab_length}')

train_docs = len(dataset_save_dict['train']['preprocessed_docs'])
print(f'Number of training documents: {train_docs}')

if dataset_save_dict['test']['preprocessed_docs'] is not None:
    test_docs = len(dataset_save_dict['test']['preprocessed_docs'])
    print(f'Number of test documents: {test_docs}')

print()

train_unpre_len = [len(doc.split()) for doc in dataset_save_dict['train']['unpreprocessed_docs']]
print(f'Mean unpreprocessed training document length: {np.round_(np.mean(train_unpre_len), decimals=decimals)}')
print(f'St. dev. unpreprocessed training document length: {np.round_(np.std(train_unpre_len), decimals=decimals)}')
print(f'Min unpreprocessed training document length: {min(train_unpre_len)}')
print(f'Max unpreprocessed training document length: {max(train_unpre_len)}')
print()

train_pre_len = [len(doc.split()) for doc in dataset_save_dict['train']['preprocessed_docs']]
print(f'Mean preprocessed training document length: {np.round_(np.mean(train_pre_len), decimals=decimals)}')
print(f'St. dev. preprocessed training document length: {np.round_(np.std(train_pre_len), decimals=decimals)}')
print(f'Min preprocessed training document length: {min(train_pre_len)}')
print(f'Max preprocessed training document length: {max(train_pre_len)}')
print()

if dataset_save_dict['test']['preprocessed_docs'] is not None:
    test_unpre_len = [len(doc.split()) for doc in dataset_save_dict['test']['unpreprocessed_docs']]
    print(f'Mean unpreprocessed test document length: {np.round_(np.mean(test_unpre_len), decimals=decimals)}')
    print(f'St. dev. unpreprocessed test document length: {np.round_(np.std(test_unpre_len), decimals=decimals)}')
    print(f'Min unpreprocessed test document length: {min(test_unpre_len)}')
    print(f'Max unpreprocessed test document length: {max(test_unpre_len)}')
    print()

    test_pre_len = [len(doc.split()) for doc in dataset_save_dict['test']['preprocessed_docs']]
    print(f'Mean preprocessed test document length: {np.round_(np.mean(test_pre_len), decimals=decimals)}')
    print(f'St. dev. preprocessed test document length: {np.round_(np.std(test_pre_len), decimals=decimals)}')
    print(f'Min preprocessed test document length: {min(test_pre_len)}')
    print(f'Max preprocessed test document length: {max(test_pre_len)}')
