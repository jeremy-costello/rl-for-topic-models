# https://github.com/maifeng/Examples_UMass-Coherence/blob/master/umass_coherence.py
import math
import torch
import pickle
import numpy as np
from itertools import combinations
from collections import namedtuple

import gensim.downloader as api
from gensim.models import KeyedVectors

from data.dataset import get_dataset


# wikipedia coherence
# https://github.com/dice-group/Palmetto
# https://aclanthology.org/N10-1012.pdf
# https://github.com/dice-group/palmetto-py
# c_v coherence
# https://stats.stackexchange.com/questions/406216/what-is-the-formula-for-c-v-coherence
# don't have to use either of these (c_v bad and no public code for the rl paper)

# other coherence measures
# see papers folder

# most of umass and uci are the same, have type as input and merge them
# make the for loop ranges the same
# use itertools combinations instead of for loops

# add perplexity
# https://stats.stackexchange.com/questions/18167/how-to-calculate-perplexity-of-a-holdout-with-latent-dirichlet-allocation
# https://homepages.inf.ed.ac.uk/imurray2/pub/09etm/
# can calculate reward based on perplexity during training
# and do early stopping based on a held out validation set
# https://github.com/shion-h/TopicModels/blob/master/scripts/calcldappl.py
# phi == beta


def main():
    pickle_name = 'data/dbpedia_test'
    dataset_save_dict = get_dataset(pickle_name)

    # word dist shape is (batch_size, vocab_size)
    # for full theta, is (num_documents, vocab_size)
    word_dist = torch.exp(torch.randn((1024, 2000)))
    bow = dataset_save_dict['train']['bow_embeddings']
    bow = bow[:1024, :]
    ll1 = calculate_log_likelihood_batch(bow, word_dist)
    ll2 = calculate_log_likelihood(bow, word_dist)
    print(ll1, ll2)
    assert np.isclose(ll1, ll2)
    perplexity = calculate_perplexity_batch(bow, word_dist)
    print(perplexity)


def calculate_log_likelihood(bow, word_dist):
    bow = bow.detach().cpu().numpy()
    word_dist = word_dist.detach().cpu().numpy()
    log_likelihood = 0.0
    for d, document in enumerate(bow):
        document_log_likelihood = 0
        for i, word in enumerate(document):
            if word > 0:
                document_log_likelihood += word * np.log(word_dist[d, i])
        log_likelihood += document_log_likelihood
    return log_likelihood


def calculate_log_likelihood_batch(bow, word_dist, epsilon=1e-8):
    log_likelihood_tensor = bow * torch.log(word_dist + epsilon)
    log_likelihood = torch.sum(log_likelihood_tensor)
    return log_likelihood.item()


# check if this is implemented correctly
# is lower for final (smaller) batch in an epoch
def calculate_perplexity_batch(bow, word_dist):
    total_words = torch.sum(bow)
    log_likelihood = calculate_log_likelihood_batch(bow, word_dist)
    perplexity = torch.exp(-1.0 * log_likelihood / total_words)
    return perplexity.item()


class UMASSCoherence:
    def __init__(self, sparse_corpus_bow, epsilon=1):
        self.sparse_corpus_bow = sparse_corpus_bow
        self.epsilon = epsilon

        self.topk_indices_cache = dict()
        self.word_indices_cache = dict()
        self.p_w1w2_cache = dict()
        self.p_w_cache = dict()

    def get_coherence(self, beta, topk):
        beta = beta.detach()
        topic_coherences = []

        topk_indices = torch.topk(beta, k=topk, dim=1).indices.cpu().numpy()
        topk_indices_tuple = tuple(topk_indices.flatten().tolist())
        if topk_indices_tuple in self.topk_indices_cache.keys():
            return self.topk_indices_cache[topk_indices_tuple]
        else:
            for word_indices in topk_indices:
                C = 0
                word_indices_tuple = tuple(word_indices.tolist())
                if word_indices_tuple in self.word_indices_cache.keys():
                    topic_coherences.append(self.word_indices_cache[word_indices_tuple])
                else:
                    for w1_id, w2_id in combinations(word_indices_tuple, r=2):
                        p_w1w2_cache_key = f'{w1_id}.{w2_id}'
                        if p_w1w2_cache_key in self.p_w1w2_cache.keys():
                            p_w1w2 = self.p_w1w2_cache[p_w1w2_cache_key]
                        else:
                            co_freq_array = (self.sparse_corpus_bow[:, [w1_id, w2_id]] > 0).sum(axis=1).A1
                            p_w1w2 = np.count_nonzero(co_freq_array == 2)
                            self.p_w1w2_cache[p_w1w2_cache_key] = p_w1w2
                        if w1_id in self.p_w_cache.keys():
                            p_w1 = self.p_w_cache[w1_id]
                        else:
                            p_w1 = (self.sparse_corpus_bow[:, w1_id] > 0).sum()
                            self.p_w_cache[w1_id] = p_w1
                        C = C + math.log((p_w1w2 + self.epsilon) / p_w1)

                    C = 2 * C / (topk * (topk - 1))
                    self.word_indices_cache[word_indices_tuple] = C
                    topic_coherences.append(C)

        mean_coherence = np.mean(topic_coherences).item()
        self.topk_indices_cache[topk_indices_tuple] = mean_coherence
        return mean_coherence


class NPMICoherence:
    def __init__(self, sparse_corpus_bow, epsilon=0):
        self.sparse_corpus_bow = sparse_corpus_bow
        self.num_docs = sparse_corpus_bow.shape[0]
        self.epsilon = epsilon

        self.topk_indices_cache = dict()
        self.word_indices_cache = dict()
        self.p_w1w2_cache = dict()
        self.p_w_cache = dict()

    def get_coherence(self, beta, topk):
        beta = beta.detach()
        topic_coherences = []

        topk_indices = torch.topk(beta, k=topk, dim=1).indices.cpu().numpy()
        topk_indices_tuple = tuple(topk_indices.flatten().tolist())
        if topk_indices_tuple in self.topk_indices_cache.keys():
            return self.topk_indices_cache[topk_indices_tuple]
        else:
            for word_indices in topk_indices:
                C = 0
                word_indices_tuple = tuple(word_indices.tolist())
                if word_indices_tuple in self.word_indices_cache.keys():
                    topic_coherences.append(self.word_indices_cache[word_indices_tuple])
                else:
                    for w1_id, w2_id in combinations(word_indices_tuple, r=2):
                    #for i, w1_id in enumerate(list(word_indices_tuple)[1:]):
                        #for j, w2_id in enumerate(list(word_indices_tuple)[:i]):
                            p_w1w2_cache_key = f'{w1_id}.{w2_id}'
                            if p_w1w2_cache_key in self.p_w1w2_cache.keys():
                                p_w1w2 = self.p_w1w2_cache[p_w1w2_cache_key]
                            else:
                                co_freq_array = (self.sparse_corpus_bow[:, [w1_id, w2_id]] > 0).sum(axis=1).A1
                                p_w1w2 = np.count_nonzero(co_freq_array == 2) / self.num_docs
                                self.p_w1w2_cache[p_w1w2_cache_key] = p_w1w2
                            if w1_id in self.p_w_cache.keys():
                                p_w1 = self.p_w_cache[w1_id]
                            else:
                                p_w1 = (self.sparse_corpus_bow[:, w1_id] > 0).sum() / self.num_docs
                                self.p_w_cache[w1_id] = p_w1
                            if w2_id in self.p_w_cache.keys():
                                p_w2 = self.p_w_cache[w2_id]
                            else:
                                p_w2 = (self.sparse_corpus_bow[:, w2_id] > 0).sum() / self.num_docs
                                self.p_w_cache[w2_id] = p_w2
                            # prevent logging zero
                            if p_w1w2 != 0:
                                numerator = math.log(p_w1w2 / (p_w1 * p_w2))
                                denominator = -1.0 * math.log(p_w1w2)
                                # prevent division by zero
                                if denominator != 0:
                                    C = C + numerator / denominator

                    C = 2 * C / (topk * (topk - 1))
                    self.word_indices_cache[word_indices_tuple] = C
                    topic_coherences.append(C)

        mean_coherence = np.mean(topic_coherences).item()
        self.topk_indices_cache[topk_indices_tuple] = mean_coherence
        return mean_coherence


# this has no sliding window
class UCICoherence:
    def __init__(self, topk, sparse_corpus_bow, epsilon=1):
        self.topk = topk
        self.sparse_corpus_bow = sparse_corpus_bow
        self.epsilon = epsilon

        self.topk_indices_cache = dict()
        self.word_indices_cache = dict()
        self.p_w1w2_cache = dict()
        self.p_w_cache = dict()

    def get_uci_coherence(self, beta):
        topic_coherences = []

        topk_indices = torch.topk(beta, k=self.topk, dim=1).indices.cpu().numpy()
        topk_indices_tuple = tuple(topk_indices.flatten().tolist())
        if topk_indices_tuple in self.topk_indices_cache.keys():
            return self.topk_indices_cache[topk_indices_tuple]
        else:
            for word_indices in topk_indices:
                C = 0
                word_indices_tuple = tuple(word_indices.tolist())
                if word_indices_tuple in self.word_indices_cache.keys():
                    topic_coherences.append(self.word_indices_cache[word_indices_tuple])
                else:
                    for w1_id, w2_id in combinations(word_indices_tuple, r=2):
                        p_w1w2_cache_key = f'{w1_id}.{w2_id}'
                        if p_w1w2_cache_key in self.p_w1w2_cache.keys():
                            p_w1w2 = self.p_w1w2_cache[p_w1w2_cache_key]
                        else:
                            co_freq_array = (self.sparse_corpus_bow[:, [w1_id, w2_id]] > 0).sum(axis=1).A1
                            p_w1w2 = np.count_nonzero(co_freq_array == 2)
                            self.p_w1w2_cache[p_w1w2_cache_key] = p_w1w2
                        if w1_id in self.p_w_cache.keys():
                            p_w1 = self.p_w_cache[w1_id]
                        else:
                            p_w1 = (self.sparse_corpus_bow[:, w1_id] > 0).sum()
                            self.p_w_cache[w1_id] = p_w1
                        if w2_id in self.p_w_cache.keys():
                            p_w2 = self.p_w_cache[w2_id]
                        else:
                            p_w2 = (self.sparse_corpus_bow[:, w2_id] > 0).sum()
                            self.p_w_cache[w2_id] = p_w2
                        C = C + math.log((p_w1w2 + self.epsilon) / (p_w1 * p_w2))

                    C = 2 * C / (self.topk * (self.topk - 1))
                    self.word_indices_cache[word_indices_tuple] = C
                    topic_coherences.append(C)

        mean_coherence = np.mean(topic_coherences)
        self.topk_indices_cache[topk_indices_tuple] = mean_coherence
        return mean_coherence


def topic_diversity(beta, topk):
    """
    This will only work for tensors. Faster reward calculation for RL optimization.
    :param beta: torch tensor (# of topics) x (size of vocabulary)
    :param topk: int number of words per topic
    :return: topic diversity
    """
    # indices of top 'k' words for each topic
    topk_indices = torch.topk(beta, k=topk, dim=1).indices
    # set of unique indices
    unique_indices = set(topk_indices.flatten().detach().cpu().tolist())
    # calculate and return topic diversity
    return len(unique_indices) / (topk * beta.size(0))


def get_topic_words(beta, dictionary, topk):
    if isinstance(beta, np.ndarray):
        beta = torch.from_numpy(beta)
    topk_indices = torch.topk(beta, k=topk, dim=1).indices

    topics = []
    for index_array in topk_indices:
        index_list = index_array.detach().cpu().tolist()

        topic = []
        for index in index_list:
            topic.append(dictionary['id2token'][index])

        topics.append(topic)
    return topics


def get_topic_words_from_files(pickle_name, experiment_name, topk):
    dataset_save_dict = get_dataset(pickle_name)
    vocabulary = dataset_save_dict['vocabulary']

    with open(f'{experiment_name}.pkl', 'rb') as handle:
        experiment_dict = pickle.load(handle)

    beta = experiment_dict['best_epoch_beta']

    topics = get_topic_words(beta, vocabulary, topk)
    return topics


# https://github.com/MilaNLProc/contextualized-topic-models
class InvertedRBO:
    def __init__(self, topics):
        self.topics = topics

        self.RBO = namedtuple("RBO", "min res ext")
        self.RBO.__doc__ += ": Result of full RBO analysis"
        self.RBO.min.__doc__ = "Lower bound estimate"
        self.RBO.res.__doc__ = "Residual corresponding to min; min + res is an upper bound estimate"
        self.RBO.ext.__doc__ = "Extrapolated point estimate"

    def score(self, topk=10, weight=0.9):
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            collect = []
            for list1, list2 in combinations(self.topics, 2):
                rbo_val = self.rbo(list1[:topk], list2[:topk], p=weight)[2]
                collect.append(rbo_val)
            return 1 - np.mean(collect)

    def rbo(self, list1, list2, p):
        if not 0 <= p <= 1:
            raise ValueError("The ``p`` parameter must be between 0 and 1.")
        args = (list1, list2, p)
        return self.RBO(self.rbo_min(*args), self.rbo_res(*args), self.rbo_ext(*args))

    def rbo_min(self, list1, list2, p, depth=None):
        depth = min(len(list1), len(list2)) if depth is None else depth
        x_k = self.overlap(list1, list2, depth)
        log_term = x_k * math.log(1 - p)
        sum_term = sum(
            p ** d / d * (self.overlap(list1, list2, d) - x_k) for d in range(1, depth + 1)
        )
        return (1 - p) / p * (sum_term - log_term)

    def rbo_res(self, list1, list2, p):
        S, L = sorted((list1, list2), key=len)
        s, l = len(S), len(L)
        x_l = self.overlap(list1, list2, l)
        f = int(math.ceil(l + s - x_l))
        term1 = s * sum(p ** d / d for d in range(s + 1, f + 1))
        term2 = l * sum(p ** d / d for d in range(l + 1, f + 1))
        term3 = x_l * (math.log(1 / (1 - p)) - sum(p ** d / d for d in range(1, f + 1)))
        return p ** s + p ** l - p ** f - (1 - p) / p * (term1 + term2 + term3)

    def rbo_ext(self, list1, list2, p):
        S, L = sorted((list1, list2), key=len)
        s, l = len(S), len(L)
        x_l = self.overlap(list1, list2, l)
        x_s = self.overlap(list1, list2, s)
        sum1 = sum(p ** d * self.agreement(list1, list2, d) for d in range(1, l + 1))
        sum2 = sum(p ** d * x_s * (d - s) / s / d for d in range(s + 1, l + 1))
        term1 = (1 - p) / p * (sum1 + sum2)
        term2 = p ** l * ((x_l - x_s) / l + x_s / s)
        return term1 + term2

    def overlap(self, list1, list2, depth):
        return self.agreement(list1, list2, depth) * min(depth, len(list1), len(list2))

    def agreement(self, list1, list2, depth):
        len_intersection, len_set1, len_set2 = self.raw_overlap(list1, list2, depth)
        return 2 * len_intersection / (len_set1 + len_set2)

    def raw_overlap(self, list1, list2, depth):
        set1, set2 = self.set_at_depth(list1, depth), self.set_at_depth(list2, depth)
        return len(set1.intersection(set2)), len(set1), len(set2)

    @staticmethod
    def set_at_depth(lst, depth):
        ans = set()
        for v in lst[:depth]:
            if isinstance(v, set):
                ans.update(v)
            else:
                ans.add(v)
        return ans


class CoherenceWordEmbeddings:
    def __init__(self, word2vec_path=None, binary=False):
        self.binary = binary
        if word2vec_path is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = KeyedVectors.load_word2vec_format(word2vec_path, binary=binary)

    def score(self, topics, topk=10):
        if topk > len(topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            arrays = []
            for index, topic in enumerate(topics):
                if len(topic) > 0:
                    local_simi = []
                    for word1, word2 in combinations(topic[0:topk], 2):
                        if word1 in self.wv.key_to_index and word2 in self.wv.key_to_index:
                            local_simi.append(self.wv.similarity(word1, word2))
                    arrays.append(np.mean(local_simi))
            return np.mean(arrays)


if __name__ == '__main__':
    main()
