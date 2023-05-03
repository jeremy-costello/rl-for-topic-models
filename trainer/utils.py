import os

import torch
import torch.nn as nn

import trainer.parameters as parameters
from data.dataset import get_dataset


def configure_optimizers(model, train_config):
    decay = set()
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn
            # don't decay biases
            if pn.endswith('bias'):
                no_decay.add(fpn)
            # decay weights according to white/blacklist
            elif pn.endswith('weight'):
                if isinstance(m, train_config.whitelist_weight_modules):
                    decay.add(fpn)
                elif isinstance(m, train_config.blacklist_weight_modules):
                    no_decay.add(fpn)
            else:
                if fpn in train_config.whitelist_weight_names:
                    decay.add(fpn)
                elif fpn in train_config.blacklist_weight_names:
                    decay.add(fpn)
    param_dict = {pn: p for pn, p in model.named_parameters()}

    # for decay and no decay sets, ensure no intersection and union contains all parameters
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f'parameters {inter_params} made it into both decay and no_decay set'
    assert len(param_dict.keys() - union_params) == 0, \
        f'parameters {param_dict.keys() - union_params} were not separated into either decay or no_decay set'

    optim_groups = [
        {'params': [param_dict[pn] for pn in sorted(list(decay))], 'weight_decay': train_config.weight_decay},
        {'params': [param_dict[pn] for pn in sorted(list(no_decay))], 'weight_decay': 0.0}
    ]

    optimizer = parameters.get_optimizer(optim_groups, train_config)
    return optimizer


def dict_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    elif isinstance(batch, dict):
        res = {}
        for k, v in batch.items():
            res[k] = v.to(device)
        return res


def create_dir_recursive(dir_, warnings):
    dir_split = dir_.split('/')
    assert len(warnings) == len(dir_split)

    for i in range(len(dir_split)):
        dir_name = '/'.join(dir_split[:(i + 1)])
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        elif warnings[i]:
            raise OSError('Directory already exists!')


def get_save_num(experiment_num, save_num_length=6):
    exp_num_str = str(experiment_num)
    save_num = '0' * (save_num_length - len(exp_num_str)) + exp_num_str
    return save_num


def get_activation(activation_name):
    if activation_name == 'softplus':
        return nn.Softplus()
    elif activation_name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError('Invalid activation name!')


def get_valid_splits(dictionary):
    valid_splits = ['train', 'test']
    splits = [split for split in dictionary.keys() if split in valid_splits]
    return splits


class ExperimentTrainConfig:
    def __init__(self):
        pass

    def true_init(self, d, i, experiment_name, seed):
        # TRAINING CONFIGURATION
        # distributed data parallel
        self.master_address = 'localhost'
        self.master_port = '12355'  # add option to change this in run_experiments
        self.backend = 'nccl'
        self.world_size = d['num_gpus']
        self.torch_distributed_debug = 'OFF'
        self.find_unused_parameters = False
        # optimizer
        self.learning_rate = d['learning_rate']
        self.betas = d['betas']
        # weight decay (on whitelist; not on blacklist)
        self.weight_decay = d['weight_decay']
        self.whitelist_weight_modules = (nn.Linear,)
        self.blacklist_weight_modules = (nn.LayerNorm, nn.BatchNorm1d, nn.Embedding)
        self.whitelist_weight_names = ()
        self.blacklist_weight_names = ('prior_mean', 'prior_variance', 'beta')
        # dataloader
        self.batch_size = d['batch_size']
        self.num_workers = d['num_cpus']
        self.deterministic_dataloader = True
        self.seed = seed
        # training
        self.max_epochs = d['max_epochs']
        self.grad_norm_clip = d['gradient_clip']
        self.lr_decay = d['lr_decay']
        self.ckpt_path = 'model/ckpt'
        self.experiment_name = experiment_name
        self.experiment_num = i
        self.check_experiment_path = False
        self.save_outputs = True
        self.save_pickle_every = 0


class ExperimentModelConfig:
    def __init__(self):
        pass

    def true_init(self, d):
        # MODEL CONFIGURATION
        self.n_components = d['num_topics']
        # decoder network
        self.input_size = d['vocab_size']
        self.input_type = d['input_type']
        self.decoder_dropout = d['theta_dropout']
        self.initialization = d['initialization']
        self.normalization = d['normalization']
        self.affine = d['affine']
        self.loss_type = d['loss_type']
        self.lda_type = d['lda_type']
        self.theta_softmax = d['theta_softmax']
        # inference network
        self.frozen_embeddings = d['frozen_embeddings']
        self.sbert_model = d['embedding_dict']['sbert_model']
        self.hugface_model = d['embedding_dict']['hugface_model']
        self.bert_size = d['embedding_dict']['size']
        self.max_length = d['embedding_dict']['max_length']
        self.hiddens = d['hidden_layers']
        self.activation = get_activation(d['activation'])
        self.inference_dropout = d['inference_dropout']
        self.parameter_noise = d['parameter_noise']
        self.prior = d['prior']
        self.trainable_prior = d['trainable_prior']
        self.kl_mult = d['kl_mult']
        self.entropy_mult = d['entropy_mult']
        self.topk = d['top_k']
        self.pickle_name = d['data_set']
        self.get_sparse_corpus_bow()

    def get_sparse_corpus_bow(self):
        dataset_save_dict = get_dataset(self.pickle_name)
        self.sparse_corpus_bow = dataset_save_dict['sparse_corpus_bow']
