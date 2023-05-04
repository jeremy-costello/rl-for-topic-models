import math
import torch
import torch.nn as nn


class TrainConfig:
    def __init__(self):
        # TRAINING CONFIGURATION
        # distributed data parallel
        self.master_address = 'localhost'
        self.master_port = '12355'
        self.backend = 'nccl'
        self.world_size = 1  # change to number of gpus on machine (for single machine training)
        self.torch_distributed_debug = 'OFF'
        self.find_unused_parameters = False
        # optimizer
        self.learning_rate = 3e-4
        self.betas = (0.9, 0.999)
        # weight decay (on whitelist; not on blacklist)
        self.weight_decay = 0.01
        self.whitelist_weight_modules = (nn.Linear,)
        self.blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        self.whitelist_weight_names = ()
        self.blacklist_weight_names = ('prior_mean', 'prior_variance', 'beta')
        # dataloader
        self.batch_size = 1024
        self.num_workers = 0
        self.deterministic_dataloader = False
        self.seed = 0
        # training
        self.max_epochs = 100
        self.grad_norm_clip = 1.0
        self.lr_decay = False
        self.ckpt_path = 'model/ckpt'
        self.experiment_name = '20newsgroups_mwl3'
        self.experiment_num = 0
        self.check_experiment_path = True
        self.save_outputs = True
        self.save_pickle_every = 0


def get_optimizer(optim_groups, train_config):
    return torch.optim.AdamW(optim_groups,
                             lr=train_config.learning_rate,
                             betas=train_config.betas)


# no LR scheduling used in paper
def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.LinearLR(optimizer)


def step_scheduler(scheduler, output_dict):
    scheduler.step()


class LRDecayer:
    def __init__(self):
        self.tokens = 0
        self.warmup_tokens = 0
        self.final_tokens = 3e6

    def decay_learning_rate(self, batch):
        self.tokens += (batch[1] >= 0).sum()
        if self.tokens < self.warmup_tokens:
            # linear warmup
            lr_mult = float(self.tokens) / float(max(1, self.warmup_tokens))
        else:
            # cosine learning rate decay
            progress_numerator = float(self.tokens - self.warmup_tokens)
            progress_denominator = float(max(1, self.final_tokens - self.warmup_tokens))
            progress = progress_numerator / progress_denominator
            lr_mult = max(1 / 3, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return lr_mult
