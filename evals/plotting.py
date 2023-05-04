import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trainer.utils import create_dir_recursive, get_save_num, get_valid_splits


exp_dir = 'experiments'


class Outputs:
    def __init__(self, experiment_name, experiment_num, check_experiment_path):
        self.save_dict = dict()
        self.pickle_name = None

        self.create_directories(experiment_name, check_experiment_path)

        self.get_pickle_name(experiment_name, experiment_num)

    @staticmethod
    def create_directories(experiment_name, check_experiment_path):
        if check_experiment_path:
            warnings = [False] + [True] * (experiment_name.count('/') + 1)
        else:
            warnings = [False] * (experiment_name.count('/') + 2)
        create_dir_recursive(f'{exp_dir}/{experiment_name}', warnings=warnings)

    def get_pickle_name(self, experiment_name, experiment_num):
        save_num = get_save_num(experiment_num)
        self.pickle_name = f'{exp_dir}/{experiment_name}/{save_num}_training_outputs.pkl'

    def save_step(self, output_dict, split, epoch, it):
        if it == 0:
            if epoch not in self.save_dict.keys():
                self.save_dict[epoch] = dict()
            self.save_dict[epoch][split] = dict()

        self.save_dict[epoch][split][it] = {
            'loss': output_dict['loss'],
            'coherence': output_dict['coherence'],
            'diversity': output_dict['diversity'],
            'perplexity': output_dict['perplexity']
        }

    def save_epoch(self, output_dict, epoch):
        self.save_dict[epoch]['beta'] = output_dict['beta'].detach().cpu().numpy()

    def save_pickle(self):
        with open(self.pickle_name, 'wb') as handle:
            pickle.dump(self.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


class PostExperiment:
    def __init__(self):
        pass


def init_experiment(experiment_name, meta_seed, experiment_seeds, search_dict):
    warnings = [False, True]
    create_dir_recursive(f'{exp_dir}/{experiment_name}', warnings=warnings)

    with open(f'{exp_dir}/{experiment_name}/meta_seed.txt', 'w') as f:
        f.write(f'{meta_seed}')

    if isinstance(experiment_seeds, int):
        with open(f'{exp_dir}/{experiment_name}/experiment_seeds.txt', 'w') as f:
            f.write(f'{experiment_seeds}')
    elif isinstance(experiment_seeds, list):
        with open(f'{exp_dir}/{experiment_name}/experiment_seeds.txt', 'w') as f:
            for seed in experiment_seeds:
                f.write(f'{seed}\n')

    if search_dict is not None:
        with open(f'{exp_dir}/{experiment_name}/search_dict.json', 'w') as f:
            json.dump(search_dict, f, indent=4)


def init_seed_folder(experiment_name, hp_dict):
    warnings = [False] * (experiment_name.count('/') + 1) + [True]
    create_dir_recursive(f'{exp_dir}/{experiment_name}', warnings=warnings)

    with open(f'{exp_dir}/{experiment_name}/hp_dict.json', 'w') as f:
        json.dump(hp_dict, f, indent=4)


def save_training_outputs(experiment_name, exp_num, outputs):
    total_epochs = max(outputs.save_dict.keys()) + 1
    metrics = outputs.save_dict[0]['train'][0].keys()

    splits = get_valid_splits(outputs.save_dict[0])

    plotting_dict = dict()
    for split in splits:
        plotting_dict[split] = dict()
        for metric in metrics:
            plotting_dict[split][metric] = np.empty(total_epochs)

    best_epoch = None
    best_metric_value = -np.inf
    for epoch in outputs.save_dict.keys():
        if epoch > total_epochs:
            raise ValueError('Max epochs incorrect!')
        for split in splits:
            for metric in metrics:
                epoch_mean = 0
                for i, it in enumerate(outputs.save_dict[epoch][split].keys()):
                    epoch_mean = (epoch_mean * i / (i + 1)
                                  + outputs.save_dict[epoch][split][it][metric] / (i + 1))
                plotting_dict[split][metric][epoch] = epoch_mean
            # some combination of metrics to find "best" epoch
            metric_value = plotting_dict[split]['coherence'][epoch]  # + plotting_dict[split]['diversity'][epoch] \
                # - plotting_dict[split]['perplexity'][epoch] / 1000
            if metric_value > best_metric_value:
                best_epoch = epoch
                best_metric_value = metric_value

    plotting_dict['best_epoch'] = best_epoch
    plotting_dict['best_epoch_beta'] = outputs.save_dict[best_epoch]['beta']

    save_num = get_save_num(exp_num)
    pickle_name = f'{exp_dir}/{experiment_name}/{save_num}_plotting_arrays.pkl'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(plotting_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    x = np.arange(total_epochs)
    for split in splits:
        for metric in metrics:
            plt.plot(x, plotting_dict[split][metric])
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.savefig(f'{exp_dir}/{experiment_name}/{save_num}_{split}_{metric}.png')
            plt.close()


def get_plotting_dict(experiment_name, exp_num):
    save_num = get_save_num(exp_num)
    pickle_name = f'{exp_dir}/{experiment_name}/{save_num}_plotting_arrays.pkl'
    with open(f'{pickle_name}', 'rb') as handle:
        plotting_dict = pickle.load(handle)
    return plotting_dict


def save_error_text(experiment_name, error_text):
    with open(f'{exp_dir}/{experiment_name}/error.txt', 'w') as f:
        f.write(f'{error_text}')


def init_experiment_dict():
    experiment_average_output_dict = {
        'experiment_num': [],
        'best_epoch': [],
        'train_loss': [],
        'test_loss': [],
        'train_perplexity': [],
        'test_perplexity': [],
        'train_coherence': [],
        'test_coherence': [],
        'train_diversity': [],
        'test_diversity': []
    }
    return experiment_average_output_dict


def init_seed_dicts(num_seeds, max_epochs, test_dataset_is_none):
    seed_plotting_dict = {
        'train_loss': np.empty((num_seeds, max_epochs)),
        'train_perplexity': np.empty((num_seeds, max_epochs)),
        'train_coherence': np.empty((num_seeds, max_epochs)),
        'train_diversity': np.empty((num_seeds, max_epochs)),
    }

    # no test_* if test dataset is None
    seed_output_dict = {
        'seed_num': [],
        'seed': [],
        'best_epoch': [],
        'train_loss': [],
        'train_perplexity': [],
        'train_coherence': [],
        'train_diversity': [],
    }

    if not test_dataset_is_none:
        seed_plotting_dict['test_loss'] = np.empty((num_seeds, max_epochs))
        seed_plotting_dict['test_perplexity'] = np.empty((num_seeds, max_epochs))
        seed_plotting_dict['test_coherence'] = np.empty((num_seeds, max_epochs))
        seed_plotting_dict['test_diversity'] = np.empty((num_seeds, max_epochs))

        seed_output_dict['test_loss'] = []
        seed_output_dict['test_perplexity'] = []
        seed_output_dict['test_coherence'] = []
        seed_output_dict['test_diversity'] = []

    return seed_plotting_dict, seed_output_dict


def update_seed_dicts(seed_plotting_dict, seed_output_dict, seed_num, seed, plotting_dict):
    best_epoch = plotting_dict['best_epoch']

    splits = get_valid_splits(plotting_dict)

    metrics = ['loss', 'perplexity', 'coherence', 'diversity']

    seed_output_dict['seed_num'].append(seed_num)
    seed_output_dict['seed'].append(seed)
    seed_output_dict['best_epoch'].append(best_epoch)

    for split in splits:
        for metric in metrics:
            seed_plotting_dict[f'{split}_{metric}'][seed_num, :] = plotting_dict[split][metric]
            seed_output_dict[f'{split}_{metric}'].append(plotting_dict[split][metric][best_epoch])

    return seed_plotting_dict, seed_output_dict


# save per seed runs in seed folder (DONE)
# save average for each at end of all experiments in one file (DONE)
# also save df of hyperparameters for each experiment in one file (LATER)
def final_seed_saving_and_plotting(seed_plotting_dict, seed_output_dict, experiment_average_output_dict,
                                   experiment_name, total_epochs, num_seeds):
    seed_output_dict['seed_num'].insert(0, 'ALL')
    seed_output_dict['seed'].insert(0, 'AVERAGE')

    splits = list(set([key.split('_')[0] for key in seed_plotting_dict.keys()]))
    metrics = ['loss', 'perplexity', 'coherence', 'diversity']

    columns = ['best_epoch']
    for split in splits:
        for metric in metrics:
            columns.append(f'{split}_{metric}')

    for column in columns:
        seed_output_dict[column].insert(0, np.mean(seed_output_dict[column]))
        experiment_average_output_dict[column].append(np.mean(seed_output_dict[column]))

    seed_output_df = pd.DataFrame.from_dict(seed_output_dict)
    seed_output_df.to_csv(f'{exp_dir}/{experiment_name}/metric_outputs.tsv', sep="\t", index=False)

    # how make pretty error plot???
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.03-errorbars.html
    x = np.arange(total_epochs)
    for split in splits:
        for metric in metrics:
            true_key = f'{split}_{metric}'
            if true_key not in seed_plotting_dict.keys():
                continue
            y = np.mean(seed_plotting_dict[true_key], axis=0)
            y_err = np.std(seed_plotting_dict[true_key], axis=0) / np.sqrt(num_seeds)
            plt.plot(x, y)
            plt.fill_between(x, y - y_err, y + y_err, alpha=0.2)
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.savefig(f'{exp_dir}/{experiment_name}/{split}_{metric}_average.png')
            plt.close()

    return experiment_average_output_dict


def save_experiment_average_output_dict(experiment_name, experiment_average_output_dict):
    # remove empty lists from dictionary (test keys if no test dataset)
    key_list = list(experiment_average_output_dict.keys())
    for key in key_list:
        if not experiment_average_output_dict[key]:
            experiment_average_output_dict.pop(key, None)

    experiment_average_output_dict_save_name = f'{exp_dir}/{experiment_name}/average_metric_outputs.tsv'
    experiment_average_output_df = pd.DataFrame.from_dict(experiment_average_output_dict)
    experiment_average_output_df.to_csv(experiment_average_output_dict_save_name, sep="\t", index=False)
