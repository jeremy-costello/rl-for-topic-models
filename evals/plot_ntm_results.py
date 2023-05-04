import os
import json
import pickle
import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt

from trainer.utils import get_save_num


result_location = 'evals/ntm_runs'

result_dict = dict()
for file in os.scandir(result_location):
    with open(file.path, 'rb') as f:
        if file.is_file():
            filename, file_ext = os.path.splitext(file.name)
            if file_ext == '.pkl':
                results = pickle.load(f)
                coh = results['coherence']
                coh_10 = [c[1] for c in coh]
                coh_mean = np.mean(coh_10)
                coh_stdev = np.std(coh_10)

                dataset = results['option']['dataset']
                if dataset not in result_dict.keys():
                    result_dict[dataset] = dict()

                model = results['option']['model']
                if model not in result_dict[dataset].keys():
                    result_dict[dataset][model] = dict()

                num_topics = results['option']['num_topics']

                result_dict[dataset][model][num_topics] = dict()
                result_dict[dataset][model][num_topics]['mean'] = coh_mean
                result_dict[dataset][model][num_topics]['stdev'] = coh_stdev

experiment_dict = {
    'ntm_20news_sweep': [10, 20, 30, 40, 50, 60],
    'ntm_snippets_sweep': [4, 8, 12, 16, 20, 24],
    'ntm_w2e_sweep': [15, 30, 45, 60, 75, 90],
    'ntm_w2e_text_sweep': [15, 30, 45, 60, 75, 90],
}

base_model = 'RL'
lambdas = [1, 3, 5, 10]
base_dir = 'experiments'
for exp_name, num_topics_list in experiment_dict.items():
    dataset = '_'.join(exp_name.split('_')[1:-1])
    assert dataset in result_dict.keys()

    for i, num_topics in enumerate(num_topics_list):
        for j, lambda_ in enumerate(lambdas):
            if lambda_ not in [1, 3, 5, 10]:
                continue
            model = f'{base_model} (λ = {lambda_})'
            if model not in result_dict[dataset].keys():
                result_dict[dataset][model] = dict()

            dir_num = get_save_num(len(lambdas) * i + j)
            metrics_tsv = f'{base_dir}/{exp_name}/{dir_num}/metric_outputs.tsv'
            df = pd.read_csv(metrics_tsv, sep="\t")

            coh_10 = df['train_coherence']
            coh_mean = np.mean(coh_10)
            coh_stdev = np.std(coh_10)

            result_dict[dataset][model][num_topics] = dict()
            result_dict[dataset][model][num_topics]['mean'] = coh_mean
            result_dict[dataset][model][num_topics]['stdev'] = coh_stdev


# https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
color_map = 'tab20'
cm = plt.get_cmap(color_map)
fig = plt.figure(figsize=(10, 10))

title_dict = {
    'snippets': 'Snippets',
    '20news': '20 Newsgroups',
    'w2e': 'W2E-title',
    'w2e_text': 'W2E-content'
}

for i, (dataset, models) in enumerate(result_dict.items()):
    ax1 = fig.add_subplot(2, 2, i + 1)
    ax1.set_title(title_dict[dataset])

    # https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html
    num_models = len(models)
    custom_cycler = cycler(color=[cm(1. * i / num_models) for i in range(num_models)])
    ax1.set_prop_cycle(custom_cycler)

    exp_name = f'ntm_{dataset}_sweep'
    num_topics_list = experiment_dict[exp_name]
    for model in models.keys():
        coh_mean_list = []
        coh_stdev_list = []
        for num_topics in num_topics_list:
            coh_mean_list.append(result_dict[dataset][model][num_topics]['mean'])
            coh_stdev_list.append(result_dict[dataset][model][num_topics]['stdev'])

        ax1.set_xticks(num_topics_list)
        ax1.errorbar(num_topics_list, coh_mean_list, yerr=coh_stdev_list, label=model)

plt.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.1), fancybox=False, shadow=False, ncol=5)

save_name = 'evals/figures/ntm_plots.png'
plt.savefig(save_name)

with open('evals/figures/ntm_result_dict.json', 'w') as f:
    json.dump(result_dict, f, indent=4)
