import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


my_parser = argparse.ArgumentParser()
my_parser.add_argument('experiment', type=str, help='Location of the experiment directory.')

args = my_parser.parse_args()

experiment_folder = args.experiment.replace('\\', '/')
df_location = f'{experiment_folder}/average_metric_outputs.tsv'
df = pd.read_csv(df_location, sep="\t")

coherences = np.array(df['test_coherence'].tolist())
diversitys = np.array(df['test_diversity'].tolist())
qualitys = coherences * diversitys

xs = np.arange(10) / 10

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(1, 1, 1)
ax.plot(xs, coherences, label='Coherence')
ax.plot(xs, diversitys, label='Diversity')
ax.plot(xs, qualitys, label='Quality')
ax.set_xticks(xs)
ax.set_xlabel('Dropout')
ax.set_ylabel('Metric')

plt.legend(loc='upper right', fancybox=False, shadow=False, ncol=1)
plt.tight_layout()

save_name = f'evals/figures/dropout_plot.png'
plt.savefig(save_name)
