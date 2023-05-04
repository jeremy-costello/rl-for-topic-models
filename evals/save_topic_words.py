import argparse
from evals.measures import get_topic_words_from_files


my_parser = argparse.ArgumentParser()

my_parser.add_argument('topk', type=int, help='Top-k topic words for calculating coherence and diversity.')
my_parser.add_argument('data', type=str, help='Location of the data pickle file.')
my_parser.add_argument('plotting_array', type=str, help='Location of the specific experiment plotting array.')

args = my_parser.parse_args()

topk = args.topk
pickle_name = args.data.rstrip('.pkl').replace('\\', '/')
experiment_name = args.plotting_array.rstrip('.pkl').replace('\\', '/')

topics = get_topic_words_from_files(pickle_name, experiment_name, topk)

for topic in topics:
    print(f'{topic}\n')

with open('evals/figures/topics.txt', 'w') as f:
    for i, topic in enumerate(topics):
        for j, word in enumerate(topic):
            if j == 0:
                f.write(word)
            else:
                f.write(' ' + word)
        if i != len(topics) - 1:
            f.write('\n')
