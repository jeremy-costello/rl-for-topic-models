import random
import itertools
import numpy as np

import torch

from trainer.trainer import Trainer
from data.dataset import get_datapipe
from evals.plotting import (init_experiment, init_experiment_dict, init_seed_folder, init_seed_dicts,
                            update_seed_dicts, final_seed_saving_and_plotting, get_plotting_dict, save_error_text,
                            save_experiment_average_output_dict)
from model.decoder_network import DecoderNetwork as Model
from trainer.utils import ExperimentTrainConfig, ExperimentModelConfig, get_save_num


def main():
    experiment_name = 'ntm_w2e_test'
    meta_seed = None
    num_seeds = 1

    if meta_seed is None:
        meta_seed = random.randint(0, 2 ** 32)

    torch.manual_seed(meta_seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    random.seed(meta_seed)
    np.random.seed(meta_seed)

    experiment_seeds = list([random.randint(0, 2 ** 32) for _ in range(num_seeds)])

    search_dict = {
        # constant
        'num_gpus': [1],
        'num_cpus': [0],
        'max_epochs': [1000],
        'frozen_embeddings': [True],
        # data
        'data_set': ['data/ntm_w2e_3648vocab'],
        'vocab_size': [2000],
        'input_type': ['bert'],  # bow, bert
        'embedding_dict': [
            {
                'sbert_model': 'all-MiniLM-L6-v2',
                'hugface_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'size': 384,
                'max_length': 256
            }
        ],
        # model
        'num_topics': [30],
        'inference_dropout': [0.2],
        'theta_dropout': [0.0],
        'hidden_layers': [(128, 128)],
        'activation': ['gelu'],  # softplus, gelu
        'initialization': ['normal'],  # xavier, normal
        'normalization': ['layer'],  # batch, layer
        'affine': [False],  # true, false
        'prior': ['laplace'],  # laplace, gaussian
        'trainable_prior': [True],
        'kl_mult': [5.0],
        'entropy_mult': [0.0],
        'parameter_noise': [False],
        'top_k': [[10]],
        'loss_type': ['rl'],  # vae, rl
        'lda_type': ['prodlda'],  # lda, prodlda
        'theta_softmax': [False],  # true, false
        # training
        'learning_rate': [3e-4],
        'betas': [(0.9, 0.999)],
        'weight_decay': [0.01],
        'batch_size': [1024],
        'gradient_clip': [1.0],  # none, float
        'lr_decay': [False]
    }

    # save experiment stuff
    init_experiment(experiment_name, meta_seed, experiment_seeds, search_dict)
    experiment_average_output_dict = init_experiment_dict()

    # https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
    keys, values = zip(*search_dict.items())
    print(f'Total runs: {len(list(itertools.product(*values))) * num_seeds}')

    for exp_num, curr_vals in enumerate(itertools.product(*values)):
        print(f'\nExperiment {exp_num}:')

        hp_dict = dict(zip(keys, curr_vals))

        full_exp_num = get_save_num(exp_num)
        true_experiment_name = f'{experiment_name}/{full_exp_num}'
        init_seed_folder(true_experiment_name, hp_dict)
        experiment_average_output_dict['experiment_num'].append(exp_num)

        # get datasets
        train_dataset = get_datapipe(hp_dict['data_set'],
                                     split='train',
                                     frozen_embeddings=hp_dict['frozen_embeddings'],
                                     sbert_model=hp_dict['embedding_dict']['sbert_model'],
                                     hugface_model=hp_dict['embedding_dict']['hugface_model'],
                                     max_length=hp_dict['embedding_dict']['max_length'])

        test_dataset = get_datapipe(hp_dict['data_set'],
                                    split='test',
                                    frozen_embeddings=hp_dict['frozen_embeddings'],
                                    sbert_model=hp_dict['embedding_dict']['sbert_model'],
                                    hugface_model=hp_dict['embedding_dict']['hugface_model'],
                                    max_length=hp_dict['embedding_dict']['max_length'])

        test_dataset_is_none = test_dataset is None
        seed_plotting_dict, seed_output_dict = init_seed_dicts(num_seeds, max_epochs=hp_dict['max_epochs'],
                                                               test_dataset_is_none=test_dataset_is_none)

        # don't think this try-except does anything with torch multiprocessing
        try:
            for seed_num, seed in enumerate(experiment_seeds):
                print(f'\nSeed {seed_num}:')

                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)

                full_seed_num = get_save_num(seed_num)
                true_seed_experiment_name = f'{true_experiment_name}/seeds/{full_seed_num}'

                train_config = ExperimentTrainConfig()
                train_config.true_init(hp_dict, seed_num, true_seed_experiment_name, seed)
                model_config = ExperimentModelConfig()
                model_config.true_init(hp_dict)

                model = Model(model_config)
                trainer = Trainer(model=model,
                                  train_dataset=train_dataset,
                                  test_dataset=test_dataset,
                                  train_config=train_config)

                trainer.distributed_train()

                # haven't tested plotting for multi-gpu training
                # haven't tested anything for multi-machine training
                plotting_dict = get_plotting_dict(true_seed_experiment_name, seed_num)
                seed_plotting_dict, seed_output_dict = update_seed_dicts(seed_plotting_dict, seed_output_dict,
                                                                         seed_num, seed, plotting_dict)
        except ValueError as error_text:
            save_error_text(true_experiment_name, error_text)
            continue

        experiment_average_output_dict = final_seed_saving_and_plotting(seed_plotting_dict, seed_output_dict,
                                                                        experiment_average_output_dict,
                                                                        true_experiment_name,
                                                                        total_epochs=hp_dict['max_epochs'],
                                                                        num_seeds=num_seeds)

    save_experiment_average_output_dict(experiment_name, experiment_average_output_dict)


if __name__ == '__main__':
    main()
