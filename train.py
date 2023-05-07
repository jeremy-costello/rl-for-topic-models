import argparse

from model.decoder_network import DecoderNetwork as Model
from model.decoder_network import NetworkConfigs as ModelConfig
from trainer.trainer import Trainer
from trainer.config import TrainConfig
from data.dataset import get_datapipe


def main(pickle_name, use_test_data):
    pickle_name = pickle_name.rstrip('.pkl').replace('\\', '/')

    model_config = ModelConfig()
    assert pickle_name == model_config.pickle_name
    model = Model(model_config)

    train_dataset = get_datapipe(pickle_name, 'train',
                                 frozen_embeddings=model_config.frozen_embeddings,
                                 sbert_model=model_config.sbert_model,
                                 hugface_model=model_config.hugface_model,
                                 max_length=model_config.max_length)

    if use_test_data:
        test_dataset = get_datapipe(pickle_name, 'test',
                                    frozen_embeddings=model_config.frozen_embeddings,
                                    sbert_model=model_config.sbert_model,
                                    hugface_model=model_config.hugface_model,
                                    max_length=model_config.max_length)
    else:
        test_dataset = None

    train_config = TrainConfig()
    trainer = Trainer(model=model,
                      train_dataset=train_dataset,
                      test_dataset=test_dataset,
                      train_config=train_config)

    trainer.distributed_train()


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('dataset', type=str, help='Data pickle file to train with.')
    my_parser.add_argument('--test', action='store_true', help='Use a test/validation data subset.')

    args = my_parser.parse_args()

    main(args.dataset, args.test)
