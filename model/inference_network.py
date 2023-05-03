import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from model.other_networks import ReturnInput, BERTLayer


class InferenceNetwork(nn.Module):
    def __init__(self, model_config, true_input_size, output_normalization):
        super().__init__()
        self.layers = nn.ModuleList([])
        if model_config.frozen_embeddings:
            hidden_output_size = model_config.hiddens[-1]
            self.layers.append(nn.ModuleList([nn.Linear(true_input_size, model_config.hiddens[0]),
                                              model_config.activation,
                                              nn.Dropout(p=model_config.inference_dropout)]))
            for (in_feat, out_feat) in zip(model_config.hiddens[:-1], model_config.hiddens[1:]):
                self.layers.append(nn.ModuleList([nn.Linear(in_feat, out_feat),
                                                  model_config.activation,
                                                  nn.Dropout(p=model_config.inference_dropout)]))
        # fine-tuning of embeddings not used in paper
        else:
            hidden_output_size = model_config.bert_size
            self.layers.append(BERTLayer(model_config))

        self.output = nn.Linear(hidden_output_size, model_config.n_components)

        # normalization
        if output_normalization is None:
            self.norm = ReturnInput()
        else:
            assert output_normalization in ['batch', 'layer']
            if output_normalization == 'batch':
                self.norm = nn.BatchNorm1d(model_config.n_components,
                                           affine=model_config.affine)
            elif output_normalization == 'layer':
                self.norm = nn.LayerNorm(model_config.n_components,
                                         elementwise_affine=model_config.affine)

    @autocast()
    def forward(self, x):
        for linear, activation, dropout in self.layers:
            x = linear(x)
            x = activation(x)
            x = dropout(x)
        x = self.output(x)
        x = self.norm(x)
        return x


# parameter noise not used in paper
class NoisyInferenceNetwork(nn.Module):
    def __init__(self, model_config, true_input_size, output_normalization):
        super().__init__()
        self.layers = nn.ModuleList([])
        if model_config.frozen_embeddings:
            hidden_output_size = model_config.hiddens[-1]
            self.layers.append(nn.ModuleList([nn.Linear(true_input_size, model_config.hiddens[0]),
                                              nn.LayerNorm(model_config.hiddens[0], elementwise_affine=False),
                                              model_config.activation,
                                              nn.Dropout(p=model_config.inference_dropout)]))
            for (in_feat, out_feat) in zip(model_config.hiddens[:-1], model_config.hiddens[1:]):
                self.layers.append(nn.ModuleList([nn.Linear(in_feat, out_feat),
                                                  nn.LayerNorm(out_feat, elementwise_affine=False),
                                                  model_config.activation,
                                                  nn.Dropout(p=model_config.inference_dropout)]))
        # fine-tuning of embeddings not used in paper
        else:
            hidden_output_size = model_config.bert_size
            self.layers.append(BERTLayer(model_config))

        self.output = nn.Linear(hidden_output_size, model_config.n_components)

        # normalization
        if output_normalization is None:
            self.norm = ReturnInput()
        else:
            assert output_normalization in ['batch', 'layer']
            if output_normalization == 'batch':
                self.norm = nn.BatchNorm1d(model_config.n_components,
                                           affine=model_config.affine)
            elif output_normalization == 'layer':
                self.norm = nn.LayerNorm(model_config.n_components,
                                         elementwise_affine=model_config.affine)

        # noise scaling
        self.noise_scale = 0.5

    @autocast()
    def forward(self, x):
        for linear, normalization, activation, dropout in self.layers:
            x = linear(x)
            if self.training:
                x += self.noise_scale * torch.randn_like(x)
            x = normalization(x)
            x = activation(x)
            x = dropout(x)
        x = self.output(x)
        if self.training:
            x += self.noise_scale * torch.randn_like(x)
        x = self.norm(x)
        return x
