import math
import transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class ReturnInput(nn.Module):
    def __init__(self):
        super().__init__()

    @autocast()
    def forward(self, x):
        return x


class InputBOW(nn.Module):
    def __init__(self):
        super().__init__()

    @autocast()
    def forward(self, x):
        x = x['x_bow']
        return x


class InputBERT(nn.Module):
    def __init__(self):
        super().__init__()

    @autocast()
    def forward(self, x):
        x = x['x_bert']
        return x


class InputCAT(nn.Module):
    def __init__(self):
        super().__init__()

    @autocast()
    def forward(self, x):
        x = torch.cat((x['x_bow'], x['x_bert']), dim=-1)
        return x


class LDA(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        # normalization
        assert model_config.normalization in ['batch', 'layer']
        if model_config.normalization == 'batch':
            self.beta_norm = nn.BatchNorm1d(model_config.input_size,
                                            affine=model_config.affine)
        elif model_config.normalization == 'layer':
            self.beta_norm = nn.LayerNorm(model_config.input_size,
                                          elementwise_affine=model_config.affine)

    @autocast()
    def forward(self, theta, beta):
        beta = F.softmax(self.beta_norm(beta), dim=-1)
        word_dist = torch.matmul(theta, beta)
        return beta, word_dist


class ProdLDA(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        # normalization
        assert model_config.normalization in ['batch', 'layer']
        if model_config.normalization == 'batch':
            self.beta_norm = nn.BatchNorm1d(model_config.input_size,
                                            affine=model_config.affine)
        elif model_config.normalization == 'layer':
            self.beta_norm = nn.LayerNorm(model_config.input_size,
                                          elementwise_affine=model_config.affine)

    @autocast()
    def forward(self, theta, beta):
        word_dist = F.softmax(self.beta_norm(torch.matmul(theta, beta)), dim=-1)
        return beta, word_dist


class ThetaVAE(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.layers = nn.ModuleList([])
        if model_config.theta_softmax:
            self.layers.append(nn.Softmax(dim=-1))
        if model_config.decoder_dropout > 0:
            self.layers.append(nn.Dropout(p=model_config.decoder_dropout))

    @autocast()
    def forward(self, posterior_mu, posterior_log_sigma, posterior_distribution):
        theta = posterior_distribution.rsample()
        for layer in self.layers:
            theta = layer(theta)
        return theta


class ThetaRL(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.layers = nn.ModuleList([])
        if model_config.theta_softmax:
            self.layers.append(nn.Softmax(dim=-1))
        if model_config.decoder_dropout > 0:
            self.layers.append(nn.Dropout(p=model_config.decoder_dropout))

    @autocast()
    def forward(self, posterior_mu, posterior_log_sigma, posterior_distribution):
        action = posterior_distribution.rsample()
        theta = (1 / (torch.exp(posterior_log_sigma) * math.sqrt(2 * math.pi))) \
            * torch.exp(-1.0 * (action - posterior_mu) ** 2 / (2 * torch.exp(posterior_log_sigma) ** 2))
        for layer in self.layers:
            theta = layer(theta)
        return theta


# fine-tuning of embeddings not used in paper
class BERTLayer(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.layer = transformers.AutoModel.from_pretrained(model_config.model_path,
                                                            output_attentions=False,
                                                            output_hidden_states=False)

    @autocast()
    def forward(self, x):
        x_ids = x['input_ids'].squeeze(dim=1)
        x_mask = x['attention_mask'].squeeze(dim=1)

        output = self.layer(x_ids, attention_mask=x_mask)
        output = torch.mean(output.last_hidden_state, dim=1)
        return output


