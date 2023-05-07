import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.distributions.normal import Normal

import trainer.utils as utils
from model.other_networks import LDA, ProdLDA, ThetaVAE, ThetaRL, InputBOW, InputBERT, InputCAT
from data.dataset import get_dataset
from model.inference_network import InferenceNetwork, NoisyInferenceNetwork
from evals.measures import NPMICoherence, topic_diversity, calculate_perplexity_batch


class NetworkConfigs:
    def __init__(self):
        # MODEL CONFIGURATION
        self.n_components = 20  # number of topics
        # decoder network
        self.input_size = 2000  # vocabulary size
        self.input_type = 'bert'
        self.decoder_dropout = 0.2  # policy / theta dropout
        self.initialization = 'normal'
        self.normalization = 'layer'
        self.affine = False
        self.loss_type = 'rl'
        self.lda_type = 'prodlda'
        self.theta_softmax = False
        # inference network
        self.frozen_embeddings = True
        self.sbert_model = 'all-MiniLM-L6-v2'
        self.hugface_model = 'sentence-transformers/all-MiniLM-L6-v2'
        self.bert_size = 384
        self.max_length = 256
        self.hiddens = (128, 128)
        self.activation = nn.GELU()
        self.inference_dropout = 0.2
        self.parameter_noise = False
        # other
        self.prior = 'laplace'
        self.trainable_prior = True
        self.kl_mult = 1.0
        self.entropy_mult = 0.0
        self.topk = [10]
        self.sparse_corpus_bow = None
        self.pickle_name = 'data/pickles/20newsgroups_mwl3'  # don't include ".pkl"
        self.get_sparse_corpus_bow()

    def get_sparse_corpus_bow(self):
        dataset_save_dict = get_dataset(self.pickle_name)
        self.sparse_corpus_bow = dataset_save_dict['sparse_corpus_bow']


class DecoderNetwork(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        # input
        assert model_config.input_type in ['bow', 'bert', 'cat']
        if model_config.input_type == 'bow':
            true_input_size = model_config.input_size
            self.get_input = InputBOW()
        elif model_config.input_type == 'bert':
            true_input_size = model_config.bert_size
            self.get_input = InputBERT()
        elif model_config.input_type == 'cat':
            true_input_size = model_config.input_size + model_config.bert_size
            self.get_input = InputCAT()
        else:
            raise ValueError('Invalid input type.')

        # inference network (parameter noise not used in paper)
        if model_config.parameter_noise:
            self.mu_inference = NoisyInferenceNetwork(model_config, true_input_size,
                                                      output_normalization=None)
            self.sigma_inference = NoisyInferenceNetwork(model_config, true_input_size,
                                                         output_normalization=model_config.normalization)
        else:
            self.mu_inference = InferenceNetwork(model_config, true_input_size,
                                                 output_normalization=None)
            self.sigma_inference = InferenceNetwork(model_config, true_input_size,
                                                    output_normalization=model_config.normalization)

        # priors
        if model_config.prior == 'gaussian':
            prior_mean = torch.Tensor(torch.zeros(model_config.n_components))
            prior_variance = torch.Tensor(torch.ones(model_config.n_components))
        elif model_config.prior == 'laplace':
            a = 1 * torch.ones((1, model_config.n_components))
            prior_mean = torch.Tensor((torch.log(a).T - torch.mean(torch.log(a), dim=-1)).squeeze())
            prior_variance = torch.Tensor((((1.0 / a) * (1 - (2.0 / model_config.n_components))).T
                                          + (1.0 / (model_config.n_components * model_config.n_components))
                                          * torch.sum(1.0 / a, dim=-1)).squeeze())
        else:
            raise ValueError('Invalid prior.')

        if model_config.trainable_prior:
            self.register_parameter('prior_mean', nn.Parameter(prior_mean))
            self.register_parameter('prior_variance', nn.Parameter(prior_variance))
        else:
            self.register_buffer('prior_mean', prior_mean)
            self.register_buffer('prior_variance', prior_variance)

        # topic distribution inference
        assert model_config.loss_type in ['vae', 'rl']
        if model_config.loss_type == 'vae':
            self.theta = ThetaVAE(model_config)
        elif model_config.loss_type == 'rl':
            self.theta = ThetaRL(model_config)

        # initialization
        assert model_config.initialization in ['normal', 'xavier']
        if model_config.initialization == 'normal':
            self.beta = nn.Parameter(0.02 * torch.randn((model_config.n_components,
                                                         model_config.input_size)))
        elif model_config.initialization == 'xavier':
            self.beta = nn.Parameter(torch.Tensor(model_config.n_components,
                                                  model_config.input_size))
            nn.init.xavier_uniform_(self.beta)

        # word distribution inference
        assert model_config.lda_type in ['lda', 'prodlda']
        if model_config.lda_type == 'lda':
            self.lda = LDA(model_config)
        elif model_config.lda_type == 'prodlda':
            self.lda = ProdLDA(model_config)

        # other
        self.n_components = model_config.n_components
        self.kl_mult = model_config.kl_mult
        self.entropy_mult = model_config.entropy_mult
        self.topk = model_config.topk
        self.coherence = NPMICoherence(sparse_corpus_bow=model_config.sparse_corpus_bow)

        self.init_type = model_config.initialization
        if model_config.frozen_embeddings:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.init_type == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif self.init_type == 'xavier':
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        optimizer = utils.configure_optimizers(self, train_config)
        return optimizer

    def kl_divergence(self, p_mean, p_variance, q_mean, q_variance):
        var_division = torch.sum(p_variance ** 2 / q_variance ** 2, dim=-1)
        diff_term = torch.sum((q_mean - p_mean) ** 2 / q_variance ** 2, dim=-1)
        logvar_det_division = torch.sum(torch.log(q_variance ** 2) - torch.log(p_variance ** 2), dim=-1)
        return 0.5 * (var_division + diff_term - self.n_components + logvar_det_division)

    def loss_fn(self, bow, word_dist, posterior_mu, posterior_log_sigma,
                posterior_distribution, epsilon=1e-8):
        # forward KL divergence
        unscaled_kl = self.kl_divergence(posterior_mu, torch.exp(posterior_log_sigma),
                                         self.prior_mean, self.prior_variance)
        
        kl = self.kl_mult * unscaled_kl

        # entropy regularization (not used in paper)
        if self.entropy_mult > 0:
            entropy = self.entropy_mult * torch.sum(posterior_distribution.entropy(), dim=-1)
        else:
            entropy = 0
        
        # reconstruction loss (log likelihood)
        reward = -1.0 * torch.sum(bow * torch.log(word_dist + epsilon), dim=-1)

        loss = reward + entropy + kl
        return loss.mean()

    @autocast()
    def forward(self, x):
        x_bow = x['x_bow']
        x = self.get_input(x)

        posterior_mu = self.mu_inference(x)
        posterior_log_sigma = self.sigma_inference(x)
        posterior_distribution = Normal(posterior_mu, torch.exp(posterior_log_sigma))

        theta = self.theta(posterior_mu, posterior_log_sigma, posterior_distribution)

        beta, word_dist = self.lda(theta, self.beta)

        loss = self.loss_fn(x_bow, word_dist, posterior_mu, posterior_log_sigma,
                            posterior_distribution)

        coherence = 0
        diversity = 0
        for i, topk in enumerate(self.topk):
            coherence = coherence * i / (i + 1) + self.coherence.get_coherence(beta, topk) / (i + 1)
            diversity = diversity * i / (i + 1) + topic_diversity(beta, topk) / (i + 1)

        perplexity = calculate_perplexity_batch(x_bow, word_dist)

        # should change this to be updated with the tqdm bar
        print(f'COHERENCE: {coherence}')
        print(f'DIVERSITY: {diversity}')
        print(f'PERPLEXITY: {perplexity}')

        output_dict = {
            'loss': loss,
            'theta': theta,
            'beta': beta,
            'word_dist': word_dist,
            'coherence': coherence,
            'diversity': diversity,
            'perplexity': perplexity
        }
        return output_dict
