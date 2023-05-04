import os
import random
import logging
import numpy as np
from tqdm import tqdm

from evals.plotting import Outputs, save_training_outputs
from trainer.config import get_scheduler, step_scheduler

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, train_config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_config = train_config

        if train_config.save_outputs:
            self.outputs = Outputs(train_config.experiment_name,
                                   train_config.experiment_num,
                                   train_config.check_experiment_path)

    def setup(self, rank):
        os.environ['MASTER_ADDR'] = self.train_config.master_address
        os.environ['MASTER_PORT'] = self.train_config.master_port
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = self.train_config.torch_distributed_debug

        # initialize the process group
        dist.init_process_group(self.train_config.backend,
                                rank=rank, world_size=self.train_config.world_size)

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    def distributed_train(self):
        # not sure how to catch errors within this
        mp.spawn(self.train, nprocs=self.train_config.world_size, join=True)

    def save_checkpoint(self, model):
        ckpt_file = f'{self.train_config.ckpt_path}/model.ckpt'

        raw_model = model.module if hasattr(model, "module") else model
        torch.save(raw_model.state_dict(), ckpt_file)

    def train(self, rank):
        # pull model and train_config from class
        model, train_config = self.model, self.train_config
        # mixed precision training
        scaler = GradScaler()
        # distributed data parallel
        self.setup(rank)
        model = model.to(rank)
        # set find unused parameters in train config
        model = DDP(model, device_ids=[rank], find_unused_parameters=train_config.find_unused_parameters)

        raw_model = model.module if hasattr(model, "module") else model
        optimizer = raw_model.configure_optimizers(train_config)
        if train_config.lr_decay:
            scheduler = get_scheduler(optimizer)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset

            distributed_sampler = DistributedSampler(data, shuffle=is_train)

            if train_config.deterministic_dataloader:
                def seed_worker(worker_id):
                    worker_seed = torch.initial_seed() % 2 ** 32
                    np.random.seed(worker_seed)
                    random.seed(worker_seed)

                generator = torch.Generator()
                generator.manual_seed(train_config.seed)

                worker_init_fn = seed_worker
            else:
                worker_init_fn = None
                generator = None

            loader = DataLoader(data, pin_memory=True,
                                batch_size=train_config.batch_size,
                                num_workers=train_config.num_workers,
                                worker_init_fn=worker_init_fn,
                                generator=generator,
                                sampler=distributed_sampler)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if (is_train and rank == 0) else enumerate(loader)
            for it, batch in pbar:
                with torch.set_grad_enabled(is_train):
                    output_dict = model(batch)
                    loss = output_dict['loss']
                    losses.append(loss.item())
                    if train_config.save_outputs:
                        self.outputs.save_step(output_dict, split, epoch, it)

                if is_train:
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    if train_config.grad_norm_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    # report progress
                    if rank == 0:
                        # this assumes learning rate is the same across all parameter groups
                        lr = optimizer.param_groups[0]['lr']
                        pbar.set_description(f"epoch {epoch} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if train_config.lr_decay:
                step_scheduler(scheduler, output_dict)

            if is_train:
                return output_dict
            else:
                test_loss = float(np.mean(losses))
                logger.info(f'test loss: {test_loss}')
                return output_dict, test_loss

        best_loss = float('inf')
        for epoch in range(train_config.max_epochs):
            output_dict = run_epoch('train')
            if train_config.save_outputs:
                self.outputs.save_epoch(output_dict, epoch)
            if self.test_dataset is not None:
                output_dict, test_loss = run_epoch('test')
                if train_config.save_pickle_every > 0 and epoch % train_config.save_pickle_every == 0:
                    self.outputs.save_pickle()

            good_model = self.test_dataset is None or test_loss < best_loss
            if train_config.ckpt_path is not None and good_model and rank == 0:
                if self.test_dataset is not None:
                    print(f'good model: test loss = {test_loss}')
                    best_loss = test_loss
                self.save_checkpoint(model)

        if train_config.save_outputs:
            if train_config.save_pickle_every > 0:
                self.outputs.save_pickle()
            save_training_outputs(train_config.experiment_name,
                                  train_config.experiment_num,
                                  self.outputs)

        self.cleanup()
