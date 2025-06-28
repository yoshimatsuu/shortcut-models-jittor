"""
Copyright (C) 2025 Yukara Ikemiya
"""

import os
import time

import torch
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter
from einops import rearrange
import cv2


class TimestepSampler:
    def __init__(
        self,
        rate_self_consistency: float = 0.25,
        min_dt: float = 0.0078125  # 1/128
    ):
        """
        rate_self_consistency: Propotion of samples for self-consistency term (default: 0.25)
        min_dt: Minimum value of 'dt' (default: 1/128)
        """
        assert 0 <= rate_self_consistency <= 1.0
        self.rate_sc = rate_self_consistency
        self.min_dt = min_dt

    def sample_t(self, num: int, device):
        num_sc = round(num * self.rate_sc)
        num_fm = num - num_sc

        # t for flow-matching term
        t_fm = torch.rand(num_fm, device=device)  # 0 -- 1
        dt_fm = torch.zeros(num_fm, device=device)

        # t/dt for self-consistency term
        t_sc = torch.rand(num_sc, device=device) * (1 - self.min_dt)  # 0 -- 1-min_dt
        max_dt = 1. - t_sc
        dt_sc = self.min_dt + torch.rand(num_sc, device=device) * (max_dt - self.min_dt)  # min_dt -- 1-t

        t = torch.cat([t_sc, t_fm])
        dt = torch.cat([dt_sc, dt_fm])
        assert len(t) == len(dt) == num

        return t, dt, num_sc


class Trainer:
    def __init__(
            self,
            model,  # model
            optimizer,  # optimizer
            scheduler,  # scheduler
            train_dataloader,
            cfg,  # Configurations
            output_root,
            logger,
            ckpt_dir=None
    ):
        self.model = model
        self.opt = optimizer
        self.sche = scheduler
        self.train_dataloader = train_dataloader
        self.cfg = cfg
        self.cfg_t = cfg.trainer
        self.EPS = 1e-8
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.logger = logger

        # timestep sampler
        cfg_ts_sampler = self.cfg_t.timestep_sampler
        self.ts_sampler = TimestepSampler(**cfg_ts_sampler)

        self.output_root = output_root
        self.sampling_root = os.path.join(self.output_root, 'sampling')
        os.makedirs(self.sampling_root)
        self.train_writer = SummaryWriter(os.path.join(self.output_root, "train"))

        self.states = {'global_step': 0, 'best_metrics': float('inf'), 'latest_metrics': float('inf')}

        if ckpt_dir is not None:
            self.__load_ckpt(ckpt_dir)

    def start_training(self):
        """
        Start training with infinite loops
        """
        self.model.train()

        self.logger.info("\n[ Started training ]\n")

        while True:
            for index, batch in enumerate(self.train_dataloader):
                # Update
                metrics = self.run_step(batch)
                self.states['global_step'] += 1
                global_step = self.states['global_step']

                if self.__its_time(self.cfg_t.logging.n_step_log):
                    self.logger.info(f"Step {global_step}: (loss: {metrics['loss']}, loss_fm: {metrics['loss_fm']}, loss_sc: {metrics['loss_sc']}, lr: {self.sche.get_last_lr()[0]})")

                # Save checkpoint
                if self.__its_time(self.cfg_t.logging.n_step_ckpt) and index != 0:
                    self.__save_ckpt()

                # Sample
                if self.__its_time(self.cfg_t.logging.n_step_sample):
                    self.__sampling()

                self.train_writer.add_scalar('loss', metrics['loss'], self.states['global_step'])
                self.train_writer.add_scalar('loss_fm', metrics['loss_fm'], self.states['global_step'])
                self.train_writer.add_scalar('loss_sc', metrics['loss_sc'], self.states['global_step'])
                self.train_writer.add_scalar('lr', self.sche.get_last_lr()[0], self.states['global_step'])

                if self.cfg_t.max_step is not None and self.cfg_t.max_step < self.states['global_step']:
                    exit(0)

    def run_step(self, batch, train: bool = True):
        """ One training step """

        images, labels = batch

        # image normalize
        # images = images / 255.
        # images = (images - 0.5) / 0.5

        # sample timesteps
        t, dt, num_sc = self.ts_sampler.sample_t(images.shape[0], device=self.device)

        # Update

        if train:
            self.opt.zero_grad()

        output = self.model.train_step(
            images.to(self.device), labels.to(self.device), t, dt, num_self_consistency=num_sc,
            cfg_dropout_prob=self.cfg_t.cfg_dropout_prob)

        if train:
            output['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg_t.max_grad_norm)
            self.opt.step()
            self.sche.step()

        return {k: v.detach() for k, v in output.items()}

    @torch.no_grad()
    def __sampling(self):
        self.model.eval()

        device = self.device
        steps: list = self.cfg_t.logging.steps
        n_sample: int = self.cfg_t.logging.n_samples_per_step
        n_label: int = self.cfg.model.num_label

        columns = ['labels', 'images']

        for step in steps:
            labels = torch.randint(n_label, size=(n_sample,), device=device)
            labels_str = '-'.join([str(label.item()) for label in labels])
            log_str = f"Step-{step}_{labels_str}"

            # sampling
            gen_sample = self.model.sample(labels, n_step=step)

            # reshape
            gen_sample = rearrange(gen_sample, 'b c h w -> c h (b w)')
            # gen_sample = gen_sample * 0.5 + 0.5
            gen_sample = torch.clip(gen_sample, 0, 1)
            gen_sample *= 255
            image = gen_sample.squeeze(0).cpu().numpy()

            global_step = self.states['global_step']
            # image = Image.fromarray(image)
            # image.save(os.path.join(self.sampling_root, f'Batch-{global_step}_'+log_str+'.jpg'))
            cv2.imwrite(os.path.join(self.sampling_root, f'Batch-{global_step}_'+log_str+'.jpg'), image)

        self.model.train()

        self.logger.info("\t->->-> Sampled.")

    def __save_ckpt(self):
        import shutil
        import json
        from omegaconf import OmegaConf

        out_dir = self.output_root + '/ckpt'

        # save latest ckpt
        latest_dir = out_dir + '/latest'
        os.makedirs(latest_dir, exist_ok=True)
        ckpts = {'model': self.model,
                 'optimizer': self.opt,
                 'scheduler': self.sche}
        for name, m in ckpts.items():
            torch.save(m.state_dict(), f"{latest_dir}/{name}.pth")

        # save states and configuration
        OmegaConf.save(self.cfg, f"{latest_dir}/config.yaml")
        with open(f"{latest_dir}/states.json", mode="wt", encoding="utf-8") as f:
            json.dump(self.states, f, indent=2)

        self.logger.info("\t->->-> Saved checkpoints.")

    def __load_ckpt(self, dir: str):
        import json

        self.logger.info(f"\n[Resuming training from the checkpoint directory] -> {dir}")
        ckpts = {'model': self.model,
                 'optimizer': self.opt,
                 'scheduler': self.sche}

        for k, v in ckpts.items():
            v.load_state_dict(torch.load(f"{dir}/{k}.pth", weights_only=False))

        with open(f"{dir}/states.json", mode="rt", encoding="utf-8") as f:
            self.states.update(json.load(f))

    def __its_time(self, itv: int):
        return (self.states['global_step'] - 1) % itv == 0
