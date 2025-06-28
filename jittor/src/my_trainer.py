"""
Copyright (C) 2025 Yukara Ikemiya
"""

import os
import time

import numpy as np
from PIL import Image
import jittor as jt
from jittor.einops import rearrange
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

    def sample_t(self, num: int):
        num_sc = round(num * self.rate_sc)
        num_fm = num - num_sc

        # t for flow-matching term
        t_fm = jt.rand(num_fm)  # 0 -- 1
        dt_fm = jt.zeros(num_fm)

        # t/dt for self-consistency term
        t_sc = jt.rand(num_sc) * (1 - self.min_dt)  # 0 -- 1-min_dt
        max_dt = 1. - t_sc
        dt_sc = self.min_dt + jt.rand(num_sc) * (max_dt - self.min_dt)  # min_dt -- 1-t

        t = jt.concat([t_sc, t_fm])
        dt = jt.concat([dt_sc, dt_fm])
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
    ):
        self.model = model
        self.opt = optimizer
        self.sche = scheduler
        self.train_dataloader = train_dataloader
        self.cfg = cfg
        self.cfg_t = cfg.trainer
        self.EPS = 1e-8
        self.logger = logger

        # timestep sampler
        cfg_ts_sampler = self.cfg_t.timestep_sampler
        self.ts_sampler = TimestepSampler(**cfg_ts_sampler)

        self.output_root = output_root
        self.sampling_root = os.path.join(self.output_root, 'sampling')
        os.makedirs(self.sampling_root)

        self.states = {'global_step': 0, 'best_metrics': float('inf'), 'latest_metrics': float('inf')}
        self.metric_dict = {'loss': [], 'loss_fm': [], 'loss_sc': [], 'lr': []}

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
                    self.__save_model()

                self.metric_dict['loss'].append(metrics['loss'])
                self.metric_dict['loss_fm'].append(metrics['loss_fm'])
                self.metric_dict['loss_sc'].append(metrics['loss_sc'])
                self.metric_dict['lr'].append(self.sche.get_last_lr()[0])

                # Sample
                if self.__its_time(self.cfg_t.logging.n_step_sample):
                    self.__sampling()
                    np.save(os.path.join(self.output_root, 'metrics.npy'), np.array(self.metric_dict, dtype=object))

                if self.cfg_t.max_step is not None and self.cfg_t.max_step < self.states['global_step']:
                    exit(0)


    def run_step(self, batch, train: bool = True):
        """ One training step """

        images, labels = batch

        # image normalize
        # images = images / 255.
        # images = (images - 0.5) / 0.5

        # sample timesteps
        t, dt, num_sc = self.ts_sampler.sample_t(images.shape[0])

        if train:
            self.opt.zero_grad()

        output = self.model.train_step(
            images, labels, t, dt, num_self_consistency=num_sc,
            cfg_dropout_prob=self.cfg_t.cfg_dropout_prob)

        if train:
            self.opt.backward(output['loss'])
            self.opt.clip_grad_norm(self.cfg_t.max_grad_norm)
            self.opt.step()
            self.sche.step()

        return {k: v.detach().item() for k, v in output.items()}

    def __sampling(self):
        with jt.no_grad():
            self.model.eval()

            steps: list = self.cfg_t.logging.steps
            n_sample: int = self.cfg_t.logging.n_samples_per_step
            n_label: int = self.cfg.model.num_label

            for step in steps:
                labels = jt.randint(low=0, high=n_label, shape=(n_sample,))
                labels_str = '-'.join([str(label.item()) for label in labels])
                log_str = f"Step-{step}_{labels_str}"

                # sampling
                gen_sample = self.model.sample(labels, n_step=step)

                # reshape
                gen_sample = rearrange(gen_sample, 'b c h w -> c h (b w)')
                # gen_sample = gen_sample * 0.5 + 0.5
                gen_sample = jt.safe_clip(gen_sample, 0, 1)
                gen_sample *= 255
                image = gen_sample.squeeze(0).numpy()

                global_step = self.states['global_step']
                # image = Image.fromarray(image)
                # image.save(os.path.join(self.sampling_root, f'Batch-{global_step}_'+log_str+'.jpg'))
                cv2.imwrite(os.path.join(self.sampling_root, f'Batch-{global_step}_'+log_str+'.jpg'), image)

            self.model.train()

            self.logger.info("\t->->-> Sampled.")

    def __save_model(self):
        import json
        from omegaconf import OmegaConf

        out_dir = self.output_root + '/ckpt'

        # save latest ckpt
        latest_dir = out_dir + '/latest'
        os.makedirs(latest_dir, exist_ok=True)
        self.model.save(f"{latest_dir}/model.pkl")

        # save states and configuration
        OmegaConf.save(self.cfg, f"{latest_dir}/config.yaml")
        with open(f"{latest_dir}/states.json", mode="wt", encoding="utf-8") as f:
            json.dump(self.states, f, indent=2)

        self.logger.info("\t->->-> Saved checkpoints.")

    def __its_time(self, itv: int):
        return (self.states['global_step'] - 1) % itv == 0
