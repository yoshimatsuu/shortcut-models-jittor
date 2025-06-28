import typing as tp

import jittor as jt

from dit import DiT
from utils.common import exists


class ShortcutModel(jt.Module):
    def __init__(
            self,
            num_label: int,
            patch_width: int,
            model_config: dict
    ):
        self.num_label = num_label
        self.patch_width = patch_width
        self.input_size = (model_config.dim_in, model_config.input_size, model_config.input_size)

        self.model = DiT(input_size=model_config.input_size,
                         patch_size=patch_width,
                         in_channels=model_config.dim_in,
                         hidden_size=model_config.dim,
                         depth=model_config.depth,
                         num_heads=model_config.num_heads,
                         num_classes=num_label,
                         )

    def train_step(
            self,
            images: jt.Var,
            labels: jt.Var,
            t: jt.Var,
            dt: jt.Var,
            num_self_consistency: int,
            cfg_dropout_prob: float = 0.1
    ):
        """
                x1: ground-truth data (e.g. image)
                """
        # MNIST, 1 channel
        images = images[:, 0:1]

        bs, ch, H, W = images.shape

        assert len(labels) == len(t) == len(dt) == bs and jt.all(t + dt <= 1.0)
        assert num_self_consistency < bs

        x1 = images
        x0 = jt.randn_like(x1)  # noise
        x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1  # eq.(1)
        v_t = x1 - x0

        if num_self_consistency > 0:
            x_t_sc = x_t[:num_self_consistency]
            t_sc = t[:num_self_consistency]
            dt_half = dt[:num_self_consistency] * 0.5
            labels_sc = labels[:num_self_consistency]
            # calculate targets for self-consistency term (eq.(5))
            with jt.no_grad():
                self.model.eval()
                v1_sc = self.model(x_t_sc, t_sc, dt_half, labels_sc)
                v2_sc = self.model(x_t_sc + dt_half[:, None, None, None] * v1_sc, t_sc + dt_half, dt_half, labels_sc)
                self.model.train()

            v_t_sc = (v1_sc + v2_sc) / 2.

        # dt = 0.0 -> naive flow-matching
        dt[num_self_consistency:] = 0.

        # forward
        v_out = self.model(x_t, t, dt, labels)

        output = {}

        # flow-matching loss (eq.(5))
        loss = jt.nn.mse_loss(v_out[num_self_consistency:], v_t[num_self_consistency:])
        output['loss_fm'] = loss.detach()

        # self-consistency loss (eq.(5))
        if num_self_consistency > 0:
            loss_sc = jt.nn.mse_loss(v_out[:num_self_consistency], v_t_sc)
            loss += loss_sc
            output['loss_sc'] = loss_sc.detach()

        output['loss'] = loss
        return output

    def sample(
        self,
        labels,
        n_step: tp.Optional[int] = None,
        dt_list: tp.Optional[tp.List[int]] = None,
        input_shape: tp.Optional[tp.List[int]] = None,
        disable_shortcut: bool = False
    ):
        with jt.no_grad():
            assert exists(n_step) or exists(dt_list)
            num_sample = len(labels)

            if exists(input_shape):
                ch_in, H_in, W_in = input_shape
            else:
                ch_in, H_in, W_in = self.input_size

            if exists(n_step):
                dt_list = [1. / n_step] * n_step

            assert sum(dt_list) <= 1 + 1e-6

            # initial noise
            x = jt.randn(num_sample, ch_in, H_in, W_in)

            # sample
            t_cur = jt.zeros(num_sample)
            for dt_val in dt_list:
                dt = jt.full((num_sample,), dt_val)

                # predict
                if disable_shortcut:
                    dt_in = jt.zeros_like(dt)
                else:
                    dt_in = dt

                self.model.eval()
                vel = self.model(x, t_cur, dt_in, labels)

                # update
                x += vel * dt[:, None, None, None]

                t_cur = t_cur + dt

            return x