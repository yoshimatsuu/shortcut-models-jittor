from itertools import repeat
import collections.abc
from functools import partial
import math
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from enum import Enum
import jittor as jt
import numpy as np


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


def modulate(x, shift, scale):
    return x * (1 + jt.unsqueeze(scale, 1)) + jt.unsqueeze(shift, 1)


to_2tuple = _ntuple(2)


class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'


class PatchEmbed(jt.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            bias: bool = True,
    ):
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        self.proj = jt.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else jt.nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def execute(self, x):
        x = self.proj(x)

        x = jt.transpose(jt.flatten(x, start_dim=2), 1, 2)  # NCHW -> NLC

        x = self.norm(x)
        return x


class TimestepEmbedder(jt.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        self.linear1 = jt.nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.linear2 = jt.nn.Linear(hidden_size, hidden_size, bias=True)

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = (-math.log(max_period) * jt.arange(start=0, end=half, dtype=jt.float32) / half).exp()
        args = t[:, None].float() * freqs[None]
        embedding = jt.concat([args.cos(), args.sin()], dim=-1)
        if dim % 2:
            embedding = jt.concat([embedding, jt.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def execute(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.linear2(jt.nn.silu(self.linear1(t_freq)))
        return t_emb


class LabelEmbedder(jt.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = jt.nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = jt.rand(labels.shape[0]) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = jt.where(drop_ids, self.num_classes, labels)
        return labels

    def execute(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)

        return embeddings


class Attention(jt.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[jt.Module] = jt.nn.LayerNorm,
    ) -> None:
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = jt.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else jt.nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else jt.nn.Identity()
        self.attn_drop = jt.nn.Dropout(attn_drop)
        self.proj = jt.nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = jt.nn.Dropout(proj_drop)

    def execute(self, x):
        B, N, C = x.shape
        qkv = jt.reshape(self.qkv(x), (B, N, 3, self.num_heads, self.head_dim)).permute(2, 0, 3, 1, 4)
        q, k, v = jt.unbind(qkv, 0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ jt.transpose(k, -2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = jt.reshape(jt.transpose(x, 1, 2), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(jt.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(jt.nn.Conv2d, kernel_size=1) if use_conv else jt.nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.drop1 = jt.nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else jt.nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = jt.nn.Dropout(drop_probs[1])

    def execute(self, x):
        x = self.fc1(x)
        x = jt.nn.gelu(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DiTBlock(jt.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        self.norm1 = jt.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = jt.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0)
        self.linear = jt.nn.Linear(hidden_size, 6 * hidden_size, bias=True)

    def execute(self, x, c):
        c = self.linear(jt.nn.silu(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(jt.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        self.norm_final = jt.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear1 = jt.nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.linear2 = jt.nn.Linear(hidden_size, 2 * hidden_size, bias=True)

    def execute(self, x, c):
        c = self.linear2(jt.nn.silu(c))
        shift, scale = c.chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear1(x)

        return x


class DiT(jt.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=10,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.dt_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = jt.zeros(1, num_patches, hidden_size)
        self.pos_embed.requires_grad = False

        self.blocks = jt.nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        for name, module in self.named_modules():
            if isinstance(module, jt.nn.Linear):
                jt.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    jt.init.constant_(module.bias, 0)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed = jt.array(pos_embed, dtype=jt.float32).unsqueeze(0)
        self.pos_embed.requires_grad = False

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight
        jt.init.xavier_uniform_(w.view([w.shape[0], -1]))
        jt.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        jt.init.trunc_normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        jt.init.trunc_normal_(self.t_embedder.linear1.weight, std=0.02)
        jt.init.trunc_normal_(self.t_embedder.linear2.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            jt.init.constant_(block.linear.weight, 0)
            jt.init.constant_(block.linear.bias, 0)

        # Zero-out output layers:
        jt.init.constant_(self.final_layer.linear2.weight, 0)
        jt.init.constant_(self.final_layer.linear2.bias, 0)
        jt.init.constant_(self.final_layer.linear1.weight, 0)
        jt.init.constant_(self.final_layer.linear1.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = jt.linalg.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], c, h * p, h * p)
        return imgs

    def execute(self, x, t, dt, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        dt = self.dt_embedder(dt)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y + dt                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


if __name__ == '__main__':
    test_model = DiT(32, 2, 1, 32, 2, 2)

    tmp = jt.randn(4, 1, 32, 32)
    y = jt.array([1, 2, 3, 4])
    t = jt.rand(4)
    dt = jt.rand(4)

    print(test_model(tmp, t, dt, y))
    # print(test_model(tmp, c).shape)
