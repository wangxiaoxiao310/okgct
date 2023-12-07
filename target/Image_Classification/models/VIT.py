import math
import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import HeUniform, TruncatedNormal, initializer

import mindspore
import numpy as np
from mindspore import Tensor, nn, ops
from mindspore.numpy import ones
from enum import Enum


class Format(str, Enum):
    NCHW = "NCHW"
    NHWC = "NHWC"
    NCL = "NCL"
    NLC = "NLC"


def nchw_to(x: mindspore.Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(start_dim=2).transpose((0, 2, 1))
    elif fmt == Format.NCL:
        x = x.flatten(start_dim=2)
    return x


def nhwc_to(x: mindspore.Tensor, fmt: Format):
    if fmt == Format.NCHW:
        x = x.permute(0, 3, 1, 2)
    elif fmt == Format.NLC:
        x = x.flatten(start_dim=1, end_dim=2)
    elif fmt == Format.NCL:
        x = x.flatten(start_dim=1, end_dim=2).transpose((0, 2, 1))
    return x


class DropPath(mindspore.nn.Cell):
    def __init__(
        self,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True,
    ) -> None:
        super().__init__()
        self.keep_prob = 1.0 - drop_prob
        self.scale_by_keep = scale_by_keep
        self.dropout = mindspore.nn.Dropout(p=drop_prob)

    def construct(self, x: Tensor) -> Tensor:
        if self.keep_prob == 1.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.dropout(ones(shape))
        if not self.scale_by_keep:
            random_tensor = ops.mul(random_tensor, self.keep_prob)
        return x * random_tensor


class Mlp(nn.Cell):
    def __init__(
        self,
        in_features: int,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchDropout(nn.Cell):
    def __init__(
        self,
        prob: float = 0.5,
        num_prefix_tokens: int = 1,
        ordered: bool = False,
        return_indices: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens
        self.ordered = ordered
        self.return_indices = return_indices
        self.sort = ops.Sort()

    def construct(self, x):
        if not self.training or self.prob == 0.0:
            if self.return_indices:
                return x, None
            return x

        if self.num_prefix_tokens:
            prefix_tokens, x = x[:, : self.num_prefix_tokens], x[:, self.num_prefix_tokens :]
        else:
            prefix_tokens = None

        B = x.shape[0]
        L = x.shape[1]
        num_keep = max(1, int(L * (1.0 - self.prob)))
        _, indices = self.sort(ms.Tensor(np.random.rand(B, L)).astype(ms.float32))
        keep_indices = indices[:, :num_keep]
        if self.ordered:
            keep_indices, _ = self.sort(keep_indices)
        keep_indices = ops.broadcast_to(ops.expand_dims(keep_indices, axis=-1), (-1, -1, x.shape[2]))
        x = ops.gather_elements(x, dim=1, index=keep_indices)

        if prefix_tokens is not None:
            x = ops.concat((prefix_tokens, x), axis=1)

        if self.return_indices:
            return x, keep_indices
        return x


def resample_abs_pos_embed(
    posemb,
    new_size,
    old_size=None,
    num_prefix_tokens=1,
    interpolation="nearest",
):
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens

    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    interpolate = mindspore.ops.interpolate(mode=interpolation, align_corners=True)
    posemb = interpolate(posemb, size=new_size)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.astype(orig_dtype)

    if posemb_prefix is not None:
        posemb = ops.concatcat((posemb_prefix, posemb), axis=1)

    return posemb


from mindspore import Tensor, nn, ops
import collections
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


class PatchEmbed(nn.Cell):
    output_fmt: Format

    def __init__(
        self,
        image_size=224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer=None,
        flatten: bool = True,
        output_fmt=None,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if image_size is not None:
            self.image_size = to_2tuple(image_size)
            self.patches_resolution = tuple([s // p for s, p in zip(self.image_size, self.patch_size)])
            self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        else:
            self.image_size = None
            self.patches_resolution = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            self.flatten = flatten
            self.output_fmt = Format.NCHW

        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            pad_mode="pad",
            has_bias=bias,
            weight_init="TruncatedNormal",
        )

        if norm_layer is not None:
            if isinstance(embed_dim, int):
                embed_dim = (embed_dim,)
            self.norm = norm_layer(embed_dim, epsilon=1e-5)
        else:
            self.norm = None

    def construct(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        if self.image_size is not None:
            if self.strict_img_size:
                if (H, W) != (self.image_size[0], self.image_size[1]):
                    raise ValueError(
                        f"Input height and width ({H},{W}) doesn't match model ({self.image_size[0]},"
                        f"{self.image_size[1]})."
                    )
            elif not self.dynamic_img_pad:
                if H % self.patch_size[0] != 0:
                    raise ValueError(f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]}).")
                if W % self.patch_size[1] != 0:
                    raise ValueError(f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]}).")
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = ops.pad(x, (0, pad_w, 0, pad_h))

        x = self.proj(x)
        if self.flatten:
            x = ops.Reshape()(x, (B, self.embed_dim, -1))
            x = ops.Transpose()(x, (0, 2, 1))
        elif self.output_fmt != "NCHW":
            x = nchw_to(x, self.output_fmt)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Attention(nn.Cell):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = Tensor(self.head_dim**-0.5)

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.q_norm = norm_layer((self.head_dim,)) if qk_norm else nn.Identity()
        self.k_norm = norm_layer((self.head_dim,)) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(p=attn_drop, dtype=ms.float32)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop, dtype=ms.float32)

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack = ops.Unstack(axis=0)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)

    def construct(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (b, n, 3, self.num_heads, self.head_dim))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = self.unstack(qkv)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = self.q_matmul_k(q, k)
        attn = self.mul(attn, self.scale)

        attn = attn.astype(ms.float32)
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v)
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.reshape(out, (b, n, c))
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class LayerScale(nn.Cell):
    def __init__(self, dim, init_values=1e-5):
        super(LayerScale, self).__init__()
        self.gamma = Parameter(initializer(init_values, dim))

    def construct(self, x):
        return self.gamma * x


class Block(nn.Cell):
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super(Block, self).__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim=dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer((dim,))
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim=dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def construct(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Cell):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        global_pool: str = "token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        drop_rate=0.0,
        pos_drop_rate=0.0,
        patch_drop_rate=0.0,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init=True,
        init_values=None,
        no_embed_class=False,
        pre_norm=False,
        fc_norm=None,
        dynamic_img_size=False,
        dynamic_img_pad=False,
        act_layer=nn.GELU,
        embed_layer=PatchEmbed,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
        class_token=True,
        block_fn=Block,
        num_classes=1000,
    ):
        super(VisionTransformer, self).__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm

        self.global_pool = global_pool
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.dynamic_img_size = dynamic_img_size
        self.dynamic_img_pad = dynamic_img_pad

        embed_args = {}
        if dynamic_img_size:
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        elif dynamic_img_pad:
            embed_args.update(dict(output_fmt="NHWC"))

        self.patch_embed = embed_layer(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            bias=not pre_norm,
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(initializer(TruncatedNormal(0.02), (1, 1, embed_dim))) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = Parameter(initializer(TruncatedNormal(0.02), (1, embed_len, embed_dim)))
        self.pos_drop = nn.Dropout(p=pos_drop_rate, dtype=ms.float32)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()

        self.norm_pre = norm_layer((embed_dim,)) if pre_norm else nn.Identity()
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.CellList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    attn_drop=attn_drop_rate,
                    proj_drop=proj_drop_rate,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    init_values=init_values,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer((embed_dim,)) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer((embed_dim,)) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(p=drop_rate, dtype=ms.float32)
        self.head = nn.Dense(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init:
            self._init_weights()

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(TruncatedNormal(0.02), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                cell.weight.set_data(initializer(HeUniform(), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def _pos_embed(self, x):
        if self.dynamic_img_size or self.dynamic_img_pad:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = ops.reshape(x, (B, -1, C))
        else:
            pos_embed = self.pos_embed

        if self.no_embed_class:
            x = x + pos_embed
            if self.cls_token is not None:
                cls_tokens = ops.broadcast_to(self.cls_token, (x.shape[0], -1, -1))
                cls_tokens = cls_tokens.astype(x.dtype)
                x = ops.concat((cls_tokens, x), axis=1)
        else:
            if self.cls_token is not None:
                cls_tokens = ops.broadcast_to(self.cls_token, (x.shape[0], -1, -1))
                cls_tokens = cls_tokens.astype(x.dtype)
                x = ops.concat((cls_tokens, x), axis=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def construct_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def construct_head(self, x):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens :].mean(axis=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x

    def construct(self, x):
        x = self.construct_features(x)
        x = self.construct_head(x)
        return x


class VIT_model(mindspore.nn.Cell):
    def __init__(self, args):
        super(VIT_model, self).__init__()
        self.model = VisionTransformer(
            image_size=32,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_classes=args.num_classes,
        )

    def construct(self, x):
        return self.model(x)
