# Copyright 2022 Kwangyoun Kim (ASAPP inc.)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""E-Branchformer encoder definition.
Reference:
    Kwangyoun Kim, Felix Wu, Yifan Peng, Jing Pan,
    Prashant Sridhar, Kyu J. Han, Shinji Watanabe,
    "E-Branchformer: Branchformer with Enhanced merging
    for speech recognition," in SLT 2022.
"""

import logging
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from typeguard import check_argument_types

from branchformer_module.ctc import CTC
from branchformer_module.encoder.abs_encoder import AbsEncoder
from branchformer_module.layers.cgmlp import ConvolutionalGatingMLP
from branchformer_module.layers.fastformer import FastSelfAttention
from branchformer_module.nets_utils import get_activation, make_pad_mask
from branchformer_module.transformer.attention import (  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from branchformer_module.transformer.embedding import (  # noqa: H301
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from branchformer_module.transformer.layer_norm import LayerNorm
from branchformer_module.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from branchformer_module.transformer.repeat import repeat
from branchformer_module.transformer.subsampling import (
    Conv1dSubsampling2,
    Conv1dSubsampling3,
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)


class StatisticPooling(nn.Module):
    def __init__(self):
        super(StatisticPooling, self).__init__()

    def forward(self, x):
        # Assuming x has shape (batch_size, channels, height, width)

        # Mean, Variance, Skewness, Kurtosis
        mean = torch.mean(x, dim=-1, keepdim=True)
        min = torch.min(x, dim=-1, keepdim=True).values
        max = torch.max(x, dim=-1, keepdim=True).values
        variance = torch.var(x, dim=-1, keepdim=True)
        # skewness = torch.mean((x - mean) ** 3, dim=-1, keepdim=True) / (variance ** 1.5 + 1e-6)
        # kurtosis = torch.mean((x - mean) ** 4, dim=-1, keepdim=True) / (variance ** 2 + 1e-6) - 3

        # Quantiles
        # q25 = torch.quantile(x, 0.25, dim=-1, keepdim=True)
        # q75 = torch.quantile(x, 0.75, dim=-1, keepdim=True)
        # iqr = q75 - q25

        # Concatenate all statistics
        # print(">>>> mean: ", mean.size(), min.size(), max.size(), variance.size(), skewness.size(), kurtosis.size(), q25.size(), q75.size(), iqr.size())
        x = torch.cat((mean, min, max, variance), dim=1)  # , skewness, kurtosis, q25, q75, iqr), dim=1)

        return x


class EBranchformerEncoderLayer(torch.nn.Module):
    """E-Branchformer encoder layer module.

    Args:
        size (int): model dimension
        attn: standard self-attention or efficient attention
        cgmlp: ConvolutionalGatingMLP
        feed_forward: feed-forward module, optional
        feed_forward: macaron-style feed-forward module, optional
        dropout_rate (float): dropout probability
        merge_conv_kernel (int): kernel size of the depth-wise conv in merge module
    """

    def __init__(
            self,
            size: int,
            attn: torch.nn.Module,
            cgmlp: torch.nn.Module,
            feed_forward: Optional[torch.nn.Module],
            feed_forward_macaron: Optional[torch.nn.Module],
            dropout_rate: float,
            merge_conv_kernel: int = 3,
    ):
        super().__init__()

        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp

        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.ff_scale = 1.0
        if self.feed_forward is not None:
            self.norm_ff = LayerNorm(size)
        if self.feed_forward_macaron is not None:
            self.ff_scale = 0.5
            self.norm_ff_macaron = LayerNorm(size)

        self.norm_mha = LayerNorm(size)  # for the MHA module
        self.norm_mlp = LayerNorm(size)  # for the MLP module
        self.norm_final = LayerNorm(size)  # for the final output of the block

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.depthwise_conv_fusion = torch.nn.Conv1d(
            size + size,
            size + size,
            kernel_size=merge_conv_kernel,
            stride=1,
            padding=(merge_conv_kernel - 1) // 2,
            groups=size + size,
            bias=True,
        )
        self.merge_proj = torch.nn.Linear(size + size, size)

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """

        if cache is not None:
            raise NotImplementedError("cache is not None, which is not tested")

        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        if self.feed_forward_macaron is not None:
            residual = x
            x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))

        # Two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        x1 = self.norm_mha(x1)

        if isinstance(self.attn, FastSelfAttention):
            x_att = self.attn(x1, mask)
        else:
            if pos_emb is not None:
                x_att = self.attn(x1, x1, x1, pos_emb, mask)
            else:
                x_att = self.attn(x1, x1, x1, mask)

        x1 = self.dropout(x_att)

        # Branch 2: convolutional gating mlp
        x2 = self.norm_mlp(x2)

        if pos_emb is not None:
            x2 = (x2, pos_emb)
        x2 = self.cgmlp(x2, mask)
        if isinstance(x2, tuple):
            x2 = x2[0]

        x2 = self.dropout(x2)

        # Merge two branches
        x_concat = torch.cat([x1, x2], dim=-1)
        x_tmp = x_concat.transpose(1, 2)
        x_tmp = self.depthwise_conv_fusion(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)
        x = x + self.dropout(self.merge_proj(x_concat + x_tmp))

        # self.feed_forwar=None
        if self.feed_forward is not None:
            # feed forward module
            residual = x
            x = self.norm_ff(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        x = self.norm_final(x)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask


class EBranchformer(AbsEncoder):
    """E-Branchformer encoder module."""

    def __init__(
            self,
            input_dim: int,  #
            output_dim: int,  #
            encoder_dim: int = 256,  #
            attention_heads: int = 4,
            attention_layer_type: str = "rel_selfattn",
            pos_enc_layer_type: str = "rel_pos",
            rel_pos_type: str = "latest",
            cgmlp_linear_units: int = 2048,  #
            cgmlp_conv_kernel: int = 31,
            use_linear_after_conv: bool = False,
            gate_activation: str = "identity",
            num_blocks: int = 12,  #
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
            input_layer: Optional[str] = "conv2d",
            zero_triu: bool = False,
            padding_idx: int = -1,
            layer_drop_rate: float = 0.0,
            max_pos_emb_len: int = 5000,
            use_ffn: bool = False,
            macaron_ffn: bool = False,
            ffn_activation_type: str = "swish",
            linear_units: int = 2048,  #
            positionwise_layer_type: str = "linear",
            merge_conv_kernel: int = 3,
            interctc_layer_idx=None,
            interctc_use_conditioning: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        input_size = input_dim
        self._encoder_dim = encoder_dim

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if attention_layer_type == "rel_selfattn":
                attention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert attention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert attention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert attention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_dim, encoder_dim),
                torch.nn.LayerNorm(encoder_dim),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv1d2":
            self.embed = Conv1dSubsampling2(
                input_dim,
                encoder_dim,
                dropout_rate,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv1d3":
            self.embed = Conv1dSubsampling3(
                input_dim,
                encoder_dim,
                dropout_rate,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_dim,
                encoder_dim,
                dropout_rate,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(
                input_dim,
                encoder_dim,
                dropout_rate,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_dim,
                encoder_dim,
                dropout_rate,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_dim,
                encoder_dim,
                dropout_rate,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_dim,
                encoder_dim,
                dropout_rate,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_dim, encoder_dim, padding_idx=padding_idx),
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer is None:
            if input_dim == encoder_dim:
                self.embed = torch.nn.Sequential(
                    pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len)
                )
            else:
                self.embed = torch.nn.Linear(input_dim, encoder_dim)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        activation = get_activation(ffn_activation_type)
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                encoder_dim,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type is None:
            logging.warning("no macaron ffn")
        else:
            raise ValueError("Support only linear.")

        if attention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                encoder_dim,
                attention_dropout_rate,
            )
        elif attention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                encoder_dim,
                attention_dropout_rate,
            )
            logging.warning(
                "Using legacy_rel_selfattn and it will be deprecated in the future."
            )
        elif attention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                encoder_dim,
                attention_dropout_rate,
                zero_triu,
            )
        elif attention_layer_type == "fast_selfattn":
            assert pos_enc_layer_type in ["abs_pos", "scaled_abs_pos"]
            encoder_selfattn_layer = FastSelfAttention
            encoder_selfattn_layer_args = (
                encoder_dim,
                attention_heads,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + attention_layer_type)

        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (
            encoder_dim,
            cgmlp_linear_units,
            cgmlp_conv_kernel,
            dropout_rate,
            use_linear_after_conv,
            gate_activation,
        )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EBranchformerEncoderLayer(
                encoder_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                cgmlp_layer(*cgmlp_layer_args),
                positionwise_layer(*positionwise_layer_args) if use_ffn else None,
                positionwise_layer(*positionwise_layer_args)
                if use_ffn and macaron_ffn
                else None,
                dropout_rate,
                merge_conv_kernel,
            ),
            layer_drop_rate,
        )
        self.after_norm = LayerNorm(encoder_dim)

        if interctc_layer_idx is None:
            interctc_layer_idx = []
        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None

        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.statpool = StatisticPooling()
        self.fc = nn.Linear(encoder_dim, output_dim, bias=False)
        self.fc_pl = nn.Linear(2 * encoder_dim, 3)

    def encoder_dim(self) -> int:
        return self._encoder_dim

    def forward(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor,
            prev_states: torch.Tensor = None,
            ctc: CTC = None,
            max_layer: int = None,
    ):  # -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_dim).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
            ctc (CTC): Intermediate CTC module.
            max_layer (int): Layer depth below which InterCTC is applied.
        Returns:
            torch.Tensor: Output tensor (#batch, L, encoder_dim).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.
        """

        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        # print(">>>> xs_pad0: ", len(xs_pad), xs_pad[0].shape)
        if (
                isinstance(self.embed, Conv2dSubsampling)
                or isinstance(self.embed, Conv1dSubsampling2)
                or isinstance(self.embed, Conv1dSubsampling3)
                or isinstance(self.embed, Conv2dSubsampling1)
                or isinstance(self.embed, Conv2dSubsampling2)
                or isinstance(self.embed, Conv2dSubsampling6)
                or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)
        # print(">>>> xs_pad: ", xs_pad[0].shape)

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            if max_layer is not None and 0 <= max_layer < len(self.encoders):
                for layer_idx, encoder_layer in enumerate(self.encoders):
                    xs_pad, masks = encoder_layer(xs_pad, masks)
                    if layer_idx >= max_layer:
                        break
            else:
                xs_pad, masks = self.encoders(xs_pad, masks)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad

                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            xs_pad = list(xs_pad)
                            xs_pad[0] = xs_pad[0] + self.conditioning_layer(ctc_out)
                            xs_pad = tuple(xs_pad)
                        else:
                            xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]

        xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)
        # if len(intermediate_outs) > 0:
        #     return (xs_pad, intermediate_outs), olens, None
        # return xs_pad, olens, None

        encoder_outputs = self.maxpool(xs_pad.transpose(-1, -2)).squeeze(dim=-1)
        pred = self.fc(encoder_outputs)
        # _encoder_outputs = encoder_outputs.view(-1, 2 * self.encoder_dim())
        # pred_pl = self.fc_pl(_encoder_outputs)
        # return pred, pred_pl
        return pred


class EBranchformerEncoderwithAdapterLayer(torch.nn.Module):
    """E-Branchformer encoder layer module.

    Args:
        size (int): model dimension
        attn: standard self-attention or efficient attention
        cgmlp: ConvolutionalGatingMLP
        feed_forward: feed-forward module, optional
        feed_forward: macaron-style feed-forward module, optional
        dropout_rate (float): dropout probability
        merge_conv_kernel (int): kernel size of the depth-wise conv in merge module
    """

    def __init__(
            self,
            size: int,
            attn: torch.nn.Module,
            cgmlp: torch.nn.Module,
            feed_forward: Optional[torch.nn.Module],
            feed_forward_macaron: Optional[torch.nn.Module],
            dropout_rate: float,
            merge_conv_kernel: int = 3,
            adapter_dim=int,
    ):
        super().__init__()

        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp

        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.ff_scale = 1.0
        if self.feed_forward is not None:
            self.norm_ff = LayerNorm(size)
        if self.feed_forward_macaron is not None:
            self.ff_scale = 0.5
            self.norm_ff_macaron = LayerNorm(size)

        self.norm_mha = LayerNorm(size)  # for the MHA module
        self.norm_mlp = LayerNorm(size)  # for the MLP module
        self.norm_final = LayerNorm(size)  # for the final output of the block

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.depthwise_conv_fusion = torch.nn.Conv1d(
            size + size,
            size + size,
            kernel_size=merge_conv_kernel,
            stride=1,
            padding=(merge_conv_kernel - 1) // 2,
            groups=size + size,
            bias=True,
        )
        self.merge_proj = torch.nn.Linear(size + size, size)

        self.adapter = torch.nn.Sequential(torch.nn.Linear(size, adapter_dim),
                                           torch.nn.ReLU(), torch.nn.Linear(adapter_dim, size))

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """

        if cache is not None:
            raise NotImplementedError("cache is not None, which is not tested")

        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        if self.feed_forward_macaron is not None:
            residual = x
            x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))

        # Two branches
        x1 = x
        x2 = x

        # Branch 1: multi-headed attention module
        x1 = self.norm_mha(x1)

        if isinstance(self.attn, FastSelfAttention):
            x_att = self.attn(x1, mask)
        else:
            if pos_emb is not None:
                x_att = self.attn(x1, x1, x1, pos_emb, mask)
            else:
                x_att = self.attn(x1, x1, x1, mask)

        x1 = self.dropout(x_att)

        # Branch 2: convolutional gating mlp
        x2 = self.norm_mlp(x2)

        if pos_emb is not None:
            x2 = (x2, pos_emb)
        x2 = self.cgmlp(x2, mask)
        if isinstance(x2, tuple):
            x2 = x2[0]

        x2 = self.dropout(x2)

        # Merge two branches
        x_concat = torch.cat([x1, x2], dim=-1)
        x_tmp = x_concat.transpose(1, 2)
        x_tmp = self.depthwise_conv_fusion(x_tmp)
        x_tmp = x_tmp.transpose(1, 2)
        x = x + self.dropout(self.merge_proj(x_concat + x_tmp))

        # self.feed_forwar=None
        if self.feed_forward is not None:
            # feed forward module
            residual = x
            x = self.norm_ff(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        #  adapter2
        y = self.adapter(x)
        x = x + y

        x = self.norm_final(x)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask


class EBranchformerwithAdapter(AbsEncoder):
    """E-Branchformer encoder module."""

    def __init__(
            self,
            input_dim: int,  #
            output_dim: int,  #
            encoder_dim: int = 256,  #
            attention_heads: int = 4,
            attention_layer_type: str = "rel_selfattn",
            pos_enc_layer_type: str = "rel_pos",
            rel_pos_type: str = "latest",
            cgmlp_linear_units: int = 2048,  #
            cgmlp_conv_kernel: int = 31,
            use_linear_after_conv: bool = False,
            gate_activation: str = "identity",
            num_blocks: int = 12,  #
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
            input_layer: Optional[str] = "conv2d",
            zero_triu: bool = False,
            padding_idx: int = -1,
            layer_drop_rate: float = 0.0,
            max_pos_emb_len: int = 5000,
            use_ffn: bool = False,
            macaron_ffn: bool = False,
            ffn_activation_type: str = "swish",
            linear_units: int = 2048,  #
            positionwise_layer_type: str = "linear",
            merge_conv_kernel: int = 3,
            interctc_layer_idx=None,
            interctc_use_conditioning: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        input_size = input_dim
        self._encoder_dim = encoder_dim
        self.adapter_dim = [2, 4, 8, 12, 16, 32]

        # self.adapter_dim = [32, 16, 12, 8, 4, 2]
        # self.adapter_dim = [8, 8, 8, 8, 8, 8]

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if attention_layer_type == "rel_selfattn":
                attention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert attention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert attention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert attention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_dim, encoder_dim),
                torch.nn.LayerNorm(encoder_dim),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv1d2":
            self.embed = Conv1dSubsampling2(
                input_dim,
                encoder_dim,
                dropout_rate,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv1d3":
            self.embed = Conv1dSubsampling3(
                input_dim,
                encoder_dim,
                dropout_rate,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_dim,
                encoder_dim,
                dropout_rate,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(
                input_dim,
                encoder_dim,
                dropout_rate,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_dim,
                encoder_dim,
                dropout_rate,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_dim,
                encoder_dim,
                dropout_rate,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_dim,
                encoder_dim,
                dropout_rate,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_dim, encoder_dim, padding_idx=padding_idx),
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer is None:
            if input_dim == encoder_dim:
                self.embed = torch.nn.Sequential(
                    pos_enc_class(encoder_dim, positional_dropout_rate, max_pos_emb_len)
                )
            else:
                self.embed = torch.nn.Linear(input_dim, encoder_dim)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        activation = get_activation(ffn_activation_type)
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                encoder_dim,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type is None:
            logging.warning("no macaron ffn")
        else:
            raise ValueError("Support only linear.")

        if attention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                encoder_dim,
                attention_dropout_rate,
            )
        elif attention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                encoder_dim,
                attention_dropout_rate,
            )
            logging.warning(
                "Using legacy_rel_selfattn and it will be deprecated in the future."
            )
        elif attention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                encoder_dim,
                attention_dropout_rate,
                zero_triu,
            )
        elif attention_layer_type == "fast_selfattn":
            assert pos_enc_layer_type in ["abs_pos", "scaled_abs_pos"]
            encoder_selfattn_layer = FastSelfAttention
            encoder_selfattn_layer_args = (
                encoder_dim,
                attention_heads,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + attention_layer_type)

        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (
            encoder_dim,
            cgmlp_linear_units,
            cgmlp_conv_kernel,
            dropout_rate,
            use_linear_after_conv,
            gate_activation,
        )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EBranchformerEncoderwithAdapterLayer(
                encoder_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                cgmlp_layer(*cgmlp_layer_args),
                positionwise_layer(*positionwise_layer_args) if use_ffn else None,
                positionwise_layer(*positionwise_layer_args)
                if use_ffn and macaron_ffn
                else None,
                dropout_rate,
                merge_conv_kernel,
                self.adapter_dim[lnum],
            ),
            layer_drop_rate,
        )
        self.after_norm = LayerNorm(encoder_dim)

        if interctc_layer_idx is None:
            interctc_layer_idx = []
        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None

        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.statpool = StatisticPooling()
        self.fc = nn.Linear(encoder_dim, output_dim, bias=False)
        # self.fc_pl = nn.Linear(2*encoder_dim, 3)

    def encoder_dim(self) -> int:
        return self._encoder_dim

    def forward(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor,
            prev_states: torch.Tensor = None,
            ctc: CTC = None,
            max_layer: int = None,
    ):  # -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_dim).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
            ctc (CTC): Intermediate CTC module.
            max_layer (int): Layer depth below which InterCTC is applied.
        Returns:
            torch.Tensor: Output tensor (#batch, L, encoder_dim).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.
        """

        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        # print(">>>> xs_pad0: ", len(xs_pad), xs_pad[0].shape)
        if (
                isinstance(self.embed, Conv2dSubsampling)
                or isinstance(self.embed, Conv1dSubsampling2)
                or isinstance(self.embed, Conv1dSubsampling3)
                or isinstance(self.embed, Conv2dSubsampling1)
                or isinstance(self.embed, Conv2dSubsampling2)
                or isinstance(self.embed, Conv2dSubsampling6)
                or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        elif self.embed is not None:
            xs_pad = self.embed(xs_pad)
        # print(">>>> xs_pad: ", xs_pad[0].shape)

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            if max_layer is not None and 0 <= max_layer < len(self.encoders):  # max_layer=None
                for layer_idx, encoder_layer in enumerate(self.encoders):
                    xs_pad, masks = encoder_layer(xs_pad, masks)
                    if layer_idx >= max_layer:
                        break
            else:
                xs_pad, masks = self.encoders(xs_pad, masks)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad

                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            xs_pad = list(xs_pad)
                            xs_pad[0] = xs_pad[0] + self.conditioning_layer(ctc_out)
                            xs_pad = tuple(xs_pad)
                        else:
                            xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]

        xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)
        # if len(intermediate_outs) > 0:
        #     return (xs_pad, intermediate_outs), olens, None
        # return xs_pad, olens, None

        encoder_outputs = self.maxpool(xs_pad.transpose(-1, -2)).squeeze(dim=-1)
        pred = self.fc(encoder_outputs)
        # _encoder_outputs = encoder_outputs.view(-1, 2 * self.encoder_dim())
        # pred_pl = self.fc_pl(_encoder_outputs)
        # return pred, pred_pl
        return pred
