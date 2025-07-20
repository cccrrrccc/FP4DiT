# Copied and modified from Diffusers v0.19.0
import torch
from diffusers.models.attention import Attention
from typing import Optional


# NOTE Updating this to match AttnProcessor2_0 for Diffusers 0.29.2
class QuantAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        # Keeping this code from 1_0 since the 2_0 is not friendly to the activation quantization.
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # NOTE root of the problem is the separate act quantizers, 3 here, the last one is act_quantizer_w
        # TODO investigate why they change stuff from fp16 to fp32.
        if attn.use_act_quant:
            query = attn.act_quantizer_q(query).to(original_dtype)
            key = attn.act_quantizer_k(key).to(original_dtype)
            value = attn.act_quantizer_v(value).to(original_dtype)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # NOTE: This was missing from Ruichen code. This is the output of the softmax, and should be quantized prior to matmul with V matrix.
        # We might need to further play with this and reshape. See class QuantSMVMatMul(BaseQuantBlock) in quant_block.py
        # Shape of attention_prob is (B*H)xPxP where P is patches. 
        # high n_bits will cause overflow in FP quantization
        if attn.use_act_quant and attn.act_quantizer_w.n_bits <= 8:
            attention_probs = attn.act_quantizer_w(attention_probs).to(original_dtype)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states