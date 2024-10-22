# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# https://www.apache.org/licenses/LICENSE-2.0

import torch
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm

from slicegpt.model_adapter import LayerAdapter, ModelAdapter
from typing import Dict
import torch.nn.functional as F

class SRSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask, decay=0.00002):
        ctx.save_for_backward(input)
        ctx.mask = mask
        ctx.decay = decay
        
        return input * (~mask)

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors

        return grad_output + ctx.decay * ctx.mask * weight, None
    
srste_mask = SRSTE.apply

class rot_mask_Linear(Linear):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ori = 0 # 0 for right/output, 1 for left/input
        self.Q = None
        self.mask = None
        
    def forward(self, input):
        W_ = self.weight.data
        b_ = None
        if self.bias is not None:
            b_ = self.bias.data    
        if self.Q is not None:
            if self.ori == 1: # 左乘，对应输入，不需要对 b 处理
                W_ = torch.matmul(W_, self.Q) # 实际上的矩阵要转置
            else: # 右乘，对应输出，需要对 b 处理
                W_ = torch.matmul(self.Q.T, W_)
                if b_ is not None:
                    b_ = torch.matmul(self.Q.T, b_)
        if self.mask is not None:
            W_ = srste_mask(W_, self.mask)
            
        return F.linear(input, W_, b_)

class CompressedLlamaDecoderLayer(LlamaDecoderLayer):
    """
    This class simulates the LlamaDecoderLayer class from transformers
    (https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L376)
    but with the addition of a shortcut_Q attribute. This attribute is used to rotate the residual tensors.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_shortcut_Q = None
        self.mlp_shortcut_Q = None
        self.embed_Q = None
        self.head_Q = None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        past_key_value: tuple[Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs,
    ) -> tuple:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if self.embed_Q is not None:
            hidden_states = torch.matmul(hidden_states, self.embed_Q)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        if self.attn_shortcut_Q is not None:
            if isinstance(self.attn_shortcut_Q, Linear):
                rotated_residual = self.attn_shortcut_Q(residual)
            else:
                rotated_residual = matmul(residual, self.attn_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.mlp_shortcut_Q is not None:
            if isinstance(self.mlp_shortcut_Q, Linear):
                rotated_residual = self.mlp_shortcut_Q(residual)
            else:
                rotated_residual = matmul(residual, self.mlp_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states
            
        if self.head_Q is not None:
            hidden_states = torch.matmul(hidden_states, self.head_Q)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaLayerAdapter(LayerAdapter):
    def __init__(self, layer: LlamaDecoderLayer) -> None:
        super().__init__()
        self._layer: LlamaDecoderLayer = layer

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def get_first_layernorm(self) -> Module:
        return self.layer.input_layernorm

    def get_second_layernorm(self) -> Module:
        return self.layer.post_attention_layernorm

    def get_attention_inputs(self) -> list[Linear]:
        return [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        return self.layer.self_attn.o_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.gate_proj, self.layer.mlp.up_proj]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.down_proj
    
    def get_attention_inputs_dict(self) -> Dict[str, rot_mask_Linear]:
        return {'self_attn.q_proj': self.layer.self_attn.q_proj, 
                'self_attn.k_proj': self.layer.self_attn.k_proj, 
                'self_attn.v_proj': self.layer.self_attn.v_proj}
        
    def get_attention_outputs_dict(self) -> Dict[str, rot_mask_Linear]:
        return {'self_attn.o_proj': self.layer.self_attn.o_proj}
    
    def get_mlp_inputs_dict(self) -> Dict[str, rot_mask_Linear]:
        return {'mlp.gate_proj': self.layer.mlp.gate_proj,
                'mlp.up_proj': self.layer.mlp.up_proj}
        
    def get_mlp_outputs_dict(self) -> Dict[str, rot_mask_Linear]:
        return {'mlp.down_proj': self.layer.mlp.down_proj}


class LlamaModelAdapter(ModelAdapter):
    def __init__(self, model: LlamaForCausalLM) -> None:
        super().__init__()
        self._model: LlamaForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return LlamaConfig

    @property
    def parallel_blocks(self) -> bool:
        return False

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return False

    @property
    def original_layer_type(self) -> type:
        return LlamaDecoderLayer
    
    @property
    def original_linear_type(self) -> type:
        return Linear

    @property
    def original_layer_norm_type(self) -> type:
        return LlamaRMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return LlamaLayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        return CompressedLlamaDecoderLayer
    
    @property
    def rotated_linear_type(self) -> type:
        return rot_mask_Linear

    @property
    def use_cache(self) -> bool:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self.model(input_ids=input_ids).logits

    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        compressed_layer = self.compressed_layer_type(self.config, layer_idx).to(self.config.torch_dtype)
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer
    
    def convert_linear_to_rotated(self, layer: Module, layer_idx: int | None=None) -> Module: 
        rotated_linear = self.rotated_linear_type(layer.in_features, layer.out_features, bias=(layer.bias is not None)).to(self.config.torch_dtype)
        rotated_linear.load_state_dict(layer.state_dict(), strict=True)
        return rotated_linear

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.model.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> Module:
        pre_head_layernorm = self.model.model.norm
        assert isinstance(pre_head_layernorm, self.original_layer_norm_type)
        return pre_head_layernorm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # Llama-2 and Llama-3 don't have a pad tokens by default
        tokenizer.pad_token = tokenizer.eos_token
        self.config.pad_token_id = tokenizer.pad_token_id

    @classmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not (model_name.startswith("meta-llama/Llama-2") or model_name.startswith("meta-llama/Meta-Llama-3")):
            return None

        model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only,
            _attn_implementation="sdpa",
        )
        model.config.torch_dtype = dtype

        return LlamaModelAdapter(model)

    @classmethod
    def _from_uninitialized(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not (model_name.startswith("meta-llama/Llama-2") or model_name.startswith("meta-llama/Meta-Llama-3")):
            return None

        class UninitializedLlamaForCausalLM(LlamaForCausalLM):
            def _init_weights(self, _) -> None:
                # Prevent weight initialization
                pass

        config = LlamaConfig.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )
        model = UninitializedLlamaForCausalLM(config)
        model = model.to(dtype=dtype)

        return LlamaModelAdapter(model)
