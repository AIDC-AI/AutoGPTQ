# Copyright (c) 2024 AIDC-AI

from logging import getLogger

import torch

from ._base  import BaseGPTQForCausalLM
from ._const import SUPPORTED_MODELS
from ..utils.import_utils import compare_transformers_version

if compare_transformers_version("v4.28.0", op="ge"):
    from ..nn_modules.fused_llama_attn import FusedLlamaAttentionForQuantizedModel
    from ..nn_modules.fused_llama_mlp import FusedLlamaMLPForQuantizedModel
else:
    FusedLlamaAttentionForQuantizedModel = None
    FusedLlamaMLPForQuantizedModel = None

logger = getLogger(__name__)


class OvisGPTQForCausalLM(BaseGPTQForCausalLM):
    # layer_type = "Gemma2DecoderLayer"
    layers_block_name = "llm.model.layers"
    outside_layer_modules = ["llm.model.embed_tokens", "llm.model.norm", "visual_tokenizer", "vte"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    # hack so one can prepare examples outside
    def _prepare_examples_for_quantization(self, examples, batch_size: int = 1):
        return examples

    def generate(self, inputs, **kwargs):
        """shortcut for model.generate"""
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            return self.model.generate(inputs, **kwargs)


class OvisGemma2GPTQForCausalLM(OvisGPTQForCausalLM):
    layer_type = "Gemma2DecoderLayer"



class OvisLlamaGPTQForCausalLM(OvisGPTQForCausalLM):
    layer_type = "LlamaDecoderLayer"

    fused_attn_module_type = FusedLlamaAttentionForQuantizedModel
    fused_mlp_module_type = FusedLlamaMLPForQuantizedModel




SUPPORTED_MODELS.append("ovis")
__all__ = ["OvisGemma2GPTQForCausalLM", "OvisLlamaGPTQForCausalLM"]
