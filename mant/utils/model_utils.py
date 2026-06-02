import torch
import torch.nn as nn
import accelerate
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.falcon.modeling_falcon import FalconForCausalLM
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, OPTConfig, LlamaConfig

from mant.models.llama_mant import LlamaForCausalLM_mant
from mant.models.opt_mant import OPTForCausalLM_mant

import os

def load_quantized_model(model_path, args, quant_config):
    if os.path.exists(os.path.join(model_path, 'config.json')) and args.quant_mode == 'mant':
        print(f"Loading quantized model from {model_path} ...")
        enc = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        config.update({
            "a_bit": args.a_bit, 
            "w_bit": args.w_bit, 
            "k_bit": args.k_bit, 
            "v_bit": args.v_bit, 
            "group_size": args.q_group_size, 
            "quant_kv": quant_config['quant_kv']
        })
        kwargs = {"device_map": "balanced", "torch_dtype": torch.float16, "attn_implementation": "eager"}

        model_cls = AutoModelForCausalLM
        if quant_config['quant_kv']:
            if isinstance(config, OPTConfig):
                model_cls = OPTForCausalLM_mant
            elif isinstance(config, LlamaConfig):
                model_cls = LlamaForCausalLM_mant
            else:
                model_type = getattr(config, "model_type", type(config).__name__)
                raise NotImplementedError(
                    f"KV cache quantization is not supported for model_type={model_type}. "
                    "Use k16v16 for Qwen3 or add a Qwen3-specific MANT attention wrapper."
                )
        
        model = model_cls.from_pretrained(model_path, config=config, **kwargs)
        return model, enc
    return None, None

def dump_quantized_model(model, enc, dump_path):
    print(f"Dumping quantized model to {dump_path} ...")
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    model.save_pretrained(dump_path)
    enc.save_pretrained(dump_path)

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)
        
def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        layers = model.model.decoder.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    elif hasattr(model, "bert") and hasattr(model.bert, "encoder") and hasattr(model.bert.encoder, "layer"):
        layers = model.bert.encoder.layer
    else:
        raise NotImplementedError(type(model))
    return layers
