from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
import torch
import argparse
import os
import json

from mant.quantize.quantizer import (
    pseudo_quant_output_mse,
    make_quant_linear,
    pseudo_quantize_model_int,
    pseudo_quantize_model_mixed_mant,
)

import datetime
import re
import tqdm
from torch import nn

from mant.models.opt_mant import OPTForCausalLM_mant
from mant.models.llama_mant import LlamaForCausalLM_mant

from mant.utils.model_utils import load_quantized_model, dump_quantized_model

from transformers import OPTConfig, LlamaConfig

def print_time(print_str):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{timestamp} - {print_str}')

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='path of the hf model')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=0)
parser.add_argument(
    '--limit_samples',
    type=int,
    default=None,
    help='Limit evaluation to the first N samples. Useful for fast precision search.',
)

# quantization config
parser.add_argument('--quant_bit_width', type=str, default='w16a16k16v16')
parser.add_argument('--w_bit', type=int, default=None)
parser.add_argument('--a_bit', type=int, default=16)
parser.add_argument('--k_bit', type=int, default=16)
parser.add_argument('--v_bit', type=int, default=16)
parser.add_argument('--q_group_size', type=int, default=-1)

parser.add_argument('--quant_mode', type=str, default="mant")
parser.add_argument('--quant_dtype', type=str, default="int")
parser.add_argument('--w_low', type=int, default=75)
parser.add_argument('--w_high', type=int, default=150)
parser.add_argument(
    '--layer_bit_config',
    type=str,
    default=None,
    help='JSON file or comma-separated list with per-linear bit widths. Overrides --quant_bit_width w bits.',
)
parser.add_argument(
    '--layer_a_bits',
    type=str,
    choices=['global', 'follow'],
    default='global',
    help='Use global a_bit from --quant_bit_width, or follow --layer_bit_config for activation bits.',
)

parser.add_argument('--a_stride', type=int, default=10)

args = parser.parse_args()

if args.limit_samples is not None and args.limit_samples <= 0:
    raise ValueError("--limit_samples must be a positive integer")

quant_config = {
    "quant_dtype": args.quant_dtype,  # specify the data type
    "q_group_size": args.q_group_size,  # whether to use group quantization
    "w_low": args.w_low,
    "w_high": args.w_high,
    "quant_method": args.quant_mode,
    "quant_kv": False,
}

def extract_bitwidths(quantization_string):
    w_bits = int(re.search(r'w(-?\d+)', quantization_string).group(1))
    a_bits = int(re.search(r'a(-?\d+)', quantization_string).group(1))
    k_bits = int(re.search(r'k(-?\d+)', quantization_string).group(1))
    v_bits = int(re.search(r'v(-?\d+)', quantization_string).group(1))
    return w_bits, a_bits, k_bits, v_bits


def load_layer_bits(path_or_list):
    if path_or_list is None:
        return None
    if os.path.exists(path_or_list):
        with open(path_or_list, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            for key in ('w_bits', 'bits', 'pattern'):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"{path_or_list} must contain a JSON list or an object with w_bits/bits/pattern")
        return [int(x) for x in payload]
    return [int(x.strip()) for x in path_or_list.split(',') if x.strip()]


def has_low_precision(bit_spec):
    if bit_spec is None:
        return False
    if isinstance(bit_spec, int):
        return bit_spec != -1 and bit_spec < 16
    return any(int(bit) != -1 and int(bit) < 16 for bit in bit_spec)

args.w_bit, args.a_bit, args.k_bit, args.v_bit = extract_bitwidths(args.quant_bit_width)
layer_w_bits = load_layer_bits(args.layer_bit_config)
layer_a_bits = layer_w_bits if args.layer_a_bits == 'follow' else args.a_bit
if args.k_bit < 16 or args.v_bit < 16:
    quant_config['quant_kv'] = True

print("\nQuantization configuration:", quant_config)
if layer_w_bits is not None:
    print(f"Layer bit configuration: {len(layer_w_bits)} entries, activation_bits={args.layer_a_bits}")

# Build model and tokenizer
def build_model_and_enc(model_path):
    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # LOAD model for DEBUGGING purpose
    # model, enc = load_quantized_model('./quant_cache/', args, quant_config)
    # if model is not None:
    #     return model, enc

    # All hf model
    config = AutoConfig.from_pretrained(model_path)
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # Eager mode for our hacked attention implementation
    kwargs = {"device_map": "balanced", "torch_dtype": torch.float16, "attn_implementation": "eager"}

    # To modify the attention layer
    if quant_config['quant_kv']:
        if args.quant_mode != 'mant':
            raise ValueError("KV cache quantization is currently only supported for 'mant' mode.")
        config.a_bit = args.a_bit
        config.w_bit = args.w_bit
        config.k_bit = args.k_bit
        config.v_bit = args.v_bit
        config.group_size = args.q_group_size
        config.quant_kv = quant_config['quant_kv']
        if isinstance(config, OPTConfig):
            model = OPTForCausalLM_mant.from_pretrained(
                model_path, config=config, **kwargs)
        elif isinstance(config, LlamaConfig):
            model = LlamaForCausalLM_mant.from_pretrained(
                model_path, config=config, **kwargs)
        else:
            raise NotImplementedError('not support yet')
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, **kwargs)

    # Weight quantization
    weight_bit_spec = layer_w_bits if layer_w_bits is not None else args.w_bit
    activation_bit_spec = layer_a_bits

    if has_low_precision(weight_bit_spec):
        quant_mode = quant_config['quant_method']

        if quant_mode in ['ant', 'olive']:
            make_quant_linear(
                model, weight_bit_spec, activation_bit_spec, quant_config=quant_config
            )
        elif quant_mode =='mant':
            if layer_w_bits is not None:
                pseudo_quantize_model_mixed_mant(
                    model, enc, w_bits=layer_w_bits, quant_config=quant_config,
                    n_samples=512, seqlen=512, a_stride=args.a_stride
                )
            elif args.w_bit == 8:
                pseudo_quantize_model_int(model, w_bit=args.w_bit, q_group_size=args.q_group_size)
            elif args.w_bit == 4:
                pseudo_quant_output_mse(
                    model, enc, w_bit=args.w_bit, quant_config=quant_config, n_samples=512, seqlen=512, a_stride=args.a_stride
                )
            else:
                print('not supported yet')
                exit(0)
            make_quant_linear(
                model, weight_bit_spec, activation_bit_spec, quant_config=quant_config
            )
        elif quant_mode == 'int':
            pseudo_quantize_model_int(model, w_bit=weight_bit_spec, q_group_size=args.q_group_size)
            make_quant_linear(
                model, weight_bit_spec, activation_bit_spec, quant_config=quant_config
            )
        else:
            raise NotImplementedError(f"{quant_mode} not supported yet!")
        
        # dump model for DEBUGGING purpose
        # dump_quantized_model(model, enc, './quant_cache/')
        # exit(0)    
    
    return model, enc


def main():
    print("\nargs:", args, "\n")
    # A hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path)
    lm_eval_model = HFLM(pretrained=model, batch_size=args.batch_size)
    
    if args.tasks is not None:
        if args.tasks in ['wikitext', 'c4', 'ptb']:
        # Adapted from https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
            from mant.utils.dataload_utils import get_loaders
            model.seqlen = 2048
            _, testenc = get_loaders(args.tasks, model=args.model_path, seqlen=model.seqlen)
            
            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // model.seqlen
            if args.limit_samples is not None:
                nsamples = min(nsamples, args.limit_samples)
            model = model.eval()
            nlls = []
            print_time('Start a task')
            for i in tqdm.tqdm(range(1), desc="Data Type Search..."):
                batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                    model.device
                )
                with torch.no_grad():
                    lm_logits_tmp = model(batch).logits

            for i in tqdm.tqdm(range(nsamples), desc="Task Evaluating..."):
                batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                    model.device
                )
                with torch.no_grad():
                    lm_logits = model(batch).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = testenc[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                ][:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * model.seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
            print_time('Task finish!')
            print(ppl.item())    
        else:
            # Do other evaluations
            print_time('Start a task')
            task_names = args.tasks.split(",")

            results = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=task_names,
                batch_size=args.batch_size,
                num_fewshot=args.num_fewshot,
                limit=args.limit_samples,
            )
            print_time('Task finish!')
            print(make_table(results))


if __name__ == '__main__':
    main()
