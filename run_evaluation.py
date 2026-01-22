from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
import torch
import argparse
import os

from mant.quantize.quantizer import pseudo_quant_output_mse, make_quant_linear,pseudo_quantize_model_int

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

parser.add_argument('--a_stride', type=int, default=10)

args = parser.parse_args()

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

args.w_bit, args.a_bit, args.k_bit, args.v_bit = extract_bitwidths(args.quant_bit_width)
if args.k_bit < 16 or args.v_bit < 16:
    quant_config['quant_kv'] = True

print("\nQuantization configuration:", quant_config)

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
    if args.w_bit and args.w_bit != -1 and args.w_bit < 16:
        quant_mode = quant_config['quant_method']

        if quant_mode in ['ant', 'olive']:
            make_quant_linear(
                model, args.w_bit, args.a_bit, quant_config=quant_config
            )
        elif quant_mode =='mant':
            if args.w_bit == 8:
                pseudo_quantize_model_int(model, w_bit=args.w_bit, q_group_size=args.q_group_size)
            elif args.w_bit == 4:
                pseudo_quant_output_mse(
                    model, enc, w_bit=args.w_bit, quant_config=quant_config, n_samples=512, seqlen=512, a_stride=args.a_stride
                )
            else:
                print('not supported yet')
                exit(0)
            make_quant_linear(
                model, args.w_bit, args.a_bit, quant_config=quant_config
            )
        elif quant_mode == 'int':
            pseudo_quantize_model_int(model, w_bit=args.w_bit, q_group_size=args.q_group_size)
            make_quant_linear(
                model, args.w_bit, args.a_bit, quant_config=quant_config
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
            )
            print_time('Task finish!')
            print(make_table(results))


if __name__ == '__main__':
    main()
