import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from ..utils.model_utils import set_op_by_name

import functools
from collections import defaultdict
import copy

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]


def _tensor_bit(bit_spec, layer_id, tensor_id, tensor_name, default_bit):
    if bit_spec is None:
        return default_bit
    if isinstance(bit_spec, int):
        return bit_spec
    if isinstance(bit_spec, (list, tuple)):
        if tensor_id >= len(bit_spec):
            raise ValueError(
                f"Bit pattern has {len(bit_spec)} entries, but tensor index {tensor_id} was requested"
            )
        return int(bit_spec[tensor_id])
    if isinstance(bit_spec, dict):
        full_name = f"{layer_id}.{tensor_name}"
        if full_name in bit_spec:
            return int(bit_spec[full_name])
        if str(layer_id) in bit_spec:
            layer_bits = bit_spec[str(layer_id)]
            if isinstance(layer_bits, dict):
                if tensor_name in layer_bits:
                    return int(layer_bits[tensor_name])
            elif isinstance(layer_bits, (list, tuple)):
                local_idx = int(tensor_name.split(".")[-1]) if tensor_name.isdigit() else None
                if local_idx is not None and local_idx < len(layer_bits):
                    return int(layer_bits[local_idx])
        if "default" in bit_spec:
            return int(bit_spec["default"])
        return default_bit
    raise TypeError(f"Unsupported bit specification type: {type(bit_spec)}")


def _normal_group_size(module, q_group_size):
    return module.weight.data.shape[1] if q_group_size == -1 else q_group_size

@torch.no_grad()
def pseudo_quantize_model_int(
    model,
    w_bit,
    q_group_size,
):
    from ..utils.model_utils import get_blocks, get_named_linears
    from .quant_func import pseudo_quantize_int

    layers = get_blocks(model)
    tensor_id = 0
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            cur_w_bit = _tensor_bit(w_bit, i, tensor_id, n, w_bit if isinstance(w_bit, int) else 8)
            tensor_id += 1
            if cur_w_bit >= 16 or cur_w_bit == -1:
                continue
            m.weight.data = pseudo_quantize_int(
                m.weight.data, n_bit=cur_w_bit, q_group_size=q_group_size
            )


@torch.no_grad()
def pseudo_quantize_model_mixed_mant(
    model,
    enc,
    w_bits,
    quant_config,
    n_samples=512,
    seqlen=512,
    calib_data="pileval",
    a_stride=10,
):
    from ..utils.calib_data import get_calib_dataset
    from ..utils.model_utils import get_blocks, get_named_linears
    from .quant_grid import generate_quant_grid
    from .quant_func import quantize_rows, pseudo_quantize_int
    from .qmodule_mant import encode_gen

    layers = get_blocks(model)
    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen)
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError

    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:
        pass
    layers[0] = layers[0].module
    inps = inps[0]

    layer_kwargs["use_cache"] = False
    for k in ["past_key_value", "past_key_values"]:
        layer_kwargs.pop(k, None)

    quant_grid_cache = {}
    global_tensor_id = 0

    for i in tqdm(range(len(layers)), desc="Mixed MANT pseudo weight quantization..."):
        layer = layers[i]
        named_linears = get_named_linears(layer)

        def cache_input_hook(m, x, y, name, feat_dict):
            feat_dict[name].append(x[0].detach().cpu())

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name, feat_dict=input_feat)))

        inps = inps.to(next(layer.parameters()).device)
        layer_kwargs_copy = copy.deepcopy(layer_kwargs)
        _ = layer(inps, **layer_kwargs)[0]

        for h in handles:
            h.remove()
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        for name, m in named_linears.items():
            cur_w_bit = _tensor_bit(w_bits, i, global_tensor_id, name, 4)
            global_tensor_id += 1
            if cur_w_bit >= 16 or cur_w_bit == -1:
                continue
            if cur_w_bit > 4:
                m.weight.data = pseudo_quantize_int(
                    m.weight.data, n_bit=cur_w_bit, q_group_size=quant_config["q_group_size"]
                )
                continue

            input_x = input_feat[name].reshape(-1, input_feat[name].shape[-1])
            group_size = _normal_group_size(m, quant_config["q_group_size"])
            group_num = m.weight.data.shape[1] // group_size

            if cur_w_bit not in quant_grid_cache:
                quant_grid_set = encode_gen(cur_w_bit, a_stride=a_stride)
                int_grid_set = generate_quant_grid(n_bit=cur_w_bit, signed=True, quant_dtype='int')
                mode_list = list(quant_grid_set.keys()) + ['int']
                quant_grid_set['int'] = int_grid_set['int']
                quant_grid_cache[cur_w_bit] = (mode_list, quant_grid_set)
            mode_list, quant_grid_set = quant_grid_cache[cur_w_bit]

            tensor_mse = torch.tensor(0.).to(m.weight.data.device)
            for group_id in range(group_num):
                x = input_x[:, group_id * group_size: (group_id + 1) * group_size].to(m.weight.data.device)
                org_group_w = m.weight.data[:, group_id * group_size: (group_id + 1) * group_size]
                org_group_output = torch.mm(x, org_group_w.T)
                deq_w = torch.zeros_like(org_group_w, dtype=torch.half).to(m.weight.data.device)
                min_mse = torch.full([1, m.weight.data.shape[0]], 10000.0).to(m.weight.data.device)

                for mode in mode_list:
                    quant_grid = quant_grid_set[mode]
                    w_group_deq = quantize_rows(org_group_w, quant_grid).half()
                    deq_group_output = torch.mm(x, w_group_deq.T)
                    mse = (deq_group_output - org_group_output).pow(2).mean(0, keepdim=True)
                    sig = (mse <= min_mse).to(torch.half)
                    mask = sig.repeat(group_size, 1).T
                    deq_w = torch.mul(deq_w, 1.0 - mask) + torch.mul(w_group_deq, mask)
                    min_mse = torch.where(mse <= min_mse, mse, min_mse)

                tensor_mse += min_mse.mean()
                m.weight.data[:, group_id * group_size: (group_id + 1) * group_size] = deq_w
            print(f"layer: {i}, {name}, bit_width: {cur_w_bit}, tensor_mse: {tensor_mse}")

        inps = layer(inps, **layer_kwargs_copy)[0]
        del layer_kwargs_copy
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

@torch.no_grad()
def pseudo_quant_output_mse(
    model, enc,
    w_bit, quant_config,
    n_samples=512, seqlen=512,
    # some configs for ablation study
    calib_data="pileval",
    a_stride=10,
    do_stats=False
):
    from ..utils.calib_data import get_calib_dataset
    from ..utils.model_utils import get_blocks, get_named_linears
    from .quant_grid import generate_quant_grid
    from .quant_func import quantize_rows
    from .qmodule_mant import encode_gen

    layers = get_blocks(model)

    # -------------------------------------------------------------------------
    # Build calibration batch (token ids) and catch the first block input + kwargs.
    # -------------------------------------------------------------------------
    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen)
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    # Catcher: intercept the input tensor and kwargs of block-0, then early-exit.
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # Patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    layers[0] = layers[0].module # restore original block
    inps = inps[0]  # block-0 input hidden states


    # Judge store KV cache or not
    # print("use_cache =", layer_kwargs.get("use_cache", None))

    # NOTE: Do not store KV cache during weight search to avoid GPU OOM
    layer_kwargs["use_cache"] = False
    for k in ["past_key_value", "past_key_values"]:
        layer_kwargs.pop(k, None)

    gc.collect()
    torch.cuda.empty_cache()
    overall_mse = torch.tensor(0.).to(next(layers[0].parameters()).device)

    quant_grid_set = encode_gen(w_bit, a_stride=a_stride)
    int_grid_set = generate_quant_grid(n_bit=w_bit, signed=True, quant_dtype='int')

    mode_list = []
    mode_list.extend(quant_grid_set.keys())
    mode_list.append('int')
    quant_grid_set['int'] = int_grid_set['int']    

    if do_stats:
        dtype_stats = {}
        tensor_stats = {}
        for mode in mode_list:
            dtype_stats[mode] = torch.tensor(0.)
            tensor_stats[mode] = torch.tensor(0.)

    # -------------------------------------------------------------------------
    # Layer-wise search: update weights block by block, and propagate updated outputs.
    # -------------------------------------------------------------------------
    for i in tqdm(range(len(layers)), desc="Pseudo weight quantization with output MSE..."):
        layer = layers[i]
        named_linears = get_named_linears(layer)

        # Hook: cache each Linear's input activation X (saved on CPU to reduce GPU residency).
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name,
                                  feat_dict=input_feat)))
        
        # Ensure current block runs on the right device (pipeline/model parallel safety).
        inps = inps.to(next(layer.parameters()).device) 
        # Snapshot kwargs for the propagation pass (may be mutated internally by forward).
        layer_kwargs_copy = copy.deepcopy(layer_kwargs)

        # =============================================================================
        # Step (1): Calibration Pass
        # Run forward with original FP16 weights to trigger hooks and cache input X.
        # =============================================================================
        _ = layer(inps, **layer_kwargs)[0] # hook-only forward

        for h in handles:
            h.remove()

        # Merge cached X for each Linear: [B, T, K] chunks -> one tensor.
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # =====================================================================
        # (2) Weight search + in-place update:
        # For each Linear weight group, pick best coefficient by output MSE.
        # =====================================================================
        for name, m in named_linears.items():

            input_x = input_feat[name] # 66 * 512 * K, reshape to 2D (M=33792 * K) before computation
            group_size = quant_config["q_group_size"]
            if group_size == -1:
                group_size = m.weight.data.shape[1]
            group_num = m.weight.data.shape[1] // group_size

            # input_x = input_x.to(m.weight.data.device)
            input_x = input_x.reshape(-1, input_x.shape[-1]) # [M, K]

            tensor_mse = torch.tensor(0.).to(m.weight.data.device)
            for group_id in range(0, group_num):
                x = input_x[ : , group_id * group_size: (group_id + 1) * group_size ].to(m.weight.data.device) # [M, g]

                org_group_w = m.weight.data[ : ,group_id * group_size: (group_id + 1) * group_size ]   # [N, g] 
                org_group_output = torch.mm(x, org_group_w.T)

                def weight_quant():
                    # Support output MSE search for MANT
                    deq_w = torch.zeros_like(org_group_w, dtype=torch.half).to(m.weight.data.device)
                    min_mse = torch.full([1, m.weight.data.shape[0]], 10000.0).to(m.weight.data.device)  # [1, N]

                    # For stats
                    if do_stats:
                        data_type_identify = torch.zeros_like(min_mse, dtype=torch.int32)
                        mapping_list = {}

                    # Search for the candidate coefficient a
                    for idx, mode in enumerate(mode_list):

                        quant_grid = quant_grid_set[mode]
                        w_group_deq = quantize_rows(org_group_w, quant_grid)
                        w_group_deq = w_group_deq.half()

                        deq_group_output = torch.mm(x, w_group_deq.T)
                        mse = (deq_group_output - org_group_output).pow(2).mean(0, keepdim=True)

                        sig = (mse <= min_mse).to(torch.half) # [1, N]
                        mask = sig.repeat(group_size, 1).T # [N, group_size]
                        org_mask = 1.0 - mask
                        deq_w = torch.mul(deq_w, org_mask) + torch.mul(w_group_deq, mask)
                        # deq_w = torch.mul(w_group_deq, mask)
                        # For stats
                        if do_stats:
                            mapping_list[mode] = idx
                            data_type_identify = torch.where(mse < min_mse, idx, data_type_identify)

                        # Update min MSE
                        min_mse = torch.where(mse <= min_mse, mse, min_mse)

                    if do_stats:
                        # Collect stats
                        for mode in mode_list:
                            dtype_stats[mode] = dtype_stats[mode].to(data_type_identify.device)
                            tensor_stats[mode] = tensor_stats[mode].to(data_type_identify.device)
                            dtype_stats[mode] = dtype_stats[mode] + torch.count_nonzero(data_type_identify.view(-1) == mapping_list[mode]) / 1e5
                            tensor_stats[mode] = tensor_stats[mode] + torch.count_nonzero(data_type_identify.view(-1) == mapping_list[mode]) 
                    return deq_w, min_mse

                deq_w, min_mse = weight_quant()
                tensor_mse += min_mse.mean()

                # Update weights in-place
                m.weight.data[ : ,group_id * group_size: (group_id + 1) * group_size ] = deq_w

            if do_stats:
                for mode in mode_list:
                    print(f"{mode} num: {tensor_stats[mode]}")    
                    tensor_stats[mode] = torch.tensor(0.)

            print(f"layer: {i}, {name}, tensor_mse: {tensor_mse}")
            overall_mse = overall_mse.to(tensor_mse.device)
            overall_mse += tensor_mse

        # =====================================================================
        # (3) Propagation pass (quantized weights):
        # Rerun this block with the same input `inps` and propagate its output.
        # =====================================================================
        inps = layer(inps, **layer_kwargs_copy)[0]

        del layer_kwargs_copy
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    if do_stats:
        overall_select = 0
        for mode in mode_list:
            overall_select = overall_select + dtype_stats[mode]
        for mode in mode_list:
            ratio = dtype_stats[mode] / overall_select
            print(f"{mode} ratio: {ratio * 100:.3f}%")
    print(f"overall_mse: {overall_mse}")

    gc.collect()
    torch.cuda.empty_cache()

@torch.no_grad()
def make_quant_linear(
    model, w_bit, a_bit, quant_config=None,
):
    from ..utils.model_utils import get_blocks, get_named_linears

    layers = get_blocks(model)
    tensor_id = 0

    for i in tqdm(range(len(layers)), desc="make quant linear..."):
        layer = layers[i]
        named_linears = get_named_linears(layer)

        for name, module in named_linears.items():
            cur_w_bit = _tensor_bit(w_bit, i, tensor_id, name, w_bit if isinstance(w_bit, int) else 4)
            cur_a_bit = _tensor_bit(a_bit, i, tensor_id, name, a_bit if isinstance(a_bit, int) else cur_w_bit)
            tensor_id += 1
            if cur_w_bit >= 16 and cur_a_bit >= 16:
                continue

            module_quant_config = copy.deepcopy(quant_config)
            if quant_config['quant_method'] == 'ant':
                from .qmodule_ant import ANT_Linear
                q_linear = ANT_Linear.from_linear(
                    module, cur_w_bit, cur_a_bit, module_quant_config['q_group_size'], i, name, quant_config=module_quant_config)
            elif quant_config['quant_method'] == 'olive':
                from .qmodule_olive import OliVe_Linear
                q_linear = OliVe_Linear.from_linear(
                    module, cur_w_bit, cur_a_bit, module_quant_config['q_group_size'], i, name, quant_config=module_quant_config)
            elif quant_config['quant_method'] in ['mant', 'int']:
                from .qmodule_mant import MANT_Linear
                q_linear = MANT_Linear.from_linear(
                    module, cur_w_bit, cur_a_bit, module_quant_config['q_group_size'], i, name)
            else:
                raise NotImplementedError(f"{quant_config['quant_method']} not supported yet!")
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)

    torch.cuda.empty_cache()
    gc.collect()
