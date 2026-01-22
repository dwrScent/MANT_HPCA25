
import torch
import torch.nn.functional as F

@torch.no_grad()
def quantize_rows(w, quant_grid):
    '''
    Perform row-wise quantization on a 2D tensor using a specific grid.
    Returns: dequantized weight
    '''
    quant_grid = quant_grid.to(w.device)
    max_val = w.abs().amax(dim=1, keepdim=True)
    max_quant_val = max(quant_grid)
    
    # Compute the scaling factor
    scales = max_val / max_quant_val
    zeros = 0

    labels = (((w + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    w_deq = quant_grid[labels] * scales - zeros

    return w_deq

@torch.no_grad()
def pseudo_quantize_int(tensor, n_bit=8, q_group_size=-1, alpha=1.0):
    org_shape = tensor.shape
    padding_size = 0
    
    # Tensor-wise quantization
    if q_group_size == -2:
        tensor = tensor.view(-1)
        max_val = tensor.abs().amax()
        max_val = max_val.clamp(min=1e-5)
    # Channel-wise or group-wise quantization
    else:
        if q_group_size > 0:
            if org_shape[-1] % q_group_size != 0:
                # Calculate padding size
                padding_size = q_group_size - (org_shape[-1] % q_group_size)
                # Apply padding
                tensor = F.pad(tensor, (0, padding_size), "constant", 0)
                padding_shape = tensor.shape
                assert padding_shape[-1] % q_group_size == 0
            tensor = tensor.reshape(-1, q_group_size)
        assert tensor.dim() == 2
        max_val = tensor.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)

    max_int = 2 ** (n_bit - 1) - 1
    min_int = - 2 ** (n_bit - 1)
    scales = (max_val * alpha) / max_int
    zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(tensor).sum() == 0

    # Batch processing to avoid OOM
    batch_num = 32
    if tensor.shape[0] % batch_num == 0:
        batch_size = tensor.shape[0] // batch_num
        tensor_deq = torch.zeros_like(tensor)
        for i in range(batch_num):
            batch_idx = slice(i * batch_size, (i + 1) * batch_size)
            tensor_batch = tensor[batch_idx]
            
            # If tensor-wise, scales is a scalar that does not require batching
            scale_batch = scales[batch_idx] if scales.ndim > 0 else scales
            tensor_deq[batch_idx] = (torch.clamp(torch.round(tensor_batch / scale_batch) +
                                zeros, min_int, max_int) - zeros) * scale_batch
        tensor = tensor_deq
    else:
        tensor = (torch.clamp(torch.round(tensor / scales) +
                            zeros, min_int, max_int) - zeros) * scales

    assert torch.isnan(tensor).sum() == 0

    # Reshape back to original shape
    if q_group_size == -2:
        tensor = tensor.reshape(org_shape)
    elif padding_size > 0:
        tensor = tensor.reshape(padding_shape)
        tensor = tensor[:, :org_shape[-1]]
        tensor = tensor.reshape(org_shape)
    else:
        tensor = tensor.reshape(org_shape)

    return tensor

@torch.no_grad()
def pseudo_quantize_mant(tensor, q_group_size=-1):
    quantized_part_shape = tensor.shape

    if q_group_size > 0:
        quantized_part_group = tensor.reshape(-1, q_group_size)
    else:
        raise ValueError('not support yet')

    quant_grid_set = {}
    quant_grid_set['coefficient_25'] = torch.tensor([-1.0000, -0.7061, -0.5181, -0.3828, -0.2739, -0.1782, -0.0891, -0.0033, 0.0033,  0.0891,  0.1782,  0.2739,  0.3828,  0.5181,  0.7061,  1.0000])
    quant_grid_set['int'] = torch.tensor([-0., -7., -6., -5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5., 6.,  7.])
    quant_grid_set['coefficient_0'] = torch.tensor([-1.0000, -0.5000, -0.2500, -0.1250, -0.0625, -0.0312, -0.0156, -0.0078, 0.0078,  0.0156,  0.0312,  0.0625,  0.1250,  0.2500,  0.5000,  1.0000])

    quantized_part_group_deq = torch.zeros_like(quantized_part_group)
    
    # Batch processing to avoid OOM
    batch_num = 32
    assert quantized_part_group.shape[0] % batch_num == 0
    batch_size = quantized_part_group.shape[0] // batch_num
    
    for i in range(batch_num):
        batch_idx = slice(i * batch_size, (i + 1) * batch_size)
        tensor_batch = quantized_part_group[batch_idx]
        
        max_val = torch.max(torch.abs(tensor_batch), dim=1, keepdim=True).values
        value_var = torch.var(tensor_batch / max_val, dim=1, keepdim=True)

        q_nf = quantize_rows(tensor_batch, quant_grid_set['coefficient_25'])
        q_int = quantize_rows(tensor_batch, quant_grid_set['int'])
        q_pot = quantize_rows(tensor_batch, quant_grid_set['coefficient_0'])

        # Select data type based on variance
        mask_pot = (value_var < 0.05).expand_as(q_pot)
        mask_nf = ((value_var >= 0.05) & (value_var <= 0.25)).expand_as(q_nf)
        mask_int = (value_var > 0.25).expand_as(q_int)

        batch_deq = torch.zeros_like(tensor_batch)
        batch_deq = torch.where(mask_pot, q_pot, batch_deq)
        batch_deq = torch.where(mask_nf, q_nf, batch_deq)
        batch_deq = torch.where(mask_int, q_int, batch_deq)
        
        quantized_part_group_deq[batch_idx] = batch_deq

    quantized_part_deq = quantized_part_group_deq.reshape(quantized_part_shape)
    quantized_part_deq = quantized_part_deq.to(dtype=tensor.dtype, device=tensor.device)

    assert torch.isnan(quantized_part_deq).sum() == 0

    return quantized_part_deq

@torch.no_grad()
def pseudo_quantize_mant_v8(tensor, q_group_size=-1):
    quantized_part_shape = tensor.shape

    if q_group_size > 0:
        quantized_part_group = tensor.reshape(-1, q_group_size)
    else:
        raise ValueError('not support yet')

    quant_grid_set = {}
    # Define 8 types of quantization grids
    quant_grid_set['int'] = torch.tensor([-0., -7., -6., -5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5., 6.,  7.])
    quant_grid_set['coefficient_0'] = torch.tensor([-1.0000, -0.5000, -0.2500, -0.1250, -0.0625, -0.0312, -0.0156, -0.0078, 
        0.0078,  0.0156,  0.0312,  0.0625,  0.1250,  0.2500,  0.5000,  1.0000])
    quant_grid_set['coefficient_20'] = torch.tensor([-1.0000, -0.6865, -0.4924, -0.3582, -0.2537, -0.1642, -0.0821, -0.0037,
        0.0037,  0.0821,  0.1642,  0.2537,  0.3582,  0.4924,  0.6865,  1.0000])
    quant_grid_set['coefficient_40'] = torch.tensor([-1.0000, -0.7451, -0.5688, -0.4314, -0.3137, -0.2059, -0.1030, -0.0025,
        0.0025,  0.1030,  0.2059,  0.3137,  0.4314,  0.5688,  0.7451,  1.0000])
    quant_grid_set['coefficient_60'] = torch.tensor([-1.0000, -0.7739, -0.6060, -0.4670, -0.3430, -0.2263, -0.1132, -0.0018,
        0.0018,  0.1132,  0.2263,  0.3430,  0.4670,  0.6060,  0.7739,  1.0000])
    quant_grid_set['coefficient_80'] = torch.tensor([-1.0000, -0.7905, -0.6279, -0.4883, -0.3604, -0.2384, -0.1192, -0.0015,
        0.0015,  0.1192,  0.2384,  0.3604,  0.4883,  0.6279,  0.7905,  1.0000])
    quant_grid_set['coefficient_100'] = torch.tensor([-1.0000, -0.8018, -0.6426, -0.5024, -0.3721, -0.2463, -0.1232, -0.0012,
        0.0012,  0.1232,  0.2463,  0.3721,  0.5024,  0.6426,  0.8018,  1.0000])
    quant_grid_set['coefficient_120'] = torch.tensor([-1.0000, -0.8101, -0.6528, -0.5122, -0.3801, -0.2520, -0.1260, -0.0010,
        0.0010,  0.1260,  0.2520,  0.3801,  0.5122,  0.6528,  0.8101,  1.0000])

    quantized_part_group_deq = torch.zeros_like(quantized_part_group)
    
    # Define quantization modes and their variance thresholds
    modes = [
        {'name': 'coefficient_0', 'q_grid': 'coefficient_0', 'thresholds': (None, 0.0417)},
        {'name': 'coefficient_20', 'q_grid': 'coefficient_20', 'thresholds': (0.0417, 0.0797)},
        {'name': 'coefficient_40', 'q_grid': 'coefficient_40', 'thresholds': (0.0797, 0.122)},
        {'name': 'coefficient_60', 'q_grid': 'coefficient_60', 'thresholds': (0.122, 0.1368)},
        {'name': 'coefficient_80', 'q_grid': 'coefficient_80', 'thresholds': (0.1368, 0.1446)},
        {'name': 'coefficient_100', 'q_grid': 'coefficient_100', 'thresholds': (0.1446, 0.1528)},
        {'name': 'coefficient_120', 'q_grid': 'coefficient_120', 'thresholds': (0.1528, 0.1611)},
        {'name': 'int', 'q_grid': 'int', 'thresholds': (0.1611, None)},
    ]
    
    # Statistics counters
    type_counts = [0] * len(modes)
    
    # Batch processing
    batch_num = 32
    assert quantized_part_group.shape[0] % batch_num == 0
    batch_size = quantized_part_group.shape[0] // batch_num
    
    for i in range(batch_num):
        batch_idx = slice(i * batch_size, (i + 1) * batch_size)
        tensor_batch = quantized_part_group[batch_idx]
        
        max_val = torch.max(torch.abs(tensor_batch), dim=1, keepdim=True).values
        value_var = torch.var(tensor_batch / max_val, dim=1, keepdim=True)
        
        # Calculate all quantization modes
        quantized_results = {
            mode['name']: quantize_rows(tensor_batch, quant_grid_set[mode['q_grid']])
            for mode in modes
        }
        
        # Select data type based on variance thresholds
        batch_deq = torch.zeros_like(tensor_batch)
        for idx, mode in enumerate(modes):
            lower, upper = mode['thresholds']
            if lower is None:
                mask = (value_var < upper).expand_as(batch_deq)
            elif upper is None:
                mask = (value_var > lower).expand_as(batch_deq)
            else:
                mask = ((value_var >= lower) & (value_var <= upper)).expand_as(batch_deq)
            
            batch_deq = torch.where(mask, quantized_results[mode['name']], batch_deq)
            type_counts[idx] += mask.sum().item()
        
        quantized_part_group_deq[batch_idx] = batch_deq

    quantized_part_deq = quantized_part_group_deq.reshape(quantized_part_shape)
    quantized_part_deq = quantized_part_deq.to(dtype=tensor.dtype, device=tensor.device)

    assert torch.isnan(quantized_part_deq).sum() == 0

    # Print selection statistics
    total_elements = quantized_part_group.numel()
    percentages = [count / total_elements * 100 for count in type_counts]
    type_names = [mode['name'] for mode in modes]
    
    print(f"Selection percentages: {', '.join([f'Type{i+1} ({name}): {pct:.2f}%' for i, (name, pct) in enumerate(zip(type_names, percentages))])}")

    return quantized_part_deq

@torch.no_grad()
def quantize_with_grid(tensor_value, quant_grid, group_size, alpha=1.0):

    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)
    assert torch.isnan(tensor_value).sum() == 0

    max_quant_val = max(quant_grid)

    if group_size == -2:
        # Tensor-wise
        tensor_value = tensor_value.view(-1)

        max_val = tensor_value.abs().amax()
        scales = (max_val * alpha) / max_quant_val
        zeros = 0

        batch_num = 32
        assert tensor_value.shape[0] % batch_num == 0
        batch_size = tensor_value.shape[0] // batch_num
        tensor_deq = torch.zeros_like(tensor_value)
        for idx in range(batch_num):
            tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size]
            labels = (((tensor_par + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
            tensor_q_par = quant_grid[labels] * scales - zeros
            tensor_deq[idx*batch_size : (idx+1)*batch_size] = tensor_q_par
        
    # Channel or group wise
    elif group_size >= -1:

        if group_size > 0:
            assert org_shape[-1] % group_size == 0
            tensor_value = tensor_value.reshape(-1, group_size)

        assert tensor_value.dim() == 2

        max_val = tensor_value.abs().amax(dim=1, keepdim=True)
        scales = (max_val * alpha) / max_quant_val
        zeros = 0

        # Batch processing to avoid OOM
        batch_num = 32
        assert tensor_value.shape[0] % batch_num == 0
        batch_size = tensor_value.shape[0] // batch_num
        tensor_deq = torch.zeros_like(tensor_value)
        for idx in range(batch_num):
            tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
            labels = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
            tensor_q_par = quant_grid[labels] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
            tensor_deq[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par


    assert torch.isnan(tensor_deq).sum() == 0
    assert torch.isnan(scales).sum() == 0

    # tensor_deq = tensor_deq.half()
    tensor_deq = tensor_deq.reshape(org_shape)

    return tensor_deq
