"""
Centralized Model Configurations

This module consolidates benchmark model configurations by decoupling
Model Architecture (tensors, shapes) from Accelerator Configuration (bit widths).

Structure:
1. AcceleratorConfig: Defines the bit-width patterns for each model.
2. Generation: Combines Model + Accelerator to produce the final NetList.
"""

from typing import Dict, List, Any, NamedTuple

from .base_models import models

# Type alias for the final output format
NetList = List[List[Any]]

class AcceleratorConfig:
    """Defines the quantization (bit-width) policy for an accelerator."""
    def __init__(self, name: str, bit_patterns: Dict[str, List[int]]):
        self.name = name
        self.bit_patterns = bit_patterns

    def get_bits(self, model_name: str) -> List[int]:
        """Returns the bit pattern for the given model (all blocks)."""
        if model_name not in self.bit_patterns:
             # Error out if model is not defined for this accelerator.
             raise ValueError(f"No bit pattern defined for model '{model_name}' in accelerator '{self.name}'")
        return self.bit_patterns[model_name]

# ==============================================================================
# 1. Accelerator Configurations (Bit Width Patterns)
#    Define the sequence of bits for the *entire model*.
#    - For models with a single block pattern, you can specify one pattern.
#    - For models with per-block specialization, explicitly expand all blocks
#      into one flat bit-width list.
# ==============================================================================

# ANT Accelerator
ant_cfg = AcceleratorConfig("ant", {
    # 7 tensors/block × 8 blocks = 56 entries
    "llama2_7b":  [8, 8, 8, 8, 4, 4, 4] * 8,
    # 7 blocks: first 6 block is [8,4,4,8], last 1 block 4bit
    "falcon_7b":  [8, 4, 4, 8] * 7 + [4, 4, 4, 4],
    # 1 block
    "llama3_8b":  [4, 8, 8, 4, 4, 4, 8],
    # 2 blocks: [8,8,8,4,4,4,8] + [4,8,8,4,4,4,8]
    "llama3_70b": [8, 8, 8, 4, 4, 4, 8] + [4, 8, 8, 4, 4, 4, 8],
    # 7 blocks: first 6 block [8,8,4,4,4,4,8], last 1 4bit
    "mistral_7b": [8, 8, 4, 4, 4, 4, 8] * 6 + [4, 4, 4, 4, 4, 4, 4],
    # 8 blocks: first 6 block [8,8,8,4,4,4], last 2 4bit
    "opt6b7":     [8, 8, 8, 4, 4, 4] * 6 + [4, 4, 4, 4, 4, 4] * 2,
})

# MANT Accelerator
mant_cfg = AcceleratorConfig("mant", {
    # llama2_7b: first 6 block [8,8,8,8,4,4,4], last 2 block 4bit
    "llama2_7b":  [8, 8, 8, 8, 4, 4, 4] * 6 + [4, 4, 4, 4, 4, 4, 4] * 2,
    # falcon_7b: first 7 block [8,4,4,8], last 1 block [8,4,4,4]
    "falcon_7b":  [8, 4, 4, 8] * 7 + [8, 4, 4, 4],
    # 1 block
    "llama3_8b":  [4, 4, 8, 4, 4, 4, 8],
    # 2 blocks: [4,4,8,4,4,4,8] + [4,8,8,4,4,4,8]
    "llama3_70b": [4, 4, 8, 4, 4, 4, 8] + [4, 8, 8, 4, 4, 4, 8],
    # mistral_7b: first 6 block [4,8,4,4,4,4,8], last 1 block 4bit
    "mistral_7b": [4, 8, 4, 4, 4, 4, 8] * 6 + [4, 4, 4, 4, 4, 4, 4],
    # opt6b7: first 5 blocks are same, last 3 blocks is 4bit
    "opt6b7":     [8, 8, 8, 4, 4, 4] * 5 + [4, 4, 4, 4, 4, 4] * 3,
})

# M2XFP Accelerator (All 4 bits, 拓展到和 bench 同长度)
m2xfp_cfg = AcceleratorConfig("m2xfp", {
    # 7 × 8 = 56 entries
    "llama2_7b":  [4,4,4,4,4,4,4] * 8,
    # 4 × 7 = 28
    "falcon_7b":  [4,4,4,4] * 8,
    # 7
    "llama3_8b":  [4,4,4,4,4,4,4],
    # 7 × 2 = 14
    "llama3_70b": [4,4,4,4,4,4,4] * 2,
    # 7 × 7 = 49
    "mistral_7b": [4,4,4,4,4,4,4] * 7,
    # 6 × 8 = 48
    "opt6b7":     [4,4,4,4,4,4] * 8,
})


# Microscopiq Accelerator
microscopiq_cfg = AcceleratorConfig("microscopiq", {
    # llama2_7b: first 7 block [8,8,8,8,4,4,4], last 1xx block 4bit
    "llama2_7b":  [8, 8, 8, 8, 4, 4, 4] * 7 + [4, 4, 4, 4, 4, 4, 4] * 1,
    # falcon_7b: 7 block is [4,8,4,8]
    "falcon_7b":  [4, 8, 4, 8] * 8,
    # 1 block
    "llama3_8b":  [4, 8, 8, 4, 4, 4, 8],
    # 2 blocks: first 2 block is [4,8,8,4,4,4,8]
    "llama3_70b": [4, 8, 8, 4, 4, 4, 8] * 2,
    # mistral_7b: 7 block is [4,8,4,4,4,4,8]
    "mistral_7b": [4, 8, 4, 4, 4, 4, 8] * 7,
    # opt6b7: first 6 block [8,8,4,4,4,4], last 1 block 4bit
    "opt6b7":     [8, 8, 4, 4, 4, 4] * 7 + [4, 4, 4, 4, 4, 4],
})


# Olive Accelerator (Complex Mixed Precision)
olive_cfg = AcceleratorConfig("olive", {
    # llama2_7b: 8 blockis [4,8,8,4,8,4,8]
    "llama2_7b":  [4, 8, 8, 4, 8, 4, 8] * 8,
    # falcon_7b: 7 blockis [4,4,8,8]
    "falcon_7b":  [4, 4, 8, 8] * 8,
    # 1 block
    "llama3_8b":  [4, 8, 8, 4, 8, 4, 8],
    # 2 blocks: [4,8,8,4,8,4,8] + [4,4,8,4,8,4,8]
    "llama3_70b": [4, 8, 8, 4, 8, 4, 8] + [4, 4, 8, 4, 8, 4, 8],
    # mistral_7b: 7 blockis [8,8,4,4,8,4,8]
    "mistral_7b": [8, 8, 4, 4, 8, 4, 8] * 7,
    # opt6b7: 8 blockis [4,8,8,4,8,4]
    "opt6b7":     [4, 8, 8, 4, 8, 4] * 8,
})

accelerators = {
    "ant": ant_cfg,
    "mant": mant_cfg,
    "m2xfp": m2xfp_cfg,
    "microscopiq": microscopiq_cfg,
    "olive": olive_cfg,
}

# ==============================================================================
# 1.5 Sanity Check
#    For each model, the bit-pattern length must be consistent across all
#    accelerators that support this model.
# ==============================================================================
for model_name in models.keys():
    baseline_len = None
    for scheme_name, acc_conf in accelerators.items():
        pattern = acc_conf.bit_patterns.get(model_name)
        if pattern is None:
            continue
        if baseline_len is None:
            baseline_len = len(pattern)
        elif len(pattern) != baseline_len:
            raise ValueError(
                f"Inconsistent bit pattern length for model '{model_name}' across accelerators: "
                f"'{scheme_name}' has specified bit width of {len(pattern)} tensors, expected {baseline_len}."
            )
        
# ==============================================================================
# 2. Configuration Generator
# ==============================================================================

def generate_config(model_key: str, accelerator_key: str, seq_len: int = 2048, repeated_blocks: int = 1) -> NetList:
    """
    Generates the benchmark configuration list by combining model architecture
    and accelerator bit-width policies.

    The accelerator bit pattern is defined for the *entire model*:
    we assume `bits` has `num_blocks * len(model.block)` entries, where
    `model.block` describes one Transformer block.
    """
    model = models[model_key]
    accel = accelerators[accelerator_key]
    
    # Get bit patterns for the whole model
    bits = accel.get_bits(model_key)

    full_net_list = []

    tensors_per_block = len(model.block)
    if len(bits) % tensors_per_block != 0:
        raise ValueError(
            f"Length mismatch: model '{model_key}' block has {tensors_per_block} tensors, "
            f"but accelerator '{accelerator_key}' provides {len(bits)} bits."
        )

    num_blocks = len(bits) // tensors_per_block

    # Generate blocks: for each block, pick its own slice of `bits`
    for _ in range(repeated_blocks):
        for b in range(num_blocks):
            block_bits = bits[b * tensors_per_block : (b + 1) * tensors_per_block]
            for tensor, bit_width in zip(model.block, block_bits):
                # Standard GEMM shape logic:
                # Input:  [M, K]
                # Weight: [N, K] (Transposed in typical inference engines)
                # Output: [M, N]
                
                gemm_config = [
                    [seq_len, tensor.k],  # Input Shape
                    [tensor.n, tensor.k],  # Weight Shape
                    [seq_len, tensor.n],  # Output Shape
                    [],                  # Placeholder 1 (Strides/Paddings?)
                    [],                  # Placeholder 2
                    bit_width,           # Quantization Bits
                    1                    # Op Type (1 = GEMM, 0 = CONV)
                ]
                full_net_list.append(gemm_config)
            
    return full_net_list

# ==============================================================================
# 3. Exports (Backward Compatibility)
#    These variables match the structure of your original file.
# ==============================================================================

SCHEMES = ["ant", "mant", "m2xfp", "microscopiq", "olive"]
MODELS  = ["llama2_7b", "falcon_7b", "llama3_8b", "llama3_70b", "mistral_7b", "opt6b7"]

# Store configs for each scheme, e.g., scheme_configs["ant"]["llama2_7b"]
scheme_configs = {}

for scheme in SCHEMES:
    model_dict = {}
    for model in MODELS:
        cfg = generate_config(model, scheme)

        # 1) Generate original-style variable names: ant_llama2_7b / mant_llama3_70b / ...
        var_name = f"{scheme}_{model}"
        globals()[var_name] = cfg

        # 2) Fill in the per-scheme dict: ant_cfgigs["llama2_7b"], etc.
        model_dict[model] = cfg

    scheme_configs[scheme] = model_dict

# Backward-compatible top-level dicts matching your original *_configs names
ant_cfgigs         = scheme_configs["ant"]
mant_cfgigs        = scheme_configs["mant"]
m2xfp_cfgigs       = scheme_configs["m2xfp"]
microscopiq_cfgigs = scheme_configs["microscopiq"]
olive_cfgigs       = scheme_configs["olive"]

if __name__ == '__main__':
    # Example usage
    config = generate_config("llama2_7b", "ant")
    for layer in config:
        print(layer)