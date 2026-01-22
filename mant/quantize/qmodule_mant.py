
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant_func import pseudo_quantize_int

def encode_gen(w_bit, return_list=False, a_stride=10):
    """
    Generate the computable codebook from the index list
    Computable codebook: a*index + 2^index * b
    """
    coefficient_list = []
    # Select coefficient a
    for coefficient in range(0, 128, a_stride):
        coefficient_list.append(coefficient)
    # supply some specific data type, merge them after removing duplicates
    supply_list = []
    if a_stride == 10:
        supply_list = [5, 17]
    merged_list = list(set(coefficient_list + supply_list))

    codebook_dict = {}
    b = 1
    for coefficient in merged_list:
    # for coefficient in coefficient_list:
        codebook_list = []
        for item in range(2 ** w_bit):
            # 0~15 maps to -8~7
            index = item - (2 ** (w_bit-1))
            if index < 0:
                index = (-index) - 1
                codebook_list.append(-(coefficient * index + (2 ** index * b)))
            elif index >= 0:
                assert index < (2 ** w_bit // 2)
                codebook_list.append(coefficient * index + (2 ** index * b))
            else:
                raise ValueError(f"Index {index} out of range")

        codebook_list = torch.tensor(codebook_list).to(dtype=torch.half)
        codebook_list, _ = codebook_list.sort()
        # Normalization
        codebook_list = codebook_list / codebook_list.max()
        # to list if need
        if return_list:
            codebook_list = codebook_list.tolist()

        codebook_dict[f"coefficient_{coefficient}"] = codebook_list
    return codebook_dict

class MANT_Linear(nn.Module):
    def __init__(self, w_bit, a_bit, group_size, in_features, out_features, bias, dev, layer_id, layer_name):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.group_size = group_size if group_size != -1 else in_features

        self.layer_id = layer_id
        self.layer_name = layer_name

        assert self.in_features % self.group_size == 0

        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.float16, device=dev))

        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, a_bit, group_size, layer_id, layer_name):

        mant_linear = cls(w_bit, a_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device, layer_id, layer_name)

        mant_linear.weight = linear.weight.data.clone().half()
        if linear.bias is not None:
            mant_linear.bias = linear.bias.clone().half()
        
        return mant_linear
    
    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        input = x.reshape(-1, x.shape[-1])

        # quantize activation to INT8
        if self.a_bit < 16 and self.a_bit != -1:
            input = pseudo_quantize_int(input, n_bit=self.a_bit, q_group_size=self.group_size)
 
        input = input.to(device=self.weight.device)
        out = F.linear(input, self.weight)

        out = out + self.bias if self.bias is not None else out

        return out.reshape(out_shape)
