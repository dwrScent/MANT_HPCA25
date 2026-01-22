import torch

def int_value(n_bit, signed=True):
    B = n_bit - 1 if signed else n_bit
    values = [0.] + list(range(1, 2 ** B))
    if signed:
        values += [-i for i in range(1, 2 ** B)]
        values.append(-2 ** B)
    values.remove(-8)
    return values

def pot_value(n_bit, signed=True):
    B = n_bit - 1 if signed else n_bit
    exp_bit = B
    values = []
    values.append(0.)
    # values.append(-0.)
    for i in range(0, 2 ** exp_bit - 1):
        values.append(2 ** i)
        if signed:
            values.append(-2 ** i)
    return values

def flint_value(n_bit, signed=True, exp_base=0):

    B = n_bit - 1 if signed else n_bit

    value_bit = B
    assert(value_bit >= 2)

    exp_num =     value_bit * 2 - 1
    neg_exp_num = value_bit - 1
    pos_exp_num = value_bit - 1
    
    
    exp_max = pos_exp_num + exp_base
    exp_min = -neg_exp_num

    ## Append zero value
    values = [0., -0.]
    values = [0.]

    # values = [0.]
    ## exponent negative
    for i in range(0, neg_exp_num + 1):
        exp_bit = i + 2
        exp_value = -(exp_bit - 1)
        mant_bit = value_bit - exp_bit
        for j in range(int(2 ** mant_bit)):
            v = 2 ** (exp_value + exp_base) * (1 + 2 ** (-mant_bit) * j)
            values.append(v)
            if signed:
                values.append(-v)

    ## exponent zero
    exp_bit = 2
    exp_value = 0
    mant_bit = value_bit - exp_bit
    for j in range(int(2 ** mant_bit)):
        v = 2 ** (exp_value + exp_base) * (1 + 2 ** (-mant_bit) * j)
        values.append(v)
        if signed:
            values.append(-v)

    ## exponent positive     
    for i in range(1, pos_exp_num):
        exp_bit = i + 2
        exp_value = i
        mant_bit = value_bit - exp_bit
        for j in range(int(2 ** mant_bit)):
            v = 2 ** (exp_value + exp_base) * (1 + 2 ** (-mant_bit) * j)
            values.append(v)
            if signed:
                values.append(-v)
    ## Append max value
    values.append(2 ** exp_max)
    if signed:
        values.append(-2 ** exp_max)

    return values

def float_value(n_bit, signed=True, exp_field=2):
    B = n_bit - 1 if signed else n_bit

    # mapping, total_bit: exponent_bit
    exp_field_map = {3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4}
    if n_bit in exp_field_map:
        exp_field = exp_field_map[n_bit]
    else:
        raise ValueError("Not support this bit width")
    exp_bit = exp_field

    man_bit = B - exp_bit
    values = []
    min_to_zero = True
    subnormal = True
    for i in range(2 ** exp_bit):
        for j in range(2 ** man_bit):
            if min_to_zero:
                values.append(0.)
                values.append(-0.)
                min_to_zero = False
            else:
                if subnormal:
                    values.append((2 ** i) * (j * 2 ** (-man_bit)))
                else:
                    values.append((2 ** (i - 1)) * (1 + j * 2 ** (-man_bit)))

                if signed:
                    if subnormal:
                        values.append(-(2 ** i) * (j * 2 ** (-man_bit)))
                    else:
                        values.append(-(2 ** (i - 1)) * (1 + j * 2 ** (-man_bit)))
        subnormal = False

    return torch.tensor(values)

from scipy.stats import norm
def normal_float_value(n_bit, signed=True, offset=0.9677083, use_extra_value=True):

    if use_extra_value:
        # one more positive value, this is an asymmetric type
        v1 = norm.ppf(torch.linspace(offset, 0.5, 2 ** (n_bit - 1) + 1)[:-1]).tolist()
        v2 = [0] ## we have 15 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 2 ** (n_bit - 1))[:-1])).tolist()
    else:
        v1 = norm.ppf(torch.linspace(offset, 0.5, 2 ** (n_bit - 1))[:-1]).tolist()
        v2 = [0] ## we have 14 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 2 ** (n_bit - 1))[:-1])).tolist()

    v = v1 + v2 + v3
    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()

    assert values.numel() == 2 ** n_bit 
    # print(values)
    return values

def generate_quant_grid(n_bit=4, signed=True, quant_dtype="flint"):
    quant_grid_set = {}
    quant_grid_funcs = {
        "int": int_value,
        "flint": flint_value,
        "pot": pot_value,
        "float": float_value,
        "nf": normal_float_value,
    }
    mode_list = quant_dtype.split('-')
    if "kmeans" in mode_list:
        mode_list.remove("kmeans")
    if "weighted_kmeans" in mode_list:
        mode_list.remove("weighted_kmeans")

    for mode in mode_list:
        if mode in quant_grid_funcs:
            quant_grid_set[mode] = quant_grid_funcs[mode](n_bit=n_bit, signed=signed)
        elif mode == 'meta_flint':
            pass
        else:
            raise ValueError(f"Invalid mode: {mode}")

    # Convert list to tensor
    for key, value in quant_grid_set.items():
        quant_grid_set[key] = torch.tensor(value)
    return quant_grid_set
