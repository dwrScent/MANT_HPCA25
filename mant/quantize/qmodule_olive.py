import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant_grid import flint_value as ant_flint_value

def int_value(n_bit, signed=True):
    B = n_bit - 1 if signed else n_bit

    values = []
    values.append(0.)
    for i in range(1, 2 ** B):
        values.append(i)
        if signed:
            values.append(-i)

    values = torch.tensor(values)
    values, _ = torch.sort(values)
    # add a bias to normalize the codebook (the threshold between outliers and normal values is 32)
    values *= 32 / (2 ** B)
    return values


def flint_value(n_bit, signed=True, exp_base=0):
    # Reuse quant_grid's generation logic
    values = torch.tensor(ant_flint_value(n_bit, signed, exp_base))
    values, _ = torch.sort(values)
    
    # Apply OliVe-specific scaling
    B = n_bit - 1 if signed else n_bit
    exp_max = (B - 1) + exp_base
    
    # add a bias to normalize the codebook (the threshold between outliers and normal values is 32)
    values *= 32 / (2 ** exp_max)
    return values


def outlier_value(n_bit, signed=True, exp_bit=2, exp_base=5):
    B = n_bit - 1 if signed else n_bit

    value_bit = B
    mant_bit = value_bit - exp_bit
    values = []

    for i in range(exp_base, exp_base + 2 ** exp_bit):
        for j in range(int(2 ** mant_bit)):
            if i == exp_base and j == 0:
                continue

            v = 2 ** i * (1 + 2 ** (-mant_bit) * j)
            values.append(v)
            if signed:
                values.append(-v)

    values = torch.tensor(values)
    values, _ = torch.sort(values)

    return values


@torch.no_grad()
def get_quant(tensor_value, quant_grid, outlier_grid, alpha=1.0, group_size=-1):
    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)
    outlier_grid = outlier_grid.to(tensor_value.device)
    merge_grid = torch.cat((quant_grid, outlier_grid), dim=0)

    # Unify processing logic
    if group_size == -2:
        # Tensor-wise: treat as 1D flat
        target = tensor_value.view(-1)
        mean = target.mean()
        std = target.std()
        scales = None # Scalar, will be computed below
    elif group_size >= -1:
        # Group-wise: treat as (N, GroupSize)
        if group_size > 0:
            assert org_shape[-1] % group_size == 0
            target = tensor_value.reshape(-1, group_size)
        else:
            target = tensor_value
        mean = target.mean(dim=1, keepdim=True)
        std = target.std(dim=1, keepdim=True)

    normal_max = torch.maximum((mean + 3 * std).abs(), (mean - 3 * std).abs())
    max_quant_val = max(quant_grid)
    scales = (normal_max * alpha) / max_quant_val
    zeros = 0

    # Batch processing to avoid OOM
    batch_num = 32
    assert target.shape[0] % batch_num == 0
    batch_size = target.shape[0] // batch_num
    tensor_q = torch.zeros_like(target)
    
    for idx in range(batch_num):
        start, end = idx * batch_size, (idx + 1) * batch_size
        tensor_par = target[start:end]
        
        # Handle scalar vs tensor scales
        scale_par = scales[start:end] if (isinstance(scales, torch.Tensor) and scales.ndim > 0) else scales
        
        labels = (((tensor_par + zeros) / scale_par).unsqueeze(-1) - merge_grid).abs().argmin(dim=-1)
        tensor_q[start:end] = merge_grid[labels]

    tensor_q = tensor_q.view(-1)

    # Outlier Victim Pair Encoding
    mask = tensor_q.abs() > 32
    victim_odd = torch.roll(mask, 1, -1)
    victim_odd[::2] = 0
    victim_even = torch.roll(mask & (~victim_odd), -1, -1)
    victim_even[1::2] = 0
    victim = victim_even | victim_odd
    tensor_q = tensor_q * (~victim)

    # Reshape back and dequantize
    if group_size >= -1:
        # Restore group shape for correct broadcasting with scales
        tensor_q = tensor_q.view(target.shape)

    tensor_deq = tensor_q * scales - zeros

    tensor_deq = tensor_deq.to(tensor_value.device).half()
    tensor_deq = tensor_deq.reshape(org_shape)

    return tensor_deq

class OliVeQuantizer:
    """
    Holds quant params (quant_grid/outlier_grid/alpha) and provides:
      - search()          : replicate olive_quant()
      - runtime_quantize(): replicate forward() else-branch quantization behavior
      - finalize_weight() : replicate "group_size>-1 => alpha forced to 1.0" weight materialization
    """
    def __init__(self, bit, group_size, quant_config, is_weight: bool):
        self.bit = bit
        self.group_size = group_size
        self.quant_config = quant_config
        self.is_weight = is_weight

        # calibrated
        self.quant_grid = None
        self.outlier_grid = None
        self.alpha = -1.0
        self.mode = None  # "int" or "flint"

        # cached candidates (computed once per bit/exp_base per search call)
        self._int_grid = None
        self._flint_grid = None

    @staticmethod
    def _build_outlier_grid(n_bit, exp_base):
        if n_bit == 8:
            return outlier_value(n_bit, signed=True, exp_bit=4)
        return outlier_value(n_bit, signed=True, exp_base=exp_base)

    @torch.no_grad()
    def search(self, *, weight_ref, input_ref, tensor_value, group_size_for_search,
               exp_base, is_input, layer_id, layer_name):
        """
        Strictly replicate olive_quant(self, n_bit, weight, input, quant_config, group_size, ...)
        but store results into this quantizer and return best dequantized tensor.
        """
        mode_list = self.quant_config['quant_dtype'].split('-')

        # For MSE computation (original uses float, not float64)
        org_output = torch.mm(input_ref, weight_ref.T).to(torch.float)

        int_grid = int_value(self.bit, signed=True)
        flint_grid = flint_value(self.bit, signed=True)
        outlier_grid = self._build_outlier_grid(self.bit, exp_base)

        # init final tensor (match original dtype/device behavior)
        final_tensor = torch.zeros_like(tensor_value, dtype=torch.half).to(tensor_value.device)

        min_mse = float('inf')
        best_mode = 'null'
        best_alpha = -1.0

        lb = self.quant_config['w_low']
        ub = self.quant_config['w_high']

        for mode in mode_list:
            # support flint and int in OliVe
            if mode == 'int':
                quant_grid = int_grid
            elif mode == 'flint':
                quant_grid = flint_grid
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            for i in range(lb, ub, 10):
                search_alpha = i * 0.01

                tensor_deq = get_quant(
                    tensor_value,
                    quant_grid,
                    outlier_grid,
                    alpha=search_alpha,
                    group_size=group_size_for_search
                )

                if is_input:
                    deq_output = torch.mm(tensor_deq, weight_ref.T).to(torch.float)
                else:
                    deq_output = torch.mm(input_ref, tensor_deq.T).to(torch.float)

                mse = (deq_output - org_output).pow(2).mean()

                if mse < min_mse:
                    min_mse = mse
                    final_tensor = tensor_deq
                    best_mode = mode
                    best_alpha = search_alpha

        # commit
        self._int_grid = int_grid
        self._flint_grid = flint_grid

        self.mode = best_mode
        self.alpha = best_alpha
        self.quant_grid = int_grid if best_mode == 'int' else flint_grid
        self.outlier_grid = outlier_grid

        quant_obj = 'input' if is_input else 'weight'
        print(
            f"layer: {layer_id}, tensor: {layer_name}, {quant_obj} quant, "
            f"best mode: {best_mode}, mse: {min_mse}, alpha: {best_alpha}, bit_width: {self.bit}"
        )

        # preserve original extra prints
        if is_input:
            print(f"input max: {tensor_value.max()}, deq_max: {final_tensor.max()}, exp_base: {exp_base}")
        else:
            print(f"weight max: {tensor_value.max()}, deq_max: {final_tensor.max()}, exp_base: {exp_base}")

        return final_tensor

    @torch.no_grad()
    def finalize_weight_materialization(self, original_weight):
        """
        Replicate original first-forward:
          if group_size > -1:
              deq_weight = get_quant(weight, weight_quant_grid, weight_outlier_grid, alpha=1.0, group_size=self.group_size)
              self.weight = deq_weight
          else:
              self.weight = searched_deq_weight
        """
        assert self.is_weight
        assert self.quant_grid is not None and self.outlier_grid is not None
        if self.group_size > -1:
            return get_quant(
                original_weight,
                self.quant_grid,
                self.outlier_grid,
                alpha=1.0,  # IMPORTANT: forced to 1.0 (matches original)
                group_size=self.group_size
            )
        return None  # caller keeps searched weight

    @torch.no_grad()
    def runtime_quantize(self, tensor_value, *, group_size_override=None):
        """
        Replicate original runtime quantization:
          if module.group_size > -1:
              get_quant(..., group_size=self.group_size)
          else:
              get_quant(..., group_size=-2)
        """
        assert self.quant_grid is not None and self.outlier_grid is not None
        gs = self.group_size if group_size_override is None else group_size_override
        return get_quant(tensor_value, self.quant_grid, self.outlier_grid, alpha=self.alpha, group_size=gs)


class OliVeQuantConfig:
    """
    Owns weight+act quantizers and reproduces original 'first inference does search' logic.
    """
    def __init__(self, w_bit, a_bit, group_size, quant_config, layer_id, layer_name):
        self.quant_config = quant_config
        self.group_size = group_size
        self.layer_id = layer_id
        self.layer_name = layer_name

        self.w_quant = OliVeQuantizer(w_bit, group_size, quant_config, is_weight=True)
        self.a_quant = OliVeQuantizer(a_bit, group_size, quant_config, is_weight=False)

        self.calibrated = False

    @torch.no_grad()
    def calibrate_first_forward(self, weight, input_2d):
        """
        Strictly replicate first-forward branch in original OliVe_Linear.forward():
          - weight search: group_size=-1, exp_base=5
          - if group_size>-1: materialize weight with group_size=self.group_size, alpha=1.0
          - input search: group_size=-2 always, but exp_base depends on group_size>-1 ? 5 : 7
        Returns (final_weight, deq_input_for_this_forward)
        """
        # 1) channel-wise search for weight (group_size=-1), exp_base=5
        searched_w = self.w_quant.search(
            weight_ref=weight,
            input_ref=input_2d,
            tensor_value=weight,
            group_size_for_search=-1,
            exp_base=5,
            is_input=False,
            layer_id=self.layer_id,
            layer_name=self.layer_name
        )

        # 2) materialize weight (if group_size>-1: group quant with alpha=1.0)
        final_w = searched_w
        maybe_mat = self.w_quant.finalize_weight_materialization(weight)
        if maybe_mat is not None:
            final_w = maybe_mat

        # 3) input search is tensor-wise (group_size=-2), exp_base depends on group_size>-1
        exp_base_in = 5 if self.group_size > -1 else 7
        searched_inp = self.a_quant.search(
            weight_ref=final_w,       # IMPORTANT: use deq_weight reference (matches original)
            input_ref=input_2d,
            tensor_value=input_2d,
            group_size_for_search=-2,
            exp_base=exp_base_in,
            is_input=True,
            layer_id=self.layer_id,
            layer_name=self.layer_name
        )

        self.calibrated = True
        return final_w, searched_inp

    @torch.no_grad()
    def runtime_quantize_input(self, input_2d):
        """
        Strictly replicate original else-branch:
          if group_size>-1 => group quant for activation (group_size=self.group_size)
          else => tensor-wise (group_size=-2)
        """
        if self.group_size > -1:
            return self.a_quant.runtime_quantize(input_2d, group_size_override=self.group_size)
        return self.a_quant.runtime_quantize(input_2d, group_size_override=-2)


class OliVe_Linear(nn.Module):
    def __init__(self, w_bit, a_bit, group_size, in_features, out_features, bias, dev, quant_config, layer_id, layer_name):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.group_size = group_size
        self.quant_config = quant_config
        self.layer_id = layer_id
        self.layer_name = layer_name

        assert self.in_features % self.group_size == 0

        self.quant_cfg = OliVeQuantConfig(w_bit, a_bit, group_size, quant_config, layer_id, layer_name)

        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features,), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, a_bit, group_size, layer_id, layer_name, quant_config=None):
        olive_linear = cls(
            w_bit, a_bit, group_size,
            linear.in_features, linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            quant_config,
            layer_id,
            layer_name
        )

        olive_linear.weight = linear.weight.data.clone().half()
        if linear.bias is not None:
            olive_linear.bias = linear.bias.clone().half()

        if w_bit > 6:
            olive_linear.quant_config['quant_dtype'] = 'int'
        return olive_linear

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        input_2d = x.reshape(-1, x.shape[-1])

        # First forward: search + use searched deq_input for THIS forward
        if not self.quant_cfg.calibrated:
            deq_weight, deq_input = self.quant_cfg.calibrate_first_forward(self.weight, input_2d)
            self.weight = deq_weight
            print("olive search data type and alpha.")
        else:
            deq_input = self.quant_cfg.runtime_quantize_input(input_2d)

        out = F.linear(deq_input, self.weight)
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
