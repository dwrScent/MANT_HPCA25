import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_func import quantize_with_grid, pseudo_quantize_int


META_FLINT_SET = {
    "flint_0": [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0],
    "flint_1": [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 8.0, -8.0],
    "flint_2": [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 12.0, -12.0],
    "flint_3": [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0],
    "flint_4": [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 2.5, -2.5, 3.0, -3.0, 3.5, -3.5],
    "flint_5": [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 6.0, -6.0, 8.0, -8.0, 12.0, -12.0],
    "flint_6": [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 6.0, -6.0, 8.0, -8.0, 16.0, -16.0],
    "flint_7": [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 24.0, -24.0],
    "flint_8": [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 32.0, -32.0],
    "flint_9": [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 5.0, -5.0, 6.0, -6.0, 7.0, -7.0],
    "flint_10": [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0, 8.0, -8.0, 12.0, -12.0],
    "flint_11": [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0, 8.0, -8.0, 16.0, -16.0],
    "flint_12": [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 24.0, -24.0],
    "flint_13": [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 32.0, -32.0],
    "flint_14": [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0, 6.0, -6.0, 7.0, -7.0],
    "flint_15": [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 12.0, -12.0, 16.0, -16.0, 24.0, -24.0],
    "flint_16": [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 12.0, -12.0, 16.0, -16.0, 32.0, -32.0],
    "flint_17": [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 32.0, -32.0, 48.0, -48.0],
    "flint_18": [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 32.0, -32.0, 64.0, -64.0],
    "flint_19": [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 10.0, -10.0, 12.0, -12.0, 14.0, -14.0],
    "flint_20": [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0],
    "flint_21": [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 4.0, -4.0],
    "flint_22": [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 6.0, -6.0],
    "flint_23": [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0],
    "flint_24": [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 1.25, -1.25, 1.5, -1.5, 1.75, -1.75],
    "flint_25": [0.0, 0.125, -0.125, 0.25, -0.25, 0.375, -0.375, 0.5, -0.5, 0.625, -0.625, 0.75, -0.75, 0.875, -0.875],
}


def meta_flint_grid_set():
    return {
        name: torch.tensor(values)
        for name, values in META_FLINT_SET.items()
    }


class MetaFlintQuantizer:
    def __init__(self, bit, group_size, quant_config, is_weight):
        self.bit = bit
        self.group_size = group_size
        self.quant_config = quant_config
        self.is_weight = is_weight

        self.quant_grid = None
        self.mode = None
        self.alpha = -1.0

    @torch.no_grad()
    def search(self, weight_ref, input_ref, *, tensor_value, group_size_for_search, is_input, layer_id, layer_name):
        assert torch.isnan(weight_ref).sum() == 0
        assert torch.isnan(input_ref).sum() == 0

        if self.bit > 6:
            mode_list = ["int"]
            quant_grid_set = {}
        else:
            quant_grid_set = meta_flint_grid_set()
            mode_list = list(quant_grid_set.keys())

        org_output = torch.mm(input_ref, weight_ref.T).to(torch.float64)
        lb = self.quant_config["w_low"]
        ub = self.quant_config["w_high"]
        if group_size_for_search > 0:
            lb = 100
            ub = 105

        min_mse = float("inf")
        best_mode = "null"
        best_alpha = -1.0
        best_tensor_deq = None
        mse_cal = nn.MSELoss()

        for mode in mode_list:
            quant_grid = quant_grid_set.get(mode)
            for i in range(lb, ub, 10):
                search_alpha = i * 0.01
                if self.bit > 6:
                    tensor_deq = pseudo_quantize_int(
                        tensor_value,
                        n_bit=self.bit,
                        q_group_size=group_size_for_search,
                    )
                else:
                    tensor_deq = quantize_with_grid(
                        tensor_value,
                        quant_grid,
                        group_size=group_size_for_search,
                        alpha=search_alpha,
                    )

                if is_input:
                    deq_output = torch.mm(tensor_deq, weight_ref.T).to(torch.float64)
                else:
                    deq_output = torch.mm(input_ref, tensor_deq.T).to(torch.float64)

                mse = mse_cal(deq_output, org_output)
                if mse < min_mse:
                    min_mse = mse
                    best_mode = mode
                    best_alpha = search_alpha
                    best_tensor_deq = tensor_deq

        self.mode = best_mode
        self.alpha = best_alpha
        self.quant_grid = quant_grid_set.get(best_mode)

        quant_obj = "input" if is_input else "weight"
        print(
            f"layer: {layer_id}, tensor: {layer_name}, {quant_obj} quant, "
            f"best mode: {best_mode}, mse: {min_mse}, alpha: {best_alpha}, "
            f"bit_width: {self.bit}, group_size: {group_size_for_search}"
        )
        return best_tensor_deq

    @torch.no_grad()
    def runtime_quantize_activation(self, input_tensor):
        assert not self.is_weight
        assert self.mode is not None

        if self.bit > 6:
            return pseudo_quantize_int(
                input_tensor,
                n_bit=self.bit,
                q_group_size=self.group_size,
                alpha=1.0,
            )

        if self.group_size == -1:
            return quantize_with_grid(input_tensor, self.quant_grid, -2, alpha=self.alpha)
        return quantize_with_grid(input_tensor, self.quant_grid, self.group_size, alpha=1.0)


class MetaFlintQuantConfig:
    def __init__(self, w_bit, a_bit, group_size, quant_config, layer_id, layer_name):
        self.group_size = group_size
        self.layer_id = layer_id
        self.layer_name = layer_name
        self.w_quant = MetaFlintQuantizer(w_bit, group_size, quant_config, is_weight=True)
        self.a_quant = MetaFlintQuantizer(a_bit, group_size, quant_config, is_weight=False)
        self.calibrated = False

    @torch.no_grad()
    def calibrate_first_forward(self, weight, input_2d):
        deq_weight = self.w_quant.search(
            weight_ref=weight,
            input_ref=input_2d,
            tensor_value=weight,
            group_size_for_search=self.group_size,
            is_input=False,
            layer_id=self.layer_id,
            layer_name=self.layer_name,
        )

        input_group_size = -2 if self.group_size == -1 else self.group_size
        deq_input = self.a_quant.search(
            weight_ref=deq_weight,
            input_ref=input_2d,
            tensor_value=input_2d,
            group_size_for_search=input_group_size,
            is_input=True,
            layer_id=self.layer_id,
            layer_name=self.layer_name,
        )

        self.calibrated = True
        return deq_weight, deq_input


class MetaFlint_Linear(nn.Module):
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

        self.quant_cfg = MetaFlintQuantConfig(w_bit, a_bit, group_size, quant_config, layer_id, layer_name)

        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer("bias", torch.zeros((out_features,), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, a_bit, group_size, layer_id, layer_name, quant_config=None):
        meta_linear = cls(
            w_bit,
            a_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            quant_config,
            layer_id,
            layer_name,
        )

        meta_linear.weight = linear.weight.data.clone().half()
        if linear.bias is not None:
            meta_linear.bias = linear.bias.clone().half()

        return meta_linear

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        input_2d = x.reshape(-1, x.shape[-1])

        assert torch.isnan(self.weight).sum() == 0
        assert torch.isnan(input_2d).sum() == 0

        if not self.quant_cfg.calibrated:
            deq_weight, deq_input = self.quant_cfg.calibrate_first_forward(self.weight, input_2d)
            self.weight = deq_weight
            print("meta-flint search data type and alpha.")
        else:
            deq_input = self.quant_cfg.a_quant.runtime_quantize_activation(input_2d)

        out = F.linear(deq_input, self.weight)
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
