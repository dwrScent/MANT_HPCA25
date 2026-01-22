import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_grid import generate_quant_grid
from .quant_func import quantize_with_grid, pseudo_quantize_int
class ANTQuantizer:
    """
    Hold calibrated params (mode / alpha / quant_grid) and provide:
      - search()  : replicate original quant_grid() behavior (MSE search)
      - runtime_quantize_act(): replicate original forward() else-branch behavior for activation
      - finalize_weight_deq(): replicate original "group_size > -1 => alpha forced to 1.0" weight path
    """
    def __init__(self, bit, group_size, quant_config, is_weight: bool):
        self.bit = bit
        self.group_size = group_size
        self.quant_config = quant_config
        self.is_weight = is_weight

        # calibrated params
        self.quant_grid = None          # selected grid (quant_grid_set[mode])
        self.quant_grid_set = None      # grid set produced by generate_quant_grid(...)
        self.mode = None
        self.alpha = -1.0

    def _assert_no_nan(self, tensor, name="tensor"):
        assert torch.isnan(tensor).sum() == 0, f"Found NaN in {name}"

    @torch.no_grad()
    def search(self, weight_ref, input_ref, *, group_size_for_search: int, is_input: bool,
               layer_id: int, layer_name: str):
        """
        Strictly replicate your original quant_grid(self, n_bit, weight, input, quant_config, group_size, ...)
        except:
          - 'self' is now this quantizer, and we return best_deq_tensor.
          - We still print exactly the same message.
        """
        self._assert_no_nan(weight_ref, "weight_ref")
        self._assert_no_nan(input_ref, "input_ref")

        mode_list = self.quant_config['quant_dtype'].split('-')
        quant_grid_set = generate_quant_grid(n_bit=self.bit, signed=True, quant_dtype=self.quant_config['quant_dtype'])
        self.quant_grid_set = quant_grid_set

        # NOTE: replicate original: mm first, then .to(float64)
        org_output = torch.mm(input_ref, weight_ref.T).to(torch.float64)

        lb = self.quant_config['w_low']
        ub = self.quant_config['w_high']

        tensor_value = input_ref if is_input else weight_ref

        min_mse = float('inf')
        best_mode = 'null'
        best_alpha = -1.0
        best_tensor_deq = None

        mse_cal = nn.MSELoss()

        for mode in mode_list:
            quant_grid = quant_grid_set[mode]
            for i in range(lb, ub, 10):
                search_alpha = i * 0.01

                if self.bit > 6:
                    # replicate original behavior:
                    #   - assert mode == 'int'
                    #   - pseudo_quantize_int DOES NOT receive alpha here
                    assert mode == 'int'
                    tensor_deq = pseudo_quantize_int(
                        tensor_value,
                        n_bit=self.bit,
                        q_group_size=group_size_for_search
                    )
                else:
                    tensor_deq = quantize_with_grid(
                        tensor_value,
                        quant_grid,
                        group_size=group_size_for_search,
                        alpha=search_alpha
                    )

                if is_input:
                    deq_output = torch.mm(tensor_deq, weight_ref.T).to(torch.float64)
                else:
                    deq_output = torch.mm(input_ref, tensor_deq.T).to(torch.float64)

                mse = mse_cal(deq_output, org_output)

                # replicate original: strictly "<" (not <=)
                if mse < min_mse:
                    min_mse = mse
                    best_mode = mode
                    best_alpha = search_alpha
                    best_tensor_deq = tensor_deq

        # commit calibrated params
        self.mode = best_mode
        self.alpha = best_alpha
        self.quant_grid = quant_grid_set[best_mode]

        obj_name = "input" if is_input else "weight"
        print(
            f"layer: {layer_id}, tensor: {layer_name}, {obj_name} quant, "
            f"best mode: {best_mode}, mse: {min_mse}, alpha: {best_alpha}, bit_width: {self.bit}"
        )
        return best_tensor_deq

    @torch.no_grad()
    def finalize_weight_deq(self, original_weight):
        """
        Replicate original forward() first-pass weight handling:
          after search(), if group_size > -1:
            quant_grid_set = generate_quant_grid(self.w_bit, quant_dtype=self.weight_mode)
            deq_weight = quantize_with_grid(self.weight, quant_grid_set[self.weight_mode], self.group_size, alpha=1.0)
        """
        assert self.is_weight, "finalize_weight_deq is only for weight quantizer"
        assert self.mode is not None, "Weight quantizer not searched yet"
        if self.group_size > -1:
            # IMPORTANT: replicate original call signature (no signed=True here)
            qset = generate_quant_grid(self.bit, quant_dtype=self.mode)
            return quantize_with_grid(original_weight, qset[self.mode], self.group_size, alpha=1.0)
        else:
            # group_size == -1 => keep search result (per-channel)
            return None

    @torch.no_grad()
    def runtime_quantize_activation(self, input_tensor):
        """
        Replicate original forward() else-branch activation quantization.
        This MUST use module's group_size semantics:
          - if group_size == -1 => tensor-wise (use -2) and use stored alpha/mode/grid
          - if group_size > 0  => group-wise and force alpha=1.0; grid may be regenerated
        """
        assert not self.is_weight, "runtime_quantize_activation is only for activation quantizer"
        assert self.mode is not None, "Activation quantizer not calibrated yet"

        org_shape = input_tensor.shape
        inp = input_tensor

        if self.bit > 6:
            if self.group_size == -1:
                deq = pseudo_quantize_int(inp, n_bit=self.bit, q_group_size=-2, alpha=self.alpha)
            elif self.group_size > 0:
                deq = pseudo_quantize_int(inp, n_bit=self.bit, q_group_size=self.group_size, alpha=1.0)
            else:
                raise NotImplementedError('Not supported yet')
        else:
            if self.group_size == -1:
                # replicate original: use stored input_quant_grid, group_size=-2, alpha=self.input_alpha
                deq = quantize_with_grid(inp, self.quant_grid, -2, alpha=self.alpha)
            elif self.group_size > 0:
                # replicate original: regenerate quant_grid_set with quant_dtype=self.input_mode, alpha=1.0
                qset = generate_quant_grid(self.bit, quant_dtype=self.mode)
                deq = quantize_with_grid(inp, qset[self.mode], self.group_size, alpha=1.0)
            else:
                raise NotImplementedError('Not supported yet')

        return deq.reshape(org_shape)

class ANTQuantConfig:
    """
    Owns weight & activation quantizers and reproduces original 'first inference does search' flow.
    """
    def __init__(self, w_bit, a_bit, group_size, quant_config, layer_id, layer_name):
        self.quant_config = quant_config
        self.layer_id = layer_id
        self.layer_name = layer_name

        self.w_quant = ANTQuantizer(w_bit, group_size, quant_config, is_weight=True)
        self.a_quant = ANTQuantizer(a_bit, group_size, quant_config, is_weight=False)

        self.calibrated = False

    @torch.no_grad()
    def calibrate_first_forward(self, weight, input_2d):
        """
        Replicate original first-forward behavior:
          1) search weight using group_size=self.group_size
          2) if group_size > -1 => recompute weight with alpha=1.0 using quantize_with_grid(...)
          3) search input tensor-wise (group_size=-2) using the finalized deq_weight as reference
          4) return (final_deq_weight, searched_deq_input_for_this_forward)
        """
        # 1) search weight
        searched_w_deq = self.w_quant.search(
            weight_ref=weight,
            input_ref=input_2d,
            group_size_for_search=self.w_quant.group_size,
            is_input=False,
            layer_id=self.layer_id,
            layer_name=self.layer_name
        )

        # 2) finalize weight if needed (group_size > -1 => alpha forced to 1.0)
        final_w_deq = searched_w_deq
        maybe_final = self.w_quant.finalize_weight_deq(weight)
        if maybe_final is not None:
            final_w_deq = maybe_final

        # 3) search input tensor-wise (group_size=-2), exactly as original
        searched_inp_deq = self.a_quant.search(
            weight_ref=final_w_deq,
            input_ref=input_2d,
            group_size_for_search=-2,
            is_input=True,
            layer_id=self.layer_id,
            layer_name=self.layer_name
        )

        self.calibrated = True
        return final_w_deq, searched_inp_deq


class ANT_Linear(nn.Module):
    """
    Same external behavior as your original ANT_Linear, but forward() is slim.
    """
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

        self.quant_cfg = ANTQuantConfig(w_bit, a_bit, group_size, quant_config, layer_id, layer_name)

        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features,), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, a_bit, group_size, layer_id, layer_name, quant_config=None):
        ant_linear = cls(
            w_bit, a_bit, group_size,
            linear.in_features, linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            quant_config,
            layer_id,
            layer_name
        )

        ant_linear.weight = linear.weight.data.clone().half()
        if linear.bias is not None:
            ant_linear.bias = linear.bias.clone().half()

        # replicate original behavior: if w_bit > 6 force quant_dtype='int'
        if w_bit > 6:
            ant_linear.quant_config['quant_dtype'] = 'int'

        return ant_linear

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        input_2d = x.reshape(-1, x.shape[-1])

        assert torch.isnan(self.weight).sum() == 0
        assert torch.isnan(input_2d).sum() == 0

        # First forward: search (weight + input) and use searched deq_input for THIS forward
        if not self.quant_cfg.calibrated:
            deq_weight, deq_input = self.quant_cfg.calibrate_first_forward(self.weight, input_2d)
            self.weight = deq_weight
            print("ant search data type and alpha.")
        else:
            # Subsequent forwards: runtime quantization for activation
            deq_input = self.quant_cfg.a_quant.runtime_quantize_activation(input_2d)

        # replicate original: F.linear without bias then add bias
        out = F.linear(deq_input, self.weight)
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
