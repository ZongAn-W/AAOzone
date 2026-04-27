import torch
import torch.nn as nn


DEFAULT_AUXILIARY_VARIABLE_SPECS = (
    {"key": "u", "display_name": "U_Wind"},
    {"key": "v", "display_name": "V_Wind"},
    {"key": "t", "display_name": "Temperature"},
    {"key": "f", "display_name": "Solar_Flux"},
)


def _build_param_names(auxiliary_variable_specs):
    param_names = ["w1_o3", "b1_o3", "k1_o3"]
    for spec in auxiliary_variable_specs:
        key = spec["key"]
        param_names.extend(
            [
                f"w1_{key}",
                f"w2_{key}",
                f"b1_{key}",
                f"b2_{key}",
                f"k1_{key}",
                f"k2_{key}",
            ]
        )
    return tuple(param_names)


PHASE_WARP_PARAM_NAMES = _build_param_names(DEFAULT_AUXILIARY_VARIABLE_SPECS)


class PhaseWarpFrontEnd(nn.Module):
    def __init__(self, spatial_shape=(), auxiliary_variable_specs=None):
        super().__init__()
        self.spatial_shape = tuple(spatial_shape)
        self.auxiliary_variable_specs = tuple(
            auxiliary_variable_specs or DEFAULT_AUXILIARY_VARIABLE_SPECS
        )
        self.expected_input_channels = 1 + len(self.auxiliary_variable_specs)
        param_shape = (1, 1, 1, *self.spatial_shape)

        self.w1_o3 = nn.Parameter(torch.zeros(param_shape))
        self.b1_o3 = nn.Parameter(torch.zeros(param_shape))
        self.k1_o3 = nn.Parameter(torch.zeros(param_shape))

        for spec in self.auxiliary_variable_specs:
            key = spec["key"]
            setattr(self, f"w1_{key}", nn.Parameter(torch.ones(param_shape)))
            setattr(self, f"w2_{key}", nn.Parameter(torch.ones(param_shape)))
            setattr(self, f"b1_{key}", nn.Parameter(torch.zeros(param_shape)))
            setattr(self, f"b2_{key}", nn.Parameter(torch.zeros(param_shape)))
            setattr(self, f"k1_{key}", nn.Parameter(torch.zeros(param_shape)))
            setattr(self, f"k2_{key}", nn.Parameter(torch.zeros(param_shape)))

    def get_parameter_names(self):
        return _build_param_names(self.auxiliary_variable_specs)

    def get_output_channels(self):
        return 1 + 2 * len(self.auxiliary_variable_specs)

    def _ls_to_radians(self, ls):
        if ls.dim() != 2:
            raise ValueError(f"Expected ls with shape [B, T], got {tuple(ls.shape)}")

        extra_dims = [1] * len(self.spatial_shape)
        return (ls * (torch.pi / 180.0)).view(ls.size(0), ls.size(1), 1, *extra_dims)

    def _modulate_pair(self, x_var, ls_rad, w_sin, w_cos, b_sin, b_cos, k_sin, k_cos):
        k_safe_sin = torch.tanh(k_sin)
        k_safe_cos = torch.tanh(k_cos)
        warp_sin = k_safe_sin * torch.cos(ls_rad)
        warp_cos = k_safe_cos * torch.sin(ls_rad)

        x_sin = x_var * (w_sin * torch.sin(ls_rad + warp_sin + b_sin))
        x_cos = x_var * (w_cos * torch.cos(ls_rad + warp_cos + b_cos))
        return x_sin, x_cos

    def fuse_o3(self, o3, ls):
        ls_rad = self._ls_to_radians(ls) if ls.dim() == 2 else ls
        k_safe_o3 = torch.tanh(self.k1_o3)
        warp_o3 = k_safe_o3 * torch.cos(ls_rad)
        return o3 + (self.w1_o3 * torch.sin(ls_rad + warp_o3 + self.b1_o3))

    def forward(self, x, ls):
        if x.size(2) != self.expected_input_channels:
            raise ValueError(
                f"Expected {self.expected_input_channels} raw input channels, got {x.size(2)}"
            )

        ls_rad = self._ls_to_radians(ls)
        o3 = x[:, :, 0:1, ...]
        outputs = [self.fuse_o3(o3, ls_rad)]

        for channel_idx, spec in enumerate(self.auxiliary_variable_specs, start=1):
            key = spec["key"]
            x_var = x[:, :, channel_idx:channel_idx + 1, ...]
            outputs.extend(
                self._modulate_pair(
                    x_var,
                    ls_rad,
                    getattr(self, f"w1_{key}"),
                    getattr(self, f"w2_{key}"),
                    getattr(self, f"b1_{key}"),
                    getattr(self, f"b2_{key}"),
                    getattr(self, f"k1_{key}"),
                    getattr(self, f"k2_{key}"),
                )
            )

        return torch.cat(outputs, dim=2)

    def get_plot_configs(self):
        configs = [("O3", self.w1_o3, self.b1_o3, self.k1_o3)]
        for spec in self.auxiliary_variable_specs:
            key = spec["key"]
            display_name = spec["display_name"]
            configs.extend(
                [
                    (
                        f"{display_name}_Sin",
                        getattr(self, f"w1_{key}"),
                        getattr(self, f"b1_{key}"),
                        getattr(self, f"k1_{key}"),
                    ),
                    (
                        f"{display_name}_Cos",
                        getattr(self, f"w2_{key}"),
                        getattr(self, f"b2_{key}"),
                        getattr(self, f"k2_{key}"),
                    ),
                ]
            )
        return configs

    def get_auxiliary_param_triplets(self):
        triplets = {}
        for spec in self.auxiliary_variable_specs:
            key = spec["key"]
            triplets[spec["display_name"]] = {
                "sin": (
                    getattr(self, f"w1_{key}"),
                    getattr(self, f"b1_{key}"),
                    getattr(self, f"k1_{key}"),
                ),
                "cos": (
                    getattr(self, f"w2_{key}"),
                    getattr(self, f"b2_{key}"),
                    getattr(self, f"k2_{key}"),
                ),
            }
        return triplets
