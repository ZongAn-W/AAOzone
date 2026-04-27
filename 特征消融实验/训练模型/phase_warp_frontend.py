import torch
import torch.nn as nn


UVST_ORDER = ("U", "V", "S", "T")
UVST_TO_ATTR_SUFFIX = {
    "U": "u",
    "V": "v",
    "S": "f",
    "T": "t",
}
UVST_TO_PLOT_LABEL = {
    "U": "U_Wind",
    "V": "V_Wind",
    "S": "Solar_Flux",
    "T": "Temperature",
}

PHASE_WARP_PARAM_NAMES = (
    "w1_o3", "b1_o3", "k1_o3",
    "w1_u", "w2_u", "b1_u", "b2_u", "k1_u", "k2_u",
    "w1_v", "w2_v", "b1_v", "b2_v", "k1_v", "k2_v",
    "w1_f", "w2_f", "b1_f", "b2_f", "k1_f", "k2_f",
    "w1_t", "w2_t", "b1_t", "b2_t", "k1_t", "k2_t",
)


def normalize_active_uvst(active_uvst):
    requested = tuple(active_uvst or ())
    invalid = [name for name in requested if name not in UVST_ORDER]
    if invalid:
        raise ValueError(f"Unsupported UVST channel(s): {invalid}. Expected a subset of {UVST_ORDER}.")
    if len(set(requested)) != len(requested):
        raise ValueError(f"Duplicate UVST channels are not allowed: {requested}")
    return tuple(name for name in UVST_ORDER if name in requested)


class PhaseWarpFrontEnd(nn.Module):
    def __init__(self, spatial_shape=(), active_uvst=()):
        super().__init__()
        self.spatial_shape = tuple(spatial_shape)
        self.active_uvst = normalize_active_uvst(active_uvst)
        self.expected_input_channels = 1 + len(self.active_uvst)
        param_shape = (1, 1, 1, *self.spatial_shape)

        self.w1_o3 = nn.Parameter(torch.zeros(param_shape))
        self.b1_o3 = nn.Parameter(torch.zeros(param_shape))
        self.k1_o3 = nn.Parameter(torch.zeros(param_shape))

        for uvst_name in self.active_uvst:
            suffix = UVST_TO_ATTR_SUFFIX[uvst_name]
            setattr(self, f"w1_{suffix}", nn.Parameter(torch.ones(param_shape)))
            setattr(self, f"w2_{suffix}", nn.Parameter(torch.ones(param_shape)))
            setattr(self, f"b1_{suffix}", nn.Parameter(torch.zeros(param_shape)))
            setattr(self, f"b2_{suffix}", nn.Parameter(torch.zeros(param_shape)))
            setattr(self, f"k1_{suffix}", nn.Parameter(torch.zeros(param_shape)))
            setattr(self, f"k2_{suffix}", nn.Parameter(torch.zeros(param_shape)))

    def _ls_to_radians(self, ls):
        if ls.dim() != 2:
            raise ValueError(f"Expected ls with shape [B, T], got {tuple(ls.shape)}")

        extra_dims = [1] * len(self.spatial_shape)
        return (ls * (torch.pi / 180.0)).view(ls.size(0), ls.size(1), 1, *extra_dims)

    def _get_pair_params(self, uvst_name):
        suffix = UVST_TO_ATTR_SUFFIX[uvst_name]
        return (
            getattr(self, f"w1_{suffix}"),
            getattr(self, f"w2_{suffix}"),
            getattr(self, f"b1_{suffix}"),
            getattr(self, f"b2_{suffix}"),
            getattr(self, f"k1_{suffix}"),
            getattr(self, f"k2_{suffix}"),
        )

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

        for channel_index, uvst_name in enumerate(self.active_uvst, start=1):
            x_var = x[:, :, channel_index:channel_index + 1, ...]
            outputs.extend(self._modulate_pair(x_var, ls_rad, *self._get_pair_params(uvst_name)))

        return torch.cat(outputs, dim=2)

    def get_plot_configs(self):
        configs = [
            ("O3", self.w1_o3, self.b1_o3, self.k1_o3),
        ]
        for uvst_name in self.active_uvst:
            suffix = UVST_TO_ATTR_SUFFIX[uvst_name]
            plot_label = UVST_TO_PLOT_LABEL[uvst_name]
            configs.append((f"{plot_label}_Sin", getattr(self, f"w1_{suffix}"), getattr(self, f"b1_{suffix}"), getattr(self, f"k1_{suffix}")))
            configs.append((f"{plot_label}_Cos", getattr(self, f"w2_{suffix}"), getattr(self, f"b2_{suffix}"), getattr(self, f"k2_{suffix}")))
        return configs
