"""
PhaseWarp module-level ablation runner.

This script keeps the same data protocol and PredRNNv2 backbone used by the
current compare scripts, then changes only the seasonal/PhaseWarp front-end.
It is intended to answer whether the gain comes from Ls features, fixed
Fourier modulation, learnable phase offset, learnable bias, O3 fusion, or
spatially varying parameters.
"""

import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

from phase_warp_frontend import UVST_ORDER, UVST_TO_ATTR_SUFFIX, normalize_active_uvst
from predrnnv2_phasewarp_compare import (
    EarlyStopping,
    Logger,
    SpatioTemporalLSTMCellV2,
    build_grid_dataloaders,
    evaluate_metrics,
    load_aligned_cube,
    set_random_seed,
)


ACTIVE_UVST_SUBSET = ("U", "V", "S", "T")
RUN_VARIANTS = None

SUMMARY_COLUMNS = (
    "variant",
    "description",
    "active_uvst",
    "output_channels",
    "trainable_params",
    "rmse",
    "mae",
    "r2",
    "smape",
    "rmse_delta_vs_raw",
    "mae_delta_vs_raw",
    "r2_delta_vs_raw",
    "smape_delta_vs_raw",
)

VARIANT_SPECS = OrderedDict(
    [
        (
            "raw",
            {
                "kind": "identity",
                "description": "Raw O3+UVST inputs, no explicit Ls or PhaseWarp.",
            },
        ),
        (
            "ls_concat",
            {
                "kind": "ls_concat",
                "description": "Raw inputs plus one normalized Ls channel.",
            },
        ),
        (
            "sincos_concat",
            {
                "kind": "sincos_concat",
                "description": "Raw inputs plus sin(Ls) and cos(Ls) channels.",
            },
        ),
        (
            "fixed_fourier",
            {
                "kind": "fixed_fourier",
                "description": "Fixed O3+sin(Ls) and UVST*sin/cos(Ls), no learnable front-end params.",
            },
        ),
        (
            "phasewarp_full",
            {
                "kind": "phasewarp",
                "description": "Full learnable PhaseWarp: W, B, K, O3 fusion, spatially varying params.",
                "use_b": True,
                "use_k": True,
                "fuse_o3": True,
                "spatially_shared": False,
            },
        ),
        (
            "phasewarp_no_k",
            {
                "kind": "phasewarp",
                "description": "PhaseWarp without learnable K phase-warp terms.",
                "use_b": True,
                "use_k": False,
                "fuse_o3": True,
                "spatially_shared": False,
            },
        ),
        (
            "phasewarp_no_b",
            {
                "kind": "phasewarp",
                "description": "PhaseWarp without learnable B phase-bias terms.",
                "use_b": False,
                "use_k": True,
                "fuse_o3": True,
                "spatially_shared": False,
            },
        ),
        (
            "phasewarp_no_o3_fusion",
            {
                "kind": "phasewarp",
                "description": "PhaseWarp modulates UVST only; O3 history is passed through unchanged.",
                "use_b": True,
                "use_k": True,
                "fuse_o3": False,
                "spatially_shared": False,
            },
        ),
        (
            "phasewarp_shared_spatial",
            {
                "kind": "phasewarp",
                "description": "Full PhaseWarp with one shared W/B/K set across all lat-lon cells.",
                "use_b": True,
                "use_k": True,
                "fuse_o3": True,
                "spatially_shared": True,
            },
        ),
    ]
)


def format_active_uvst(active_uvst):
    active_uvst = normalize_active_uvst(active_uvst)
    return "()" if not active_uvst else ",".join(active_uvst)


def expand_phase_channel(values, spatial_shape):
    return values.view(values.size(0), values.size(1), 1, 1, 1).expand(
        -1,
        -1,
        1,
        spatial_shape[0],
        spatial_shape[1],
    )


class IdentityFrontEnd(nn.Module):
    def __init__(self, raw_input_dim):
        super().__init__()
        self.expected_input_channels = raw_input_dim
        self.output_channels = raw_input_dim

    def forward(self, x, ls):
        if x.size(2) != self.expected_input_channels:
            raise ValueError(f"Expected {self.expected_input_channels} channels, got {x.size(2)}")
        return x


class LsConcatFrontEnd(nn.Module):
    def __init__(self, raw_input_dim, spatial_shape):
        super().__init__()
        self.expected_input_channels = raw_input_dim
        self.output_channels = raw_input_dim + 1
        self.spatial_shape = tuple(spatial_shape)

    def forward(self, x, ls):
        if x.size(2) != self.expected_input_channels:
            raise ValueError(f"Expected {self.expected_input_channels} channels, got {x.size(2)}")
        ls_normalized = (torch.remainder(ls, 360.0) / 180.0) - 1.0
        return torch.cat([x, expand_phase_channel(ls_normalized, self.spatial_shape)], dim=2)


class SinCosConcatFrontEnd(nn.Module):
    def __init__(self, raw_input_dim, spatial_shape):
        super().__init__()
        self.expected_input_channels = raw_input_dim
        self.output_channels = raw_input_dim + 2
        self.spatial_shape = tuple(spatial_shape)

    def forward(self, x, ls):
        if x.size(2) != self.expected_input_channels:
            raise ValueError(f"Expected {self.expected_input_channels} channels, got {x.size(2)}")
        ls_rad = ls * (torch.pi / 180.0)
        sin_channel = expand_phase_channel(torch.sin(ls_rad), self.spatial_shape)
        cos_channel = expand_phase_channel(torch.cos(ls_rad), self.spatial_shape)
        return torch.cat([x, sin_channel, cos_channel], dim=2)


class FixedFourierFrontEnd(nn.Module):
    def __init__(self, raw_input_dim, active_uvst, spatial_shape):
        super().__init__()
        self.active_uvst = normalize_active_uvst(active_uvst)
        self.expected_input_channels = raw_input_dim
        self.output_channels = 1 + 2 * len(self.active_uvst)
        self.spatial_shape = tuple(spatial_shape)

    def forward(self, x, ls):
        if x.size(2) != self.expected_input_channels:
            raise ValueError(f"Expected {self.expected_input_channels} channels, got {x.size(2)}")
        ls_rad = ls * (torch.pi / 180.0)
        sin_ls = expand_phase_channel(torch.sin(ls_rad), self.spatial_shape)
        cos_ls = expand_phase_channel(torch.cos(ls_rad), self.spatial_shape)

        outputs = [x[:, :, 0:1] + sin_ls]
        for channel_index in range(1, 1 + len(self.active_uvst)):
            x_var = x[:, :, channel_index:channel_index + 1]
            outputs.extend([x_var * sin_ls, x_var * cos_ls])
        return torch.cat(outputs, dim=2)


class ConfigurablePhaseWarpFrontEnd(nn.Module):
    def __init__(
        self,
        spatial_shape,
        active_uvst,
        use_b=True,
        use_k=True,
        fuse_o3=True,
        spatially_shared=False,
    ):
        super().__init__()
        self.spatial_shape = tuple(spatial_shape)
        self.active_uvst = normalize_active_uvst(active_uvst)
        self.expected_input_channels = 1 + len(self.active_uvst)
        self.output_channels = 1 + 2 * len(self.active_uvst)
        self.use_b = use_b
        self.use_k = use_k
        self.fuse_o3 = fuse_o3
        self.spatially_shared = spatially_shared

        param_shape = (1, 1, 1) if spatially_shared else (1, 1, 1, *self.spatial_shape)
        if self.fuse_o3:
            self.w1_o3 = nn.Parameter(torch.zeros(param_shape))
            self._add_optional_phase_param("b1_o3", param_shape, self.use_b)
            self._add_optional_phase_param("k1_o3", param_shape, self.use_k)

        for uvst_name in self.active_uvst:
            suffix = UVST_TO_ATTR_SUFFIX[uvst_name]
            setattr(self, f"w1_{suffix}", nn.Parameter(torch.ones(param_shape)))
            setattr(self, f"w2_{suffix}", nn.Parameter(torch.ones(param_shape)))
            self._add_optional_phase_param(f"b1_{suffix}", param_shape, self.use_b)
            self._add_optional_phase_param(f"b2_{suffix}", param_shape, self.use_b)
            self._add_optional_phase_param(f"k1_{suffix}", param_shape, self.use_k)
            self._add_optional_phase_param(f"k2_{suffix}", param_shape, self.use_k)

    def _add_optional_phase_param(self, name, shape, trainable):
        value = torch.zeros(shape)
        if trainable:
            setattr(self, name, nn.Parameter(value))
        else:
            self.register_buffer(name, value)

    def _ls_to_radians(self, ls):
        if ls.dim() != 2:
            raise ValueError(f"Expected ls with shape [B, T], got {tuple(ls.shape)}")
        return (ls * (torch.pi / 180.0)).view(ls.size(0), ls.size(1), 1, 1, 1)

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
        warp_sin = torch.tanh(k_sin) * torch.cos(ls_rad)
        warp_cos = torch.tanh(k_cos) * torch.sin(ls_rad)
        x_sin = x_var * (w_sin * torch.sin(ls_rad + warp_sin + b_sin))
        x_cos = x_var * (w_cos * torch.cos(ls_rad + warp_cos + b_cos))
        return x_sin, x_cos

    def forward(self, x, ls):
        if x.size(2) != self.expected_input_channels:
            raise ValueError(f"Expected {self.expected_input_channels} channels, got {x.size(2)}")
        ls_rad = self._ls_to_radians(ls)
        o3 = x[:, :, 0:1]

        if self.fuse_o3:
            warp_o3 = torch.tanh(self.k1_o3) * torch.cos(ls_rad)
            o3_out = o3 + self.w1_o3 * torch.sin(ls_rad + warp_o3 + self.b1_o3)
        else:
            o3_out = o3

        outputs = [o3_out]
        for channel_index, uvst_name in enumerate(self.active_uvst, start=1):
            x_var = x[:, :, channel_index:channel_index + 1]
            outputs.extend(self._modulate_pair(x_var, ls_rad, *self._get_pair_params(uvst_name)))
        return torch.cat(outputs, dim=2)


def build_frontend(variant_spec, raw_input_dim, active_uvst, spatial_shape):
    kind = variant_spec["kind"]
    if kind == "identity":
        return IdentityFrontEnd(raw_input_dim)
    if kind == "ls_concat":
        return LsConcatFrontEnd(raw_input_dim, spatial_shape)
    if kind == "sincos_concat":
        return SinCosConcatFrontEnd(raw_input_dim, spatial_shape)
    if kind == "fixed_fourier":
        return FixedFourierFrontEnd(raw_input_dim, active_uvst, spatial_shape)
    if kind == "phasewarp":
        return ConfigurablePhaseWarpFrontEnd(
            spatial_shape=spatial_shape,
            active_uvst=active_uvst,
            use_b=variant_spec.get("use_b", True),
            use_k=variant_spec.get("use_k", True),
            fuse_o3=variant_spec.get("fuse_o3", True),
            spatially_shared=variant_spec.get("spatially_shared", False),
        )
    raise ValueError(f"Unknown variant kind: {kind}")


class PredRNNv2ModuleAblationForecaster(nn.Module):
    def __init__(self, variant_spec, pred_len, lat_size, lon_size, active_uvst, hidden_dims, filter_size=3):
        super().__init__()
        self.pred_len = pred_len
        self.lat_size = lat_size
        self.lon_size = lon_size
        self.active_uvst = normalize_active_uvst(active_uvst)
        self.hidden_dims = list(hidden_dims)
        self.raw_input_dim = 1 + len(self.active_uvst)

        self.frontend = build_frontend(
            variant_spec=variant_spec,
            raw_input_dim=self.raw_input_dim,
            active_uvst=self.active_uvst,
            spatial_shape=(lat_size, lon_size),
        )
        self.model_input_dim = self.frontend.output_channels

        self.cells = nn.ModuleList()
        for layer_idx, hidden_dim in enumerate(self.hidden_dims):
            cur_input_dim = self.model_input_dim if layer_idx == 0 else self.hidden_dims[layer_idx - 1]
            self.cells.append(SpatioTemporalLSTMCellV2(cur_input_dim, hidden_dim, filter_size=filter_size))

        self.forecast_head = nn.Conv2d(self.hidden_dims[-1], pred_len, kernel_size=1)

    def _init_states(self, batch_size, device):
        h_states = []
        c_states = []
        for hidden_dim in self.hidden_dims:
            h_state = torch.zeros(batch_size, hidden_dim, self.lat_size, self.lon_size, device=device)
            c_state = torch.zeros_like(h_state)
            h_states.append(h_state)
            c_states.append(c_state)
        memory = torch.zeros(batch_size, self.hidden_dims[0], self.lat_size, self.lon_size, device=device)
        return h_states, c_states, memory

    def forward(self, x, ls):
        if x.size(2) != self.raw_input_dim:
            raise ValueError(f"Expected {self.raw_input_dim} raw channels, got {x.size(2)}")
        features = self.frontend(x, ls)
        if features.size(2) != self.model_input_dim:
            raise ValueError(f"Expected {self.model_input_dim} model channels, got {features.size(2)}")

        batch_size, seq_len, _, _, _ = features.shape
        h_states, c_states, memory = self._init_states(batch_size, features.device)
        for time_idx in range(seq_len):
            current = features[:, time_idx]
            for layer_idx, cell in enumerate(self.cells):
                h_next, c_next, memory = cell(current, h_states[layer_idx], c_states[layer_idx], memory)
                h_states[layer_idx] = h_next
                c_states[layer_idx] = c_next
                current = h_next
        return self.forecast_head(h_states[-1])


def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def train_and_evaluate_variant(
    variant_name,
    variant_spec,
    active_uvst,
    train_loader,
    test_loader,
    device,
    y_std,
    y_mean,
    lat_size,
    lon_size,
    horizon,
    hidden_dims,
    filter_size,
    epochs,
    learning_rate,
    early_stopping_patience,
    results_dir,
):
    display_name = f"PredRNNv2_ModuleAblation_{variant_name}"
    print(f"\n[Experiment] {display_name}")
    print(f"Description: {variant_spec['description']}")

    model = PredRNNv2ModuleAblationForecaster(
        variant_spec=variant_spec,
        pred_len=horizon,
        lat_size=lat_size,
        lon_size=lon_size,
        active_uvst=active_uvst,
        hidden_dims=hidden_dims,
        filter_size=filter_size,
    ).to(device)
    trainable_params = count_trainable_params(model)
    print(f"Front-end output channels: {model.model_input_dim}")
    print(f"Trainable params: {trainable_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()
    checkpoint_path = os.path.join(results_dir, f"{display_name.lower()}_checkpoint.pth")
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=checkpoint_path)

    for epoch_idx in range(epochs):
        model.train()
        train_loss_sum = 0.0
        for xb, lsb, yb in train_loader:
            xb = xb.to(device)
            lsb = lsb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb, lsb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, lsb, yb in test_loader:
                xb = xb.to(device)
                lsb = lsb.to(device)
                yb = yb.to(device)
                pred = model(xb, lsb)
                val_loss_sum += criterion(pred, yb).item()

        avg_val_loss = val_loss_sum / len(test_loader)
        print(
            f"{display_name} | Epoch {epoch_idx + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"{display_name} triggered early stopping.")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    metrics = evaluate_metrics(model, test_loader, device, y_std, y_mean)
    save_path = os.path.join(results_dir, f"{display_name.lower()}.pth")
    torch.save(model.state_dict(), save_path)

    print(f"{display_name} weights saved to: {save_path}")
    print(
        f"{display_name} Metrics | RMSE: {metrics['rmse']:.4f} | "
        f"MAE: {metrics['mae']:.4f} | R^2: {metrics['r2']:.4f} | SMAPE: {metrics['smape']:.2%}"
    )
    return metrics, model.model_input_dim, trainable_params


def build_summary_record(variant_name, variant_spec, active_uvst, metrics, output_channels, trainable_params, raw_metrics):
    if raw_metrics is None:
        rmse_delta = mae_delta = r2_delta = smape_delta = 0.0
    else:
        rmse_delta = raw_metrics["rmse"] - metrics["rmse"]
        mae_delta = raw_metrics["mae"] - metrics["mae"]
        r2_delta = metrics["r2"] - raw_metrics["r2"]
        smape_delta = raw_metrics["smape"] - metrics["smape"]

    return {
        "variant": variant_name,
        "description": variant_spec["description"],
        "active_uvst": format_active_uvst(active_uvst),
        "output_channels": str(output_channels),
        "trainable_params": str(trainable_params),
        "rmse": f"{metrics['rmse']:.6f}",
        "mae": f"{metrics['mae']:.6f}",
        "r2": f"{metrics['r2']:.6f}",
        "smape": f"{metrics['smape']:.6f}",
        "rmse_delta_vs_raw": f"{rmse_delta:.6f}",
        "mae_delta_vs_raw": f"{mae_delta:.6f}",
        "r2_delta_vs_raw": f"{r2_delta:.6f}",
        "smape_delta_vs_raw": f"{smape_delta:.6f}",
    }


def update_summary_file(summary_path, record):
    records = OrderedDict()
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                if line.split("\t")[0] == SUMMARY_COLUMNS[0]:
                    continue
                parts = line.split("\t")
                if len(parts) == len(SUMMARY_COLUMNS):
                    parsed = dict(zip(SUMMARY_COLUMNS, parts))
                    records[parsed["variant"]] = parsed

    records[record["variant"]] = record
    ordered_variant_names = [name for name in VARIANT_SPECS if name in records]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\t".join(SUMMARY_COLUMNS) + "\n")
        for variant_name in ordered_variant_names:
            row = records[variant_name]
            f.write("\t".join(row[column] for column in SUMMARY_COLUMNS) + "\n")


def resolve_variant_names():
    if RUN_VARIANTS is None:
        return tuple(VARIANT_SPECS.keys())

    invalid = [name for name in RUN_VARIANTS if name not in VARIANT_SPECS]
    if invalid:
        raise ValueError(f"Unknown variant(s): {invalid}. Expected one of {tuple(VARIANT_SPECS)}")
    return tuple(RUN_VARIANTS)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logs_dir = os.path.join(base_dir, "models", "训练过程")
    results_dir = os.path.join(base_dir, "models", "训练结果")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, "PredRNNv2_PhaseWarp_ModuleAblation.txt")
    summary_path = os.path.join(results_dir, "predrnnv2_phasewarp_module_ablation_summary.txt")
    original_stdout = sys.stdout
    logger = Logger(log_path)
    sys.stdout = logger

    try:
        seed = 42
        set_random_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        active_uvst = normalize_active_uvst(ACTIVE_UVST_SUBSET)
        variant_names = resolve_variant_names()

        print(f"Training Device: {device}")
        print(f"Active UVST subset: {active_uvst}")
        print(f"Variant count: {len(variant_names)}")
        print(f"Log path: {log_path}")
        print(f"Summary path: {summary_path}")

        window = 3
        horizon = 3
        batch_size = 4
        hidden_dim = 32
        num_layers = 2
        hidden_dims = [hidden_dim] * num_layers
        filter_size = 3
        epochs = 15
        learning_rate = 1e-3
        early_stopping_patience = 5

        feature_cubes, y_raw, ls_raw = load_aligned_cube(base_dir)
        lat_size, lon_size = y_raw.shape[1], y_raw.shape[2]
        train_loader, test_loader, y_mean, y_std, input_keys = build_grid_dataloaders(
            feature_cubes=feature_cubes,
            y_raw=y_raw,
            ls_raw=ls_raw,
            active_uvst=active_uvst,
            window=window,
            horizon=horizon,
            batch_size=batch_size,
        )
        print(f"Input channels before front-end: {input_keys}")

        raw_metrics = None
        for variant_index, variant_name in enumerate(variant_names, start=1):
            set_random_seed(seed)
            variant_spec = VARIANT_SPECS[variant_name]
            print(f"\n[Batch] Starting variant {variant_index}/{len(variant_names)}: {variant_name}")
            metrics, output_channels, trainable_params = train_and_evaluate_variant(
                variant_name=variant_name,
                variant_spec=variant_spec,
                active_uvst=active_uvst,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                y_std=y_std,
                y_mean=y_mean,
                lat_size=lat_size,
                lon_size=lon_size,
                horizon=horizon,
                hidden_dims=hidden_dims,
                filter_size=filter_size,
                epochs=epochs,
                learning_rate=learning_rate,
                early_stopping_patience=early_stopping_patience,
                results_dir=results_dir,
            )

            if variant_name == "raw":
                raw_metrics = metrics

            record = build_summary_record(
                variant_name=variant_name,
                variant_spec=variant_spec,
                active_uvst=active_uvst,
                metrics=metrics,
                output_channels=output_channels,
                trainable_params=trainable_params,
                raw_metrics=raw_metrics,
            )
            update_summary_file(summary_path, record)
            print(f"Summary updated: {summary_path}")

        print("\nAll PhaseWarp module ablation runs completed.")
    finally:
        sys.stdout = original_stdout
        logger.close()


if __name__ == "__main__":
    main()
