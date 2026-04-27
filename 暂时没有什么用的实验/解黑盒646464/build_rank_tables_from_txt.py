import csv
import re
from pathlib import Path


EXPERIMENT_TAG = "PredRNNv2_PhaseWarp_SyncVerification"


def parse_float(value):
    value = value.strip()
    if value.endswith("%"):
        return float(value[:-1])
    return float(value)


def parse_key_value_block(lines, start_idx):
    data = {}
    idx = start_idx
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            break
        if " = " not in line:
            break
        key, value = [part.strip() for part in line.split("=", 1)]
        data[key] = parse_float(value)
        idx += 1
    return data, idx


def parse_fixed_table(lines, start_idx, columns):
    rows = []
    idx = start_idx
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            break
        parts = line.split()
        if len(parts) != len(columns):
            break
        row = {}
        for column, value in zip(columns, parts):
            if column.endswith("_rank"):
                row[column] = int(value)
            elif column in {"feature_name", "component"}:
                row[column] = value
            else:
                row[column] = float(value)
        rows.append(row)
        idx += 1
    return rows, idx


def parse_summary_file(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    feature_name = None
    synthetic_stats = {}
    feature_table_rows = []
    extra_table_rows_1 = []
    extra_table_rows_2 = []
    branch_rows = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Feature:"):
            feature_name = stripped.split(":", 1)[1].strip()
        elif stripped == "Synthetic variable generation:":
            synthetic_stats, _ = parse_key_value_block(lines, idx + 1)
        elif stripped == "Feature-level comparison of |W|, |B| and |K| (sin/cos combined):":
            feature_table_rows, _ = parse_fixed_table(
                lines,
                idx + 2,
                [
                    "feature_name",
                    "mean_abs_W",
                    "mean_abs_W_rank",
                    "mean_abs_B",
                    "mean_abs_B_rank",
                    "mean_abs_K",
                    "mean_abs_K_rank",
                ],
            )
        elif stripped == "Additional channel metrics for |B| and |K|:":
            extra_table_rows_1, next_idx = parse_fixed_table(
                lines,
                idx + 2,
                [
                    "feature_name",
                    "median_abs_B",
                    "median_abs_K",
                    "std_abs_B",
                    "std_abs_K",
                    "rms_abs_B",
                    "rms_abs_K",
                ],
            )
            extra_table_rows_2, _ = parse_fixed_table(
                lines,
                next_idx + 2,
                [
                    "feature_name",
                    "p99_abs_B",
                    "p99_abs_K",
                    "frac_abs_B_gt_0.10",
                    "frac_abs_K_gt_0.10",
                    "frac_abs_B_gt_0.20",
                    "frac_abs_K_gt_0.20",
                ],
            )
        elif stripped == "Branch-level comparison of |W|, |B| and |K|:":
            branch_rows, _ = parse_fixed_table(
                lines,
                idx + 2,
                ["feature_name", "component", "mean_abs_W", "mean_abs_B", "mean_abs_K"],
            )
            break

    if not feature_name:
        raise ValueError(f"Failed to parse feature name from {path}")

    merged_feature_rows = {}
    for source_rows in (feature_table_rows, extra_table_rows_1, extra_table_rows_2):
        for row in source_rows:
            feature = row["feature_name"]
            merged_feature_rows.setdefault(feature, {"feature_name": feature}).update(row)

    return {
        "source_path": str(path),
        "feature_name": feature_name,
        "synthetic_stats": synthetic_stats,
        "feature_rows": list(merged_feature_rows.values()),
        "branch_rows": branch_rows,
    }


def parse_training_log(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    performance = {}
    pfi_rows = []

    metric_patterns = {
        "rmse": re.compile(r"^RMSE:\s*([0-9.]+)$"),
        "mae": re.compile(r"^MAE\s*:\s*([0-9.]+)$"),
        "r2": re.compile(r"^R2\s*:\s*([0-9.]+)$"),
        "filtered_mape_pct": re.compile(r"^Filtered MAPE \(>0\.1\):\s*([0-9.]+)%$"),
        "smape_pct": re.compile(r"^SMAPE:\s*([0-9.]+)%$"),
        "baseline_rmse": re.compile(r"^Baseline RMSE:\s*([0-9.]+)$"),
    }
    pfi_pattern = re.compile(
        r"^Feature \[(?P<feature>.+?)\] -> permuted RMSE: (?P<permuted>[0-9.]+), increase: (?P<increase>[0-9.]+)$"
    )

    for raw_line in lines:
        line = raw_line.strip()
        for key, pattern in metric_patterns.items():
            match = pattern.match(line)
            if match:
                performance[key] = float(match.group(1))
        pfi_match = pfi_pattern.match(line)
        if pfi_match:
            pfi_rows.append(
                {
                    "feature_name": pfi_match.group("feature"),
                    "permuted_rmse": float(pfi_match.group("permuted")),
                    "increase_rmse": float(pfi_match.group("increase")),
                }
            )

    return performance, pfi_rows


def assign_ranks(rows, metric_name, rank_name, reverse=True):
    ordered = sorted(rows, key=lambda row: row[metric_name], reverse=reverse)
    for rank_idx, row in enumerate(ordered, start=1):
        row[rank_name] = rank_idx


def filter_sync_rows(rows):
    return [row for row in rows if row["feature_name"].startswith("V_sync")]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, columns):
    if not rows:
        return "_No data_\n"

    labels = [label for _, label in columns]
    widths = [len(label) for label in labels]
    for row in rows:
        for idx, (key, _) in enumerate(columns):
            value = row.get(key, "")
            if isinstance(value, float):
                text = f"{value:.6f}"
            else:
                text = str(value)
            widths[idx] = max(widths[idx], len(text))

    def format_row(values):
        return "| " + " | ".join(str(value).ljust(width) for value, width in zip(values, widths)) + " |"

    header = format_row(labels)
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        body.append(format_row(values))
    return "\n".join([header, divider, *body]) + "\n"


def main():
    current_dir = Path(__file__).resolve().parent
    base_dir = current_dir.parent.parent
    result_dir = base_dir / "models" / "训练结果"
    train_log_dir = base_dir / "models" / "训练过程"

    summary_paths = sorted(result_dir.glob(f"{EXPERIMENT_TAG}_*_SyncSummary.txt"))
    if not summary_paths:
        raise FileNotFoundError(f"No summary txt files found in {result_dir}")

    parsed_summaries = [parse_summary_file(path) for path in summary_paths]
    common_feature_rows = parsed_summaries[0]["feature_rows"]
    common_branch_rows = parsed_summaries[0]["branch_rows"]

    generation_rows = []
    for summary in parsed_summaries:
        generation_rows.append(
            {
                "feature_name": summary["feature_name"],
                **summary["synthetic_stats"],
            }
        )

    assign_ranks(generation_rows, "noise_std", "rank_noise_std_asc", reverse=False)
    assign_ranks(generation_rows, "train_corr_with_o3", "rank_train_corr_desc", reverse=True)

    all_feature_rows = [dict(row) for row in common_feature_rows]
    feature_rank_metrics = [
        "mean_abs_W",
        "mean_abs_B",
        "mean_abs_K",
        "median_abs_B",
        "median_abs_K",
        "std_abs_B",
        "std_abs_K",
        "rms_abs_B",
        "rms_abs_K",
        "p99_abs_B",
        "p99_abs_K",
        "frac_abs_B_gt_0.10",
        "frac_abs_K_gt_0.10",
        "frac_abs_B_gt_0.20",
        "frac_abs_K_gt_0.20",
    ]
    for metric in feature_rank_metrics:
        assign_ranks(all_feature_rows, metric, f"rank_{metric}_all", reverse=True)

    sync_only_rows = [dict(row) for row in filter_sync_rows(all_feature_rows)]
    for metric in feature_rank_metrics:
        assign_ranks(sync_only_rows, metric, f"rank_{metric}_sync", reverse=True)

    sync_branch_rows = []
    for row in common_branch_rows:
        if row["feature_name"].startswith("V_sync"):
            sync_branch_rows.append(
                {
                    "branch_name": f"{row['feature_name']}_{row['component']}",
                    **row,
                }
            )
    for metric in ("mean_abs_W", "mean_abs_B", "mean_abs_K"):
        assign_ranks(sync_branch_rows, metric, f"rank_{metric}_sync_branch", reverse=True)

    performance, pfi_rows = parse_training_log(train_log_dir / f"{EXPERIMENT_TAG}.txt")
    assign_ranks(pfi_rows, "increase_rmse", "rank_increase_rmse", reverse=True)

    write_csv(result_dir / f"{EXPERIMENT_TAG}_SyncGenerationRanks.csv", generation_rows)
    write_csv(result_dir / f"{EXPERIMENT_TAG}_AllFeatureMetricRanks.csv", all_feature_rows)
    write_csv(result_dir / f"{EXPERIMENT_TAG}_SyncOnlyMetricRanks.csv", sync_only_rows)
    write_csv(result_dir / f"{EXPERIMENT_TAG}_SyncBranchMetricRanks.csv", sync_branch_rows)
    write_csv(result_dir / f"{EXPERIMENT_TAG}_PFIRanks.csv", pfi_rows)
    write_csv(result_dir / f"{EXPERIMENT_TAG}_PerformanceMetrics.csv", [performance])

    report_lines = []
    report_lines.append(f"# {EXPERIMENT_TAG} Rank Tables")
    report_lines.append("")
    report_lines.append("## Model Performance")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            [performance],
            [
                ("rmse", "RMSE"),
                ("mae", "MAE"),
                ("r2", "R2"),
                ("filtered_mape_pct", "FilteredMAPE(%)"),
                ("smape_pct", "SMAPE(%)"),
                ("baseline_rmse", "BaselineRMSE"),
            ],
        )
    )

    report_lines.append("## PFI Ranking")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            sorted(pfi_rows, key=lambda row: row["rank_increase_rmse"]),
            [
                ("feature_name", "Feature"),
                ("increase_rmse", "IncreaseRMSE"),
                ("rank_increase_rmse", "Rank"),
                ("permuted_rmse", "PermutedRMSE"),
            ],
        )
    )

    report_lines.append("## Sync Generation Ranking")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            sorted(generation_rows, key=lambda row: row["rank_train_corr_desc"]),
            [
                ("feature_name", "Feature"),
                ("alpha", "Alpha"),
                ("signal_std", "SignalStd"),
                ("noise_std", "NoiseStd"),
                ("rank_noise_std_asc", "NoiseRankAsc"),
                ("train_corr_with_o3", "TrainCorr"),
                ("rank_train_corr_desc", "CorrRankDesc"),
            ],
        )
    )

    report_lines.append("## All Features Core Metrics")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            sorted(all_feature_rows, key=lambda row: row["rank_mean_abs_B_all"]),
            [
                ("feature_name", "Feature"),
                ("mean_abs_W", "mean|W|"),
                ("rank_mean_abs_W_all", "W_Rank"),
                ("mean_abs_B", "mean|B|"),
                ("rank_mean_abs_B_all", "B_Rank"),
                ("mean_abs_K", "mean|K|"),
                ("rank_mean_abs_K_all", "K_Rank"),
                ("median_abs_B", "median|B|"),
                ("median_abs_K", "median|K|"),
            ],
        )
    )

    report_lines.append("## All Features Tail Metrics")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            sorted(all_feature_rows, key=lambda row: row["rank_p99_abs_B_all"]),
            [
                ("feature_name", "Feature"),
                ("p99_abs_B", "p99|B|"),
                ("rank_p99_abs_B_all", "p99B_Rank"),
                ("p99_abs_K", "p99|K|"),
                ("rank_p99_abs_K_all", "p99K_Rank"),
                ("frac_abs_B_gt_0.10", "frac|B|>0.10"),
                ("frac_abs_K_gt_0.10", "frac|K|>0.10"),
            ],
        )
    )

    report_lines.append("## Sync Only Core Metrics")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            sorted(sync_only_rows, key=lambda row: row["rank_mean_abs_B_sync"]),
            [
                ("feature_name", "Feature"),
                ("mean_abs_W", "mean|W|"),
                ("rank_mean_abs_W_sync", "W_RankSync"),
                ("mean_abs_B", "mean|B|"),
                ("rank_mean_abs_B_sync", "B_RankSync"),
                ("mean_abs_K", "mean|K|"),
                ("rank_mean_abs_K_sync", "K_RankSync"),
                ("median_abs_B", "median|B|"),
                ("median_abs_K", "median|K|"),
            ],
        )
    )

    report_lines.append("## Sync Only Tail Metrics")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            sorted(sync_only_rows, key=lambda row: row["rank_p99_abs_B_sync"]),
            [
                ("feature_name", "Feature"),
                ("p99_abs_B", "p99|B|"),
                ("rank_p99_abs_B_sync", "p99B_RankSync"),
                ("p99_abs_K", "p99|K|"),
                ("rank_p99_abs_K_sync", "p99K_RankSync"),
                ("frac_abs_B_gt_0.10", "frac|B|>0.10"),
                ("rank_frac_abs_B_gt_0.10_sync", "fracB>0.10_Rank"),
                ("frac_abs_K_gt_0.10", "frac|K|>0.10"),
                ("rank_frac_abs_K_gt_0.10_sync", "fracK>0.10_Rank"),
            ],
        )
    )

    report_lines.append("## Sync Branch Metrics")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            sorted(sync_branch_rows, key=lambda row: row["rank_mean_abs_B_sync_branch"]),
            [
                ("branch_name", "Branch"),
                ("mean_abs_W", "mean|W|"),
                ("rank_mean_abs_W_sync_branch", "W_Rank"),
                ("mean_abs_B", "mean|B|"),
                ("rank_mean_abs_B_sync_branch", "B_Rank"),
                ("mean_abs_K", "mean|K|"),
                ("rank_mean_abs_K_sync_branch", "K_Rank"),
            ],
        )
    )

    report_path = result_dir / f"{EXPERIMENT_TAG}_RankTables.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved markdown report: {report_path}")
    print(f"Saved CSV tables to: {result_dir}")


if __name__ == "__main__":
    main()
