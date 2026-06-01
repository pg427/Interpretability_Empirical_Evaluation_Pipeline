from pathlib import Path
import json
import math
from scipy.stats import spearmanr, wilcoxon
import numpy as np
import os
import pandas as pd


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def spearman_classes_vs_ria(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    ria_metric="ARIA_mean",
    alpha=0.05,
    output_filename="ria_classes_spearman.json",
):
    """
    H2.3: Datasets with a higher number of classes exhibit lower interpretability.

    Computes Spearman correlation between:
        x = number of classes
        y = RIA metric, default ARIA_mean
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [
            "iris",
            "wine",
            "breast_cancer",
            "german_credit",
            "darwin",
            "sepsis",
            "yeast",
        ]

    dataset_classes = {
        "iris": 3,
        "wine": 3,
        "breast_cancer": 2,
        "german_credit": 2,
        "darwin": 2,
        "sepsis": 2,
        "yeast": 10,
    }

    rows = []

    for ds in datasets:
        ria_path = base_dir / ds / f"{ds}_ria.json"

        if not ria_path.exists():
            print(f"Skipping {ds}: RIA file not found at {ria_path}")
            continue

        ria_data = load_json(ria_path)
        agg = ria_data.get("aggregations", {})

        n_classes = dataset_classes.get(ds)
        ria_value = agg.get(ria_metric)

        if n_classes is None or ria_value is None:
            print(f"Skipping {ds}: missing n_classes or {ria_metric}")
            continue

        rows.append({
            "dataset": ds,
            "n_classes": int(n_classes),
            ria_metric: float(ria_value),
        })

    if len(rows) < 3:
        raise ValueError("At least 3 datasets are required for Spearman correlation.")

    x = [r["n_classes"] for r in rows]
    y = [r[ria_metric] for r in rows]

    rho, p_value = spearmanr(x, y)

    rho = float(rho) if not math.isnan(rho) else None
    p_value = float(p_value) if not math.isnan(p_value) else None

    is_significant = False if p_value is None else p_value < alpha

    if p_value is None:
        significance_statement = (
            "Statistical significance could not be determined because the p-value is undefined."
        )
    elif is_significant:
        significance_statement = f"The result is statistically significant (p = {p_value:.4f} < {alpha})."
    else:
        significance_statement = f"The result is not statistically significant (p = {p_value:.4f} ≥ {alpha})."

    if rho is None:
        direction_statement = (
            "The direction of the relationship could not be determined because Spearman rho is undefined."
        )
        hypothesis_statement = "This does not provide sufficient evidence to support H2.3."
    elif rho < 0:
        direction_statement = (
            "A negative correlation indicates that datasets with more classes tend to have lower RIA values."
        )
        hypothesis_statement = (
            "This supports H2.3."
            if is_significant
            else "This shows the expected negative direction, but does not provide statistically significant evidence to support H2.3."
        )
    else:
        direction_statement = (
            "A positive correlation indicates that datasets with more classes tend to have higher RIA values."
        )
        hypothesis_statement = "This does not support H2.3."

    result = {
        "hypothesis": "H2.3",
        "description": "Spearman correlation between number of classes and RIA.",
        "ria_metric": ria_metric,
        "n_datasets": len(rows),
        "spearman_rho": rho,
        "p_value": p_value,
        "alpha": alpha,
        "is_statistically_significant": bool(is_significant),
        "interpretation": {
            "significance": significance_statement,
            "direction": direction_statement,
            "hypothesis_support": hypothesis_statement,
        },
        "rows": rows,
    }

    output_path = stats_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result

def spearman_features_vs_ria(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    ria_metric="ARIA_mean",
    alpha=0.05,
    output_filename="ria_spearman.json",
):
    """
    H2.1: Datasets with fewer features exhibit higher interpretability.

    Computes Spearman correlation between:
        x = number of features
        y = RIA metric, default ARIA_mean

    Expects files:
        trained_models/<dataset>/<dataset>_ria.json
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [
            "iris",
            "wine",
            "breast_cancer",
            "german_credit",
            "darwin",
            "sepsis",
            "yeast",
        ]

    rows = []

    for ds in datasets:
        ria_path = base_dir / ds / f"{ds}_ria.json"

        if not ria_path.exists():
            print(f"Skipping {ds}: RIA file not found at {ria_path}")
            continue

        ria_data = load_json(ria_path)
        agg = ria_data.get("aggregations", {})

        n_features = agg.get("n_features")
        ria_value = agg.get(ria_metric)

        if n_features is None or ria_value is None:
            print(f"Skipping {ds}: missing n_features or {ria_metric}")
            continue

        rows.append({
            "dataset": ds,
            "n_features": int(n_features),
            ria_metric: float(ria_value),
        })

    if len(rows) < 3:
        raise ValueError("At least 3 datasets are required for Spearman correlation.")

    x = [r["n_features"] for r in rows]
    y = [r[ria_metric] for r in rows]

    rho, p_value = spearmanr(x, y)

    rho = float(rho) if not math.isnan(rho) else None
    p_value = float(p_value) if not math.isnan(p_value) else None

    is_significant = (
        False if p_value is None else p_value < alpha
    )

    if p_value is None:
        significance_statement = "Statistical significance could not be determined because the p-value is undefined."
    elif is_significant:
        significance_statement = f"The result is statistically significant (p = {p_value:.4f} < {alpha})."
    else:
        significance_statement = f"The result is not statistically significant (p = {p_value:.4f} ≥ {alpha})."

    if rho is None:
        direction_statement = "The direction of the relationship could not be determined because Spearman rho is undefined."
        hypothesis_statement = "This does not provide sufficient evidence to support H2.1."
    elif rho < 0:
        direction_statement = "A negative correlation indicates that datasets with more features tend to have lower RIA values."
        hypothesis_statement = (
            "This supports H2.1."
            if is_significant
            else "This shows the expected negative direction, but does not provide statistically significant evidence to support H2.1."
        )
    else:
        direction_statement = "A positive correlation indicates that datasets with more features tend to have higher RIA values."
        hypothesis_statement = "This does not support H2.1."

    result = {
        "hypothesis": "H2.1",
        "description": "Spearman correlation between number of features and RIA.",
        "ria_metric": ria_metric,
        "n_datasets": len(rows),
        "spearman_rho": rho,
        "p_value": p_value,
        "alpha": alpha,
        "is_statistically_significant": bool(is_significant),
        "interpretation": {
            "significance": significance_statement,
            "direction": direction_statement,
            "hypothesis_support": hypothesis_statement,
        },
        "rows": rows,
    }

    output_path = stats_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result

def _mean_soc_for_method(soc_data, method):
    values = [float(x["soc"]) for x in soc_data["by_method"][method]]
    return float(np.mean(values))


def wilcoxon_h1_1_soc(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    alpha=0.05,
    output_filename="soc_h1_1_wilcoxon.json",
):
    """
    H1.1: Interpretability measures will assign higher interpretability to
    inherently transparent models such as Decision Trees and Case-Based Reasoning
    compared to neural network-based models such as MLP and DNN.

    Since SOC is lower = more interpretable, support for H1.1 means:
        DT SOC < MLP SOC
        DT SOC < DNN SOC
        CBR SOC < MLP SOC
        CBR SOC < DNN SOC

    Uses paired dataset-level mean SOC values across datasets.
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [
            "iris",
            "wine",
            "breast_cancer",
            "german_credit",
            "sepsis",
            "darwin",
            "yeast",
        ]

    comparisons = [
        ("dt", "mlp"),
        ("dt", "dnn"),
        ("cbr", "mlp"),
        ("cbr", "dnn"),
    ]

    rows = []
    by_comparison = {f"{a}_vs_{b}": [] for a, b in comparisons}

    for ds in datasets:
        soc_path = base_dir / ds / f"{ds}_soc_all_methods.json"

        if not soc_path.exists():
            print(f"Skipping {ds}: SOC file not found at {soc_path}")
            continue

        soc_data = load_json(soc_path)

        for method_a, method_b in comparisons:
            if method_a not in soc_data["by_method"] or method_b not in soc_data["by_method"]:
                print(f"Skipping {ds} {method_a} vs {method_b}: missing method")
                continue

            soc_a = _mean_soc_for_method(soc_data, method_a)
            soc_b = _mean_soc_for_method(soc_data, method_b)

            supports_expected_direction = soc_a < soc_b

            row = {
                "dataset": ds,
                "method_a": method_a,
                "method_b": method_b,
                "method_a_mean_soc": soc_a,
                "method_b_mean_soc": soc_b,
                "difference_a_minus_b": soc_a - soc_b,
                "supports_expected_direction": bool(supports_expected_direction),
            }

            by_comparison[f"{method_a}_vs_{method_b}"].append(row)
            rows.append(row)

    results = {}

    for comparison_name, comp_rows in by_comparison.items():
        if len(comp_rows) < 3:
            results[comparison_name] = {
                "error": "At least 3 paired datasets are required for Wilcoxon signed-rank test.",
                "n_datasets": len(comp_rows),
                "rows": comp_rows,
            }
            continue

        x = [r["method_a_mean_soc"] for r in comp_rows]
        y = [r["method_b_mean_soc"] for r in comp_rows]

        # alternative="less" tests whether method_a SOC < method_b SOC
        stat, p_value = wilcoxon(x, y, alternative="less", zero_method="wilcox")

        stat = float(stat) if not math.isnan(stat) else None
        p_value = float(p_value) if not math.isnan(p_value) else None

        is_significant = False if p_value is None else p_value < alpha
        win_count = sum(r["supports_expected_direction"] for r in comp_rows)

        if p_value is None:
            significance_statement = (
                "Statistical significance could not be determined because the p-value is undefined."
            )
        elif is_significant:
            significance_statement = (
                f"The result is statistically significant (p = {p_value:.4f} < {alpha})."
            )
        else:
            significance_statement = (
                f"The result is not statistically significant (p = {p_value:.4f} ≥ {alpha})."
            )

        if win_count == len(comp_rows):
            direction_statement = (
                "The expected direction holds for all datasets: method_a has lower SOC than method_b."
            )
        elif win_count > len(comp_rows) / 2:
            direction_statement = (
                "The expected direction holds for a majority of datasets, but not all."
            )
        else:
            direction_statement = (
                "The expected direction does not hold for a majority of datasets."
            )

        hypothesis_statement = (
            "This supports H1.1 for this comparison."
            if is_significant and win_count > len(comp_rows) / 2
            else "This does not provide sufficient evidence to support H1.1 for this comparison."
        )

        results[comparison_name] = {
            "test": "Wilcoxon signed-rank test",
            "alternative": "less",
            "direction_tested": "method_a_mean_soc < method_b_mean_soc",
            "n_datasets": len(comp_rows),
            "wilcoxon_statistic": stat,
            "p_value": p_value,
            "alpha": alpha,
            "is_statistically_significant": bool(is_significant),
            "win_count": int(win_count),
            "total_pairs": int(len(comp_rows)),
            "interpretation": {
                "significance": significance_statement,
                "direction": direction_statement,
                "hypothesis_support": hypothesis_statement,
            },
            "rows": comp_rows,
        }

    result = {
        "hypothesis": "H1.1",
        "description": (
            "Wilcoxon signed-rank tests comparing SOC values for transparent models "
            "Decision Tree and CBR against neural network models MLP and DNN. "
            "Because lower SOC indicates higher interpretability, the expected direction is "
            "transparent model SOC < neural model SOC."
        ),
        "alpha": alpha,
        "comparisons": results,
        "all_rows": rows,
    }

    output_path = stats_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result

def wilcoxon_h1_2_soc(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    alpha=0.05,
    output_filename="soc_h1_2_wilcoxon.json",
):
    """
    H1.2: Interpretability measures will assign higher interpretability to
    simpler variants of a method compared to more complex variants within
    the same family, such as assigning higher interpretability to Decision
    Trees compared to XGBoost.

    In this work, ProtoPNet is treated as a more complex, neural/prototype-based
    variant of Case-Based Reasoning because its interpretability intuition is
    inspired by classical case-based and prototype-based reasoning.

    Since SOC is lower = more interpretable, support for H1.2 means:
        DT SOC < XGB SOC
        CBR SOC < ProtoPNet SOC
        MLP SOC < DNN SOC
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [
            "iris",
            "wine",
            "breast_cancer",
            "german_credit",
            "darwin",
            "sepsis",
            "yeast",
        ]

    comparisons = [
        ("dt", "xgb"),
        ("cbr", "proto"),
        ("mlp", "dnn"),
    ]

    rows = []
    by_comparison = {f"{a}_vs_{b}": [] for a, b in comparisons}

    for ds in datasets:
        soc_path = base_dir / ds / f"{ds}_soc_all_methods.json"

        if not soc_path.exists():
            print(f"Skipping {ds}: SOC file not found at {soc_path}")
            continue

        soc_data = load_json(soc_path)

        for method_a, method_b in comparisons:
            if method_a not in soc_data["by_method"] or method_b not in soc_data["by_method"]:
                print(f"Skipping {ds} {method_a} vs {method_b}: missing method")
                continue

            soc_a = _mean_soc_for_method(soc_data, method_a)
            soc_b = _mean_soc_for_method(soc_data, method_b)

            supports_expected_direction = soc_a < soc_b

            row = {
                "dataset": ds,
                "method_a": method_a,
                "method_b": method_b,
                "method_a_mean_soc": soc_a,
                "method_b_mean_soc": soc_b,
                "difference_a_minus_b": soc_a - soc_b,
                "supports_expected_direction": bool(supports_expected_direction),
            }

            by_comparison[f"{method_a}_vs_{method_b}"].append(row)
            rows.append(row)

    results = {}

    for comparison_name, comp_rows in by_comparison.items():
        if len(comp_rows) < 3:
            results[comparison_name] = {
                "error": "At least 3 paired datasets are required for Wilcoxon signed-rank test.",
                "n_datasets": len(comp_rows),
                "rows": comp_rows,
            }
            continue

        x = [r["method_a_mean_soc"] for r in comp_rows]
        y = [r["method_b_mean_soc"] for r in comp_rows]

        # alternative="less" tests whether simpler method SOC < complex method SOC
        stat, p_value = wilcoxon(x, y, alternative="less", zero_method="wilcox")

        stat = float(stat) if not math.isnan(stat) else None
        p_value = float(p_value) if not math.isnan(p_value) else None

        is_significant = False if p_value is None else p_value < alpha
        win_count = sum(r["supports_expected_direction"] for r in comp_rows)

        if p_value is None:
            significance_statement = (
                "Statistical significance could not be determined because the p-value is undefined."
            )
        elif is_significant:
            significance_statement = f"The result is statistically significant (p = {p_value:.4f} < {alpha})."
        else:
            significance_statement = f"The result is not statistically significant (p = {p_value:.4f} ≥ {alpha})."

        if win_count == len(comp_rows):
            direction_statement = (
                "The expected direction holds for all datasets: the simpler method has lower SOC than the more complex method."
            )
        elif win_count > len(comp_rows) / 2:
            direction_statement = (
                "The expected direction holds for a majority of datasets, but not all."
            )
        else:
            direction_statement = (
                "The expected direction does not hold for a majority of datasets."
            )

        hypothesis_statement = (
            "This supports H1.2 for this comparison."
            if is_significant and win_count > len(comp_rows) / 2
            else "This does not provide sufficient evidence to support H1.2 for this comparison."
        )

        results[comparison_name] = {
            "test": "Wilcoxon signed-rank test",
            "alternative": "less",
            "direction_tested": "method_a_mean_soc < method_b_mean_soc",
            "n_datasets": len(comp_rows),
            "wilcoxon_statistic": stat,
            "p_value": p_value,
            "alpha": alpha,
            "is_statistically_significant": bool(is_significant),
            "win_count": int(win_count),
            "total_pairs": int(len(comp_rows)),
            "interpretation": {
                "significance": significance_statement,
                "direction": direction_statement,
                "hypothesis_support": hypothesis_statement,
            },
            "rows": comp_rows,
        }

    result = {
        "hypothesis": "H1.2",
        "description": (
            "Wilcoxon signed-rank tests comparing SOC values for simpler model variants "
            "against more complex variants within the same method family. In this work, "
            "ProtoPNet is treated as a more complex neural/prototype-based variant of CBR. "
            "Because lower SOC indicates higher interpretability, the expected direction is "
            "simpler method SOC < complex method SOC."
        ),
        "alpha": alpha,
        "comparisons": results,
        "all_rows": rows,
    }

    output_path = stats_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result

def wilcoxon_h1_3_soc(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    alpha=0.05,
    output_filename="soc_h1_3_wilcoxon.json",
):
    """
    H1.3: Interpretability measures will assign ProtoPNet an intermediate
    level of interpretability: less interpretable than classical CBR, but
    more interpretable than standard neural network models such as MLP and DNN.

    Since SOC is lower = more interpretable, support for H1.3 means:
        CBR SOC < ProtoPNet SOC
        ProtoPNet SOC < MLP SOC
        ProtoPNet SOC < DNN SOC
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [
            "iris",
            "wine",
            "breast_cancer",
            "german_credit",
            "darwin",
            "sepsis",
            "yeast",
        ]

    comparisons = [
        ("dt", "proto"),
        ("cbr", "proto"),    # CBR should have lower SOC than ProtoPNet
        ("proto", "mlp"),   # ProtoPNet should have lower SOC than MLP
        ("proto", "dnn"),   # ProtoPNet should have lower SOC than DNN
    ]

    rows = []
    by_comparison = {f"{a}_vs_{b}": [] for a, b in comparisons}

    for ds in datasets:
        soc_path = base_dir / ds / f"{ds}_soc_all_methods.json"

        if not soc_path.exists():
            print(f"Skipping {ds}: SOC file not found at {soc_path}")
            continue

        soc_data = load_json(soc_path)

        for method_a, method_b in comparisons:
            if method_a not in soc_data["by_method"] or method_b not in soc_data["by_method"]:
                print(f"Skipping {ds} {method_a} vs {method_b}: missing method")
                continue

            soc_a = _mean_soc_for_method(soc_data, method_a)
            soc_b = _mean_soc_for_method(soc_data, method_b)

            supports_expected_direction = soc_a < soc_b

            row = {
                "dataset": ds,
                "method_a": method_a,
                "method_b": method_b,
                "method_a_mean_soc": soc_a,
                "method_b_mean_soc": soc_b,
                "difference_a_minus_b": soc_a - soc_b,
                "supports_expected_direction": bool(supports_expected_direction),
            }

            by_comparison[f"{method_a}_vs_{method_b}"].append(row)
            rows.append(row)

    results = {}

    for comparison_name, comp_rows in by_comparison.items():
        if len(comp_rows) < 3:
            results[comparison_name] = {
                "error": "At least 3 paired datasets are required for Wilcoxon signed-rank test.",
                "n_datasets": len(comp_rows),
                "rows": comp_rows,
            }
            continue

        x = [r["method_a_mean_soc"] for r in comp_rows]
        y = [r["method_b_mean_soc"] for r in comp_rows]

        # alternative="less" tests whether method_a SOC < method_b SOC
        stat, p_value = wilcoxon(x, y, alternative="less", zero_method="wilcox")

        stat = float(stat) if not math.isnan(stat) else None
        p_value = float(p_value) if not math.isnan(p_value) else None

        is_significant = False if p_value is None else p_value < alpha
        win_count = sum(r["supports_expected_direction"] for r in comp_rows)

        if p_value is None:
            significance_statement = (
                "Statistical significance could not be determined because the p-value is undefined."
            )
        elif is_significant:
            significance_statement = f"The result is statistically significant (p = {p_value:.4f} < {alpha})."
        else:
            significance_statement = f"The result is not statistically significant (p = {p_value:.4f} ≥ {alpha})."

        if win_count == len(comp_rows):
            direction_statement = (
                "The expected direction holds for all datasets: method_a has lower SOC than method_b."
            )
        elif win_count > len(comp_rows) / 2:
            direction_statement = (
                "The expected direction holds for a majority of datasets, but not all."
            )
        else:
            direction_statement = (
                "The expected direction does not hold for a majority of datasets."
            )

        hypothesis_statement = (
            "This supports H1.3 for this comparison."
            if is_significant and win_count > len(comp_rows) / 2
            else "This does not provide sufficient evidence to support H1.3 for this comparison."
        )

        results[comparison_name] = {
            "test": "Wilcoxon signed-rank test",
            "alternative": "less",
            "direction_tested": "method_a_mean_soc < method_b_mean_soc",
            "n_datasets": len(comp_rows),
            "wilcoxon_statistic": stat,
            "p_value": p_value,
            "alpha": alpha,
            "is_statistically_significant": bool(is_significant),
            "win_count": int(win_count),
            "total_pairs": int(len(comp_rows)),
            "interpretation": {
                "significance": significance_statement,
                "direction": direction_statement,
                "hypothesis_support": hypothesis_statement,
            },
            "rows": comp_rows,
        }

    result = {
        "hypothesis": "H1.3",
        "description": (
            "Wilcoxon signed-rank tests evaluating whether ProtoPNet occupies an "
            "intermediate SOC position: higher SOC than CBR, but lower SOC than "
            "MLP and DNN. Because lower SOC indicates higher interpretability, "
            "the expected directions are CBR SOC < ProtoPNet SOC, "
            "ProtoPNet SOC < MLP SOC, and ProtoPNet SOC < DNN SOC."
        ),
        "alpha": alpha,
        "comparisons": results,
        "all_rows": rows,
    }

    output_path = stats_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result


def _mean_feature_synergy_for_method(fs_data, method):
    """
    Computes mean Feature Synergy for one method using the synergy_matrix.

    The diagonal is excluded because it represents self-synergy.
    Since the matrix is symmetric, only the upper-triangular off-diagonal
    values are used.
    """

    method_data = fs_data["by_method"][method]

    if "synergy_matrix" not in method_data:
        raise KeyError(f"'synergy_matrix' not found for method: {method}")

    matrix = np.asarray(method_data["synergy_matrix"], dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"synergy_matrix for method {method} must be a square matrix."
        )

    upper_tri_indices = np.triu_indices_from(matrix, k=1)
    upper_tri_values = matrix[upper_tri_indices]

    if upper_tri_values.size == 0:
        return 0.0

    return float(np.nanmean(upper_tri_values))


def wilcoxon_h1_1_feature_synergy(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    alpha=0.05,
    output_filename="feature_synergy_h1_1_wilcoxon.json",
):
    """
    H1.1 tested using both:
    1. Mean FS
    2. FS Density

    Lower values indicate higher interpretability.
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [
            "iris", "wine", "breast_cancer", "german_credit",
            "darwin", "sepsis", "yeast",
        ]

    comparisons = [
        ("dt", "mlp"),
        ("dt", "dnn"),
        ("cbr", "mlp"),
        ("cbr", "dnn"),
    ]

    metrics = {
        "mean_feature_synergy": {
            "label": "Mean FS",
            "a_key": "method_a_mean_feature_synergy",
            "b_key": "method_b_mean_feature_synergy",
        },
        "interaction_density": {
            "label": "FS Density",
            "a_key": "method_a_fs_density",
            "b_key": "method_b_fs_density",
        },
    }

    final_results = {}
    final_rows = {}

    for metric_name, metric_info in metrics.items():

        rows = []
        by_comparison = {f"{a}_vs_{b}": [] for a, b in comparisons}

        for ds in datasets:

            fs_path = base_dir / ds / f"{ds}_fs_all_methods.json"

            if not fs_path.exists():
                print(f"Skipping {ds}: Feature Synergy file not found at {fs_path}")
                continue

            fs_data = load_json(fs_path)

            if "by_method" not in fs_data:
                print(f"Skipping {ds}: 'by_method' not found in {fs_path}")
                continue

            for method_a, method_b in comparisons:

                if (
                    method_a not in fs_data["by_method"]
                    or method_b not in fs_data["by_method"]
                ):
                    print(f"Skipping {ds} {method_a} vs {method_b}: missing method")
                    continue

                try:
                    mean_a, density_a = _feature_synergy_statistics(fs_data, method_a)
                    mean_b, density_b = _feature_synergy_statistics(fs_data, method_b)

                    if metric_name == "mean_feature_synergy":
                        value_a = mean_a
                        value_b = mean_b
                    else:
                        value_a = density_a
                        value_b = density_b

                except Exception as e:
                    print(f"Skipping {ds} {method_a} vs {method_b}: {e}")
                    continue

                supports_expected_direction = value_a < value_b

                row = {
                    "dataset": ds,
                    "metric": metric_info["label"],
                    "method_a": method_a,
                    "method_b": method_b,
                    metric_info["a_key"]: value_a,
                    metric_info["b_key"]: value_b,
                    "difference_a_minus_b": value_a - value_b,
                    "supports_expected_direction": bool(supports_expected_direction),
                }

                by_comparison[f"{method_a}_vs_{method_b}"].append(row)
                rows.append(row)

        results = {}

        for comparison_name, comp_rows in by_comparison.items():

            if len(comp_rows) < 3:
                results[comparison_name] = {
                    "error": "At least 3 paired datasets are required for Wilcoxon signed-rank test.",
                    "n_datasets": len(comp_rows),
                    "rows": comp_rows,
                }
                continue

            x = [r[metric_info["a_key"]] for r in comp_rows]
            y = [r[metric_info["b_key"]] for r in comp_rows]

            try:
                stat, p_value = wilcoxon(
                    x,
                    y,
                    alternative="less",
                    zero_method="wilcox",
                )
            except ValueError as e:
                results[comparison_name] = {
                    "error": str(e),
                    "n_datasets": len(comp_rows),
                    "rows": comp_rows,
                }
                continue

            stat = float(stat) if not math.isnan(stat) else None
            p_value = float(p_value) if not math.isnan(p_value) else None
            is_significant = False if p_value is None else p_value < alpha

            win_count = sum(r["supports_expected_direction"] for r in comp_rows)

            results[comparison_name] = {
                "test": "Wilcoxon signed-rank test",
                "metric": metric_info["label"],
                "alternative": "less",
                "direction_tested": f"{metric_info['a_key']} < {metric_info['b_key']}",
                "n_datasets": len(comp_rows),
                "wilcoxon_statistic": stat,
                "p_value": p_value,
                "alpha": alpha,
                "is_statistically_significant": bool(is_significant),
                "win_count": int(win_count),
                "total_pairs": int(len(comp_rows)),
                "hypothesis_support": bool(is_significant and win_count > len(comp_rows) / 2),
                "rows": comp_rows,
            }

        final_results[metric_info["label"]] = results
        final_rows[metric_info["label"]] = rows

    result = {
        "hypothesis": "H1.1",
        "description": (
            "Wilcoxon signed-rank tests for Feature Synergy using both Mean FS "
            "and FS Density. Lower values indicate higher interpretability."
        ),
        "alpha": alpha,
        "comparisons": final_results,
        "all_rows": final_rows,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result

def wilcoxon_h1_2_feature_synergy(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    alpha=0.05,
    output_filename="feature_synergy_h1_2_wilcoxon.json",
):
    """
    H1.2 tested using both:
    1. Mean FS
    2. FS Density

    Lower values indicate higher interpretability.
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [
            "iris", "wine", "breast_cancer", "german_credit",
            "darwin", "sepsis", "yeast",
        ]

    comparisons = [
        ("dt", "xgb"),
        ("cbr", "proto"),
        ("mlp", "dnn"),
    ]

    metrics = {
        "mean_feature_synergy": {
            "label": "Mean FS",
            "a_key": "method_a_mean_feature_synergy",
            "b_key": "method_b_mean_feature_synergy",
        },
        "interaction_density": {
            "label": "FS Density",
            "a_key": "method_a_fs_density",
            "b_key": "method_b_fs_density",
        },
    }

    final_results = {}
    final_rows = {}

    for metric_name, metric_info in metrics.items():

        rows = []
        by_comparison = {f"{a}_vs_{b}": [] for a, b in comparisons}

        for ds in datasets:

            fs_path = base_dir / ds / f"{ds}_fs_all_methods.json"

            if not fs_path.exists():
                print(f"Skipping {ds}: Feature Synergy file not found at {fs_path}")
                continue

            fs_data = load_json(fs_path)

            if "by_method" not in fs_data:
                print(f"Skipping {ds}: 'by_method' not found in {fs_path}")
                continue

            for method_a, method_b in comparisons:

                if (
                    method_a not in fs_data["by_method"]
                    or method_b not in fs_data["by_method"]
                ):
                    print(f"Skipping {ds} {method_a} vs {method_b}: missing method")
                    continue

                try:
                    mean_a, density_a = _feature_synergy_statistics(fs_data, method_a)
                    mean_b, density_b = _feature_synergy_statistics(fs_data, method_b)

                    if metric_name == "mean_feature_synergy":
                        value_a = mean_a
                        value_b = mean_b
                    else:
                        value_a = density_a
                        value_b = density_b

                except Exception as e:
                    print(f"Skipping {ds} {method_a} vs {method_b}: {e}")
                    continue

                supports_expected_direction = value_a < value_b

                row = {
                    "dataset": ds,
                    "metric": metric_info["label"],
                    "method_a": method_a,
                    "method_b": method_b,
                    metric_info["a_key"]: value_a,
                    metric_info["b_key"]: value_b,
                    "difference_a_minus_b": value_a - value_b,
                    "supports_expected_direction": bool(supports_expected_direction),
                }

                by_comparison[f"{method_a}_vs_{method_b}"].append(row)
                rows.append(row)

        results = {}

        for comparison_name, comp_rows in by_comparison.items():

            if len(comp_rows) < 3:
                results[comparison_name] = {
                    "error": "At least 3 paired datasets are required for Wilcoxon signed-rank test.",
                    "n_datasets": len(comp_rows),
                    "rows": comp_rows,
                }
                continue

            x = [r[metric_info["a_key"]] for r in comp_rows]
            y = [r[metric_info["b_key"]] for r in comp_rows]

            try:
                stat, p_value = wilcoxon(
                    x,
                    y,
                    alternative="less",
                    zero_method="wilcox",
                )
            except ValueError as e:
                results[comparison_name] = {
                    "error": str(e),
                    "n_datasets": len(comp_rows),
                    "rows": comp_rows,
                }
                continue

            stat = float(stat) if not math.isnan(stat) else None
            p_value = float(p_value) if not math.isnan(p_value) else None
            is_significant = False if p_value is None else p_value < alpha

            win_count = sum(r["supports_expected_direction"] for r in comp_rows)

            results[comparison_name] = {
                "test": "Wilcoxon signed-rank test",
                "metric": metric_info["label"],
                "alternative": "less",
                "direction_tested": f"{metric_info['a_key']} < {metric_info['b_key']}",
                "n_datasets": len(comp_rows),
                "wilcoxon_statistic": stat,
                "p_value": p_value,
                "alpha": alpha,
                "is_statistically_significant": bool(is_significant),
                "win_count": int(win_count),
                "total_pairs": int(len(comp_rows)),
                "hypothesis_support": bool(is_significant and win_count > len(comp_rows) / 2),
                "rows": comp_rows,
            }

        final_results[metric_info["label"]] = results
        final_rows[metric_info["label"]] = rows

    result = {
        "hypothesis": "H1.2",
        "description": (
            "Wilcoxon signed-rank tests comparing simpler and more complex "
            "model variants using both Mean FS and FS Density."
        ),
        "alpha": alpha,
        "comparisons": final_results,
        "all_rows": final_rows,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result

def wilcoxon_h1_3_feature_synergy(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    alpha=0.05,
    output_filename="feature_synergy_h1_3_wilcoxon.json",
):
    """
    H1.3 tested using both:
    1. Mean FS
    2. FS Density

    Lower values indicate higher interpretability.
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [
            "iris", "wine", "breast_cancer", "german_credit",
            "darwin", "sepsis", "yeast",
        ]

    comparisons = [
        ("cbr", "proto"),
        ("mlp", "proto"),
        ("dt", "proto"),
        ("proto", "dnn"),
    ]

    metrics = {
        "mean_feature_synergy": {
            "label": "Mean FS",
            "a_key": "method_a_mean_feature_synergy",
            "b_key": "method_b_mean_feature_synergy",
        },
        "interaction_density": {
            "label": "FS Density",
            "a_key": "method_a_fs_density",
            "b_key": "method_b_fs_density",
        },
    }

    final_results = {}
    final_rows = {}

    for metric_name, metric_info in metrics.items():

        rows = []
        by_comparison = {f"{a}_vs_{b}": [] for a, b in comparisons}

        for ds in datasets:

            fs_path = base_dir / ds / f"{ds}_fs_all_methods.json"

            if not fs_path.exists():
                print(f"Skipping {ds}: Feature Synergy file not found at {fs_path}")
                continue

            fs_data = load_json(fs_path)

            if "by_method" not in fs_data:
                print(f"Skipping {ds}: 'by_method' not found in {fs_path}")
                continue

            for method_a, method_b in comparisons:

                if (
                    method_a not in fs_data["by_method"]
                    or method_b not in fs_data["by_method"]
                ):
                    print(f"Skipping {ds} {method_a} vs {method_b}: missing method")
                    continue

                try:
                    mean_a, density_a = _feature_synergy_statistics(fs_data, method_a)
                    mean_b, density_b = _feature_synergy_statistics(fs_data, method_b)

                    if metric_name == "mean_feature_synergy":
                        value_a = mean_a
                        value_b = mean_b
                    else:
                        value_a = density_a
                        value_b = density_b

                except Exception as e:
                    print(f"Skipping {ds} {method_a} vs {method_b}: {e}")
                    continue

                supports_expected_direction = value_a < value_b

                row = {
                    "dataset": ds,
                    "metric": metric_info["label"],
                    "method_a": method_a,
                    "method_b": method_b,
                    metric_info["a_key"]: value_a,
                    metric_info["b_key"]: value_b,
                    "difference_a_minus_b": value_a - value_b,
                    "supports_expected_direction": bool(supports_expected_direction),
                }

                by_comparison[f"{method_a}_vs_{method_b}"].append(row)
                rows.append(row)

        results = {}

        for comparison_name, comp_rows in by_comparison.items():

            if len(comp_rows) < 3:
                results[comparison_name] = {
                    "error": "At least 3 paired datasets are required for Wilcoxon signed-rank test.",
                    "n_datasets": len(comp_rows),
                    "rows": comp_rows,
                }
                continue

            x = [r[metric_info["a_key"]] for r in comp_rows]
            y = [r[metric_info["b_key"]] for r in comp_rows]

            try:
                stat, p_value = wilcoxon(
                    x,
                    y,
                    alternative="less",
                    zero_method="wilcox",
                )
            except ValueError as e:
                results[comparison_name] = {
                    "error": str(e),
                    "n_datasets": len(comp_rows),
                    "rows": comp_rows,
                }
                continue

            stat = float(stat) if not math.isnan(stat) else None
            p_value = float(p_value) if not math.isnan(p_value) else None
            is_significant = False if p_value is None else p_value < alpha

            win_count = sum(r["supports_expected_direction"] for r in comp_rows)

            results[comparison_name] = {
                "test": "Wilcoxon signed-rank test",
                "metric": metric_info["label"],
                "alternative": "less",
                "direction_tested": f"{metric_info['a_key']} < {metric_info['b_key']}",
                "n_datasets": len(comp_rows),
                "wilcoxon_statistic": stat,
                "p_value": p_value,
                "alpha": alpha,
                "is_statistically_significant": bool(is_significant),
                "win_count": int(win_count),
                "total_pairs": int(len(comp_rows)),
                "hypothesis_support": bool(is_significant and win_count > len(comp_rows) / 2),
                "rows": comp_rows,
            }

        final_results[metric_info["label"]] = results
        final_rows[metric_info["label"]] = rows

    result = {
        "hypothesis": "H1.3",
        "description": (
            "Wilcoxon signed-rank tests evaluating ProtoPNet's intermediate "
            "position using both Mean FS and FS Density."
        ),
        "alpha": alpha,
        "comparisons": final_results,
        "all_rows": final_rows,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result

def spearman_h2_1_feature_synergy(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    methods=None,
    alpha=0.05,
    output_filename="feature_synergy_h2_1_spearman.json",
):
    """
    H2.1: Higher number of features should correspond to lower interpretability.

    Tested separately by AI method for:
        1. Mean FS
        2. FS Density
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    dataset_metadata = {
        "iris": {"n_features": 4, "n_classes": 3},
        "wine": {"n_features": 13, "n_classes": 3},
        "breast_cancer": {"n_features": 30, "n_classes": 2},
        "german_credit": {"n_features": 20, "n_classes": 2},
        "darwin": {"n_features": 450, "n_classes": 2},
        "sepsis": {"n_features": 3, "n_classes": 2},
        "yeast": {"n_features": 8, "n_classes": 10},
    }

    if datasets is None:
        datasets = list(dataset_metadata.keys())

    if methods is None:
        methods = ["dt", "xgb", "cbr", "proto", "mlp", "dnn"]

    rows = []

    for ds in datasets:

        if ds not in dataset_metadata:
            print(f"Skipping {ds}: dataset metadata not found.")
            continue

        fs_path = base_dir / ds / f"{ds}_fs_all_methods.json"

        if not fs_path.exists():
            print(f"Skipping {ds}: Feature Synergy file not found at {fs_path}")
            continue

        fs_data = load_json(fs_path)

        if "by_method" not in fs_data:
            print(f"Skipping {ds}: 'by_method' not found in {fs_path}")
            continue

        for method in methods:

            if method not in fs_data["by_method"]:
                print(f"Skipping {ds} {method}: method not found.")
                continue

            try:
                mean_fs, fs_density = _feature_synergy_statistics(fs_data, method)

            except Exception as e:
                print(f"Skipping {ds} {method}: {e}")
                continue

            rows.append({
                "dataset": ds,
                "method": method,
                "n_features": dataset_metadata[ds]["n_features"],
                "mean_feature_synergy": mean_fs,
                "fs_density": fs_density,
            })

    metric_specs = {
        "Mean FS": {
            "row_key": "mean_feature_synergy",
            "direction_tested": "n_features positively correlated with mean_feature_synergy",
        },
        "FS Density": {
            "row_key": "fs_density",
            "direction_tested": "n_features positively correlated with fs_density",
        },
    }

    results_by_aggregate = {}

    for metric_label, spec in metric_specs.items():

        method_results = {}

        for method in methods:

            method_rows = [
                r for r in rows
                if r["method"] == method
            ]

            if len(method_rows) < 3:
                method_results[method] = {
                    "error": "At least 3 datasets are required for Spearman correlation.",
                    "n_observations": len(method_rows),
                    "rows": method_rows,
                }
                continue

            x = [r["n_features"] for r in method_rows]
            y = [r[spec["row_key"]] for r in method_rows]

            rho, p_value = spearmanr(x, y)

            rho = float(rho) if not math.isnan(rho) else None
            p_value = float(p_value) if not math.isnan(p_value) else None

            is_significant = False if p_value is None else p_value < alpha

            if rho is None:
                direction_statement = (
                    "The direction of the relationship could not be determined."
                )
            elif rho > 0:
                direction_statement = (
                    f"For {method}, datasets with more features tend to have higher {metric_label}."
                )
            elif rho < 0:
                direction_statement = (
                    f"For {method}, datasets with more features tend to have lower {metric_label}."
                )
            else:
                direction_statement = (
                    f"For {method}, no monotonic relationship is observed between number of features and {metric_label}."
                )

            if p_value is None:
                significance_statement = (
                    "Statistical significance could not be determined because the p-value is undefined."
                )
            elif is_significant:
                significance_statement = (
                    f"The result is statistically significant (p = {p_value:.4f} < {alpha})."
                )
            else:
                significance_statement = (
                    f"The result is not statistically significant (p = {p_value:.4f} ≥ {alpha})."
                )

            hypothesis_statement = (
                "This supports H2.1 for this method and aggregate."
                if rho is not None and rho > 0 and is_significant
                else "This does not provide sufficient evidence to support H2.1 for this method and aggregate."
            )

            method_results[method] = {
                "test": "Spearman rank correlation",
                "method": method,
                "aggregate": metric_label,
                "direction_tested": spec["direction_tested"],
                "n_observations": len(method_rows),
                "spearman_rho": rho,
                "p_value": p_value,
                "alpha": alpha,
                "is_statistically_significant": bool(is_significant),
                "interpretation": {
                    "direction": direction_statement,
                    "significance": significance_statement,
                    "hypothesis_support": hypothesis_statement,
                },
                "rows": method_rows,
            }

        results_by_aggregate[metric_label] = method_results

    result = {
        "hypothesis": "H2.1",
        "description": (
            "Spearman rank correlation between number of dataset features and "
            "Feature Synergy aggregates, computed separately for each AI method."
        ),
        "alpha": alpha,
        "results_by_aggregate": results_by_aggregate,
        "rows": rows,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result
def spearman_h2_2_feature_synergy(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    methods=None,
    alpha=0.05,
    output_filename="feature_synergy_h2_2_spearman.json",
):
    """
    H2.2: Higher number of classes should correspond to lower interpretability.

    Tested separately by AI method for:
        1. Mean FS
        2. FS Density
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    dataset_metadata = {
        "iris": {"n_features": 4, "n_classes": 3},
        "wine": {"n_features": 13, "n_classes": 3},
        "breast_cancer": {"n_features": 30, "n_classes": 2},
        "german_credit": {"n_features": 20, "n_classes": 2},
        "darwin": {"n_features": 450, "n_classes": 2},
        "sepsis": {"n_features": 3, "n_classes": 2},
        "yeast": {"n_features": 8, "n_classes": 10},
    }

    if datasets is None:
        datasets = list(dataset_metadata.keys())

    if methods is None:
        methods = ["dt", "xgb", "cbr", "proto", "mlp", "dnn"]

    rows = []

    for ds in datasets:

        if ds not in dataset_metadata:
            print(f"Skipping {ds}: dataset metadata not found.")
            continue

        fs_path = base_dir / ds / f"{ds}_fs_all_methods.json"

        if not fs_path.exists():
            print(f"Skipping {ds}: Feature Synergy file not found at {fs_path}")
            continue

        fs_data = load_json(fs_path)

        if "by_method" not in fs_data:
            print(f"Skipping {ds}: 'by_method' not found in {fs_path}")
            continue

        for method in methods:

            if method not in fs_data["by_method"]:
                print(f"Skipping {ds} {method}: method not found.")
                continue

            try:
                mean_fs, fs_density = _feature_synergy_statistics(fs_data, method)

            except Exception as e:
                print(f"Skipping {ds} {method}: {e}")
                continue

            rows.append({
                "dataset": ds,
                "method": method,
                "n_classes": dataset_metadata[ds]["n_classes"],
                "mean_feature_synergy": mean_fs,
                "fs_density": fs_density,
            })

    metric_specs = {
        "Mean FS": {
            "row_key": "mean_feature_synergy",
            "direction_tested": "n_classes positively correlated with mean_feature_synergy",
        },
        "FS Density": {
            "row_key": "fs_density",
            "direction_tested": "n_classes positively correlated with fs_density",
        },
    }

    results_by_aggregate = {}

    for metric_label, spec in metric_specs.items():

        method_results = {}

        for method in methods:

            method_rows = [
                r for r in rows
                if r["method"] == method
            ]

            if len(method_rows) < 3:
                method_results[method] = {
                    "error": "At least 3 datasets are required for Spearman correlation.",
                    "n_observations": len(method_rows),
                    "rows": method_rows,
                }
                continue

            x = [r["n_classes"] for r in method_rows]
            y = [r[spec["row_key"]] for r in method_rows]

            rho, p_value = spearmanr(x, y)

            rho = float(rho) if not math.isnan(rho) else None
            p_value = float(p_value) if not math.isnan(p_value) else None

            is_significant = False if p_value is None else p_value < alpha

            if rho is None:
                direction_statement = (
                    "The direction of the relationship could not be determined."
                )
            elif rho > 0:
                direction_statement = (
                    f"For {method}, datasets with more classes tend to have higher {metric_label}."
                )
            elif rho < 0:
                direction_statement = (
                    f"For {method}, datasets with more classes tend to have lower {metric_label}."
                )
            else:
                direction_statement = (
                    f"For {method}, no monotonic relationship is observed between number of classes and {metric_label}."
                )

            if p_value is None:
                significance_statement = (
                    "Statistical significance could not be determined because the p-value is undefined."
                )
            elif is_significant:
                significance_statement = (
                    f"The result is statistically significant (p = {p_value:.4f} < {alpha})."
                )
            else:
                significance_statement = (
                    f"The result is not statistically significant (p = {p_value:.4f} ≥ {alpha})."
                )

            hypothesis_statement = (
                "This supports H2.2 for this method and aggregate."
                if rho is not None and rho > 0 and is_significant
                else "This does not provide sufficient evidence to support H2.2 for this method and aggregate."
            )

            method_results[method] = {
                "test": "Spearman rank correlation",
                "method": method,
                "aggregate": metric_label,
                "direction_tested": spec["direction_tested"],
                "n_observations": len(method_rows),
                "spearman_rho": rho,
                "p_value": p_value,
                "alpha": alpha,
                "is_statistically_significant": bool(is_significant),
                "interpretation": {
                    "direction": direction_statement,
                    "significance": significance_statement,
                    "hypothesis_support": hypothesis_statement,
                },
                "rows": method_rows,
            }

        results_by_aggregate[metric_label] = method_results

    result = {
        "hypothesis": "H2.2",
        "description": (
            "Spearman rank correlation between number of dataset classes and "
            "Feature Synergy aggregates, computed separately for each AI method."
        ),
        "alpha": alpha,
        "results_by_aggregate": results_by_aggregate,
        "rows": rows,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result

def feature_synergy_structure_analysis(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    methods=None,
    output_filename="feature_synergy_structure_analysis.json",
):
    """
    Feature Synergy structure analysis.

    Evaluates:
    - mean synergy
    - max synergy
    - interaction density
    - number of non-zero interaction pairs
    - strongest interaction pairs
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)

    stats_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [
            "iris",
            "wine",
            "breast_cancer",
            "german_credit",
            "darwin",
            "sepsis",
            "yeast",
        ]

    if methods is None:
        methods = [
            "dt",
            "xgb",
            "cbr",
            "proto",
            "mlp",
            "dnn",
        ]

    rows = []

    for ds in datasets:

        fs_path = (
            base_dir
            / ds
            / f"{ds}_fs_all_methods.json"
        )

        if not fs_path.exists():
            continue

        fs_data = load_json(fs_path)

        if "by_method" not in fs_data:
            continue

        for method in methods:

            if method not in fs_data["by_method"]:
                continue

            method_data = fs_data["by_method"][method]

            if "synergy_matrix" not in method_data:
                continue

            matrix = np.asarray(
                method_data["synergy_matrix"],
                dtype=float,
            )

            if matrix.ndim != 2:
                continue

            upper_tri_indices = np.triu_indices_from(matrix, k=1)

            values = matrix[upper_tri_indices]

            if values.size == 0:
                continue

            nonzero_values = values[values > 0]

            mean_synergy = float(np.nanmean(values))

            max_synergy = float(np.nanmax(values))

            interaction_density = float(
                np.sum(values > 0) / len(values)
            )

            n_nonzero_pairs = int(
                np.sum(values > 0)
            )

            strongest_pair_index = int(
                np.nanargmax(values)
            )

            pair_i = int(
                upper_tri_indices[0][strongest_pair_index]
            )

            pair_j = int(
                upper_tri_indices[1][strongest_pair_index]
            )

            strongest_pair_value = float(
                values[strongest_pair_index]
            )

            rows.append({
                "dataset": ds,
                "method": method,
                "mean_feature_synergy": mean_synergy,
                "max_feature_synergy": max_synergy,
                "interaction_density": interaction_density,
                "n_nonzero_interaction_pairs": n_nonzero_pairs,
                "strongest_feature_pair": [
                    pair_i,
                    pair_j,
                ],
                "strongest_pair_synergy": strongest_pair_value,
            })

    overall_summary = {}

    for method in methods:

        method_rows = [
            r for r in rows
            if r["method"] == method
        ]

        if len(method_rows) == 0:
            continue

        overall_summary[method] = {

            "mean_feature_synergy": float(np.mean([
                r["mean_feature_synergy"]
                for r in method_rows
            ])),

            "mean_max_feature_synergy": float(np.mean([
                r["max_feature_synergy"]
                for r in method_rows
            ])),

            "mean_interaction_density": float(np.mean([
                r["interaction_density"]
                for r in method_rows
            ])),

            "mean_nonzero_interaction_pairs": float(np.mean([
                r["n_nonzero_interaction_pairs"]
                for r in method_rows
            ])),
        }

    result = {
        "analysis": "Feature Synergy Structure Analysis",

        "description": (
            "Analysis of interaction structure density and "
            "distribution across AI methods and datasets."
        ),

        "rows": rows,

        "overall_summary": overall_summary,
    }

    output_path = (
        stats_dir / output_filename
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result


def robustness_value(data, method):
    """
    Extracts the average robustness score for a method.

    Expected robustness JSON structure:
        data["by_method"][method]["avg_robustness"]

    If your key name differs, add it to the possible_keys list.
    """

    if "by_method" not in data:
        return None

    if method not in data["by_method"]:
        return None

    method_data = data["by_method"][method]

    possible_keys = [
        "overall_robustness",
    ]

    for key in possible_keys:
        if key in method_data:
            value = method_data[key]
            if value is not None:
                return float(value)

    raise KeyError(
        f"No robustness score found for method '{method}'. "
        f"Available keys: {list(method_data.keys())}"
    )


def _run_robustness_wilcoxon(
    hypothesis,
    description,
    comparisons,
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    alpha=0.05,
    output_filename="robustness_wilcoxon.json",
):
    """
    Generic Wilcoxon test for robustness hypotheses.

    For Robustness:
        higher robustness -> higher interpretability

    Therefore, expected direction is:
        method_a_robustness > method_b_robustness
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [
            "iris", "wine", "breast_cancer", "german_credit",
            "darwin", "sepsis", "yeast",
        ]

    metric_info = {
        "label": "Robustness",
        "a_key": "method_a_robustness",
        "b_key": "method_b_robustness",
    }

    rows = []
    by_comparison = {f"{a}_vs_{b}": [] for a, b in comparisons}

    for ds in datasets:

        rob_path = base_dir / ds / f"{ds}_rs_all_methods.json"

        if not rob_path.exists():
            print(f"Skipping {ds}: Robustness file not found at {rob_path}")
            continue

        rob_data = load_json(rob_path)

        if "by_method" not in rob_data:
            print(f"Skipping {ds}: 'by_method' not found in {rob_path}")
            continue

        for method_a, method_b in comparisons:

            if (
                method_a not in rob_data["by_method"]
                or method_b not in rob_data["by_method"]
            ):
                print(f"Skipping {ds} {method_a} vs {method_b}: missing method")
                continue

            try:
                value_a = robustness_value(rob_data, method_a)
                value_b = robustness_value(rob_data, method_b)
            except Exception as e:
                print(f"Skipping {ds} {method_a} vs {method_b}: {e}")
                continue

            if value_a is None or value_b is None:
                print(
                    f"Skipping {ds} {method_a} vs {method_b}: "
                    f"{method_a}={value_a}, {method_b}={value_b}"
                )
                continue

            value_a = float(value_a)
            value_b = float(value_b)

            supports_expected_direction = value_a > value_b

            row = {
                "dataset": ds,
                "metric": metric_info["label"],
                "method_a": method_a,
                "method_b": method_b,
                metric_info["a_key"]: value_a,
                metric_info["b_key"]: value_b,
                "difference_a_minus_b": value_a - value_b,
                "supports_expected_direction": bool(supports_expected_direction),
            }

            by_comparison[f"{method_a}_vs_{method_b}"].append(row)
            rows.append(row)

    results = {}

    for comparison_name, comp_rows in by_comparison.items():

        if len(comp_rows) < 3:
            results[comparison_name] = {
                "error": "At least 3 paired datasets are required for Wilcoxon signed-rank test.",
                "n_datasets": len(comp_rows),
                "rows": comp_rows,
            }
            continue

        x = [r[metric_info["a_key"]] for r in comp_rows]
        y = [r[metric_info["b_key"]] for r in comp_rows]

        try:
            stat, p_value = wilcoxon(
                x,
                y,
                alternative="greater",
                zero_method="wilcox",
            )
        except ValueError as e:
            results[comparison_name] = {
                "error": str(e),
                "n_datasets": len(comp_rows),
                "rows": comp_rows,
            }
            continue

        stat = float(stat) if not math.isnan(stat) else None
        p_value = float(p_value) if not math.isnan(p_value) else None
        is_significant = False if p_value is None else p_value < alpha

        win_count = sum(r["supports_expected_direction"] for r in comp_rows)

        results[comparison_name] = {
            "test": "Wilcoxon signed-rank test",
            "metric": metric_info["label"],
            "alternative": "greater",
            "direction_tested": f"{metric_info['a_key']} > {metric_info['b_key']}",
            "n_datasets": len(comp_rows),
            "wilcoxon_statistic": stat,
            "p_value": p_value,
            "alpha": alpha,
            "is_statistically_significant": bool(is_significant),
            "win_count": int(win_count),
            "total_pairs": int(len(comp_rows)),
            "hypothesis_support": bool(
                is_significant and win_count > len(comp_rows) / 2
            ),
            "rows": comp_rows,
        }

    result = {
        "hypothesis": hypothesis,
        "description": description,
        "alpha": alpha,
        "interpretation": "Higher robustness indicates higher interpretability.",
        "comparisons": results,
        "all_rows": rows,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result
def wilcoxon_h1_1_robustness(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    alpha=0.05,
    output_filename="robustness_h1_1_wilcoxon.json",
):
    comparisons = [
        ("dt", "mlp"),
        ("dt", "dnn"),
        ("cbr", "mlp"),
        ("cbr", "dnn"),
    ]

    return _run_robustness_wilcoxon(
        hypothesis="H1.1",
        description=(
            "Wilcoxon signed-rank tests comparing inherently transparent "
            "models against neural network-based models using Robustness."
        ),
        comparisons=comparisons,
        base_dir=base_dir,
        stats_dir=stats_dir,
        datasets=datasets,
        alpha=alpha,
        output_filename=output_filename,
    )
def wilcoxon_h1_2_robustness(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    alpha=0.05,
    output_filename="robustness_h1_2_wilcoxon.json",
):
    comparisons = [
        ("dt", "xgb"),
        ("cbr", "proto"),
        ("mlp", "dnn"),
    ]

    return _run_robustness_wilcoxon(
        hypothesis="H1.2",
        description=(
            "Wilcoxon signed-rank tests comparing simpler and more complex "
            "model variants using Robustness."
        ),
        comparisons=comparisons,
        base_dir=base_dir,
        stats_dir=stats_dir,
        datasets=datasets,
        alpha=alpha,
        output_filename=output_filename,
    )
def wilcoxon_h1_3_robustness(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    alpha=0.05,
    output_filename="robustness_h1_3_wilcoxon.json",
):
    comparisons = [
        ("dt", "proto"),
        ("cbr", "proto"),
        ("proto", "mlp"),
        ("proto", "dnn"),
    ]

    return _run_robustness_wilcoxon(
        hypothesis="H1.3",
        description=(
            "Wilcoxon signed-rank tests evaluating whether ProtoPNet occupies "
            "an intermediate interpretability position using Robustness."
        ),
        comparisons=comparisons,
        base_dir=base_dir,
        stats_dir=stats_dir,
        datasets=datasets,
        alpha=alpha,
        output_filename=output_filename,
    )

def spearman_h2_1_robustness(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    methods=None,
    alpha=0.05,
    output_filename="robustness_h2_1_spearman.json",
):
    """
    H2.1:
    Interpretability measures will assign lower interpretability
    to datasets with a larger number of features.

    For Robustness:
        higher robustness -> higher interpretability

    Expected relationship:
        more features -> lower robustness

    Tested separately by AI method.
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    dataset_metadata = {
        "iris": {"n_features": 4, "n_classes": 3},
        "wine": {"n_features": 13, "n_classes": 3},
        "breast_cancer": {"n_features": 30, "n_classes": 2},
        "german_credit": {"n_features": 20, "n_classes": 2},
        "darwin": {"n_features": 450, "n_classes": 2},
        "sepsis": {"n_features": 3, "n_classes": 2},
        "yeast": {"n_features": 8, "n_classes": 10},
    }

    if datasets is None:
        datasets = list(dataset_metadata.keys())

    if methods is None:
        methods = ["dt", "xgb", "cbr", "proto", "mlp", "dnn"]

    rows = []

    for ds in datasets:

        if ds not in dataset_metadata:
            print(f"Skipping {ds}: dataset metadata not found.")
            continue

        rob_path = base_dir / ds / f"{ds}_rs_all_methods.json"

        if not rob_path.exists():
            print(f"Skipping {ds}: Robustness file not found at {rob_path}")
            continue

        rob_data = load_json(rob_path)

        if "by_method" not in rob_data:
            print(f"Skipping {ds}: 'by_method' not found in {rob_path}")
            continue

        for method in methods:

            if method not in rob_data["by_method"]:
                print(f"Skipping {ds} {method}: method not found.")
                continue

            try:
                robustness_score = robustness_value(rob_data, method)

            except Exception as e:
                print(f"Skipping {ds} {method}: {e}")
                continue

            if robustness_score is None:
                print(f"Skipping {ds} {method}: robustness score is None.")
                continue

            rows.append({
                "dataset": ds,
                "method": method,
                "n_features": dataset_metadata[ds]["n_features"],
                "robustness": float(robustness_score),
            })

    metric_specs = {
        "Robustness": {
            "row_key": "robustness",
            "direction_tested": "n_features negatively correlated with robustness",
        },
    }

    results_by_aggregate = {}

    for metric_label, spec in metric_specs.items():

        method_results = {}

        for method in methods:

            method_rows = [
                r for r in rows
                if r["method"] == method
            ]

            if len(method_rows) < 3:
                method_results[method] = {
                    "error": "At least 3 datasets are required for Spearman correlation.",
                    "n_observations": len(method_rows),
                    "rows": method_rows,
                }
                continue

            x = [r["n_features"] for r in method_rows]
            y = [r[spec["row_key"]] for r in method_rows]

            rho, p_value = spearmanr(x, y)

            rho = float(rho) if not math.isnan(rho) else None
            p_value = float(p_value) if not math.isnan(p_value) else None

            is_significant = False if p_value is None else p_value < alpha

            if rho is None:
                direction_statement = (
                    "The direction of the relationship could not be determined."
                )
            elif rho < 0:
                direction_statement = (
                    f"For {method}, datasets with more features tend to have lower {metric_label}."
                )
            elif rho > 0:
                direction_statement = (
                    f"For {method}, datasets with more features tend to have higher {metric_label}."
                )
            else:
                direction_statement = (
                    f"For {method}, no monotonic relationship is observed between number of features and {metric_label}."
                )

            if p_value is None:
                significance_statement = (
                    "Statistical significance could not be determined because the p-value is undefined."
                )
            elif is_significant:
                significance_statement = (
                    f"The result is statistically significant (p = {p_value:.4f} < {alpha})."
                )
            else:
                significance_statement = (
                    f"The result is not statistically significant (p = {p_value:.4f} ≥ {alpha})."
                )

            hypothesis_statement = (
                "This supports H2.1 for this method."
                if rho is not None and rho < 0 and is_significant
                else "This does not provide sufficient evidence to support H2.1 for this method."
            )

            method_results[method] = {
                "test": "Spearman rank correlation",
                "method": method,
                "aggregate": metric_label,
                "direction_tested": spec["direction_tested"],
                "n_observations": len(method_rows),
                "spearman_rho": rho,
                "p_value": p_value,
                "alpha": alpha,
                "is_statistically_significant": bool(is_significant),
                "hypothesis_support": bool(
                    rho is not None and rho < 0 and is_significant
                ),
                "interpretation": {
                    "direction": direction_statement,
                    "significance": significance_statement,
                    "hypothesis_support": hypothesis_statement,
                },
                "rows": method_rows,
            }

        results_by_aggregate[metric_label] = method_results

    result = {
        "hypothesis": "H2.1",
        "description": (
            "Spearman rank correlation between number of dataset features "
            "and Robustness, computed separately for each AI method."
        ),
        "alpha": alpha,
        "interpretation": (
            "Because higher robustness indicates higher interpretability, "
            "support for H2.1 requires a statistically significant negative "
            "correlation between number of features and robustness."
        ),
        "results_by_aggregate": results_by_aggregate,
        "rows": rows,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result

def spearman_h2_3_robustness(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    alpha=0.05,
    output_filename="robustness_h2_3_spearman.json",
):
    """
    H2.3:
    Interpretability measures will assign lower interpretability
    to datasets with a larger number of classes.

    For Robustness:
        higher robustness -> higher interpretability

    Expected relationship:
        more classes -> lower robustness
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Excluding Sepsis from formal statistical analysis
    dataset_class_counts = {
        "iris": 3,
        "wine": 3,
        "breast_cancer": 2,
        "german_credit": 2,
        "darwin": 2,
        "yeast": 10,
    }

    methods = [
        "dt",
        "xgb",
        "cbr",
        "proto",
        "mlp",
        "dnn",
    ]

    class_counts = []
    robustness_scores = []

    rows = []

    for ds, n_classes in dataset_class_counts.items():

        rob_path = (
            base_dir / ds / f"{ds}_rs_all_methods.json"
        )

        if not rob_path.exists():
            print(f"Skipping {ds}: file not found.")
            continue

        data = load_json(rob_path)

        dataset_scores = []

        for method in methods:

            if method not in data["by_method"]:
                continue

            score = robustness_value(data, method)

            # Skip missing/null values
            if score is None:
                continue

            dataset_scores.append(float(score))

        if len(dataset_scores) == 0:
            print(f"Skipping {ds}: no valid robustness values.")
            continue

        dataset_mean_robustness = np.mean(dataset_scores)

        class_counts.append(n_classes)
        robustness_scores.append(dataset_mean_robustness)

        rows.append({
            "dataset": ds,
            "num_classes": n_classes,
            "mean_robustness": float(dataset_mean_robustness),
        })

    correlation, p_value = spearmanr(
        class_counts,
        robustness_scores
    )

    is_significant = p_value < alpha

    result = {
        "hypothesis": "H2.3",
        "description": (
            "Spearman correlation between number of classes "
            "and mean robustness across datasets."
        ),
        "test": "Spearman rank correlation",
        "expected_relationship": (
            "more classes -> lower robustness"
        ),
        "spearman_correlation": float(correlation),
        "p_value": float(p_value),
        "alpha": alpha,
        "is_statistically_significant": bool(is_significant),
        "hypothesis_support": bool(
            is_significant and correlation < 0
        ),
        "rows": rows,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    return result

def fold_measure_values(data, method, measure):
    vals = []

    for fold in data["by_method"][method]["folds"]:
        if measure == "nf":
            val = fold.get("nf", {}).get("NF", None)
        elif measure == "ias":
            val = fold.get("ias_overall", None)
        elif measure == "mec":
            val = fold.get("mec", None)
        else:
            raise ValueError("measure must be one of: nf, ias, mec")

        if val is not None:
            vals.append(float(val))

    if len(vals) == 0:
        return None

    return sum(vals) / len(vals)


def wilcoxon_h1_1_functional_complexity_all_measures(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    alpha=0.05,
    output_filename="functional_complexity_h1_1_wilcoxon.json",
    exclude_zero_values=True,
    zero_tol=1e-12,
):
    """
    H1.1:
    Interpretability measures will assign higher interpretability
    to inherently transparent models such as Decision Trees and
    Case-Based Reasoning compared to neural network-based models
    such as MLP and DNN.

    For NF, IAS, and MEC:
        lower values -> higher interpretability

    If exclude_zero_values=True:
        Any dataset pair where either method has value 0.00 for the
        current measure is excluded from the Wilcoxon test.
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [
            "iris",
            "wine",
            "breast_cancer",
            "german_credit",
            "darwin",
            "yeast",
            "sepsis",
        ]

    measures = ["nf", "ias", "mec"]

    comparisons = {
        "dt_vs_mlp": ("dt", "mlp"),
        "dt_vs_dnn": ("dt", "dnn"),
        "cbr_vs_mlp": ("cbr", "mlp"),
        "cbr_vs_dnn": ("cbr", "dnn"),
    }

    all_results = {}

    for measure in measures:
        measure_results = {}

        for comparison_name, (transparent_method, neural_method) in comparisons.items():
            transparent_vals = []
            neural_vals = []
            rows_used = []
            rows_excluded = []

            for ds in datasets:
                path = base_dir / ds / f"{ds}_mec_all_methods.json"

                if not path.exists():
                    print(f"Skipping {ds}: file not found.")
                    continue

                data = load_json(path)

                if (
                    transparent_method not in data["by_method"]
                    or neural_method not in data["by_method"]
                ):
                    print(f"Skipping {ds}: missing methods for {comparison_name}")
                    continue

                transparent_score = fold_measure_values(
                    data, transparent_method, measure
                )
                neural_score = fold_measure_values(
                    data, neural_method, measure
                )

                if transparent_score is None or neural_score is None:
                    print(
                        f"Skipping {ds}: "
                        f"{transparent_method}={transparent_score}, "
                        f"{neural_method}={neural_score}"
                    )
                    continue

                transparent_score = float(transparent_score)
                neural_score = float(neural_score)

                row = {
                    "dataset": ds,
                    "transparent_method": transparent_method,
                    "transparent_score": transparent_score,
                    "neural_method": neural_method,
                    "neural_score": neural_score,
                    "difference_transparent_minus_neural": (
                        transparent_score - neural_score
                    ),
                }

                if exclude_zero_values and (
                    abs(transparent_score) <= zero_tol
                    or abs(neural_score) <= zero_tol
                ):
                    row["excluded_from_test"] = True
                    row["exclusion_reason"] = (
                        f"Zero value detected for {measure}; "
                        "treated as undetected/invalid for Wilcoxon testing."
                    )
                    rows_excluded.append(row)
                    continue

                row["excluded_from_test"] = False
                rows_used.append(row)

                transparent_vals.append(transparent_score)
                neural_vals.append(neural_score)

            if len(transparent_vals) < 2:
                measure_results[comparison_name] = {
                    "transparent_method": transparent_method,
                    "neural_method": neural_method,
                    "wilcoxon_statistic": None,
                    "p_value": None,
                    "alpha": alpha,
                    "is_statistically_significant": False,
                    "hypothesis_support": False,
                    "n_valid_pairs": len(transparent_vals),
                    "n_excluded_pairs": len(rows_excluded),
                    "note": (
                        "Not enough valid paired observations for Wilcoxon test "
                        "after excluding zero values."
                    ),
                    "rows_used_for_test": rows_used,
                    "rows_excluded_from_test": rows_excluded,
                }
                continue

            stat, p_value = wilcoxon(
                transparent_vals,
                neural_vals,
                alternative="less"
            )

            measure_results[comparison_name] = {
                "transparent_method": transparent_method,
                "neural_method": neural_method,
                "test": "Wilcoxon signed-rank test",
                "alternative": "transparent value < neural value",
                "wilcoxon_statistic": float(stat),
                "p_value": float(p_value),
                "alpha": alpha,
                "is_statistically_significant": bool(p_value < alpha),
                "hypothesis_support": bool(p_value < alpha),
                "n_valid_pairs": len(transparent_vals),
                "n_excluded_pairs": len(rows_excluded),
                "zero_values_excluded": exclude_zero_values,
                "rows_used_for_test": rows_used,
                "rows_excluded_from_test": rows_excluded,
            }

        all_results[measure] = measure_results

    output = {
        "hypothesis": "H1.1",
        "description": (
            "Wilcoxon signed-rank tests comparing transparent models "
            "(DT, CBR) against neural models (MLP, DNN) for NF, IAS, and MEC."
        ),
        "direction": "lower NF, IAS, and MEC indicate higher interpretability",
        "expected_relationship": "DT/CBR < MLP/DNN",
        "alpha": alpha,
        "zero_values_excluded": exclude_zero_values,
        "zero_exclusion_note": (
            "Zero-valued measure observations were excluded from Wilcoxon testing "
            "because they represent undetected feature utilization, interaction effects, "
            "or main effect complexity rather than meaningful low complexity."
        ),
        "results": all_results,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    return output

def wilcoxon_functional_complexity_all_measures(
    hypothesis_id,
    description,
    expected_relationship,
    comparisons,
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    alpha=0.05,
    output_filename=None,
    exclude_zero_values=True,
    zero_tol=1e-12,
):
    """
    Generic Wilcoxon function for NF, IAS, and MEC.

    For NF, IAS, and MEC:
        lower values -> higher interpretability

    If exclude_zero_values=True:
        Any dataset pair where either method has value 0.00 for the
        current measure is excluded from the Wilcoxon test.
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [
            "iris",
            "wine",
            "breast_cancer",
            "german_credit",
            "darwin",
            "yeast",
            "sepsis",
        ]

    if output_filename is None:
        output_filename = (
            f"functional_complexity_{hypothesis_id.lower().replace('.', '_')}_wilcoxon.json"
        )

    measures = ["nf", "ias", "mec"]
    all_results = {}

    for measure in measures:
        measure_results = {}

        for comparison_name, (method_a, method_b) in comparisons.items():
            vals_a = []
            vals_b = []
            rows = []
            excluded_rows = []

            for ds in datasets:
                path = base_dir / ds / f"{ds}_mec_all_methods.json"

                if not path.exists():
                    print(f"Skipping {ds}: file not found.")
                    continue

                data = load_json(path)

                if (
                    method_a not in data["by_method"]
                    or method_b not in data["by_method"]
                ):
                    print(f"Skipping {ds}: missing methods for {comparison_name}")
                    continue

                score_a = fold_measure_values(data, method_a, measure)
                score_b = fold_measure_values(data, method_b, measure)

                if score_a is None or score_b is None:
                    print(
                        f"Skipping {ds}: "
                        f"{method_a}={score_a}, {method_b}={score_b}"
                    )
                    continue

                score_a = float(score_a)
                score_b = float(score_b)

                row = {
                    "dataset": ds,
                    "method_a": method_a,
                    "score_a": score_a,
                    "method_b": method_b,
                    "score_b": score_b,
                    "difference_method_a_minus_method_b": score_a - score_b,
                }

                # Exclude zero-valued observations from the statistical test
                if exclude_zero_values and (
                    abs(score_a) <= zero_tol or abs(score_b) <= zero_tol
                ):
                    row["excluded_from_test"] = True
                    row["exclusion_reason"] = (
                        f"Zero value detected for {measure}; "
                        "treated as undetected/invalid for Wilcoxon testing."
                    )
                    excluded_rows.append(row)
                    continue

                row["excluded_from_test"] = False
                rows.append(row)

                vals_a.append(score_a)
                vals_b.append(score_b)

            if len(vals_a) < 2:
                measure_results[comparison_name] = {
                    "method_a": method_a,
                    "method_b": method_b,
                    "wilcoxon_statistic": None,
                    "p_value": None,
                    "alpha": alpha,
                    "is_statistically_significant": False,
                    "hypothesis_support": False,
                    "n_valid_pairs": len(vals_a),
                    "n_excluded_pairs": len(excluded_rows),
                    "note": "Not enough valid paired observations for Wilcoxon test after excluding zero values.",
                    "rows_used_for_test": rows,
                    "rows_excluded_from_test": excluded_rows,
                }
                continue

            stat, p_value = wilcoxon(
                vals_a,
                vals_b,
                alternative="less"
            )

            measure_results[comparison_name] = {
                "method_a": method_a,
                "method_b": method_b,
                "test": "Wilcoxon signed-rank test",
                "alternative": "method_a value < method_b value",
                "wilcoxon_statistic": float(stat),
                "p_value": float(p_value),
                "alpha": alpha,
                "is_statistically_significant": bool(p_value < alpha),
                "hypothesis_support": bool(p_value < alpha),
                "n_valid_pairs": len(vals_a),
                "n_excluded_pairs": len(excluded_rows),
                "zero_values_excluded": exclude_zero_values,
                "rows_used_for_test": rows,
                "rows_excluded_from_test": excluded_rows,
            }

        all_results[measure] = measure_results

    output = {
        "hypothesis": hypothesis_id,
        "description": description,
        "direction": "lower NF, IAS, and MEC indicate higher interpretability",
        "expected_relationship": expected_relationship,
        "alpha": alpha,
        "zero_values_excluded": exclude_zero_values,
        "zero_exclusion_note": (
            "Zero-valued measure observations were excluded from Wilcoxon testing "
            "because they represent undetected feature utilization, interaction effects, "
            "or main effect complexity rather than meaningful low complexity."
        ),
        "results": all_results,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    return output

def spearman_h2_1_functional_complexity_all_measures(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    alpha=0.05,
    output_filename="functional_complexity_h2_1_spearman.json",
    exclude_zero_values=True,
    zero_tol=1e-12,
):
    """
    H2.1:
    Interpretability measures will assign lower interpretability
    to datasets with larger numbers of features.

    For NF, IAS, MEC:
        higher values -> lower interpretability

    Expected relationship:
        #features positively correlates with NF, IAS, MEC

    If exclude_zero_values=True:
        For each method-measure pair, datasets where the measure value is 0.00
        are excluded before computing Spearman correlation.
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [
            "iris",
            "wine",
            "breast_cancer",
            "german_credit",
            "darwin",
            "yeast",
            "sepsis",
        ]

    dataset_feature_counts = {
        "iris": 4,
        "wine": 13,
        "breast_cancer": 30,
        "german_credit": 20,
        "darwin": 450,
        "yeast": 8,
        "sepsis": 3,
    }

    measures = ["nf", "ias", "mec"]

    methods = [
        "dt",
        "xgb",
        "cbr",
        "proto",
        "mlp",
        "dnn",
    ]

    all_results = {}

    for measure in measures:
        method_results = {}

        for method in methods:
            feature_counts = []
            measure_values = []
            rows_used = []
            rows_excluded = []

            for ds in datasets:
                path = base_dir / ds / f"{ds}_mec_all_methods.json"

                if not path.exists():
                    print(f"Skipping {ds}: file not found.")
                    continue

                if ds not in dataset_feature_counts:
                    print(f"Skipping {ds}: missing feature count.")
                    continue

                data = load_json(path)

                if method not in data["by_method"]:
                    print(f"Skipping {ds}: missing method {method}")
                    continue

                score = fold_measure_values(data, method, measure)

                if score is None:
                    print(f"Skipping {ds}: score is None for {method}-{measure}")
                    continue

                score = float(score)
                feature_count = dataset_feature_counts[ds]

                row = {
                    "dataset": ds,
                    "num_features": feature_count,
                    "measure_value": score,
                }

                if exclude_zero_values and abs(score) <= zero_tol:
                    row["excluded_from_test"] = True
                    row["exclusion_reason"] = (
                        f"Zero value detected for {measure}; "
                        "treated as undetected/invalid for Spearman testing."
                    )
                    rows_excluded.append(row)
                    continue

                row["excluded_from_test"] = False
                rows_used.append(row)

                feature_counts.append(feature_count)
                measure_values.append(score)

            if len(feature_counts) < 3:
                method_results[method] = {
                    "spearman_correlation": None,
                    "p_value": None,
                    "alpha": alpha,
                    "is_statistically_significant": False,
                    "hypothesis_support": False,
                    "n_valid_datasets": len(feature_counts),
                    "n_excluded_datasets": len(rows_excluded),
                    "note": (
                        "Not enough valid non-zero observations "
                        "for Spearman correlation."
                    ),
                    "rows_used_for_test": rows_used,
                    "rows_excluded_from_test": rows_excluded,
                }
                continue

            corr, p_value = spearmanr(
                feature_counts,
                measure_values,
                alternative="greater",
            )

            method_results[method] = {
                "test": "Spearman rank correlation",
                "alternative": "positive correlation: #features vs measure value",
                "spearman_correlation": float(corr),
                "p_value": float(p_value),
                "alpha": alpha,
                "is_statistically_significant": bool(p_value < alpha),
                "hypothesis_support": bool((corr > 0) and (p_value < alpha)),
                "n_valid_datasets": len(feature_counts),
                "n_excluded_datasets": len(rows_excluded),
                "zero_values_excluded": exclude_zero_values,
                "rows_used_for_test": rows_used,
                "rows_excluded_from_test": rows_excluded,
            }

        all_results[measure] = method_results

    output = {
        "hypothesis": "H2.1",
        "description": (
            "Spearman correlations evaluating whether datasets with larger "
            "numbers of features produce larger NF, IAS, and MEC values."
        ),
        "direction": (
            "larger NF, IAS, and MEC correspond to lower interpretability"
        ),
        "expected_relationship": (
            "#features positively correlates with NF, IAS, MEC"
        ),
        "alpha": alpha,
        "zero_values_excluded": exclude_zero_values,
        "zero_exclusion_note": (
            "Zero-valued measure observations were excluded from Spearman testing "
            "because they represent undetected feature utilization, interaction effects, "
            "or main effect complexity rather than meaningful low complexity."
        ),
        "results": all_results,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    return output

# def spearman_h2_2_functional_complexity_all_measures(
#     base_dir="trained_models",
#     stats_dir="statistical_calculations",
#     datasets=None,
#     alpha=0.05,
#     output_filename="functional_complexity_h2_2_spearman.json",
# ):
#     """
#     H2.3:
#     Interpretability measures will assign lower interpretability
#     to dataset-method pairs involving datasets with a higher number
#     of classes compared to those with fewer classes.
#
#     For NF, IAS, MEC:
#         higher values -> lower interpretability
#
#     Expected relationship:
#         #classes positively correlates with NF, IAS, MEC
#     """
#
#     base_dir = Path(base_dir)
#     stats_dir = Path(stats_dir)
#     stats_dir.mkdir(parents=True, exist_ok=True)
#
#     if datasets is None:
#         datasets = [
#             "iris",
#             "wine",
#             "breast_cancer",
#             "german_credit",
#             "darwin",
#             "yeast",
#         ]
#
#     dataset_class_counts = {
#         "iris": 3,
#         "wine": 3,
#         "breast_cancer": 2,
#         "german_credit": 2,
#         "darwin": 2,
#         "yeast": 10,
#     }
#
#     measures = ["nf", "ias", "mec"]
#     methods = ["dt", "xgb", "cbr", "proto", "mlp", "dnn"]
#
#     all_results = {}
#
#     for measure in measures:
#         method_results = {}
#
#         for method in methods:
#             class_counts = []
#             measure_values = []
#             rows = []
#
#             for ds in datasets:
#                 path = base_dir / ds / f"{ds}_mec_all_methods.json"
#
#                 if not path.exists():
#                     print(f"Skipping {ds}: file not found.")
#                     continue
#
#                 data = load_json(path)
#
#                 if method not in data["by_method"]:
#                     print(f"Skipping {ds}: missing method {method}")
#                     continue
#
#                 score = fold_measure_values(data, method, measure)
#
#                 if score is None:
#                     continue
#
#                 class_count = dataset_class_counts[ds]
#
#                 class_counts.append(class_count)
#                 measure_values.append(score)
#
#                 rows.append({
#                     "dataset": ds,
#                     "num_classes": class_count,
#                     "measure_value": score,
#                 })
#
#             if len(class_counts) < 3:
#                 method_results[method] = {
#                     "spearman_correlation": None,
#                     "p_value": None,
#                     "alpha": alpha,
#                     "is_statistically_significant": False,
#                     "hypothesis_support": False,
#                     "note": "Not enough observations for Spearman correlation.",
#                     "rows": rows,
#                 }
#                 continue
#
#             corr, p_value = spearmanr(
#                 class_counts,
#                 measure_values,
#                 alternative="greater"
#             )
#
#             method_results[method] = {
#                 "spearman_correlation": float(corr),
#                 "p_value": float(p_value),
#                 "alpha": alpha,
#                 "is_statistically_significant": bool(p_value < alpha),
#                 "hypothesis_support": bool((corr > 0) and (p_value < alpha)),
#                 "rows": rows,
#             }
#
#         all_results[measure] = method_results
#
#     output = {
#         "hypothesis": "H2.2",
#         "description": (
#             "Spearman correlations evaluating whether datasets with "
#             "larger numbers of classes produce larger NF, IAS, and MEC values."
#         ),
#         "direction": "larger NF, IAS, and MEC correspond to lower interpretability",
#         "expected_relationship": "#classes positively correlates with NF, IAS, MEC",
#         "alpha": alpha,
#         "results": all_results,
#     }
#
#     output_path = stats_dir / output_filename
#
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(output, f, indent=4)
#
#     return output

def shap_measure_value(data, measure):
    """
    Extract average fold-level value for a SHAP measure.
    """

    measure_key_map = {
        "similarity": [
            "fold_avg_similarity",
            "similarity_total_test_avg",
        ],
        "stability": [
            "fold_avg_stability",
            "stability_score",
            "stability_total_test_avg",
        ],
        "parsimony": [
            "fold_avg_parsimony",
            "total_parsimony_averaged_test",
            "parsimony_total_test_avg",
        ],
        "faithfulness": [
            "fold_avg_faithfulness",
            "faithfulness_score",
            "faithfulness_total_test_avg",
        ],
    }

    vals = []

    if measure not in data:
        return None

    candidate_keys = measure_key_map[measure]

    for fold_entry in data[measure]:
        for key in candidate_keys:
            if key in fold_entry and fold_entry[key] is not None:
                vals.append(float(fold_entry[key]))
                break

    if len(vals) == 0:
        return None

    return float(np.mean(vals))


def wilcoxon_shap_all_measures(
    hypothesis_id,
    description,
    expected_relationship,
    comparisons,
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    datasets=None,
    alpha=0.05,
    output_filename="shap_h1_1_wilcoxon.json",
):
    """
    Generic Wilcoxon analysis for SHAP-based measures.

    Measures:
        similarity
        stability
        parsimony
        faithfulness

    Assumption:
        Higher values -> higher interpretability
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = [
            "iris",
            "wine",
            "breast_cancer",
            "german_credit",
            "darwin",
            "yeast",
            "sepsis",
        ]

    measures = [
        "similarity",
        "stability",
        "parsimony",
        "faithfulness",
    ]

    all_results = {}

    for measure in measures:

        measure_results = {}

        for comparison_name, (method_a, method_b) in comparisons.items():

            vals_a = []
            vals_b = []
            rows = []

            for ds in datasets:

                path_a = (
                    base_dir / ds / f"{ds}_{method_a}_shap_measures.json"
                )

                path_b = (
                    base_dir / ds / f"{ds}_{method_b}_shap_measures.json"
                )

                if not path_a.exists():
                    print(f"Skipping {ds}: missing {path_a}")
                    continue

                if not path_b.exists():
                    print(f"Skipping {ds}: missing {path_b}")
                    continue

                data_a = load_json(path_a)
                data_b = load_json(path_b)

                score_a = shap_measure_value(data_a, measure)
                score_b = shap_measure_value(data_b, measure)

                if score_a is None or score_b is None:
                    print(
                        f"Skipping {ds} for {comparison_name}, {measure}: "
                        f"{method_a}={score_a}, "
                        f"{method_b}={score_b}"
                    )
                    continue

                vals_a.append(float(score_a))
                vals_b.append(float(score_b))

                rows.append({
                    "dataset": ds,
                    "method_a": method_a,
                    "score_a": float(score_a),
                    "method_b": method_b,
                    "score_b": float(score_b),
                    "difference_a_minus_b": float(score_a - score_b),
                })

            if len(vals_a) < 2:

                measure_results[comparison_name] = {
                    "method_a": method_a,
                    "method_b": method_b,
                    "wilcoxon_statistic": None,
                    "p_value": None,
                    "alpha": alpha,
                    "is_statistically_significant": False,
                    "hypothesis_support": False,
                    "note": (
                        "Not enough valid paired observations "
                        "for Wilcoxon test."
                    ),
                    "rows": rows,
                }

                continue

            stat, p_value = wilcoxon(
                vals_a,
                vals_b,
                alternative="greater"
            )

            measure_results[comparison_name] = {
                "method_a": method_a,
                "method_b": method_b,
                "test": "Wilcoxon signed-rank test",
                "alternative": "method_a > method_b",
                "wilcoxon_statistic": float(stat),
                "p_value": float(p_value),
                "alpha": alpha,
                "is_statistically_significant": bool(
                    p_value < alpha
                ),
                "hypothesis_support": bool(
                    p_value < alpha
                ),
                "rows": rows,
            }

        all_results[measure] = measure_results

    output = {
        "hypothesis": hypothesis_id,
        "description": description,
        "direction": (
            "Higher Similarity, Stability, Parsimony, "
            "and Faithfulness correspond to higher "
            "post-hoc interpretability."
        ),
        "expected_relationship": expected_relationship,
        "alpha": alpha,
        "results": all_results,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    return output

def shap_measure_value(data, measure):
    """
    Extract average fold-level value for a SHAP measure.
    """

    measure_key_map = {
        "similarity": [
            "fold_avg_similarity",
            "similarity_total_test_avg",
        ],
        "stability": [
            "fold_avg_stability",
            "stability_score",
            "stability_total_test_avg",
        ],
        "parsimony": [
            "fold_avg_parsimony",
            "total_parsimony_averaged_test",
            "parsimony_total_test_avg",
        ],
        "faithfulness": [
            "fold_avg_faithfulness",
            "faithfulness_score",
            "faithfulness_total_test_avg",
        ],
    }

    vals = []

    if measure not in data:
        return None

    candidate_keys = measure_key_map[measure]

    for fold_entry in data[measure]:

        for key in candidate_keys:

            if key in fold_entry and fold_entry[key] is not None:
                vals.append(float(fold_entry[key]))
                break

    if len(vals) == 0:
        return None

    return float(np.mean(vals))


def spearman_features_vs_shap_measures(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    alpha=0.05,
    output_filename="shap_h2_1_spearman.json",
):
    """
    H2.1:
    Interpretability measures will assign lower interpretability
    to datasets with larger numbers of features.

    Since higher SHAP values correspond to higher interpretability,
    expected relationship:
        number_of_features ↑ -> interpretability ↓
    therefore:
        expected negative Spearman correlation
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)

    stats_dir.mkdir(parents=True, exist_ok=True)

    dataset_features = {
        "iris": 4,
        "wine": 13,
        "breast_cancer": 30,
        "german_credit": 20,
        "darwin": 451,
        "yeast": 8,
        "sepsis": 3,
    }

    methods = [
        "dt",
        "xgb",
        "cbr",
        "proto",
        "mlp",
        "dnn",
    ]

    measures = [
        "similarity",
        "stability",
        "parsimony",
        "faithfulness",
    ]

    all_results = {}

    for measure in measures:

        measure_results = {}

        for method in methods:

            feature_counts = []
            measure_scores = []
            rows = []

            for ds, n_features in dataset_features.items():

                json_path = (
                    base_dir / ds / f"{ds}_{method}_shap_measures.json"
                )

                if not json_path.exists():
                    print(f"Missing file: {json_path}")
                    continue

                data = load_json(json_path)

                score = shap_measure_value(data, measure)

                if score is None:
                    print(
                        f"Skipping {ds}, {method}, {measure}: score=None"
                    )
                    continue

                feature_counts.append(n_features)
                measure_scores.append(score)

                rows.append({
                    "dataset": ds,
                    "method": method,
                    "num_features": n_features,
                    "measure_score": float(score),
                })

            if len(feature_counts) < 3:

                measure_results[method] = {
                    "spearman_correlation": None,
                    "p_value": None,
                    "alpha": alpha,
                    "is_statistically_significant": False,
                    "hypothesis_support": False,
                    "note": (
                        "Not enough observations "
                        "for Spearman correlation."
                    ),
                    "rows": rows,
                }

                continue

            corr, p_value = spearmanr(
                feature_counts,
                measure_scores
            )

            measure_results[method] = {
                "spearman_correlation": float(corr),
                "p_value": float(p_value),
                "alpha": alpha,
                "expected_direction": "negative",
                "is_statistically_significant": bool(
                    p_value < alpha
                ),
                "hypothesis_support": bool(
                    (corr < 0) and (p_value < alpha)
                ),
                "rows": rows,
            }

        all_results[measure] = measure_results

    output = {
        "hypothesis": "H2.1",
        "description": (
            "Spearman correlation between dataset feature count "
            "and SHAP-based interpretability measures."
        ),
        "expected_relationship": (
            "More dataset features should correspond "
            "to lower interpretability."
        ),
        "direction": (
            "Higher Similarity, Stability, Parsimony, "
            "and Faithfulness correspond to higher "
            "post-hoc interpretability."
        ),
        "alpha": alpha,
        "results": all_results,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    return output

def spearman_classes_vs_shap_measures(
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    alpha=0.05,
    output_filename="shap_h2_2_classes_spearman.json",
):
    """
    H2.2 / class-count hypothesis:
    Interpretability measures will assign lower interpretability
    to dataset–method pairs involving datasets with a larger number
    of classes compared to those with fewer classes.

    Since higher SHAP values correspond to higher interpretability,
    expected relationship:
        number_of_classes ↑ -> interpretability ↓

    Therefore:
        expected negative Spearman correlation.
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    dataset_classes = {
        "iris": 3,
        "wine": 3,
        "breast_cancer": 2,
        "german_credit": 2,
        "darwin": 2,
        "yeast": 10,
        "sepsis": 2,
    }

    methods = [
        "dt",
        "xgb",
        "cbr",
        "proto",
        "mlp",
        "dnn",
    ]

    measures = [
        "similarity",
        "stability",
        "parsimony",
        "faithfulness",
    ]

    all_results = {}

    for measure in measures:
        measure_results = {}

        for method in methods:
            class_counts = []
            measure_scores = []
            rows = []

            for ds, n_classes in dataset_classes.items():

                json_path = (
                    base_dir / ds / f"{ds}_{method}_shap_measures.json"
                )

                if not json_path.exists():
                    print(f"Missing file: {json_path}")
                    continue

                data = load_json(json_path)

                score = shap_measure_value(data, measure)

                if score is None:
                    print(
                        f"Skipping {ds}, {method}, {measure}: score=None"
                    )
                    continue

                class_counts.append(n_classes)
                measure_scores.append(score)

                rows.append({
                    "dataset": ds,
                    "method": method,
                    "num_classes": n_classes,
                    "measure_score": float(score),
                })

            if len(class_counts) < 3:
                measure_results[method] = {
                    "spearman_correlation": None,
                    "p_value": None,
                    "alpha": alpha,
                    "expected_direction": "negative",
                    "is_statistically_significant": False,
                    "hypothesis_support": False,
                    "note": (
                        "Not enough observations for Spearman correlation."
                    ),
                    "rows": rows,
                }
                continue

            corr, p_value = spearmanr(
                class_counts,
                measure_scores
            )

            measure_results[method] = {
                "spearman_correlation": float(corr),
                "p_value": float(p_value),
                "alpha": alpha,
                "expected_direction": "negative",
                "is_statistically_significant": bool(p_value < alpha),
                "hypothesis_support": bool(
                    (corr < 0) and (p_value < alpha)
                ),
                "rows": rows,
            }

        all_results[measure] = measure_results

    output = {
        "hypothesis": "H2.2",
        "description": (
            "Spearman correlation between dataset class count "
            "and SHAP-based interpretability measures."
        ),
        "expected_relationship": (
            "More dataset classes should correspond to lower interpretability."
        ),
        "direction": (
            "Higher Similarity, Stability, Parsimony, and Faithfulness "
            "correspond to higher post-hoc interpretability."
        ),
        "alpha": alpha,
        "results": all_results,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    return output


def surrogate_measure_value(data, measure, method_name):
    """
    Extract average surrogate measure value across folds.
    """

    if method_name not in data:
        return None

    folds = data[method_name]

    if not isinstance(folds, list):
        return None

    key_map = {
        "neighborhood_fidelity": "fold_neighborhood_fidelity",
        "stability": "fold_stability",
        "comprehensibility": "fold_mean_nonzero_coefficients",
    }

    measure_key = key_map[measure]

    values = []

    for fold in folds:

        if measure_key in fold and fold[measure_key] is not None:
            values.append(float(fold[measure_key]))

    if len(values) == 0:
        return None

    return float(np.mean(values))


def wilcoxon_surrogate_all_measures(
    hypothesis_id,
    description,
    expected_relationship,
    comparisons,
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    alpha=0.05,
    output_filename="surrogate_wilcoxon.json",
):
    """
    Performs Wilcoxon signed-rank tests across
    all surrogate explanation measures simultaneously.
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)

    stats_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        "iris",
        "wine",
        "breast_cancer",
        "german_credit",
        "darwin",
        "yeast",
        "sepsis",
    ]

    measures = [
        "neighborhood_fidelity",
        "stability",
        "comprehensibility",
    ]

    all_results = {}

    for measure in measures:

        measure_results = {}

        for comparison_name, (method_a, method_b) in comparisons.items():

            vals_a = []
            vals_b = []
            rows = []

            for ds in datasets:

                path = base_dir / ds / f"{ds}_surrogate_all_measures.json"

                if not path.exists():
                    continue

                data = load_json(path)

                val_a = surrogate_measure_value(data, measure, method_a)
                val_b = surrogate_measure_value(data, measure, method_b)

                if val_a is None or val_b is None:
                    continue

                vals_a.append(val_a)
                vals_b.append(val_b)

                rows.append({
                    "dataset": ds,
                    "method_a": method_a,
                    "method_b": method_b,
                    "value_a": float(val_a),
                    "value_b": float(val_b),
                })

            if len(vals_a) < 3:

                measure_results[comparison_name] = {
                    "wilcoxon_statistic": None,
                    "p_value": None,
                    "rows": rows,
                }

                continue

            stat, p_value = wilcoxon(vals_a, vals_b)

            measure_results[comparison_name] = {
                "wilcoxon_statistic": float(stat),
                "p_value": float(p_value),
                "alpha": alpha,
                "is_statistically_significant": bool(
                    p_value < alpha
                ),
                "rows": rows,
            }

        all_results[measure] = measure_results

    output = {
        "hypothesis": hypothesis_id,
        "description": description,
        "expected_relationship": expected_relationship,
        "results": all_results,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    return output

from scipy.stats import spearmanr


def spearman_surrogate_measures(
    hypothesis_id,
    description,
    dataset_property_map,
    property_name,
    expected_direction="negative",
    base_dir="trained_models",
    stats_dir="statistical_calculations",
    alpha=0.05,
    output_filename="surrogate_spearman.json",
):
    """
    Spearman correlations for all surrogate measures simultaneously.
    """

    base_dir = Path(base_dir)
    stats_dir = Path(stats_dir)

    stats_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        "dt",
        "xgb",
        "cbr",
        "proto",
        "mlp",
        "dnn",
    ]

    measures = [
        "neighborhood_fidelity",
        "stability",
        "comprehensibility",
    ]

    all_results = {}

    for measure in measures:

        measure_results = {}

        for method in methods:

            prop_vals = []
            measure_vals = []
            rows = []

            for ds, prop in dataset_property_map.items():

                path = (
                    base_dir /
                    ds /
                    f"{ds}_surrogate_all_measures.json"
                )

                if not path.exists():
                    continue

                data = load_json(path)

                score = surrogate_measure_value(
                    data,
                    measure,
                    method
                )

                if score is None:
                    continue

                prop_vals.append(prop)
                measure_vals.append(score)

                rows.append({
                    "dataset": ds,
                    property_name: prop,
                    "measure_score": float(score),
                })

            if len(prop_vals) < 3:

                measure_results[method] = {
                    "spearman_correlation": None,
                    "p_value": None,
                    "rows": rows,
                }

                continue

            corr, p_value = spearmanr(
                prop_vals,
                measure_vals
            )

            measure_results[method] = {
                "spearman_correlation": float(corr),
                "p_value": float(p_value),
                "alpha": alpha,
                "expected_direction": expected_direction,
                "is_statistically_significant": bool(
                    p_value < alpha
                ),
                "rows": rows,
            }

        all_results[measure] = measure_results

    output = {
        "hypothesis": hypothesis_id,
        "description": description,
        "results": all_results,
    }

    output_path = stats_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    return output

def run_cross_measure_consistency(
    trained_models_dir,
    datasets,
    output_json_path
):
    """
    Runs cross-measure consistency analysis across dataset-method level
    interpretability measures.

    Excludes RIA because RIA is dataset-level, while SOC, Robustness,
    Feature Synergy, MEC, IAS, and NF are dataset-method level measures.

    Produces one JSON file containing:
    - merged dataset-method table
    - Spearman correlation matrix
    - p-value matrix
    - pairwise statistical results
    """

    def mean_upper_triangle(matrix):
        arr = np.array(matrix, dtype=float)

        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            return np.nan

        idx = np.triu_indices_from(arr, k=1)
        vals = arr[idx]

        return float(np.nanmean(vals)) if len(vals) > 0 else np.nan

    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    def extract_soc(path):
        data = load_json(path)
        dataset = data["dataset_name"]
        rows = []

        for method, folds in data["by_method"].items():
            soc_vals = [
                fold.get("soc")
                for fold in folds
                if fold.get("soc") is not None
            ]

            soc_mean = np.nanmean(soc_vals) if soc_vals else np.nan

            rows.append({
                "Dataset": dataset,
                "AI Method": method,
                "SOC_raw": float(soc_mean),
                "SOC_interp": float(1 / (1 + soc_mean)) if not np.isnan(soc_mean) else None
            })

        return pd.DataFrame(rows)

    def extract_robustness(path):
        data = load_json(path)
        dataset = data["dataset_name"]
        rows = []

        for method, values in data["by_method"].items():
            robustness = values.get("overall_robustness", np.nan)

            rows.append({
                "Dataset": dataset,
                "AI Method": method,
                "Robustness": float(robustness) if robustness is not None else None
            })

        return pd.DataFrame(rows)

    def extract_feature_synergy(path):
        data = load_json(path)
        dataset = data["dataset_name"]
        rows = []

        for method, values in data["by_method"].items():
            synergy_raw = mean_upper_triangle(values.get("synergy_matrix", []))

            rows.append({
                "Dataset": dataset,
                "AI Method": method,
                "Feature_Synergy_raw": synergy_raw,
                "Feature_Synergy_interp": (
                    float(1 / (1 + synergy_raw))
                    if not np.isnan(synergy_raw)
                    else None
                )
            })

        return pd.DataFrame(rows)

    def extract_mec(path):
        data = load_json(path)
        dataset = data["dataset_name"]
        rows = []

        for method, values in data["by_method"].items():
            agg = values.get("aggregate", {})

            mec = agg.get("mec_mean", np.nan)
            ias = agg.get("ias_mean", np.nan)
            nf = agg.get("nf_mean", np.nan)

            rows.append({
                "Dataset": dataset,
                "AI Method": method,

                "MEC_raw": float(mec) if mec is not None else None,
                "IAS_raw": float(ias) if ias is not None else None,
                "NF_raw": float(nf) if nf is not None else None,

                "MEC_interp": float(1 / (1 + mec)) if mec is not None and not np.isnan(mec) else None,
                "IAS_interp": float(1 / (1 + ias)) if ias is not None and not np.isnan(ias) else None,
                "NF_interp": float(1 / (1 + nf)) if nf is not None and not np.isnan(nf) else None
            })

        return pd.DataFrame(rows)

    def build_dataset_table(dataset):
        folder = os.path.join(trained_models_dir, dataset)

        soc_path = os.path.join(folder, f"{dataset}_soc_all_methods.json")
        rs_path = os.path.join(folder, f"{dataset}_rs_all_methods.json")
        fs_path = os.path.join(folder, f"{dataset}_fs_all_methods.json")
        mec_path = os.path.join(folder, f"{dataset}_mec_all_methods.json")

        soc_df = extract_soc(soc_path)
        rs_df = extract_robustness(rs_path)
        fs_df = extract_feature_synergy(fs_path)
        mec_df = extract_mec(mec_path)

        merged = soc_df.merge(rs_df, on=["Dataset", "AI Method"], how="inner")
        merged = merged.merge(fs_df, on=["Dataset", "AI Method"], how="inner")
        merged = merged.merge(mec_df, on=["Dataset", "AI Method"], how="inner")

        return merged

    all_tables = []

    for dataset in datasets:
        all_tables.append(build_dataset_table(dataset))

    cross_measure_df = pd.concat(all_tables, ignore_index=True)

    measure_cols = [
        "SOC_interp",
        "Robustness",
        "Feature_Synergy_interp",
        "MEC_interp",
        "IAS_interp",
        "NF_interp"
    ]

    corr = pd.DataFrame(index=measure_cols, columns=measure_cols, dtype=float)
    pvals = pd.DataFrame(index=measure_cols, columns=measure_cols, dtype=float)
    pairwise_results = []

    for m1 in measure_cols:
        for m2 in measure_cols:
            valid = cross_measure_df[[m1, m2]].dropna()

            if len(valid) < 3:
                rho = np.nan
                p = np.nan

            elif m1 == m2:
                rho = 1.0
                p = 0.0

            else:
                x = pd.to_numeric(valid[m1], errors="coerce").to_numpy(dtype=float).ravel()
                y = pd.to_numeric(valid[m2], errors="coerce").to_numpy(dtype=float).ravel()

                mask = np.isfinite(x) & np.isfinite(y)
                x = x[mask]
                y = y[mask]

                if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                    rho = np.nan
                    p = np.nan
                else:
                    result = spearmanr(x, y)
                    rho = float(result.statistic)
                    p = float(result.pvalue)

            corr.loc[m1, m2] = rho
            pvals.loc[m1, m2] = p

            if m1 != m2:
                pairwise_results.append({
                    "measure_1": m1,
                    "measure_2": m2,
                    "n_pairs": int(len(valid)),
                    "spearman_rho": None if np.isnan(rho) else float(rho),
                    "p_value": None if np.isnan(p) else float(p),
                    "significant_at_0_05": None if np.isnan(p) else bool(p < 0.05)
                })

    results = {
        "analysis": "cross_measure_consistency",

        "excluded_measure": {
            "measure": "RIA",
            "reason": (
                "RIA was excluded because it is a dataset-level measure, "
                "whereas SOC, Robustness, Feature Synergy, MEC, IAS, and NF "
                "operate at the dataset-method level."
            )
        },

        "directionality": {
            "SOC_interp": "1 / (1 + SOC_raw); higher values indicate higher interpretability",
            "Robustness": "Used directly; higher values indicate higher interpretability",
            "Feature_Synergy_interp": "1 / (1 + Feature_Synergy_raw); higher values indicate higher interpretability",
            "MEC_interp": "1 / (1 + MEC_raw); higher values indicate higher interpretability",
            "IAS_interp": "1 / (1 + IAS_raw); higher values indicate higher interpretability",
            "NF_interp": "1 / (1 + NF_raw); higher values indicate higher interpretability"
        },

        "datasets": datasets,
        "n_dataset_method_pairs": int(len(cross_measure_df)),
        "measures_used": measure_cols,

        "merged_dataset_method_table": cross_measure_df.where(
            pd.notnull(cross_measure_df), None
        ).to_dict(orient="records"),

        "correlation_matrix": corr.where(
            pd.notnull(corr), None
        ).to_dict(),

        "p_value_matrix": pvals.where(
            pd.notnull(pvals), None
        ).to_dict(),

        "pairwise_results": pairwise_results
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)

    return results

def run_direct_vs_posthoc_alignment(
    trained_models_dir,
    datasets,
    output_json_path
):
    import os
    import json
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    def mean_upper_triangle(matrix):
        arr = np.array(matrix, dtype=float)

        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            return np.nan

        idx = np.triu_indices_from(arr, k=1)
        vals = arr[idx]

        return float(np.nanmean(vals)) if len(vals) > 0 else np.nan

    def safe_interp(x):
        if x is None:
            return np.nan
        try:
            x = float(x)
            if np.isnan(x):
                return np.nan
            return 1 / (1 + x)
        except Exception:
            return np.nan

    def avg_from_shap_list(shap_json, section, possible_keys):
        vals = []

        for fold in shap_json.get(section, []):
            for key in possible_keys:
                if key in fold and fold.get(key) is not None:
                    vals.append(fold.get(key))
                    break

        return float(np.nanmean(vals)) if vals else np.nan

    def avg_from_surrogate_list(surrogate_folds, possible_keys):
        vals = []

        for fold in surrogate_folds:
            for key in possible_keys:
                if key in fold and fold.get(key) is not None:
                    vals.append(fold.get(key))
                    break

        return float(np.nanmean(vals)) if vals else np.nan

    all_rows = []

    for dataset in datasets:

        dataset_folder = os.path.join(trained_models_dir, dataset)

        soc_path = os.path.join(dataset_folder, f"{dataset}_soc_all_methods.json")
        rs_path = os.path.join(dataset_folder, f"{dataset}_rs_all_methods.json")
        fs_path = os.path.join(dataset_folder, f"{dataset}_fs_all_methods.json")
        mec_path = os.path.join(dataset_folder, f"{dataset}_mec_all_methods.json")
        surrogate_path = os.path.join(dataset_folder, f"{dataset}_surrogate_all_measures.json")

        soc_json = load_json(soc_path)
        rs_json = load_json(rs_path)
        fs_json = load_json(fs_path)
        mec_json = load_json(mec_path)
        surrogate_json = load_json(surrogate_path) if os.path.exists(surrogate_path) else {}

        direct_rows = {}

        for method, folds in soc_json["by_method"].items():

            soc_vals = [
                fold.get("soc")
                for fold in folds
                if fold.get("soc") is not None
            ]

            soc_mean = float(np.nanmean(soc_vals)) if soc_vals else np.nan

            direct_rows[method] = {
                "Dataset": dataset,
                "AI Method": method,
                "SOC_raw": soc_mean,
                "SOC_interp": safe_interp(soc_mean)
            }

        for method, values in rs_json["by_method"].items():
            if method in direct_rows:
                robustness = values.get("overall_robustness", np.nan)
                direct_rows[method]["Robustness"] = (
                    float(robustness) if robustness is not None else np.nan
                )

        for method, values in fs_json["by_method"].items():
            if method in direct_rows:
                fs_raw = mean_upper_triangle(values.get("synergy_matrix", []))
                direct_rows[method]["Feature_Synergy_raw"] = fs_raw
                direct_rows[method]["Feature_Synergy_interp"] = safe_interp(fs_raw)

        for method, values in mec_json["by_method"].items():
            if method in direct_rows:
                agg = values.get("aggregate", {})

                mec = agg.get("mec_mean", np.nan)
                ias = agg.get("ias_mean", np.nan)
                nf = agg.get("nf_mean", np.nan)

                direct_rows[method]["MEC_raw"] = mec
                direct_rows[method]["IAS_raw"] = ias
                direct_rows[method]["NF_raw"] = nf

                direct_rows[method]["MEC_interp"] = safe_interp(mec)
                direct_rows[method]["IAS_interp"] = safe_interp(ias)
                direct_rows[method]["NF_interp"] = safe_interp(nf)

        for method in direct_rows.keys():

            row = direct_rows[method].copy()

            shap_path = os.path.join(
                dataset_folder,
                f"{dataset}_{method}_shap_measures.json"
            )

            if os.path.exists(shap_path):

                shap_json = load_json(shap_path)

                row["SHAP_Similarity"] = avg_from_shap_list(
                    shap_json,
                    "similarity",
                    [
                        "similarity_total_test_avg",
                        "total_similarity_averaged_test",
                        "fold_avg_similarity"
                    ]
                )

                row["SHAP_Stability"] = avg_from_shap_list(
                    shap_json,
                    "stability",
                    [
                        "stability_total_test_avg",
                        "total_stability_averaged_test",
                        "fold_avg_stability"
                    ]
                )

                row["SHAP_Parsimony"] = avg_from_shap_list(
                    shap_json,
                    "parsimony",
                    [
                        "total_parsimony_averaged_test",
                        "parsimony_total_test_avg",
                        "fold_avg_parsimony"
                    ]
                )

                row["SHAP_Faithfulness"] = avg_from_shap_list(
                    shap_json,
                    "faithfulness",
                    [
                        "faithfulness_total_test_avg",
                        "total_faithfulness_averaged_test",
                        "fold_avg_faithfulness"
                    ]
                )

            if method in surrogate_json:

                surrogate_folds = surrogate_json[method]

                row["Surrogate_Fidelity"] = avg_from_surrogate_list(
                    surrogate_folds,
                    [
                        "fold_neighborhood_fidelity",
                        "fold_avg_neighborhood_fidelity",
                        "neighborhood_fidelity"
                    ]
                )

                row["Surrogate_Stability"] = avg_from_surrogate_list(
                    surrogate_folds,
                    [
                        "fold_stability",
                        "fold_avg_stability",
                        "stability"
                    ]
                )

                row["Surrogate_Comprehensibility"] = avg_from_surrogate_list(
                    surrogate_folds,
                    [
                        "fold_mean_nonzero_coefficients",
                        "fold_avg_comprehensibility",
                        "comprehensibility"
                    ]
                )

            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    direct_measures = [
        "SOC_interp",
        "Robustness",
        "Feature_Synergy_interp",
        "MEC_interp",
        "IAS_interp",
        "NF_interp"
    ]

    posthoc_measures = [
        "SHAP_Similarity",
        "SHAP_Stability",
        "SHAP_Parsimony",
        "SHAP_Faithfulness",
        "Surrogate_Fidelity",
        "Surrogate_Stability",
        "Surrogate_Comprehensibility"
    ]

    for col in direct_measures + posthoc_measures:
        if col not in df.columns:
            df[col] = np.nan

    pairwise_results = []

    for direct_measure in direct_measures:

        for posthoc_measure in posthoc_measures:

            valid = df[[direct_measure, posthoc_measure]].dropna()

            if len(valid) < 3:
                rho = np.nan
                p = np.nan
            else:
                x = pd.to_numeric(
                    valid[direct_measure],
                    errors="coerce"
                ).to_numpy(dtype=float).ravel()

                y = pd.to_numeric(
                    valid[posthoc_measure],
                    errors="coerce"
                ).to_numpy(dtype=float).ravel()

                mask = np.isfinite(x) & np.isfinite(y)
                x = x[mask]
                y = y[mask]

                if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                    rho = np.nan
                    p = np.nan
                else:
                    result = spearmanr(x, y)
                    rho = float(result.statistic)
                    p = float(result.pvalue)

            pairwise_results.append({
                "direct_measure": direct_measure,
                "posthoc_measure": posthoc_measure,
                "n_pairs": int(len(valid)),
                "spearman_rho": None if np.isnan(rho) else float(rho),
                "p_value": None if np.isnan(p) else float(p),
                "significant_at_0_05": None if np.isnan(p) else bool(p < 0.05)
            })

    results = {
        "analysis": "direct_vs_posthoc_alignment",
        "datasets": datasets,
        "n_dataset_method_pairs": int(len(df)),
        "direct_measures": direct_measures,
        "posthoc_measures": posthoc_measures,
        "merged_dataset_method_table": df.where(
            pd.notnull(df), None
        ).to_dict(orient="records"),
        "pairwise_results": pairwise_results
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)

    return results

def run_performance_interpretability_relationship(
    trained_models_dir,
    datasets,
    output_json_path
):

    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    def safe_interp(x):
        if x is None:
            return np.nan
        try:
            x = float(x)
            if np.isnan(x):
                return np.nan
            return 1 / (1 + x)
        except Exception:
            return np.nan

    def mean_upper_triangle(matrix):
        arr = np.array(matrix, dtype=float)

        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            return np.nan

        idx = np.triu_indices_from(arr, k=1)
        vals = arr[idx]

        return float(np.nanmean(vals)) if len(vals) > 0 else np.nan

    def avg_metric_from_fold_model(model_path, metric_name):
        if not os.path.exists(model_path):
            return np.nan

        model_json = load_json(model_path)

        vals = []

        for fold in model_json:
            perf = fold.get("performance_metrics", {})

            if perf.get(metric_name) is not None:
                vals.append(perf.get(metric_name))

        return float(np.nanmean(vals)) if vals else np.nan

    all_rows = []

    method_to_model_file = {
        "dt": "dt_fold_model.json",
        "xgb": "xgb_fold_model.json",
        "cbr": "cbr_fold_model.json",
        "proto": "proto_fold_model.json",
        "mlp": "mlp_fold_model.json",
        "dnn": "dnn_fold_model.json"
    }

    for dataset in datasets:

        dataset_folder = os.path.join(trained_models_dir, dataset)

        soc_json = load_json(
            os.path.join(dataset_folder, f"{dataset}_soc_all_methods.json")
        )

        rs_json = load_json(
            os.path.join(dataset_folder, f"{dataset}_rs_all_methods.json")
        )

        fs_json = load_json(
            os.path.join(dataset_folder, f"{dataset}_fs_all_methods.json")
        )

        mec_json = load_json(
            os.path.join(dataset_folder, f"{dataset}_mec_all_methods.json")
        )

        direct_rows = {}

        for method, folds in soc_json["by_method"].items():
            soc_vals = [
                fold.get("soc")
                for fold in folds
                if fold.get("soc") is not None
            ]

            soc_mean = float(np.nanmean(soc_vals)) if soc_vals else np.nan

            direct_rows[method] = {
                "Dataset": dataset,
                "AI Method": method,
                "SOC_raw": soc_mean,
                "SOC_interp": safe_interp(soc_mean)
            }

        for method, values in rs_json["by_method"].items():
            if method in direct_rows:
                robustness = values.get("overall_robustness", np.nan)

                direct_rows[method]["Robustness"] = (
                    float(robustness)
                    if robustness is not None and not np.isnan(robustness)
                    else np.nan
                )

        for method, values in fs_json["by_method"].items():
            if method in direct_rows:
                fs_raw = mean_upper_triangle(
                    values.get("synergy_matrix", [])
                )

                direct_rows[method]["Feature_Synergy_raw"] = fs_raw
                direct_rows[method]["Feature_Synergy_interp"] = safe_interp(fs_raw)

        for method, values in mec_json["by_method"].items():
            if method in direct_rows:
                agg = values.get("aggregate", {})

                mec = agg.get("mec_mean", np.nan)
                ias = agg.get("ias_mean", np.nan)
                nf = agg.get("nf_mean", np.nan)

                direct_rows[method]["MEC_raw"] = mec
                direct_rows[method]["IAS_raw"] = ias
                direct_rows[method]["NF_raw"] = nf

                direct_rows[method]["MEC_interp"] = safe_interp(mec)
                direct_rows[method]["IAS_interp"] = safe_interp(ias)
                direct_rows[method]["NF_interp"] = safe_interp(nf)

        for method in direct_rows.keys():

            row = direct_rows[method].copy()

            model_file = method_to_model_file.get(method)
            model_path = (
                os.path.join(dataset_folder, model_file)
                if model_file is not None
                else None
            )

            row["Accuracy"] = (
                avg_metric_from_fold_model(model_path, "accuracy")
                if model_path is not None else np.nan
            )

            row["Precision"] = (
                avg_metric_from_fold_model(model_path, "precision")
                if model_path is not None else np.nan
            )

            row["Recall"] = (
                avg_metric_from_fold_model(model_path, "recall")
                if model_path is not None else np.nan
            )

            row["F1"] = (
                avg_metric_from_fold_model(model_path, "f1")
                if model_path is not None else np.nan
            )

            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    interpretability_measures = [
        "SOC_interp",
        "Robustness",
        "Feature_Synergy_interp",
        "MEC_interp",
        "IAS_interp",
        "NF_interp"
    ]

    performance_measures = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    for col in interpretability_measures + performance_measures:
        if col not in df.columns:
            df[col] = np.nan

    pairwise_results = []

    for interp_measure in interpretability_measures:

        for perf_measure in performance_measures:

            valid = df[[interp_measure, perf_measure]].dropna()

            if len(valid) < 3:
                rho = np.nan
                p = np.nan
            else:
                x = pd.to_numeric(
                    valid[interp_measure],
                    errors="coerce"
                ).to_numpy(dtype=float).ravel()

                y = pd.to_numeric(
                    valid[perf_measure],
                    errors="coerce"
                ).to_numpy(dtype=float).ravel()

                mask = np.isfinite(x) & np.isfinite(y)

                x = x[mask]
                y = y[mask]

                if (
                    len(x) < 3 or
                    len(np.unique(x)) < 2 or
                    len(np.unique(y)) < 2
                ):
                    rho = np.nan
                    p = np.nan
                else:
                    result = spearmanr(x, y)
                    rho = float(result.statistic)
                    p = float(result.pvalue)

            pairwise_results.append({
                "interpretability_measure": interp_measure,
                "performance_measure": perf_measure,
                "n_pairs": int(len(valid)),
                "spearman_rho": None if np.isnan(rho) else float(rho),
                "p_value": None if np.isnan(p) else float(p),
                "significant_at_0_05": None if np.isnan(p) else bool(p < 0.05)
            })

    results = {
        "analysis": "performance_interpretability_relationship",
        "datasets": datasets,
        "n_dataset_method_pairs": int(len(df)),
        "interpretability_measures": interpretability_measures,
        "performance_measures": performance_measures,
        "merged_dataset_method_table": df.where(
            pd.notnull(df), None
        ).to_dict(orient="records"),
        "pairwise_results": pairwise_results
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)

    return results

def run_ranking_agreement_analysis(
    trained_models_dir,
    datasets,
    output_json_path
):
    import os
    import json
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    def safe_interp(x):
        if x is None:
            return np.nan
        try:
            x = float(x)
            if np.isnan(x):
                return np.nan
            return 1 / (1 + x)
        except Exception:
            return np.nan

    def mean_upper_triangle(matrix):
        arr = np.array(matrix, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            return np.nan
        vals = arr[np.triu_indices_from(arr, k=1)]
        return float(np.nanmean(vals)) if len(vals) > 0 else np.nan

    def avg_from_list(items, possible_keys):
        vals = []
        for item in items:
            for key in possible_keys:
                if key in item and item.get(key) is not None:
                    vals.append(item.get(key))
                    break
        return float(np.nanmean(vals)) if vals else np.nan

    all_rows = []

    for dataset in datasets:
        dataset_folder = os.path.join(trained_models_dir, dataset)

        soc_json = load_json(os.path.join(dataset_folder, f"{dataset}_soc_all_methods.json"))
        rs_json = load_json(os.path.join(dataset_folder, f"{dataset}_rs_all_methods.json"))
        fs_json = load_json(os.path.join(dataset_folder, f"{dataset}_fs_all_methods.json"))
        mec_json = load_json(os.path.join(dataset_folder, f"{dataset}_mec_all_methods.json"))

        surrogate_path = os.path.join(dataset_folder, f"{dataset}_surrogate_all_measures.json")
        surrogate_json = load_json(surrogate_path) if os.path.exists(surrogate_path) else {}

        rows = {}

        for method, folds in soc_json["by_method"].items():
            soc_vals = [
                fold.get("soc")
                for fold in folds
                if fold.get("soc") is not None
            ]
            soc_mean = float(np.nanmean(soc_vals)) if soc_vals else np.nan

            rows[method] = {
                "Dataset": dataset,
                "AI Method": method,
                "SOC_interp": safe_interp(soc_mean)
            }

        for method, values in rs_json["by_method"].items():
            if method in rows:
                robustness = values.get("overall_robustness", np.nan)
                rows[method]["Robustness"] = (
                    float(robustness)
                    if robustness is not None and not np.isnan(robustness)
                    else np.nan
                )

        for method, values in fs_json["by_method"].items():
            if method in rows:
                fs_raw = mean_upper_triangle(values.get("synergy_matrix", []))
                rows[method]["Feature_Synergy_interp"] = safe_interp(fs_raw)

        for method, values in mec_json["by_method"].items():
            if method in rows:
                agg = values.get("aggregate", {})

                mec = agg.get("mec_mean", np.nan)
                ias = agg.get("ias_mean", np.nan)
                nf = agg.get("nf_mean", np.nan)

                rows[method]["MEC_interp"] = safe_interp(mec)
                rows[method]["IAS_interp"] = safe_interp(ias)
                rows[method]["NF_interp"] = safe_interp(nf)

        for method in rows.keys():

            shap_path = os.path.join(
                dataset_folder,
                f"{dataset}_{method}_shap_measures.json"
            )

            if os.path.exists(shap_path):
                shap_json = load_json(shap_path)

                rows[method]["SHAP_Similarity"] = avg_from_list(
                    shap_json.get("similarity", []),
                    [
                        "similarity_total_test_avg",
                        "total_similarity_averaged_test",
                        "fold_avg_similarity"
                    ]
                )

                rows[method]["SHAP_Stability"] = avg_from_list(
                    shap_json.get("stability", []),
                    [
                        "stability_total_test_avg",
                        "total_stability_averaged_test",
                        "fold_avg_stability"
                    ]
                )

                rows[method]["SHAP_Parsimony"] = avg_from_list(
                    shap_json.get("parsimony", []),
                    [
                        "total_parsimony_averaged_test",
                        "parsimony_total_test_avg",
                        "fold_avg_parsimony"
                    ]
                )

                rows[method]["SHAP_Faithfulness"] = avg_from_list(
                    shap_json.get("faithfulness", []),
                    [
                        "faithfulness_total_test_avg",
                        "total_faithfulness_averaged_test",
                        "fold_avg_faithfulness"
                    ]
                )

            if method in surrogate_json:
                surrogate_folds = surrogate_json[method]

                rows[method]["Surrogate_Fidelity"] = avg_from_list(
                    surrogate_folds,
                    [
                        "fold_neighborhood_fidelity",
                        "fold_avg_neighborhood_fidelity",
                        "neighborhood_fidelity"
                    ]
                )

                rows[method]["Surrogate_Stability"] = avg_from_list(
                    surrogate_folds,
                    [
                        "fold_stability",
                        "fold_avg_stability",
                        "stability"
                    ]
                )

                rows[method]["Surrogate_Comprehensibility"] = avg_from_list(
                    surrogate_folds,
                    [
                        "fold_mean_nonzero_coefficients",
                        "fold_avg_comprehensibility",
                        "comprehensibility"
                    ]
                )

        all_rows.extend(rows.values())

    df = pd.DataFrame(all_rows)

    measures = [
        "SOC_interp",
        "Robustness",
        "Feature_Synergy_interp",
        "MEC_interp",
        "IAS_interp",
        "NF_interp",
        "SHAP_Similarity",
        "SHAP_Stability",
        "SHAP_Parsimony",
        "SHAP_Faithfulness",
        "Surrogate_Fidelity",
        "Surrogate_Stability",
        "Surrogate_Comprehensibility"
    ]

    for col in measures:
        if col not in df.columns:
            df[col] = np.nan

    ranked_tables = []

    for dataset in datasets:
        sub = df[df["Dataset"] == dataset].copy()

        for measure in measures:
            rank_col = f"{measure}_rank"
            sub[rank_col] = sub[measure].rank(
                ascending=False,
                method="average"
            )

        ranked_tables.append(sub)

    ranked_df = pd.concat(ranked_tables, ignore_index=True)

    per_dataset_results = []

    for dataset in datasets:
        sub = ranked_df[ranked_df["Dataset"] == dataset].copy()

        for i, m1 in enumerate(measures):
            for m2 in measures[i + 1:]:

                r1 = f"{m1}_rank"
                r2 = f"{m2}_rank"

                valid = sub[[r1, r2]].dropna()

                if len(valid) < 3:
                    rho = np.nan
                    p = np.nan
                else:
                    x = valid[r1].to_numpy(dtype=float).ravel()
                    y = valid[r2].to_numpy(dtype=float).ravel()

                    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                        rho = np.nan
                        p = np.nan
                    else:
                        result = spearmanr(x, y)
                        rho = float(result.statistic)
                        p = float(result.pvalue)

                per_dataset_results.append({
                    "dataset": dataset,
                    "measure_1": m1,
                    "measure_2": m2,
                    "n_methods": int(len(valid)),
                    "spearman_rho": None if np.isnan(rho) else float(rho),
                    "p_value": None if np.isnan(p) else float(p),
                    "significant_at_0_05": None if np.isnan(p) else bool(p < 0.05)
                })

    summary_rows = []

    for i, m1 in enumerate(measures):
        for m2 in measures[i + 1:]:

            vals = [
                r["spearman_rho"]
                for r in per_dataset_results
                if r["measure_1"] == m1
                and r["measure_2"] == m2
                and r["spearman_rho"] is not None
            ]

            summary_rows.append({
                "measure_1": m1,
                "measure_2": m2,
                "n_datasets": int(len(vals)),
                "mean_spearman_rho": float(np.nanmean(vals)) if vals else None,
                "median_spearman_rho": float(np.nanmedian(vals)) if vals else None,
                "std_spearman_rho": float(np.nanstd(vals)) if vals else None
            })

    results = {
        "analysis": "ranking_agreement_analysis_extended",
        "datasets": datasets,
        "n_dataset_method_pairs": int(len(df)),
        "measures": measures,
        "directionality": {
            "note": "All measures are transformed or used so that higher values indicate higher interpretability.",
            "SOC_interp": "1 / (1 + SOC_raw)",
            "Robustness": "used directly",
            "Feature_Synergy_interp": "1 / (1 + Feature_Synergy_raw)",
            "MEC_interp": "1 / (1 + MEC_raw)",
            "IAS_interp": "1 / (1 + IAS_raw)",
            "NF_interp": "1 / (1 + NF_raw)",
            "SHAP_Similarity": "used directly",
            "SHAP_Stability": "used directly",
            "SHAP_Parsimony": "used directly",
            "SHAP_Faithfulness": "used directly",
            "Surrogate_Fidelity": "used directly",
            "Surrogate_Stability": "used directly",
            "Surrogate_Comprehensibility": "used directly"
        },
        "rank_definition": (
            "Within each dataset, AI methods are ranked separately for each "
            "interpretability or explanation measure. Rank 1 indicates the most "
            "interpretable method according to that measure."
        ),
        "ranked_dataset_method_table": ranked_df.where(
            pd.notnull(ranked_df), None
        ).to_dict(orient="records"),
        "per_dataset_pairwise_rank_agreement": per_dataset_results,
        "summary_pairwise_rank_agreement": summary_rows
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)

    return results

# (ARCHIVE) def spearman_h2_2_feature_synergy(
#     base_dir="trained_models",
#     stats_dir="statistical_calculations",
#     datasets=None,
#     methods=None,
#     alpha=0.05,
#     output_filename="(ARCHIVE) feature_synergy_h2_2_spearman.json",
# ):
#     """
#     H2.2:
#     Lower interpretability should correspond to
#     stronger or more complex feature interactions.
#
#     Since Feature Synergy operationalizes interaction
#     dependence directly, this analysis evaluates whether
#     higher interaction density corresponds to higher
#     Feature Synergy values.
#     """
#
#     base_dir = Path(base_dir)
#     stats_dir = Path(stats_dir)
#
#     stats_dir.mkdir(parents=True, exist_ok=True)
#
#     if datasets is None:
#         datasets = [
#             "iris",
#             "wine",
#             "breast_cancer",
#             "german_credit",
#             "darwin",
#             "sepsis",
#             "yeast",
#         ]
#
#     if methods is None:
#         methods = [
#             "dt",
#             "xgb",
#             "cbr",
#             "proto",
#             "mlp",
#             "dnn",
#         ]
#
#     rows = []
#
#     for ds in datasets:
#
#         fs_path = (
#             base_dir
#             / ds
#             / f"{ds}_fs_all_methods.json"
#         )
#
#         if not fs_path.exists():
#             continue
#
#         fs_data = load_json(fs_path)
#
#         for method in methods:
#
#             if method not in fs_data["by_method"]:
#                 continue
#
#             try:
#                 mean_fs, interaction_density = (
#                     _feature_synergy_statistics(
#                         fs_data,
#                         method,
#                     )
#                 )
#
#             except Exception:
#                 continue
#
#             rows.append({
#                 "dataset": ds,
#                 "method": method,
#                 "mean_feature_synergy": mean_fs,
#                 "interaction_density": interaction_density,
#             })
#
#     x = [
#         r["interaction_density"]
#         for r in rows
#     ]
#
#     y = [
#         r["mean_feature_synergy"]
#         for r in rows
#     ]
#
#     rho, p_value = spearmanr(x, y)
#
#     rho = float(rho) if not math.isnan(rho) else None
#     p_value = float(p_value) if not math.isnan(p_value) else None
#
#     is_significant = (
#         False if p_value is None
#         else p_value < alpha
#     )
#
#     result = {
#         "hypothesis": "H2.2",
#
#         "description": (
#             "Spearman rank correlation between "
#             "interaction density and Feature "
#             "Synergy values."
#         ),
#
#         "test": "Spearman rank correlation",
#
#         "direction_tested": (
#             "interaction_density positively "
#             "correlated with mean_feature_synergy"
#         ),
#
#         "n_observations": len(rows),
#
#         "spearman_rho": rho,
#
#         "p_value": p_value,
#
#         "alpha": alpha,
#
#         "is_statistically_significant": bool(
#             is_significant
#         ),
#
#         "rows": rows,
#     }
#
#     output_path = (
#         stats_dir / output_filename
#     )
#
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=4)
#
#     return result
def _feature_synergy_statistics(fs_data, method):
    """
    Computes:
    1. Mean Feature Synergy
    2. Interaction density

    Interaction density =
    proportion of non-zero pairwise interactions.
    """

    matrix = np.asarray(
        fs_data["by_method"][method]["synergy_matrix"],
        dtype=float,
    )

    upper_tri_indices = np.triu_indices_from(matrix, k=1)

    values = matrix[upper_tri_indices]

    if values.size == 0:
        return 0.0, 0.0

    mean_fs = float(np.nanmean(values))

    interaction_density = float(
        np.sum(values > 0) / len(values)
    )

    return mean_fs, interaction_density