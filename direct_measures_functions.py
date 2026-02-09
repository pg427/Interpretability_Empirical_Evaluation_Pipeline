from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif
from pathlib import Path
from model_save_functions import save_model, load_model

base_dir = Path.cwd()/"trained_models"

def ria_measure(dataset: Tuple[np.ndarray, np.ndarray, List[str]], dataset_name:str,
    mi_discrete_features: Union[str, np.ndarray] = "auto",
    random_state: int = 42, aria_thresh: float = 0.7,):
    """
            Compute Pawlicki (2024) RIA pre-model interpretability metrics:
              - ARIA  = arithmetic mean of MaxAbs-scaled MI and MaxAbs-scaled ANOVA-F
              - HaRIA = harmonic mean of the same two scaled terms
              - GeRIA = geometric mean of the same two scaled terms

            Paper basis:
              - MaxAbs scaling for MI and F terms (Eq. 3 and Eq. 4)
              - Metric definitions (Eq. 5, 6, 7)

            Parameters
            ----------
            dataset : (X, y, feature_names)
                X: (n_samples, n_features) float/num array
                y: (n_samples,) int labels
                feature_names: list of str length n_features

            Returns
            -------
            dict with keys:
              - "feature_names"
              - "mi_raw", "f_raw"
              - "mi_scaled", "f_scaled"
              - "ARIA", "HaRIA", "GeRIA"
              - optionally "df" (pandas DataFrame) if return_dataframe=True and pandas available
            """

    def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
        wsum = float(np.sum(w))
        if wsum <= 0:
            return float(np.mean(x))  # fallback to unweighted mean
        return float(np.sum(x * w) / wsum)

    def _topk_mean(x: np.ndarray, k: int) -> float:
        k = int(max(1, min(k, x.size)))
        return float(np.mean(np.sort(x)[-k:]))


    X, y, feature_names = dataset
    dataset_ria = {}
    ria_path = base_dir/dataset_name/f"{dataset_name}_ria.joblib"
    if ria_path.exists():
        dataset_ria = load_model(ria_path)
    else:
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X rows ({X.shape[0]}) must match y length ({y.shape[0]}).")
        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match X cols ({X.shape[1]})."
            )

        # --- Term 1: Mutual Information between each feature and target
        mi_raw = mutual_info_classif(
                X,
                y,
                discrete_features=mi_discrete_features,
                random_state=random_state
            ).astype(np.float64)

        # --- Term 2: ANOVA F-values for each feature vs target
        # (sklearn computes the standard ANOVA F-test statistic per feature)
        f_raw, _p = f_classif(X, y)
        f_raw = np.asarray(f_raw, dtype=np.float64)
        f_raw = np.nan_to_num(f_raw, nan=0.0, posinf=0.0, neginf=0.0)  # safety

        # --- MaxAbs scaling: divide by maximum value among features (Eq. 3 & Eq. 4)
        mi_max = float(np.max(np.abs(mi_raw))) if mi_raw.size else 0.0
        f_max = float(np.max(np.abs(f_raw))) if f_raw.size else 0.0

        mi_scaled = mi_raw / mi_max if mi_max > 0 else np.zeros_like(mi_raw)
        f_scaled = f_raw / f_max if f_max > 0 else np.zeros_like(f_raw)

        aria = 0.5 * (mi_scaled + f_scaled)
        geria = np.sqrt(mi_scaled * f_scaled)
        haria = (2.0 * mi_scaled * f_scaled) / (mi_scaled + f_scaled + 1e-12)

        # AVG calculation
        aria_mean = float(np.mean(aria))
        geria_mean = float(np.mean(geria))
        haria_mean = float(np.mean(haria))

        # Standard Deviation and Coefficient of Variation (for ARIA)
        aria_std = float(np.std(aria))
        aria_cv = float(aria_std / (aria_mean + 1e-12))

        # Weighted Mean by normalized MI or ANOVA (for ARIA)
        aria_weighted_by_mi = _weighted_mean(aria, mi_scaled)
        aria_weighted_by_f = _weighted_mean(aria, f_scaled)

        # Top-k Mean i.e. mean of top-k RIA features
        k5 = min(5, aria.size)
        k10 = min(10, aria.size)
        aria_top5_mean = _topk_mean(aria, k5)
        aria_top10_mean = _topk_mean(aria, k10)

        # count of features above some threshold (0.7) and count of features above the mean ARIA
        aria_count_ge_thresh = int(np.sum(aria >= aria_thresh))
        aria_count_ge_mean = int(np.sum(aria >= aria_mean))

        n_features = int(aria.size)

        dataset_ria={
            "dataset_name": dataset_name,
            "feature_names": feature_names,
            "MI_raw": mi_raw,
            "MI_scaled": mi_scaled,
            "F_raw": f_raw,
            "F_scaled": f_scaled,
            "ARIA": aria,
            "GeRIA": geria,
            "HaRIA": haria,
            "aggregations":{
                "n_features": n_features,
                "ARIA_mean": aria_mean,
                "GeRIA_mean": geria_mean,
                "HaRIA_mean": haria_mean,
                "ARIA_std": aria_std,
                "ARIA_cv": aria_cv,

                "ARIA_weighted_by_MI": aria_weighted_by_mi,
                "ARIA_weighted_by_F": aria_weighted_by_f,

                "ARIA_top5_mean": aria_top5_mean,
                "ARIA_top10_mean": aria_top10_mean,

                "ARIA_thresh": float(aria_thresh),
                "ARIA_count_ge_thresh": aria_count_ge_thresh,
                "ARIA_count_ge_mean": aria_count_ge_mean
            }
        }
        save_model(dataset_ria, ria_path)
    return dataset_ria

def soc_measure(*, method_name:str, fold_result: Dict[str, Any],
    distance_type: str = "euclidean") -> Dict[str, Any]:
    """
        Compute #SOC-based interpretability for ONE fold result dict (like outputs of CART_DT_5FOLD,
        XGB_5FOLD, CBR_5FOLD, MLP_5FOLD, DNN_8HL_5fold, PROTOPNET_5FOLD in method_functions.py). :contentReference[oaicite:10]{index=10}

        Returns:
          dict with:
            - soc (int)
            - interpretability_score (float): 1 / (1 + soc)  (higher is "more interpretable")
            - details (method-specific)
        """
    # Distance type cost: Park et al. use ~3 ops/feature for Euclidean distance (per-feature) :contentReference[oaicite:3]{index=3}
    _DIST_COST = {
        "euclidean": 3,
        "manhattan": 3,  # also listed as 3 in their table snippet :contentReference[oaicite:4]{index=4}
    }

    _ACT_COST = {
        "relu": 1,
        "logistic": 6,  # sigmoid
        "tanh": 8,
        # fallback:
        None: 1,
    }


    def _soc_decision_tree(depth: int) -> int:
        # Park et al.: #SOC(DT) = 2*D + 1
        return int(2 * depth + 1)

    def _xgb_tree_depths_sklearn(xgb_model) -> List[int]:
        """
        Get per-tree max depth from an XGBoost sklearn wrapper.
        Uses trees_to_dataframe if available.
        """
        booster = xgb_model.get_booster()
        try:
            df = booster.trees_to_dataframe()
            # df columns typically include: Tree, Node, Depth, Feature, ...
            depths = (
                df.groupby("Tree")["Depth"]
                .max()
                .fillna(0)
                .astype(int)
                .tolist()
            )
            return depths if depths else [0]
        except Exception:
            pass

        dumps = booster.get_dump(with_stats=False)
        depths = []
        for s in dumps:
            max_depth = 0
            for line in s.splitlines():
                # XGBoost uses tabs for depth indentation in dump
                d = len(line) - len(line.lstrip("\t"))
                if d > max_depth:
                    max_depth = d
            depths.append(max_depth)
        return depths if depths else [0]

    def _soc_knn(I: int, P: int, dist_cost: int) -> int:
        # Park et al.: #SOC(KNN) = I * P * Dt
        return int(I * P * dist_cost)

    def _soc_mlp(layer_sizes: List[int], activation: str) -> int:
        """
        Park et al. MLP(H=h): sum_{h}(2*N_h + At)*N_{h+1} + 2*N_{h+1}
        Here layer_sizes includes [N1, N2, ..., N_{H+1}] where N1=P and N_{H+1}=#outputs.
        """
        At = _ACT_COST.get(activation, 1)
        total = 0
        for i in range(len(layer_sizes) - 1):
            Nh = int(layer_sizes[i])
            Nnext = int(layer_sizes[i + 1])
            total += (2 * Nh + At) * Nnext + 2 * Nnext
        return int(total)

    def _soc_protopnet(model, input_dim: int, n_classes: int) -> int:
        """
        Not in Park et al. (they don't cover ProtoPNet), so this is a transparent extension:
          SOC = SOC(encoder as MLP) + SOC(prototype distance+exp+max pooling)

        Your TabularProtoPNet encoder is:
          input_dim -> 64 -> 16 -> input_dim  (with ReLU after first two layers)
        Then it computes distances to Pprotos, exp(-dist/tau), and per-class max.
        """
        # Encoder MLP SOC (treat as ReLU MLP with 3 affine layers)
        encoder_layers = [input_dim, 64, 16, input_dim]
        soc_encoder = _soc_mlp(encoder_layers, activation="relu")

        # Prototype part:
        # - squared L2 distance to each prototype: ~3*input_dim ops (sub, square, add accumulation) per prototype
        # - exp: treat as ~3 ops (Park uses #exp separately; we keep a small constant)
        # - per class max over prototypes: comparisons ~(n_protos_per_class - 1) per class
        n_protos = int(getattr(model, "n_prototypes", getattr(model, "prototypes", np.zeros((1, input_dim))).shape[0]))
        if n_protos <= 0:
            n_protos = 1

        dist_ops = n_protos * (3 * input_dim)  # per-proto distance work
        exp_ops = n_protos * 3  # rough constant for exp
        # class-wise max: comparisons; approximate n_protos per class
        ppc = max(1, n_protos // max(1, n_classes))
        max_ops = n_classes * max(0, ppc - 1)

        return int(soc_encoder + dist_ops + exp_ops + max_ops)

    m = method_name.lower().strip()
    model = fold_result.get("model", None)

    X_train = fold_result.get("X_train")
    y_train = fold_result.get("y_train")

    if X_train is None or y_train is None:
        raise ValueError("fold_result must include X_train and y_train (from stratified_5fold_standardize).")

    P = int(np.asarray(X_train).shape[1])
    I = int(np.asarray(X_train).shape[0])
    n_classes = int(len(np.unique(np.asarray(y_train))))

    dist_cost = _DIST_COST.get(distance_type, 3)
    details: Dict[str, Any] = {}
    soc: int
    method_soc = {}

    # -----------------
    # Decision Tree (DT)
    # -----------------
    if m in {"dt", "cart", "decisiontree"}:
        if model is None:
            raise ValueError("fold_result['model'] is required for DT SOC computation.")
        depth = int(model.get_depth())  # sklearn tree depth
        soc = _soc_decision_tree(depth)
        details.update({"depth": depth, "formula": "2*D + 1"})

    # -----------------
    # XGBoost (XGB)
    # -----------------
    elif m in {"xgb", "xgboost"}:
        if model is None:
            raise ValueError("fold_result['model'] is required for XGB SOC computation.")
        tree_depths = _xgb_tree_depths_sklearn(model)
        soc = int(sum(_soc_decision_tree(d) for d in tree_depths))
        details.update(
            {
                "n_trees": int(len(tree_depths)),
                "tree_depths": tree_depths,
                "extension": "sum over trees of (2*D_t + 1)",
            }
        )

    # -----------------
    # CBR (treat as KNN)
    # -----------------
    elif m in {"cbr"}:
        # Park et al. KNN SOC: I*P*Dt
        soc = _soc_knn(I=I, P=P, dist_cost=dist_cost)
        details.update({"I": I, "P": P, "Dt_cost": dist_cost, "formula": "I*P*Dt (KNN proxy)"})

    # -----------------
    # MLP / DNN (sklearn MLPClassifier)
    # -----------------
    elif m in {"mlp", "dnn"}:
        if model is None:
            raise ValueError("fold_result['model'] is required for MLP/DNN SOC computation.")
        # sklearn MLPClassifier stores hidden_layer_sizes; output size = n_classes for classification
        hidden = getattr(model, "hidden_layer_sizes", ())
        if isinstance(hidden, int):
            hidden_layers = [hidden]
        else:
            hidden_layers = list(hidden)

        activation = getattr(model, "activation", "relu")
        layer_sizes = [P] + hidden_layers + [n_classes]
        soc = _soc_mlp(layer_sizes, activation=activation)
        details.update(
            {
                "layer_sizes": layer_sizes,
                "activation": activation,
                "At_cost": _ACT_COST.get(activation, 1),
                "formula": "sum_h (2*N_h + At)*N_{h+1} + 2*N_{h+1}",
            }
        )

    # -----------------
    # ProtoPNet (extension)
    # -----------------
    elif m in {"proto", "protopnet", "protopnet_tabular"}:
        if model is None:
            raise ValueError("fold_result['model'] is required for ProtoPNet SOC computation.")
        soc = _soc_protopnet(model, input_dim=P, n_classes=n_classes)
        details.update(
            {
                "input_dim": P,
                "n_classes": n_classes,
                "extension": "SOC(encoder MLP) + SOC(prototype distance+exp+max)",
            }
        )

    else:
        raise ValueError(f"Unknown method_name='{method_name}'. Expected one of: dt, xgb, cbr, mlp, dnn, proto.")

    interpretability_score = float(1.0 / (1.0 + soc))

    method_soc = {
        "method_name": method_name,
        "soc": int(soc),
        "interpretability_score": interpretability_score,
        "details": details,
        "fold": fold_result.get("fold", None),
    }

    return method_soc

def soc_all_methods_for_dataset(
    *,
    dataset_name: str,
    method_fold_results: Dict[str, Any],
    distance_type: str = "euclidean",
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """
    Create ONE file per dataset with SOC results for all methods across folds.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier (e.g., "iris", "breast_cancer")
    method_fold_results : dict
        Mapping:
            method_name -> folds
        where folds is either:
            - dict: {fold_id: fold_result_dict, ...}
            - list: [fold_result_dict, ...]
    distance_type : str
        Used for CBR/KNN SOC (euclidean/manhattan)
    force_recompute : bool
        If True, ignore cached aggregate file and rebuild.

    Returns
    -------
    dict with:
        - "dataset_name"
        - "by_method" : method -> list of per-fold SOC dicts
        - "summary" : method -> aggregate stats (mean SOC, mean score, etc.)
    """

    out_path = base_dir / dataset_name / f"{dataset_name}_soc_all_methods.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force_recompute:
        return load_model(out_path)

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    summary: Dict[str, Dict[str, Any]] = {}

    for method_name, folds in method_fold_results.items():
        fold_items = []

        # normalize folds structure to iterable of (fold_id, fold_result)
        if isinstance(folds, dict):
            iterable = list(folds.items())
        elif isinstance(folds, list):
            iterable = list(enumerate(folds))
        else:
            raise TypeError(f"folds for method '{method_name}' must be dict or list, got {type(folds)}")

        for fold_id, fold_res in iterable:
            # ensure fold id is present for caching + reporting
            if isinstance(fold_res, dict) and "fold" not in fold_res:
                fold_res = dict(fold_res)
                fold_res["fold"] = fold_id

            fold_soc = soc_measure(
                method_name=method_name,
                fold_result=fold_res,
                distance_type=distance_type,
            )
            fold_items.append(fold_soc)

        by_method[method_name] = fold_items

        soc_vals = np.array([d["soc"] for d in fold_items], dtype=float)
        score_vals = np.array([d["interpretability_score"] for d in fold_items], dtype=float)

        summary[method_name] = {
            "n_folds": int(len(fold_items)),
            "soc_mean": float(np.mean(soc_vals)) if soc_vals.size else None,
            "soc_std": float(np.std(soc_vals)) if soc_vals.size else None,
            "score_mean": float(np.mean(score_vals)) if score_vals.size else None,
            "score_std": float(np.std(score_vals)) if score_vals.size else None,
        }

    result = {
        "dataset_name": dataset_name,
        "by_method": by_method,
        "summary": summary,
    }

    save_model(result, out_path)
    return result