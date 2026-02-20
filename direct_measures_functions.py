from typing import Any, Dict, List, Tuple, Union, Optional, Iterable
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif
from pathlib import Path
from model_save_functions import save_model, load_model

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from method_functions import CBR, TabularProtoPNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from dataset_functions import load_dataset
from dataclasses import dataclass
import skexplain
from itertools import combinations

from tqdm.auto import tqdm
import time


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

def feature_synergy_measure(
    dataset,
    method,
    folds,
    *,
    pop_size: int = 200,
    n_generations: int = 100,
    n_runs: int = 10,
    parent_count: int = 100,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.02,  # kept for API compatibility; repo uses 1-point mutation (always), see below.
    include_accuracy_factor: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Robertson & Hu (2021) Feature Synergy (GECCO'21) – implemented to be faithful to the provided GA_feature_synergy repo
    mechanics (selection/recombination/mutation/population update), but integrated into YOUR pipeline:

      - Uses your `folds` (list/dict of fold_result dicts) and trains/evaluates subsets for:
            DT, XGB, CBR, ProtoPNet, MLP, DNN
      - Drops KNN/SVC and drops the repo's k_value gene entirely.
      - GA core mechanics are repo-faithful:
          * dominance rule: error_i < error_j AND size_i <= size_j
          * selection: fast non-dominated sort, fill fronts; last front is randomly shuffled and truncated
          * recombination: one-point crossover, create exactly (pop_size - parent_count) children from shuffled parents
          * mutation: one-point bit flip per child with a guard to avoid all-zero subsets
          * next generation: parents + children

    Caching:
      trained_models/<dataset>/<dataset>_feature_synergy.joblib
      stores results under ["by_method"][method]

    Notes:
      - This metric requires evaluating MANY feature subsets. Re-using the already-trained fold["model"] is not valid
        for “drop columns” evaluation (shape mismatch) for DT/XGB/MLP/DNN/ProtoPNet and usually wrong for CBR too.
        Therefore, this implementation retrains a fresh model per mask per fold (paper-faithful, expensive).
    """

    fs_path = base_dir / dataset / f"{dataset}_feature_synergy.joblib"
    fs_path.parent.mkdir(parents=True, exist_ok=True)

    # normalize folds input (list of dicts)
    if isinstance(folds, dict):
        fold_list = list(folds.values())
    elif isinstance(folds, list):
        fold_list = folds
    else:
        raise TypeError(f"folds must be a list or dict, got {type(folds)}")

    X_full, y_full, feature_names = load_dataset(dataset)
    n_features = len(feature_names)

    m = str(method).lower().strip()
    if m not in {"dt", "xgb", "cbr", "proto", "mlp", "dnn"}:
        raise ValueError(f"Unknown method='{method}'. Expected one of: dt, xgb, cbr, proto, mlp, dnn")

    dataset_feature_synergy: Dict[str, Any]
    if fs_path.exists():
        dataset_feature_synergy = load_model(fs_path)
    else:
        dataset_feature_synergy = {"dataset_name": dataset, "by_method": {}}

    if dataset_feature_synergy.get("by_method", {}).get(m) is not None:
        return dataset_feature_synergy["by_method"][m]

    rng = np.random.default_rng(random_state)

    # -----------------------------
    # Repo-faithful GA helpers
    # -----------------------------
    def _apply_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return X[:, mask.astype(bool)]

    def _dominates(err_i: float, size_i: int, err_j: float, size_j: int) -> bool:
        """
        Repo dominance:
            i dominates j if error_i < error_j AND size_i <= size_j
        """
        return (err_i < err_j) and (size_i <= size_j)

    def _fast_non_dominated_sort(errors: List[float], sizes: List[int]) -> List[List[int]]:
        """
        Repo-style fast non-dominated sorting.
        Returns fronts as list of lists of population indices.
        """
        N = len(errors)
        S = [[] for _ in range(N)]
        n_dom = [0] * N
        fronts: List[List[int]] = [[]]

        for p in range(N):
            for q in range(N):
                if p == q:
                    continue
                if _dominates(errors[p], sizes[p], errors[q], sizes[q]):
                    S[p].append(q)
                elif _dominates(errors[q], sizes[q], errors[p], sizes[p]):
                    n_dom[p] += 1
            if n_dom[p] == 0:
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and len(fronts[i]) > 0:
            next_front: List[int] = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        return fronts

    def _select_parents(population: np.ndarray, errors: List[float], sizes: List[int], n_parents: int) -> np.ndarray:
        """
        Repo selection behavior:
          - fill parents by entire fronts
          - if last front overflows, randomly shuffle that front and take remainder
        """
        fronts = _fast_non_dominated_sort(errors, sizes)
        chosen: List[int] = []

        for front in fronts:
            if len(chosen) + len(front) <= n_parents:
                chosen.extend(front)
            else:
                # random shuffle boundary front, then take needed
                front = list(front)
                rng.shuffle(front)
                need = n_parents - len(chosen)
                chosen.extend(front[:need])
                break

        return population[np.array(chosen, dtype=int)]

    def _one_point_crossover(mother: np.ndarray, father: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Repo crossover:
          - one point split i, child1 = mother[:i] + father[i:], child2 swapped
          - applied with probability crossover_prob; else clone parents
        """
        if rng.random() > crossover_prob:
            return mother.copy(), father.copy()

        if n_features <= 1:
            return mother.copy(), father.copy()

        i = int(rng.integers(1, n_features))  # [1, n_features-1] effectively for n_features>1
        c1 = np.concatenate([mother[:i], father[i:]]).astype(np.int8, copy=False)
        c2 = np.concatenate([father[:i], mother[i:]]).astype(np.int8, copy=False)
        return c1, c2

    def _one_point_mutation(child: np.ndarray) -> np.ndarray:
        """
        Repo mutation:
          - choose one random index j and flip bit
          - guard: if flipping produces all-zeros, undo the flip (keep at least one feature selected)
        Note: repo always mutates once per child. We mimic that.
        """
        if n_features <= 0:
            return child

        j = int(rng.integers(0, n_features))
        mutated = child.copy()
        mutated[j] = 1 - mutated[j]

        if mutated.sum() == 0:
            # undo (repo prevents empty solution)
            mutated[j] = 1
        return mutated

    def _init_population() -> np.ndarray:
        """
        Repo initialisation uses random binary strings.
        We ensure no all-zero individuals.
        """
        pop = rng.integers(0, 2, size=(pop_size, n_features), dtype=np.int8)
        # fix all-zero rows
        zero_rows = np.where(pop.sum(axis=1) == 0)[0]
        for r in zero_rows:
            pop[r, int(rng.integers(0, n_features))] = 1
        return pop

    # -----------------------------
    # Subset evaluation (per mask)
    # -----------------------------
    def _accuracy_for_fold(mask: np.ndarray, fold: Dict[str, Any]) -> float:
        """
        Train a fresh model for the requested method on masked features and return test accuracy.
        This is the analog of repo's "evaluate error by CV" but integrated with your fold splits.
        """
        if mask.sum() <= 0:
            return 0.0

        X_train = np.asarray(fold["X_train"])
        X_test = np.asarray(fold["X_test"])
        y_train = np.asarray(fold["y_train"])
        y_test = np.asarray(fold["y_test"])

        Xtr = _apply_mask(X_train, mask)
        Xte = _apply_mask(X_test, mask)

        # ---- DT ----
        if m == "dt":
            model = DecisionTreeClassifier(criterion="entropy", random_state=random_state)
            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)
            return float(accuracy_score(y_test, y_pred))

        # ---- XGB ----
        if m == "xgb":
            # Ensure 1D labels
            ytr = np.asarray(y_train).ravel()
            yte = np.asarray(y_test).ravel()

            n_classes = np.unique(ytr).size
            # if n_classes <= 0:
            #     return 0.0  # defensive; should never happen now
            # if n_classes == 1:
            #     # Degenerate fold (only one class in train). Accuracy = fraction of that class in test.
            #     majority = 0
            #     y_pred = np.full_like(yte, fill_value=majority)
            #     return float(accuracy_score(yte, y_pred))
            if n_classes == 2:
                objective = "binary:logistic"
                eval_metric = "logloss"
                extra = {}
            else:
                objective = "multi:softprob"
                eval_metric = "mlogloss"
                extra = {"num_class": n_classes}

            model = XGBClassifier(
                n_estimators=50,
                early_stopping_rounds=1,
                random_state=random_state,
                objective=objective,
                eval_metric=eval_metric,
                **extra,
            )

            model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
            y_pred = model.predict(Xte)
            return float(accuracy_score(yte, y_pred))

        # ---- MLP (1 hidden layer) ----
        if m == "mlp":
            # match your pipeline style: one hidden layer
            model = MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                max_iter=500,
                random_state=random_state,
            )
            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)
            return float(accuracy_score(y_test, y_pred))

        # ---- DNN (8 hidden layers) ----
        if m == "dnn":
            model = MLPClassifier(
                hidden_layer_sizes=(512, 512, 256, 256, 128, 128, 64, 64),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                max_iter=500,
                random_state=random_state,
            )
            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)
            return float(accuracy_score(y_test, y_pred))

        # ---- CBR ----
        if m == "cbr":
            Xtr_obj = np.asarray(Xtr, dtype=object)
            Xte_obj = np.asarray(Xte, dtype=object)
            model = CBR(categorical_idx=None, k=3, importance_type="gain", random_state=random_state)
            model.fit(Xtr_obj, y_train)
            y_pred = model.predict(Xte_obj)
            return float(accuracy_score(y_test, y_pred))

        # ---- ProtoPNet ----
        if m == "proto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            use_amp = bool(device.startswith("cuda"))

            # keep training modest; GA calls this a lot
            epochs = 30
            batch_size = 32
            lr = 1e-3
            weight_decay = 1e-4

            Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
            ytr_t = torch.tensor(y_train, dtype=torch.long)
            Xte_t = torch.tensor(Xte, dtype=torch.float32).to(device)
            yte_t = torch.tensor(y_test, dtype=torch.long).to(device)

            n_classes = int(torch.unique(ytr_t).numel())
            model = TabularProtoPNet(
                input_dim=Xtr_t.shape[1],
                n_classes=n_classes,
                n_prototypes_per_class=3,
            ).to(device)
            model.prototypes_class_identity = model.prototypes_class_identity.to(device)

            train_ds = TensorDataset(Xtr_t, ytr_t)
            train_loader = DataLoader(
                train_ds,
                batch_size=min(batch_size, len(train_ds)),
                shuffle=True,
                pin_memory=device.startswith("cuda"),
                num_workers=0,
            )

            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            model.train()
            for _ in range(epochs):
                for xb, yb in train_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits, _, _ = model(xb)
                        loss = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            model.eval()
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, _, _ = model(Xte_t)
                y_pred = torch.argmax(logits, dim=1)
            return float((y_pred == yte_t).float().mean().detach().cpu().item())

        raise RuntimeError("Unreachable")

    def _eval_mask(mask: np.ndarray) -> Tuple[float, int]:
        """
        Return (error, size) for the mask, matching repo objective semantics:
          - size = number of selected features
          - error = 1 - mean_accuracy_over_folds
        """
        size = int(mask.sum())
        if size <= 0:
            return 1.0, 0

        accs = []
        for fold in fold_list:
            accs.append(_accuracy_for_fold(mask, fold))
        mean_acc = float(np.mean(accs)) if accs else 0.0
        error = float(1.0 - mean_acc)
        return error, size

    _eval_calls = 0
    _eval_time_total = 0.0
    _last_log = time.time()
    _eval_cache: dict[bytes, tuple[float, int]] = {}

    def _timed_eval_mask(mask: np.ndarray) -> Tuple[float, int]:
        nonlocal _eval_calls, _eval_time_total
        key = mask.tobytes()

        cached = _eval_cache.get(key)
        if cached is not None:
            return cached

        t0 = time.time()
        out = _eval_mask(mask)
        dt = time.time() - t0

        _eval_calls += 1
        _eval_time_total += dt

        _eval_cache[key] = out
        return out

    def _avg_eval_sec() -> float:
        return (_eval_time_total / _eval_calls) if _eval_calls else 0.0

    # -----------------------------
    # GA execution (n_runs)
    # -----------------------------
    all_A_masks: List[np.ndarray] = []
    run_summaries: List[Dict[str, Any]] = []

    children_size = int(pop_size - parent_count)
    if children_size <= 0:
        raise ValueError("pop_size must be > parent_count")

    for run in tqdm(range(n_runs), desc=f"[{dataset}/{m}] runs", leave=True):
        pop = _init_population()

        # evaluate initial population
        errors = []
        sizes = []
        for i in range(pop.shape[0]):
            e, s = _timed_eval_mask(pop[i])
            errors.append(e)
            sizes.append(s)

        gen_bar = tqdm(range(n_generations), desc=f"run {run} gens", leave=False)
        for gen in gen_bar:
            # select parents (repo selection)
            parents = _select_parents(pop, errors, sizes, n_parents=parent_count)

            # recombination (repo: shuffle parents and pair off; create children_size children)
            parents_shuffled = parents.copy()
            rng.shuffle(parents_shuffled)

            children: List[np.ndarray] = []
            i = 0
            while len(children) < children_size:
                mother = parents_shuffled[i % parents_shuffled.shape[0]]
                father = parents_shuffled[(i + 1) % parents_shuffled.shape[0]]
                c1, c2 = _one_point_crossover(mother, father)
                children.append(c1)
                if len(children) < children_size:
                    children.append(c2)
                i += 2

            children_arr = np.vstack(children).astype(np.int8, copy=False)

            # mutation (repo: one-point mutation on each child; always applied once)
            for ci in range(children_arr.shape[0]):
                children_arr[ci] = _one_point_mutation(children_arr[ci])

            # evaluate parents ONCE (needed because parents is a subset of pop)
            parents_errors = []
            parents_sizes = []
            for pi in range(parents.shape[0]):
                e, s = _timed_eval_mask(parents[pi])
                parents_errors.append(e)
                parents_sizes.append(s)

            # evaluate children ONCE
            children_errors = []
            children_sizes = []
            for ci in range(children_arr.shape[0]):
                e, s = _timed_eval_mask(children_arr[ci])
                children_errors.append(e)
                children_sizes.append(s)

            # next population = parents + children (repo)
            pop = np.vstack([parents, children_arr])

            # reuse computed fitness (NO re-eval)
            errors = parents_errors + children_errors
            sizes = parents_sizes + children_sizes

            gen_bar.set_postfix({
                "evals": _eval_calls,
                "avg_eval_s": f"{_avg_eval_sec():.3f}",
                "pop": pop_size,
                "parents": parent_count,
                "children": children_size,
            })

        # final non-dominated set A = first front (repo uses all final population for stats; for synergy we need A)
        final_errors = errors
        final_sizes = sizes
        # for i in tqdm(range(pop.shape[0]), desc="init pop eval", leave=False):
        #     e, s = _timed_eval_mask(pop[i])
        #     final_errors.append(e)
        #     final_sizes.append(s)

        fronts = _fast_non_dominated_sort(final_errors, final_sizes)
        A_idx = fronts[0] if fronts else list(range(pop.shape[0]))
        A = pop[np.array(A_idx, dtype=int)]

        # deduplicate A
        A_unique = np.unique(A, axis=0)
        all_A_masks.extend([a.copy() for a in A_unique])

        run_summaries.append(
            {
                "run": int(run),
                "A_count": int(A_unique.shape[0]),
                "final_front_error_min": float(np.min([final_errors[i] for i in A_idx])) if A_idx else None,
                "final_front_size_min": int(np.min([final_sizes[i] for i in A_idx])) if A_idx else None,
            }
        )

    # Combine unique masks across runs
    if len(all_A_masks) == 0:
        raise RuntimeError("GA produced no candidate subsets A (unexpected).")

    A_all = np.unique(np.vstack(all_A_masks), axis=0)

    # Compute error/acc/size for A
    A_errors = []
    A_sizes = []
    for i in range(A_all.shape[0]):
        e, s = _timed_eval_mask(A_all[i])
        A_errors.append(e)
        A_sizes.append(s)
    A_errors = np.array(A_errors, dtype=float)
    A_sizes = np.array(A_sizes, dtype=float)
    A_acc = 1.0 - A_errors

    # -----------------------------
    # Feature importance (paper/repo formula)
    #   FI(i) = sum_{a in A, i in a} (1 - error(a)) / |a|
    # -----------------------------
    scores_a = A_acc / (A_sizes + 1e-12)

    feature_importance = np.zeros(n_features, dtype=float)
    for i in range(A_all.shape[0]):
        mask = A_all[i].astype(bool)
        feature_importance[mask] += scores_a[i]

    fi_sum = float(feature_importance.sum())
    feature_importance_norm = feature_importance / fi_sum if fi_sum > 0 else feature_importance

    # -----------------------------
    # Synergy matrix via Jaccard over co-occurrence sets
    # -----------------------------
    co = A_all.astype(bool)  # (|A|, P)
    inter = co.T @ co
    counts = co.sum(axis=0).astype(float)
    union = counts[:, None] + counts[None, :] - inter

    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(jaccard, 0.0)

    synergy = jaccard.copy()

    # Optional: scale synergy(i,j) by accuracy of subset {i,j}
    pair_acc = None
    if include_accuracy_factor:
        pair_acc = np.zeros((n_features, n_features), dtype=float)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                mask = np.zeros(n_features, dtype=np.int8)
                mask[i] = 1
                mask[j] = 1
                # mean acc over folds
                accs = [_accuracy_for_fold(mask, fold) for fold in fold_list]
                acc_ij = float(np.mean(accs)) if accs else 0.0
                pair_acc[i, j] = acc_ij
                pair_acc[j, i] = acc_ij
        synergy = synergy * pair_acc

    # top pairs
    pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            pairs.append(
                {
                    "i": int(i),
                    "j": int(j),
                    "feature_i": feature_names[i],
                    "feature_j": feature_names[j],
                    "synergy": float(synergy[i, j]),
                    "jaccard_raw": float(jaccard[i, j]),
                    "pair_accuracy": float(pair_acc[i, j]) if pair_acc is not None else None,
                }
            )
    pairs.sort(key=lambda d: d["synergy"], reverse=True)

    result = {
        "dataset_name": dataset,
        "method_name": m,
        "feature_names": feature_names,
        "ga_params": {
            "pop_size": int(pop_size),
            "n_generations": int(n_generations),
            "n_runs": int(n_runs),
            "parent_count": int(parent_count),
            "children_size": int(children_size),
            "crossover_prob": float(crossover_prob),
            # mutation_prob kept for API compatibility, but repo uses one-point mutation per child.
            "mutation_prob_param": float(mutation_prob),
            "mutation_operator": "one_point_flip_per_child_with_no_empty_guard",
            "dominance_rule": "error_i < error_j AND size_i <= size_j",
            "boundary_front_tiebreak": "random_shuffle_then_take",
            "include_accuracy_factor": bool(include_accuracy_factor),
            "random_state": int(random_state),
        },
        "ga_run_summaries": run_summaries,
        "A_size": int(A_all.shape[0]),
        "A_subsets": [np.where(A_all[i].astype(bool))[0].astype(int).tolist() for i in range(A_all.shape[0])],
        "A_subset_sizes": A_sizes.astype(int).tolist(),
        "A_subset_accuracy": A_acc.astype(float).tolist(),
        "feature_importance": feature_importance.astype(float).tolist(),
        "feature_importance_normalized": feature_importance_norm.astype(float).tolist(),
        "synergy_matrix": synergy.astype(float).tolist(),
        "top_pairs": pairs[: min(50, len(pairs))],
    }

    dataset_feature_synergy.setdefault("by_method", {})
    dataset_feature_synergy["by_method"][m] = result
    # save_model(dataset_feature_synergy, fs_path)

    return result

def feature_synergy_all_methods_for_dataset(
    dataset: str,
    fold_models_for_dataset: Dict[str, Any],
    *,
    methods: Optional[Iterable[str]] = None,
    pop_size: int = 200,
    n_generations: int = 150,
    n_runs: int = 10,
    parent_count: int = 100,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.02,
    include_accuracy_factor: bool = True,
    random_state: int = 42,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """
    Wrapper to compute Feature Synergy for all six AI methods for a single dataset,
    using your `fold_models_for_dataset` dict (same object you use for SOC).

    Parameters
    ----------
    dataset : str
        Dataset name (e.g., "iris")
    fold_models_for_dataset : dict
        Dict mapping method_name -> folds, where folds is list/dict of fold dicts containing:
        fold, train_idx, test_idx, X_train, X_test, y_train, y_test, scaler, model, performance_metrics, ...
    methods : iterable[str] or None
        Which methods to run. Defaults to: ["dt","xgb","cbr","proto","mlp","dnn"] (if present in input)
    force_recompute : bool
        If True, recompute even if cached artifact exists.

    Returns
    -------
    dict with:
      - dataset_name
      - by_method: {method: feature_synergy_measure_result}
    """
    out_path = base_dir / dataset / f"{dataset}_feature_synergy_all_methods.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and (not force_recompute):
        return load_model(out_path)

    if methods is None:
        methods = ["dt", "xgb", "cbr", "proto", "mlp", "dnn"]

    results: Dict[str, Any] = {"dataset_name": dataset, "by_method": {}}

    for method in methods:
        m = str(method).lower().strip()
        if m not in fold_models_for_dataset:
            # skip methods not available for this dataset
            continue

        res_m = feature_synergy_measure(
            dataset,
            m,
            fold_models_for_dataset[m],
            pop_size=pop_size,
            n_generations=n_generations,
            n_runs=n_runs,
            parent_count=parent_count,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            include_accuracy_factor=include_accuracy_factor,
            random_state=random_state,
        )
        results["by_method"][m] = res_m

    save_model(results, out_path)
    return results


@dataclass(frozen=True)
class FeatureRangeConstraint:
    """Constraint Ci: keep feature f_idx within [alpha, beta] during mutation."""
    f_idx: int
    alpha: float
    beta: float


def robustness_measure(
    dataset_name: str,
    method_name: str,
    folds: List[Dict[str, Any]],
    *,
    k_bins: int = 10,                    # number of bins per feature
    binning: str = "quantile",           # "quantile" or "uniform"
    use_constraints: bool = True,        # apply Ci to GA mutation (NeighbourRobustness)
    N: int = 200,                        # population size
    G: int = 100,                        # number of generations
    gamma: float = 0.7,
    mu: float = 0.2,
    cf_only: bool = True,                # use only label-flipping candidates for E[d(x,c)]
    min_cf_required: int = 1,            # skip x if fewer counterfactuals than this
    max_instances_per_subpop: Optional[int] = None,  # cap instances per (feature,bin) subpopulation
    random_state: int = 42,
    progress: bool = True,               # NEW: allow disabling progress bars
) -> Dict[str, Any]:
    """
    Robustness implementation (feature -> bins -> subpopulation -> instances -> GA),
    with performance improvements:
      1) Fewer tqdm objects: only folds + per-fold (feature,bin) "cells" bar.
      2) Vectorized evaluate_batch (no Python loop over population for distance).
      3) (Minor) avoid nested instance tqdm.

    NOTE: This keeps your algorithmic structure the same.
    """

    if binning not in {"quantile", "uniform"}:
        raise ValueError('binning must be "quantile" or "uniform"')

    if progress and tqdm is None:
        progress = False  # tqdm not available; silently disable

    # --------------------
    # Helpers: distance
    # --------------------
    def _var(v: np.ndarray) -> float:
        v = np.asarray(v, dtype=float).ravel()
        if v.size == 0:
            return 0.0
        mu_v = float(np.mean(v))
        return float(np.mean((v - mu_v) ** 2))

    # (Kept for use when scoring CF distances; fast enough there)
    def norm_euclid(x: np.ndarray, c: np.ndarray, eps: float = 1e-12) -> float:
        x = np.asarray(x, dtype=float).ravel()
        c = np.asarray(c, dtype=float).ravel()
        denom = _var(x) + _var(c)
        if denom < eps:
            return 0.0 if np.allclose(x, c, atol=1e-9, rtol=0.0) else 1.0
        return float(np.clip(0.5 * (_var(x - c) / denom), 0.0, 1.0))

    # --------------------
    # Helpers: binning
    # --------------------
    def make_bins(values: np.ndarray, k: int) -> List[Tuple[float, float]]:
        v = np.asarray(values, dtype=float)
        if k <= 0:
            raise ValueError("k_bins must be >= 1")

        if binning == "quantile":
            edges = np.quantile(v, q=np.linspace(0, 1, k + 1))
        else:
            edges = np.linspace(np.min(v), np.max(v), k + 1)

        edges = np.unique(edges)
        if edges.size < 2:
            return [(float(v[0]), float(v[0]))]

        return [(float(edges[i]), float(edges[i + 1])) for i in range(edges.size - 1)]

    # --------------------
    # Helpers: model
    # --------------------
    def predict_label(model: Any, X: np.ndarray) -> np.ndarray:
        y = np.asarray(model.predict(X))
        if y.ndim == 2:
            return np.argmax(y, axis=1)
        return y.astype(int)

    # --------------------
    # GA parts
    # --------------------
    def select_top(P: np.ndarray, fit: np.ndarray, N_: int) -> np.ndarray:
        order = np.argsort(-fit)
        return P[order[:N_]]

    def crossover_two_point(rng: np.random.Generator, P: np.ndarray, m_: int, gamma_: float) -> np.ndarray:
        Pc = P.copy()
        idx = rng.permutation(Pc.shape[0])
        for kk in range(0, Pc.shape[0] - 1, 2):
            if rng.random() <= gamma_:
                i1, i2 = idx[kk], idx[kk + 1]
                a = int(rng.integers(0, m_))
                b = int(rng.integers(0, m_))
                lo, hi = (a, b) if a <= b else (b, a)
                tmp = Pc[i1, lo:hi].copy()
                Pc[i1, lo:hi] = Pc[i2, lo:hi]
                Pc[i2, lo:hi] = tmp
        return Pc

    def mutate_empirical(
        rng: np.random.Generator,
        P: np.ndarray,
        feat_values: List[np.ndarray],
        m_: int,
        mu_: float,
        constraint: Optional[FeatureRangeConstraint],
    ) -> np.ndarray:
        Pm = P.copy()
        for ii in range(Pm.shape[0]):
            if rng.random() <= mu_:
                fj = int(rng.integers(0, m_))
                vals = feat_values[fj]
                Pm[ii, fj] = float(vals[int(rng.integers(0, vals.shape[0]))])

                if constraint is not None:
                    fi = constraint.f_idx
                    vals_fi = feat_values[fi]
                    mask = (vals_fi >= constraint.alpha) & (vals_fi <= constraint.beta)
                    if np.any(mask):
                        feasible = vals_fi[mask]
                        Pm[ii, fi] = float(feasible[int(rng.integers(0, feasible.shape[0]))])
                    else:
                        Pm[ii, fi] = float(np.clip(Pm[ii, fi], constraint.alpha, constraint.beta))
        return Pm

    # --------------------
    # NEW: Vectorized evaluate_batch
    # --------------------
    def evaluate_batch_vectorized(model: Any, P: np.ndarray, x: np.ndarray, y_x: int) -> np.ndarray:
        """
        One predict for the whole population + vectorized NormEuclid.
        """
        yP = predict_label(model, P)
        I_diff = (yP != y_x).astype(float)

        x_row = x.reshape(1, -1)
        I_same = np.all(P == x_row, axis=1).astype(float)

        # Vectorized NormEuclid d(x, P[i])
        mean_x = x.mean()
        var_x = ((x - mean_x) ** 2).mean()

        mean_P = P.mean(axis=1)
        var_P = ((P - mean_P[:, None]) ** 2).mean(axis=1)

        diff = x_row - P
        mean_diff = diff.mean(axis=1)
        var_diff = ((diff - mean_diff[:, None]) ** 2).mean(axis=1)

        denom = var_x + var_P
        d = np.where(
            denom < 1e-12,
            np.where(I_same > 0, 0.0, 1.0),
            0.5 * (var_diff / denom),
        )
        d = np.clip(d, 0.0, 1.0)

        return I_diff + (1.0 - d) - I_same

    def genetic_alg(
        rng: np.random.Generator,
        model: Any,
        x: np.ndarray,
        y_x: int,  # NEW: pass y_x to avoid recomputing predict_label(x)
        feat_values: List[np.ndarray],
        m_: int,
        *,
        N_: int,
        G_: int,
        gamma_: float,
        mu_: float,
        constraint: Optional[FeatureRangeConstraint],
    ) -> np.ndarray:
        P = np.repeat(x.reshape(1, -1), repeats=N_, axis=0)  # P0
        fit = evaluate_batch_vectorized(model, P, x, y_x)

        for _ in range(G_):
            P_sel = select_top(P, fit, N_)
            P_cross = crossover_two_point(rng, P_sel, m_, gamma_)
            P_mut = mutate_empirical(rng, P_cross, feat_values, m_, mu_, constraint)
            fit = evaluate_batch_vectorized(model, P_mut, x, y_x)
            P = P_mut

        return P

    def _tuple_keys_to_str(d):
        return {f"{fi}|{bj}": v for (fi, bj), v in d.items()}
    # --------------------
    # Main loop (folds)
    # --------------------
    rng_master = np.random.default_rng(random_state)
    m = int(np.asarray(folds[0]["X_test"]).shape[1])

    cell_sum: Dict[Tuple[int, int], float] = {}
    cell_count: Dict[Tuple[int, int], int] = {}
    cell_nx: Dict[Tuple[int, int], int] = {}
    per_fold_debug = []

    fold_iter = (
        tqdm(enumerate(folds), total=len(folds), desc=f"{dataset_name}/{method_name} folds")
        if progress else enumerate(folds)
    )

    for fold_idx, fold in fold_iter:
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        X_test = np.asarray(fold["X_test"], dtype=float)
        model = fold.get("model", None)
        if model is None:
            raise ValueError('fold["model"] is required.')

        # NOTE: if you want train-distribution mutation, swap to fold["X_train"]
        feat_values = [X_test[:, j].astype(float) for j in range(m)]
        fold_cells_used = 0

        # Precompute number of cells for this fold to drive a single progress bar
        if progress:
            total_cells = 0
            for fi in range(m):
                total_cells += len(make_bins(X_test[:, fi], k_bins))
            cell_pbar = tqdm(total=total_cells, desc=f"fold {fold_idx} cells", leave=False)
        else:
            cell_pbar = None

        for fi in range(m):
            bins = make_bins(X_test[:, fi], k_bins)
            for bj, (alpha, beta) in enumerate(bins):
                # Update cell progress early
                if cell_pbar is not None:
                    cell_pbar.update(1)

                # Subpopulation indices
                if bj == len(bins) - 1:
                    mask = (X_test[:, fi] >= alpha) & (X_test[:, fi] <= beta)
                else:
                    mask = (X_test[:, fi] >= alpha) & (X_test[:, fi] < beta)

                idxs = np.where(mask)[0]
                if idxs.size == 0:
                    continue

                if max_instances_per_subpop is not None and idxs.size > max_instances_per_subpop:
                    idxs = rng.choice(idxs, size=int(max_instances_per_subpop), replace=False)

                constraint = FeatureRangeConstraint(fi, alpha, beta) if use_constraints else None

                cell_scores = []

                # No nested tqdm here (much cheaper)
                for j in idxs:
                    x = X_test[j]
                    y_x = int(predict_label(model, x.reshape(1, -1))[0])

                    P_G = genetic_alg(
                        rng, model, x, y_x, feat_values, m,
                        N_=N, G_=G, gamma_=gamma, mu_=mu, constraint=constraint
                    )

                    if cf_only:
                        y_P = predict_label(model, P_G)
                        CF = P_G[y_P != y_x]
                    else:
                        CF = P_G

                    if CF.shape[0] < min_cf_required:
                        continue

                    # (This part is fine to keep simple; CF is usually smaller than N)
                    dists = np.array([norm_euclid(x, CF[t]) for t in range(CF.shape[0])], dtype=float)
                    cell_scores.append(float(np.mean(dists)))

                if not cell_scores:
                    continue

                key = (fi, bj)
                cell_sum[key] = cell_sum.get(key, 0.0) + float(np.mean(cell_scores))
                cell_count[key] = cell_count.get(key, 0) + 1
                cell_nx[key] = cell_nx.get(key, 0) + len(cell_scores)
                fold_cells_used += 1

                if cell_pbar is not None:
                    cell_pbar.set_postfix({"cells_used": fold_cells_used, "subpop_n": int(len(idxs))})

        if cell_pbar is not None:
            cell_pbar.close()

        per_fold_debug.append({"fold": int(fold.get("fold", fold_idx)), "cells_used": int(fold_cells_used)})

    cell_means = {k: (cell_sum[k] / cell_count[k]) for k in cell_sum.keys()}
    overall = float(np.mean(list(cell_means.values()))) if cell_means else float("nan")
    cell_means_json = _tuple_keys_to_str(cell_means)
    cell_nx_json = _tuple_keys_to_str(cell_nx)

    return {
        "dataset": dataset_name,
        "method": method_name,
        "overall_robustness": overall,
        "cell_means": cell_means_json,
        "cell_instances_used": cell_nx_json,
        "per_fold": per_fold_debug,
        "config": {
            "k_bins": int(k_bins),
            "binning": binning,
            "use_constraints": bool(use_constraints),
            "N": int(N), "G": int(G), "gamma": float(gamma), "mu": float(mu),
            "cf_only": bool(cf_only),
            "min_cf_required": int(min_cf_required),
            "max_instances_per_subpop": max_instances_per_subpop,
            "random_state": int(random_state),
        },
    }


def robustness_all_methods_for_dataset(
    dataset: str,
    fold_models_for_dataset: Dict[str, Any],
    *,
    methods: Optional[Iterable[str]] = None,
    k_bins: int = 10,
    binning: str = "quantile",
    use_constraints: bool = True,
    N: int = 200,
    G: int = 100,
    gamma: float = 0.7,
    mu: float = 0.2,
    cf_only: bool = True,
    min_cf_required: int = 1,
    max_instances_per_subpop: Optional[int] = None,
    random_state: int = 42,
    progress: bool = True,  # NEW
) -> Dict[str, Any]:

    out_path = base_dir / dataset / f"{dataset}_robustness_all_methods.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return load_model(out_path)

    if methods is None:
        methods = ["dt", "xgb", "cbr", "proto", "mlp", "dnn"]

    results: Dict[str, Any] = {"dataset_name": dataset, "by_method": {}}

    method_list = list(methods)
    method_iter = tqdm(method_list, desc=f"{dataset} methods") if (progress and tqdm) else method_list

    for method in method_iter:
        mth = str(method).lower().strip()
        if mth not in fold_models_for_dataset:
            continue

        res_m = robustness_measure(
            dataset,
            mth,
            fold_models_for_dataset[mth],
            k_bins=k_bins,
            binning=binning,
            use_constraints=use_constraints,
            N=N,
            G=G,
            gamma=gamma,
            mu=mu,
            cf_only=cf_only,
            min_cf_required=min_cf_required,
            max_instances_per_subpop=max_instances_per_subpop,
            random_state=random_state,
            progress=progress,
        )
        results["by_method"][mth] = res_m

    save_model(results, out_path)
    return results


def _patch_skexplain_run_parallel():
    """
    Python 3.14 + skexplain workaround:
    skexplain's run_parallel does copy(args_iterator) where args_iterator can be itertools.product,
    which raises:
        TypeError: cannot pickle 'itertools.product' object

    BUT: global_explainer often imports run_parallel directly, so patch both:
      - skexplain.common.multiprocessing_utils.run_parallel
      - skexplain.main.global_explainer.run_parallel
    """
    import skexplain.common.multiprocessing_utils as mp_utils
    import skexplain.main.global_explainer as global_explainer

    # if we've patched already, skip
    if getattr(mp_utils, "_patched_iter_copy_bug", False):
        # still ensure the global_explainer alias points to the patched version
        if getattr(global_explainer, "run_parallel", None) is not mp_utils.run_parallel:
            global_explainer.run_parallel = mp_utils.run_parallel
        return

    _orig_run_parallel = mp_utils.run_parallel

    def _run_parallel_wrapper(*args, **kwargs):
        # Handle kwargs style
        if "args_iterator" in kwargs:
            ai = kwargs["args_iterator"]
            if not isinstance(ai, list):
                kwargs["args_iterator"] = list(ai)
            return _orig_run_parallel(*args, **kwargs)

        # Handle positional style: run_parallel(func, args_iterator, ...)
        if len(args) >= 2:
            args = list(args)
            ai = args[1]
            if not isinstance(ai, list):
                args[1] = list(ai)
            return _orig_run_parallel(*args, **kwargs)

        return _orig_run_parallel(*args, **kwargs)

    # Patch in both modules
    mp_utils.run_parallel = _run_parallel_wrapper
    mp_utils._patched_iter_copy_bug = True

    # Critical: also patch the imported alias in global_explainer
    global_explainer.run_parallel = _run_parallel_wrapper


def _patch_numpy_percentile_interpolation():
    """
    NumPy 2.x removed `interpolation=` from np.percentile in favor of `method=`.
    skexplain still uses `interpolation=...`, so we shim it.
    """
    import numpy as _np

    if getattr(_np, "_patched_percentile_interpolation", False):
        return

    _orig = _np.percentile

    def _percentile_wrapper(a, q, *args, **kwargs):
        # If skexplain passes interpolation="lower", map -> method="lower"
        if "interpolation" in kwargs and "method" not in kwargs:
            kwargs["method"] = kwargs.pop("interpolation")
        elif "interpolation" in kwargs and "method" in kwargs:
            # If both are provided, drop interpolation to avoid errors
            kwargs.pop("interpolation", None)
        return _orig(a, q, *args, **kwargs)

    _np.percentile = _percentile_wrapper
    _np._patched_percentile_interpolation = True


# ============================================================
#  NF (Number of Features Used)
# ============================================================

def _number_of_features_used(
    model,
    X: np.ndarray,
    *,
    M: int = 200,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)

    X = np.asarray(X)
    n, p = X.shape
    M = min(M, n)

    idx = rng.choice(n, size=M, replace=False)
    Xs = X[idx].copy()

    base_pred = model.predict(Xs)
    used_mask = np.zeros(p, dtype=bool)

    for j in range(p):
        Xp = Xs.copy()

        donor_idx = rng.integers(0, M, size=M)
        donor_idx[donor_idx == np.arange(M)] = (donor_idx[donor_idx == np.arange(M)] + 1) % M

        Xp[:, j] = Xs[donor_idx, j]

        pert_pred = model.predict(Xp)

        if np.any(pert_pred != base_pred):
            used_mask[j] = True

    return {
        "NF": int(np.sum(used_mask)),
        "used_mask": used_mask,
    }



def mec_all_methods_for_datasets(
        dataset: str,
        fold_models_for_dataset: Dict[str, Any],
        *,
        methods: Optional[Iterable[str]] = None,
        n_bins: int = 20,
        subsample: float = 1.0,
        n_bootstrap: int = 1,
        mec_max_segments: int = 10,
        mec_approx_error: float = 0.05,
        nf_M: int = 200,
        progress: bool = True,
        n_jobs: int = 1,
) -> Dict[str, Any]:
    """
    Computes:
      - 1D ALE for all features
      - IAS (overall interaction strength) via explainer.interaction_strength(ale=ale_1d, ...)
      - MEC via explainer.main_effect_complexity(ale_1d, ...)
      - NF (Number of Features Used) via your Algorithm-1 style perturbation

    Notes:
      - Uses your existing caching pattern with base_dir (must exist as a Path in this module).
      - Requires the helper patches:
          _patch_numpy_percentile_interpolation()
          _patch_skexplain_run_parallel()
      - Requires helper:
          _number_of_features_used(...)
    """

    out_path = base_dir / dataset / f"{dataset}_mec_all_methods.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return load_model(out_path)

    if methods is None:
        methods = ["dt", "xgb", "cbr", "proto", "mlp", "dnn"]

    # Apply patches once up-front (safe to call repeatedly)
    _patch_numpy_percentile_interpolation()
    _patch_skexplain_run_parallel()

    results: Dict[str, Any] = {
        "dataset_name": dataset,
        "by_method": {},
        "config": {
            "n_bins": n_bins,
            "subsample": subsample,
            "n_bootstrap": n_bootstrap,
            "mec_max_segments": mec_max_segments,
            "mec_approx_error": mec_approx_error,
            "nf_M": nf_M,
            "n_jobs": n_jobs,
        },
    }

    method_iter = tqdm(methods, desc=f"{dataset} MEC") if (progress and tqdm) else methods

    for method in method_iter:
        mth = str(method).lower().strip()
        if mth not in fold_models_for_dataset:
            continue

        folds = fold_models_for_dataset[mth]
        if not folds:
            continue

        method_results: Dict[str, Any] = {"folds": []}

        for fold_data in folds:
            fold_id = fold_data["fold"]
            X_train = np.asarray(fold_data["X_train"])
            y_train = np.asarray(fold_data["y_train"])
            X_test = np.asarray(fold_data["X_test"])
            model = fold_data["model"]

            feature_names = fold_data.get(
                "feature_names",
                [f"f{i}" for i in range(X_train.shape[1])]
            )

            estimator_name = f"{mth}_fold{fold_id}"

            explainer = skexplain.ExplainToolkit(
                estimators=(estimator_name, model),  # your skexplain version needs a tuple
                X=X_train,
                y=y_train,
                feature_names=feature_names,         # required when X is numpy
            )

            # -----------------------------
            #  1D ALE (all features)
            # -----------------------------
            ale_1d = explainer.ale(
                features="all",
                n_bins=n_bins,
                subsample=subsample,
                n_bootstrap=n_bootstrap,
                n_jobs=n_jobs,
            )

            # -----------------------------
            #  IAS (overall) per tutorial:
            #     ias = explainer.interaction_strength(ale=ale_1d, ...)
            # -----------------------------
            ias_ds = explainer.interaction_strength(
                ale=ale_1d,
                subsample=subsample,
                n_bootstrap=n_bootstrap,
            )

            # robust scalar extraction from xarray.Dataset
            ias_numeric_vars = [
                v for v in ias_ds.data_vars
                if np.issubdtype(ias_ds[v].dtype, np.number)
            ]
            if not ias_numeric_vars:
                raise RuntimeError(
                    f"interaction_strength returned no numeric data_vars. Got: "
                    f"{[(k, str(ias_ds[k].dtype), ias_ds[k].shape) for k in ias_ds.data_vars]}"
                )
            _ias_vals = np.asarray(ias_ds[ias_numeric_vars[0]].values, dtype=float).reshape(-1)
            ias_overall = float(np.mean(_ias_vals))  # mean across bootstraps if present

            # -----------------------------
            #  MEC per docs:
            #     mec = explainer.main_effect_complexity(ale_1d, ...)
            # -----------------------------
            mec_dict = explainer.main_effect_complexity(
                ale_1d,
                estimator_names=estimator_name,
                max_segments=mec_max_segments,
                approx_error=mec_approx_error,
            )
            mec_value = float(mec_dict[estimator_name])

            # -----------------------------
            #  NF (your Algorithm 1)
            # -----------------------------
            nf_result = _number_of_features_used(
                model,
                X_test,
                M=nf_M,
            )

            method_results["folds"].append({
                "fold": fold_id,
                "feature_names": list(feature_names),
                "ale_1d": ale_1d,          # xarray Dataset (can be large)
                "ias_overall": ias_overall,
                "ias_ds": ias_ds,          # optional, but useful for debugging
                "mec": mec_value,
                "nf": nf_result,
            })

        # -----------------------------
        #  Aggregate across folds
        # -----------------------------
        mec_vals = [f["mec"] for f in method_results["folds"]]
        nf_vals = [f["nf"]["NF"] for f in method_results["folds"]]
        ias_vals = [f["ias_overall"] for f in method_results["folds"]]

        method_results["aggregate"] = {
            "ias_mean": float(np.mean(ias_vals)) if ias_vals else None,
            "ias_std": float(np.std(ias_vals)) if ias_vals else None,
            "mec_mean": float(np.mean(mec_vals)) if mec_vals else None,
            "mec_std": float(np.std(mec_vals)) if mec_vals else None,
            "nf_mean": float(np.mean(nf_vals)) if nf_vals else None,
            "nf_std": float(np.std(nf_vals)) if nf_vals else None,
            "n_folds": int(len(method_results["folds"])),
        }

        results["by_method"][mth] = method_results

    save_model(results, out_path)
    return results

