import shap
from model_save_functions import save_model, load_model
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances

base_dir = Path.cwd()/"trained_models"
#
# def shap_attributions(explainer, model, X, y_pred):
#     '''
#     Returns SHAP attributions for the predicted class for each instance in X.
#     :param explainer: SHAP explainer object
#     :param model: Trained classifier object
#     :param X: Test set features
#     :param y_pred: Predicted class by trained classifier
#     :return:
#         SHAP attributions for test set with SHAP Explainer for predicted class by trained classifier.
#     '''
#
#     sv = explainer.shap_values(X)
#     class_idx = np.array([np.where(model.classes_ == y)[0][0] for y in y_pred])
#     sv_pred = sv[np.arange(len(X)), :, class_idx]
#     return sv_pred
#
#
# def CART_DT_5fold_shap(dataset, folds, *, random_state = 42, max_samples=None):
#     '''
#         This function produces SHAP feature attributions for test set with Decision Tree Classifier.
#         :param folds: Dictionary of folds structured as:
#             fold: Fold No.
#             train_idx: Indices of training samples
#             test_idx: Indices of testing samples
#             X_train: Features of training samples
#             X_test: Features of testing samples
#             y_train: Labels of training samples
#             y_test: Labels of testing samples
#             scaler: Standard Scaler object for standardization
#             model: Decision Tree classifier object
#             performance_metrics: Dictionary of performance metrics and their corresponding values for the trained DT classifier
#
#         :return: Dictionary of SHAP feature attributions for test set with Decision Tree Classifier
#             ...
#             explainer: SHAP explainer object
#             feature_attribution_all_classes: Feature attributions for all classes for testing instances
#             feature_attribution_pred_class: Feature attributions for predicted class via Decision Tree Classifier
#             y_pred: Predicted class by Decision Tree Classifier
#         '''
#
#     post_hoc_path = base_dir/dataset/f"dt_fold_model_shap.joblib"
#     DT_5fold_model_shap = []
#     if post_hoc_path.exists():
#         DT_5fold_model_shap = load_model(post_hoc_path)
#     else:
#         for fold_idx, fold in enumerate(folds):
#             dt_model = fold['model']
#             X_test = fold['X_test']
#             y_test = fold['y_test']
#
#             if max_samples is not None and len(X_test) > max_samples:
#                 X_test = X_test[:max_samples]
#
#             explainer = shap.TreeExplainer(dt_model)
#             sv = explainer.shap_values(X_test)
#
#             y_pred = dt_model.predict(X_test)
#             class_idx = np.array([np.where(dt_model.classes_==y)[0][0] for y in y_pred])
#             sv_pred = sv[np.arange(len(X_test)), :, class_idx]
#             DT_5fold_model_shap.append({
#                 "fold": fold['fold'],
#                 "train_idx": fold['train_idx'],
#                 "test_idx": fold['test_idx'],
#                 "X_train": fold['X_train'],
#                 "X_test": fold['X_test'],
#                 "y_train": fold['y_train'],
#                 "y_test": fold['y_test'],
#                 "scaler": fold['scaler'],
#                 "model": fold['model'],
#                 "performance_metrics": fold['performance_metrics'],
#                 # "explainer": explainer,
#                 "feature_attribution_all_classes": sv,
#                 "feature_attribution_pred_class": sv_pred,
#                 "y_pred": y_pred
#             })
#         save_model(DT_5fold_model_shap, post_hoc_path)
#     return DT_5fold_model_shap
#
# def XGB_5fold_shap(dataset, folds, *, random_state = 42, max_samples=None):
#     '''
#         This function produces SHAP feature attributions for test set with XGB Classifier.
#         :param folds: Dictionary of folds structured as:
#             fold: Fold No.
#             train_idx: Indices of training samples
#             test_idx: Indices of testing samples
#             X_train: Features of training samples
#             X_test: Features of testing samples
#             y_train: Labels of training samples
#             y_test: Labels of testing samples
#             scaler: Standard Scaler object for standardization
#             model: Decision Tree classifier object
#             performance_metrics: Dictionary of performance metrics and their corresponding values for the trained XGB classifier
#
#         :return: Dictionary of SHAP feature attributions for test set with XGB Classifier
#             ...
#             explainer: SHAP explainer object
#             feature_attribution_all_classes: Feature attributions for all classes for testing instances
#             feature_attribution_pred_class: Feature attributions for predicted class via XGB Classifier
#             y_pred: Predicted class by XGB Classifier
#         '''
#
#     post_hoc_path = base_dir/dataset/f"xgb_fold_model_shap.joblib"
#     XGB_5fold_model_shap = []
#     if post_hoc_path.exists():
#         XGB_5fold_model_shap = load_model(post_hoc_path)
#     else:
#         for fold_idx, fold in enumerate(folds):
#             dt_model = fold['model']
#             X_test = fold['X_test']
#
#             if max_samples is not None and len(X_test) > max_samples:
#                 X_test = X_test[:max_samples]
#
#             explainer = shap.TreeExplainer(dt_model)
#             sv = explainer.shap_values(X_test)
#
#             y_pred = dt_model.predict(X_test)
#             class_idx = np.array([np.where(dt_model.classes_==y)[0][0] for y in y_pred])
#
#             # Case A: multiclass often returns a list: [ (n, f), (n, f), ... ] length = n_classes
#             if isinstance(sv, list):
#                 # stack to (n, f, c)
#                 sv_all = np.stack(sv, axis=2)
#                 sv_pred = sv_all[np.arange(len(X_test)), :, class_idx]
#
#             # Case B: some setups return (n, f, c) already
#             elif getattr(sv, "ndim", None) == 3:
#                 sv_all = sv
#                 sv_pred = sv_all[np.arange(len(X_test)), :, class_idx]
#
#             # Case C: binary/single-output returns (n, f) only
#             elif getattr(sv, "ndim", None) == 2:
#                 sv_all = sv
#                 # no per-class axis exists;best definition of "pred class" is just the same attribution
#                 sv_pred = sv_all
#
#             XGB_5fold_model_shap.append({
#                 "fold": fold['fold'],
#                 "train_idx": fold['train_idx'],
#                 "test_idx": fold['test_idx'],
#                 "X_train": fold['X_train'],
#                 "X_test": fold['X_test'],
#                 "y_train": fold['y_train'],
#                 "y_test": fold['y_test'],
#                 "scaler": fold['scaler'],
#                 "model": fold['model'],
#                 "performance_metrics": fold['performance_metrics'],
#                 # "explainer": explainer,
#                 "feature_attribution_all_classes": sv,
#                 "feature_attribution_pred_class": sv_pred,
#                 "y_pred": y_pred
#             })
#         save_model(XGB_5fold_model_shap, post_hoc_path)
#     return XGB_5fold_model_shap
#
#
# class CBRProbaWrapper:
#     """
#     Pickle-friendly callable for SHAP KernelExplainer.
#
#     Holds:
#       - cbr_model: the trained CBR model
#       - cat_cols: list of categorical column indices (encoded as ints)
#       - round_categoricals: whether to round cat cols before calling predict_proba
#     """
#     def __init__(self, cbr_model, cat_cols, *, round_categoricals=True):
#         self.cbr_model = cbr_model
#         self.cat_cols = list(cat_cols)
#         self.round_categoricals = round_categoricals
#
#     def __call__(self, X_in):
#         X_in = np.asarray(X_in, dtype=np.float32)
#
#         if self.round_categoricals and len(self.cat_cols) > 0:
#             X_in = X_in.copy()
#             X_in[:, self.cat_cols] = np.rint(X_in[:, self.cat_cols]).astype(np.int32)
#
#         # IMPORTANT: ensure the class ordering matches self.cbr_model.classes_
#         return self.cbr_model.predict_proba(X_in, classes=self.cbr_model.classes_, round_categoricals=False)
#
#
#
# def CBR_5fold_shap(dataset, folds, *, random_state=42, max_samples=None, background_size=50, nsamples="auto"):
#     post_hoc_path = base_dir / dataset / f"cbr_fold_model_shap.joblib"
#     CBR_5fold_model_shap = []
#
#     if post_hoc_path.exists():
#         return load_model(post_hoc_path)
#
#     rng = np.random.RandomState(random_state)
#
#     for fold_idx, fold in enumerate(folds):
#         cbr_model = fold['model']
#
#         X_train_raw = np.asarray(fold["X_train"], dtype=object)
#         X_test_raw = np.asarray(fold["X_test"], dtype=object)
#
#         if max_samples is not None and len(X_test_raw) > max_samples:
#             X_test_raw = X_test_raw[:max_samples]
#
#         y_pred = cbr_model.predict(X_test_raw)
#
#         cat_cols = sorted(list(getattr(cbr_model, "categorical_idx", [])))
#
#         if len(cat_cols)>0:
#             X_train_enc, X_test_enc, enc = cbr_model.encode_categoricals_for_xgb(X_train_raw, X_test_raw, cat_cols)
#             X_train_enc = X_train_enc.astype(np.float32)
#             X_test_enc = X_test_enc.astype(np.float32)
#
#         else:
#             X_train_enc = X_train_raw.astype(np.float32)
#             X_test_enc = X_test_raw.astype(np.float32)
#             enc = None
#
#         if X_train_enc.shape[0] > background_size:
#             bg_idx = rng.choice(X_train_enc.shape[0], size=background_size, replace=False)
#             background = X_train_enc[bg_idx]
#         else:
#             background = X_train_enc
#
#         old_X_train = cbr_model.X_train_
#         try:
#             cbr_model.X_train_ = X_train_enc
#
#             f_proba=CBRProbaWrapper(cbr_model, cat_cols, round_categoricals=True)
#             explainer = shap.KernelExplainer(f_proba, background)
#
#             sv = explainer.shap_values(X_test_enc, nsamples=nsamples)
#             classes = np.asarray(cbr_model.classes_)
#             class_idx = np.array([np.where(classes == y)[0][0] for y in y_pred])
#
#             if isinstance(sv, list):
#                 sv_pred = np.stack([sv[class_idx[i]][i, :] for i in range(len(X_test_enc))], axis=0)
#             else:
#                 # fallback if SHAP returns array-like
#                 # expected shapes vary; safest is to index the last dim as class
#                 sv_pred = sv[np.arange(len(X_test_enc)), :, class_idx]
#
#             CBR_5fold_model_shap.append({
#                 "fold": fold["fold"],
#                 "train_idx": fold["train_idx"],
#                 "test_idx": fold["test_idx"],
#                 "X_train": fold["X_train"],
#                 "X_test": fold["X_test"],
#                 "y_train": fold["y_train"],
#                 "y_test": fold["y_test"],
#                 "scaler": fold.get("scaler", None),
#                 "model": fold["model"],
#                 "performance_metrics": fold.get("performance_metrics", None),
#
#                 # "explainer": None,
#                 "feature_attribution_all_classes": sv,
#                 "feature_attribution_pred_class": sv_pred,
#                 "y_pred": y_pred,
#
#                 # optional debug fields (handy)
#                 "categorical_idx": cat_cols,
#                 "encoder": enc,
#                 "X_test_encoded_used_for_shap": X_test_enc
#             })
#
#         finally:
#             cbr_model.X_train_ = old_X_train
#
#     save_model(CBR_5fold_model_shap, post_hoc_path)
#     return CBR_5fold_model_shap
#
#
#
# def rebuild_explainer_from_fold(method: str, fold: dict):
#     """
#     Rebuild a SHAP explainer from a saved fold dict.
#     - method: "dt", "xgb", "cbr"
#     """
#     model = fold["model"]
#
#     if method in ("dt", "xgb"):
#         return shap.TreeExplainer(model)
#
#     if method == "cbr":
#         # reconstruct the KernelExplainer the same way you did in CBR_5fold_shap
#         rng = np.random.RandomState(fold.get("shap_random_state", 42))
#         background_size = int(fold.get("shap_background_size", 50))
#         nsamples = fold.get("shap_nsamples", "auto")
#
#         cbr_model = model
#         X_train_raw = np.asarray(fold["X_train"], dtype=object)
#         X_test_raw  = np.asarray(fold["X_test"], dtype=object)
#
#         cat_cols = sorted(list(getattr(cbr_model, "categorical_idx", [])))
#
#         if len(cat_cols) > 0:
#             X_train_enc, X_test_enc, enc = cbr_model.encode_categoricals_for_xgb(
#                 X_train_raw, X_test_raw, cat_cols
#             )
#             X_train_enc = X_train_enc.astype(np.float32)
#         else:
#             X_train_enc = X_train_raw.astype(np.float32)
#
#         if X_train_enc.shape[0] > background_size:
#             bg_idx = rng.choice(X_train_enc.shape[0], size=background_size, replace=False)
#             background = X_train_enc[bg_idx]
#         else:
#             background = X_train_enc
#
#         old_X_train = cbr_model.X_train_
#         try:
#             cbr_model.X_train_ = X_train_enc
#             f_proba = CBRProbaWrapper(cbr_model, cat_cols, round_categoricals=True)
#             explainer = shap.KernelExplainer(f_proba, background)
#         finally:
#             cbr_model.X_train_ = old_X_train
#
#         return explainer
#
#     raise ValueError(f"Unknown method: {method}")
#
# def CART_DT_5fold_surrogate_linear(
#     dataset,
#     folds,
#     *,
#     random_state=42,
#     max_samples=None,
#     neighborhood_size=100,
#     kernel_width=None,
#     include_point=False
# ):
#     """
#     Prepare local linear surrogate models for each specific test point for a
#     trained Decision Tree classifier, following the same fold-wise structure
#     as the SHAP function.
#
#     Parameters
#     ----------
#     dataset : str
#         Dataset name used for saving/loading cached results.
#
#     folds : list[dict]
#         Dictionary/list of folds structured as:
#             fold: Fold No.
#             train_idx: Indices of training samples
#             test_idx: Indices of testing samples
#             X_train: Features of training samples
#             X_test: Features of testing samples
#             y_train: Labels of training samples
#             y_test: Labels of testing samples
#             scaler: Standard Scaler object for standardization
#             model: Decision Tree classifier object
#             performance_metrics: Dictionary of performance metrics
#
#     random_state : int, default=42
#         Random seed for reproducibility.
#
#     max_samples : int or None, default=None
#         If provided, only the first max_samples test points of each fold are used.
#
#     neighborhood_size : int, default=100
#         Number of nearest training samples used to fit the local surrogate
#         around each test point.
#
#     kernel_width : float or None, default=None
#         Width of exponential kernel for weighting neighbors.
#         If None, uses sqrt(n_features).
#
#     include_point : bool, default=True
#         Whether to include the test point itself in the surrogate fit.
#
#     Returns
#     -------
#     DT_5fold_model_surrogate : list[dict]
#         For each fold, returns a dictionary containing:
#             ...
#             surrogate_models: list of fitted local linear models, one per test point
#             surrogate_coefficients: array of shape (n_test_used, n_features)
#             surrogate_intercepts: array of shape (n_test_used,)
#             surrogate_pred_prob_at_point: surrogate prediction at each test point
#             blackbox_pred_prob_at_point: DT predicted probability at each test point
#             neighborhood_indices: training-neighborhood indices used per test point
#             neighborhood_distances: distances of local neighborhood per test point
#             y_pred: predicted class by Decision Tree classifier
#     """
#
#     rng = np.random.default_rng(random_state)
#
#     post_hoc_path = base_dir / dataset / "dt_fold_model_linear_surrogate.joblib"
#     DT_5fold_model_surrogate = []
#
#     if post_hoc_path.exists():
#         DT_5fold_model_surrogate = load_model(post_hoc_path)
#     else:
#         for fold_idx, fold in enumerate(folds):
#             dt_model = fold["model"]
#
#             X_train = np.asarray(fold["X_train"])
#             X_test_full = np.asarray(fold["X_test"])
#             y_test_full = np.asarray(fold["y_test"])
#
#             test_idx_full = np.asarray(fold["test_idx"])
#
#             if max_samples is not None and len(X_test_full) > max_samples:
#                 X_test = X_test_full[:max_samples]
#                 y_test = y_test_full[:max_samples]
#                 test_idx = test_idx_full[:max_samples]
#             else:
#                 X_test = X_test_full
#                 y_test = y_test_full
#                 test_idx = test_idx_full
#
#             y_pred = dt_model.predict(X_test)
#             pred_class_idx = np.array(
#                 [np.where(dt_model.classes_ == y)[0][0] for y in y_pred]
#             )
#
#             n_test, n_features = X_test.shape
#             if kernel_width is None:
#                 kw = np.sqrt(n_features)
#             else:
#                 kw = kernel_width
#
#             surrogate_models = []
#             surrogate_coefficients = []
#             surrogate_intercepts = []
#             surrogate_pred_prob_at_point = []
#             blackbox_pred_prob_at_point = []
#             neighborhood_indices_all = []
#             neighborhood_distances_all = []
#
#             for i in range(n_test):
#                 x0 = X_test[i].reshape(1, -1)
#                 c_idx = pred_class_idx[i]
#
#                 # ------------------------------------------------------------
#                 # Step 1: find local neighborhood from training data
#                 # ------------------------------------------------------------
#                 dists = pairwise_distances(X_train, x0, metric="euclidean").ravel()
#                 k = min(neighborhood_size, len(X_train))
#                 nn_idx = np.argsort(dists)[:k]
#
#                 X_local = X_train[nn_idx]
#                 d_local = dists[nn_idx]
#
#                 # Optional: include the point itself in the local surrogate fit
#                 if include_point:
#                     X_local_fit = np.vstack([X_local, x0])
#                     d_local_fit = np.concatenate([d_local, np.array([0.0])])
#                 else:
#                     X_local_fit = X_local
#                     d_local_fit = d_local
#
#                 # ------------------------------------------------------------
#                 # Step 2: get black-box target values in that neighborhood
#                 # Here we fit surrogate to predicted probability of the
#                 # predicted class at x0
#                 # ------------------------------------------------------------
#                 bb_probs_local = dt_model.predict_proba(X_local_fit)[:, c_idx]
#
#                 # ------------------------------------------------------------
#                 # Step 3: compute locality weights
#                 # exponential kernel like LIME-style locality weighting
#                 # ------------------------------------------------------------
#                 weights = np.exp(-(d_local_fit ** 2) / (kw ** 2 + 1e-12))
#
#                 # ------------------------------------------------------------
#                 # Step 4: fit local linear surrogate
#                 # ------------------------------------------------------------
#                 surrogate = LinearRegression()
#                 surrogate.fit(X_local_fit, bb_probs_local, sample_weight=weights)
#
#                 # ------------------------------------------------------------
#                 # Step 5: store explanation values for this specific point
#                 # ------------------------------------------------------------
#                 s_pred = surrogate.predict(x0)[0]
#                 bb_pred = dt_model.predict_proba(x0)[0, c_idx]
#
#                 surrogate_models.append(surrogate)
#                 surrogate_coefficients.append(surrogate.coef_.copy())
#                 surrogate_intercepts.append(float(surrogate.intercept_))
#                 surrogate_pred_prob_at_point.append(float(s_pred))
#                 blackbox_pred_prob_at_point.append(float(bb_pred))
#                 neighborhood_indices_all.append(nn_idx.copy())
#                 neighborhood_distances_all.append(d_local.copy())
#
#             DT_5fold_model_surrogate.append({
#                 "fold": fold["fold"],
#                 "train_idx": fold["train_idx"],
#                 "test_idx": test_idx,
#                 "X_train": fold["X_train"],
#                 "X_test": X_test,
#                 "y_train": fold["y_train"],
#                 "y_test": y_test,
#                 "scaler": fold["scaler"],
#                 "model": fold["model"],
#                 "performance_metrics": fold["performance_metrics"],
#
#                 # surrogate-specific outputs
#                 "surrogate_models": surrogate_models,
#                 "surrogate_coefficients": np.array(surrogate_coefficients),
#                 "surrogate_intercepts": np.array(surrogate_intercepts),
#                 "surrogate_pred_prob_at_point": np.array(surrogate_pred_prob_at_point),
#                 "blackbox_pred_prob_at_point": np.array(blackbox_pred_prob_at_point),
#                 "neighborhood_indices": neighborhood_indices_all,
#                 "neighborhood_distances": neighborhood_distances_all,
#                 "y_pred": y_pred
#             })
#
#         save_model(DT_5fold_model_surrogate, post_hoc_path)
#
#     return DT_5fold_model_surrogate

import numpy as np
import shap
import torch
import torch.nn.functional as F


# =========================================================
# Shared Helpers
# =========================================================
def _get_class_indices(classes, y_pred):
    classes = np.asarray(classes)
    return np.array([np.where(classes == y)[0][0] for y in y_pred])


def _normalize_shap_output(sv, y_pred, classes):
    """
    Normalize SHAP outputs into:
        sv_all  : all-class attributions when available
        sv_pred : predicted-class attribution per instance

    Handles:
      1. list of arrays: [ (n,f), (n,f), ... ] for multiclass
      2. ndarray (n,f,c)
      3. ndarray (n,c,f)
      4. ndarray (n,f) for binary/single-output cases
    """
    class_idx = _get_class_indices(classes, y_pred)

    # Case 1: list of per-class arrays
    if isinstance(sv, list):
        sv_all = np.stack(sv, axis=2)   # (n, f, c)
        sv_pred = sv_all[np.arange(len(y_pred)), :, class_idx]
        return sv_all, sv_pred

    sv = np.asarray(sv)

    # Case 2A: already (n, f, c)
    if sv.ndim == 3 and sv.shape[2] == len(classes):
        sv_all = sv
        sv_pred = sv_all[np.arange(len(y_pred)), :, class_idx]
        return sv_all, sv_pred

    # Case 2B: (n, c, f) -> transpose to (n, f, c)
    if sv.ndim == 3 and sv.shape[1] == len(classes):
        sv_all = np.transpose(sv, (0, 2, 1))
        sv_pred = sv_all[np.arange(len(y_pred)), :, class_idx]
        return sv_all, sv_pred

    # Case 3: binary/single-output
    if sv.ndim == 2:
        sv_all = sv
        sv_pred = sv
        return sv_all, sv_pred

    raise ValueError(
        f"Unsupported SHAP output type/shape. "
        f"type={type(sv)}, shape={getattr(sv, 'shape', None)}"
    )


def shap_attributions(explainer, model, X, y_pred):
    """
    Returns SHAP attributions for the predicted class for each instance in X.
    """
    sv = explainer.shap_values(X)
    _, sv_pred = _normalize_shap_output(sv, y_pred, model.classes_)
    return sv_pred


# =========================================================
# DT SHAP
# =========================================================
def CART_DT_5fold_shap(dataset, folds, *, random_state=42, max_samples=None):
    """
    SHAP feature attributions for Decision Tree folds.
    """
    post_hoc_path = base_dir / dataset / "dt_fold_model_shap.joblib"
    DT_5fold_model_shap = []

    if post_hoc_path.exists():
        return load_model(post_hoc_path)

    for fold_idx, fold in enumerate(folds):
        dt_model = fold["model"]
        X_test = np.asarray(fold["X_test"], dtype=np.float32)

        if max_samples is not None and len(X_test) > max_samples:
            X_test = X_test[:max_samples]

        explainer = shap.TreeExplainer(dt_model)
        sv = explainer.shap_values(X_test)

        y_pred = dt_model.predict(X_test)
        sv_all, sv_pred = _normalize_shap_output(sv, y_pred, dt_model.classes_)

        DT_5fold_model_shap.append({
            "fold": fold["fold"],
            "train_idx": fold["train_idx"],
            "test_idx": fold["test_idx"],
            "X_train": fold["X_train"],
            "X_test": fold["X_test"],
            "y_train": fold["y_train"],
            "y_test": fold["y_test"],
            "scaler": fold.get("scaler", None),
            "model": fold["model"],
            "performance_metrics": fold.get("performance_metrics", None),
            "feature_attribution_all_classes": sv_all,
            "feature_attribution_pred_class": sv_pred,
            "y_pred": y_pred
        })

    save_model(DT_5fold_model_shap, post_hoc_path)
    return DT_5fold_model_shap


# =========================================================
# XGB SHAP
# =========================================================
def XGB_5fold_shap(dataset, folds, *, random_state=42, max_samples=None):
    """
    SHAP feature attributions for XGB folds.
    """
    post_hoc_path = base_dir / dataset / "xgb_fold_model_shap.joblib"
    XGB_5fold_model_shap = []

    if post_hoc_path.exists():
        return load_model(post_hoc_path)

    for fold_idx, fold in enumerate(folds):
        xgb_model = fold["model"]
        X_test = np.asarray(fold["X_test"], dtype=np.float32)

        if max_samples is not None and len(X_test) > max_samples:
            X_test = X_test[:max_samples]

        explainer = shap.TreeExplainer(xgb_model)
        sv = explainer.shap_values(X_test)

        y_pred = xgb_model.predict(X_test)
        sv_all, sv_pred = _normalize_shap_output(sv, y_pred, xgb_model.classes_)

        XGB_5fold_model_shap.append({
            "fold": fold["fold"],
            "train_idx": fold["train_idx"],
            "test_idx": fold["test_idx"],
            "X_train": fold["X_train"],
            "X_test": fold["X_test"],
            "y_train": fold["y_train"],
            "y_test": fold["y_test"],
            "scaler": fold.get("scaler", None),
            "model": fold["model"],
            "performance_metrics": fold.get("performance_metrics", None),
            "feature_attribution_all_classes": sv_all,
            "feature_attribution_pred_class": sv_pred,
            "y_pred": y_pred
        })

    save_model(XGB_5fold_model_shap, post_hoc_path)
    return XGB_5fold_model_shap


# =========================================================
# CBR Wrapper + SHAP
# =========================================================
class CBRProbaWrapper:
    """
    Pickle-friendly callable for SHAP KernelExplainer.
    """
    def __init__(self, cbr_model, cat_cols, *, round_categoricals=True):
        self.cbr_model = cbr_model
        self.cat_cols = list(cat_cols)
        self.round_categoricals = round_categoricals

    def __call__(self, X_in):
        X_in = np.asarray(X_in, dtype=np.float32)

        if self.round_categoricals and len(self.cat_cols) > 0:
            X_in = X_in.copy()
            X_in[:, self.cat_cols] = np.rint(X_in[:, self.cat_cols]).astype(np.int32)

        return self.cbr_model.predict_proba(
            X_in,
            classes=self.cbr_model.classes_,
            round_categoricals=False
        )


def CBR_5fold_shap(
    dataset,
    folds,
    *,
    random_state=42,
    max_samples=None,
    background_size=50,
    nsamples="auto"
):
    post_hoc_path = base_dir / dataset / "cbr_fold_model_shap.joblib"
    CBR_5fold_model_shap = []

    if post_hoc_path.exists():
        return load_model(post_hoc_path)

    rng = np.random.RandomState(random_state)

    for fold_idx, fold in enumerate(folds):
        cbr_model = fold["model"]

        X_train_raw = np.asarray(fold["X_train"], dtype=object)
        X_test_raw = np.asarray(fold["X_test"], dtype=object)

        if max_samples is not None and len(X_test_raw) > max_samples:
            X_test_raw = X_test_raw[:max_samples]

        y_pred = cbr_model.predict(X_test_raw)

        cat_cols = sorted(list(getattr(cbr_model, "categorical_idx", [])))

        if len(cat_cols) > 0:
            X_train_enc, X_test_enc, enc = cbr_model.encode_categoricals_for_xgb(
                X_train_raw, X_test_raw, cat_cols
            )
            X_train_enc = X_train_enc.astype(np.float32)
            X_test_enc = X_test_enc.astype(np.float32)
        else:
            X_train_enc = X_train_raw.astype(np.float32)
            X_test_enc = X_test_raw.astype(np.float32)
            enc = None

        if X_train_enc.shape[0] > background_size:
            bg_idx = rng.choice(X_train_enc.shape[0], size=background_size, replace=False)
            background = X_train_enc[bg_idx]
        else:
            background = X_train_enc

        old_X_train = cbr_model.X_train_
        try:
            cbr_model.X_train_ = X_train_enc

            f_proba = CBRProbaWrapper(cbr_model, cat_cols, round_categoricals=True)
            explainer = shap.KernelExplainer(f_proba, background)

            sv = explainer.shap_values(X_test_enc, nsamples=nsamples)
            sv_all, sv_pred = _normalize_shap_output(sv, y_pred, cbr_model.classes_)

        finally:
            cbr_model.X_train_ = old_X_train

        CBR_5fold_model_shap.append({
            "fold": fold["fold"],
            "train_idx": fold["train_idx"],
            "test_idx": fold["test_idx"],
            "X_train": fold["X_train"],
            "X_test": fold["X_test"],
            "y_train": fold["y_train"],
            "y_test": fold["y_test"],
            "scaler": fold.get("scaler", None),
            "model": fold["model"],
            "performance_metrics": fold.get("performance_metrics", None),
            "feature_attribution_all_classes": sv_all,
            "feature_attribution_pred_class": sv_pred,
            "y_pred": y_pred,
            "categorical_idx": cat_cols,
            "encoder": enc,
            "X_test_encoded_used_for_shap": X_test_enc,
            "shap_random_state": random_state,
            "shap_background_size": background_size,
            "shap_nsamples": nsamples
        })

    save_model(CBR_5fold_model_shap, post_hoc_path)
    return CBR_5fold_model_shap


# =========================================================
# Generic sklearn predict_proba wrapper (MLP/DNN)
# =========================================================
class PredictProbaWrapper:
    """
    Pickle-friendly callable for SHAP KernelExplainer with sklearn-like models.
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, X_in):
        X_in = np.asarray(X_in, dtype=np.float32)
        return self.model.predict_proba(X_in)


# =========================================================
# ProtoPNet wrapper
# =========================================================
class ProtoPNetProbaWrapper:
    """
    Pickle-friendly callable for SHAP KernelExplainer with PyTorch ProtoPNet.
    Returns class probabilities.
    """
    def __init__(self, model, device="cpu", batch_size=256):
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def __call__(self, X_in):
        X_in = np.asarray(X_in, dtype=np.float32)
        self.model.eval()

        outputs = []
        with torch.no_grad():
            for start in range(0, len(X_in), self.batch_size):
                xb = torch.tensor(
                    X_in[start:start + self.batch_size],
                    dtype=torch.float32,
                    device=self.device
                )
                logits, _, _ = self.model(xb)
                probs = F.softmax(logits, dim=1)
                outputs.append(probs.detach().cpu().numpy())

        return np.vstack(outputs)


# =========================================================
# MLP SHAP
# =========================================================
def MLP_5fold_shap(
    dataset,
    folds,
    *,
    random_state=42,
    max_samples=None,
    background_size=50,
    nsamples="auto"
):
    post_hoc_path = base_dir / dataset / "mlp_fold_model_shap.joblib"
    MLP_5fold_model_shap = []

    if post_hoc_path.exists():
        return load_model(post_hoc_path)

    rng = np.random.RandomState(random_state)

    for fold_idx, fold in enumerate(folds):
        mlp_model = fold["model"]
        X_train = np.asarray(fold["X_train"], dtype=np.float32)
        X_test = np.asarray(fold["X_test"], dtype=np.float32)

        if max_samples is not None and len(X_test) > max_samples:
            X_test = X_test[:max_samples]

        y_pred = mlp_model.predict(X_test)

        if len(X_train) > background_size:
            bg_idx = rng.choice(len(X_train), size=background_size, replace=False)
            background = X_train[bg_idx]
        else:
            background = X_train

        f_proba = PredictProbaWrapper(mlp_model)
        explainer = shap.KernelExplainer(f_proba, background)

        sv = explainer.shap_values(X_test, nsamples=nsamples)
        sv_all, sv_pred = _normalize_shap_output(sv, y_pred, mlp_model.classes_)

        MLP_5fold_model_shap.append({
            "fold": fold["fold"],
            "train_idx": fold["train_idx"],
            "test_idx": fold["test_idx"],
            "X_train": fold["X_train"],
            "X_test": fold["X_test"],
            "y_train": fold["y_train"],
            "y_test": fold["y_test"],
            "scaler": fold.get("scaler", None),
            "model": fold["model"],
            "performance_metrics": fold.get("performance_metrics", None),
            "feature_attribution_all_classes": sv_all,
            "feature_attribution_pred_class": sv_pred,
            "y_pred": y_pred,
            "shap_random_state": random_state,
            "shap_background_size": background_size,
            "shap_nsamples": nsamples
        })

    save_model(MLP_5fold_model_shap, post_hoc_path)
    return MLP_5fold_model_shap


# =========================================================
# DNN SHAP
# =========================================================
def DNN_8HL_5fold_shap(
    dataset,
    folds,
    *,
    random_state=42,
    max_samples=None,
    background_size=50,
    nsamples="auto"
):
    post_hoc_path = base_dir / dataset / "dnn_fold_model_shap.joblib"
    DNN_5fold_model_shap = []

    if post_hoc_path.exists():
        return load_model(post_hoc_path)

    rng = np.random.RandomState(random_state)

    for fold_idx, fold in enumerate(folds):
        dnn_model = fold["model"]
        X_train = np.asarray(fold["X_train"], dtype=np.float32)
        X_test = np.asarray(fold["X_test"], dtype=np.float32)

        if max_samples is not None and len(X_test) > max_samples:
            X_test = X_test[:max_samples]

        y_pred = dnn_model.predict(X_test)

        if len(X_train) > background_size:
            bg_idx = rng.choice(len(X_train), size=background_size, replace=False)
            background = X_train[bg_idx]
        else:
            background = X_train

        f_proba = PredictProbaWrapper(dnn_model)
        explainer = shap.KernelExplainer(f_proba, background)

        sv = explainer.shap_values(X_test, nsamples=nsamples)
        sv_all, sv_pred = _normalize_shap_output(sv, y_pred, dnn_model.classes_)

        DNN_5fold_model_shap.append({
            "fold": fold["fold"],
            "train_idx": fold["train_idx"],
            "test_idx": fold["test_idx"],
            "X_train": fold["X_train"],
            "X_test": fold["X_test"],
            "y_train": fold["y_train"],
            "y_test": fold["y_test"],
            "scaler": fold.get("scaler", None),
            "model": fold["model"],
            "performance_metrics": fold.get("performance_metrics", None),
            "feature_attribution_all_classes": sv_all,
            "feature_attribution_pred_class": sv_pred,
            "y_pred": y_pred,
            "shap_random_state": random_state,
            "shap_background_size": background_size,
            "shap_nsamples": nsamples
        })

    save_model(DNN_5fold_model_shap, post_hoc_path)
    return DNN_5fold_model_shap


# =========================================================
# ProtoPNet SHAP
# =========================================================
def PROTOPNET_5fold_shap(
    dataset,
    folds,
    *,
    random_state=42,
    max_samples=None,
    background_size=50,
    nsamples="auto",
    device=None,
    batch_size=256
):
    post_hoc_path = base_dir / dataset / "protopnet_fold_model_shap.joblib"
    PROTOPNET_5fold_model_shap = []

    if post_hoc_path.exists():
        return load_model(post_hoc_path)

    rng = np.random.RandomState(random_state)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold_idx, fold in enumerate(folds):
        proto_model = fold["model"]
        proto_model.eval()
        proto_model.to(device)

        X_train = np.asarray(fold["X_train"], dtype=np.float32)
        X_test = np.asarray(fold["X_test"], dtype=np.float32)

        if max_samples is not None and len(X_test) > max_samples:
            X_test = X_test[:max_samples]

        with torch.no_grad():
            xb = torch.tensor(X_test, dtype=torch.float32, device=device)
            logits, _, _ = proto_model(xb)
            y_pred = torch.argmax(logits, dim=1).detach().cpu().numpy()

        n_classes = logits.shape[1]
        classes = np.arange(n_classes)

        if len(X_train) > background_size:
            bg_idx = rng.choice(len(X_train), size=background_size, replace=False)
            background = X_train[bg_idx]
        else:
            background = X_train

        f_proba = ProtoPNetProbaWrapper(proto_model, device=device, batch_size=batch_size)
        explainer = shap.KernelExplainer(f_proba, background)

        sv = explainer.shap_values(X_test, nsamples=nsamples)
        sv_all, sv_pred = _normalize_shap_output(sv, y_pred, classes)

        PROTOPNET_5fold_model_shap.append({
            "fold": fold["fold"],
            "train_idx": fold["train_idx"],
            "test_idx": fold["test_idx"],
            "X_train": fold["X_train"],
            "X_test": fold["X_test"],
            "y_train": fold["y_train"],
            "y_test": fold["y_test"],
            "scaler": fold.get("scaler", None),
            "model": fold["model"],
            "performance_metrics": fold.get("performance_metrics", None),
            "proto_outputs": fold.get("proto_outputs", None),
            "feature_attribution_all_classes": sv_all,
            "feature_attribution_pred_class": sv_pred,
            "y_pred": y_pred,
            "shap_random_state": random_state,
            "shap_background_size": background_size,
            "shap_nsamples": nsamples,
            "shap_device": device
        })

    save_model(PROTOPNET_5fold_model_shap, post_hoc_path)
    return PROTOPNET_5fold_model_shap


# =========================================================
# Optional: rebuild explainer from fold
# =========================================================
def rebuild_explainer_from_fold(method: str, fold: dict):
    model = fold["model"]

    if method in ("dt", "xgb"):
        return shap.TreeExplainer(model)

    if method == "cbr":
        rng = np.random.RandomState(fold.get("shap_random_state", 42))
        background_size = int(fold.get("shap_background_size", 50))

        cbr_model = model
        X_train_raw = np.asarray(fold["X_train"], dtype=object)
        cat_cols = sorted(list(getattr(cbr_model, "categorical_idx", [])))

        if len(cat_cols) > 0:
            X_train_enc, _, _ = cbr_model.encode_categoricals_for_xgb(
                X_train_raw, X_train_raw[:1], cat_cols
            )
            X_train_enc = X_train_enc.astype(np.float32)
        else:
            X_train_enc = X_train_raw.astype(np.float32)

        if len(X_train_enc) > background_size:
            bg_idx = rng.choice(len(X_train_enc), size=background_size, replace=False)
            background = X_train_enc[bg_idx]
        else:
            background = X_train_enc

        old_X_train = cbr_model.X_train_
        try:
            cbr_model.X_train_ = X_train_enc
            f_proba = CBRProbaWrapper(cbr_model, cat_cols, round_categoricals=True)
            explainer = shap.KernelExplainer(f_proba, background)
        finally:
            cbr_model.X_train_ = old_X_train

        return explainer

    if method in ("mlp", "dnn"):
        rng = np.random.RandomState(fold.get("shap_random_state", 42))
        background_size = int(fold.get("shap_background_size", 50))

        X_train = np.asarray(fold["X_train"], dtype=np.float32)
        if len(X_train) > background_size:
            bg_idx = rng.choice(len(X_train), size=background_size, replace=False)
            background = X_train[bg_idx]
        else:
            background = X_train

        f_proba = PredictProbaWrapper(model)
        return shap.KernelExplainer(f_proba, background)

    if method == "protopnet":
        rng = np.random.RandomState(fold.get("shap_random_state", 42))
        background_size = int(fold.get("shap_background_size", 50))
        device = fold.get("shap_device", "cuda" if torch.cuda.is_available() else "cpu")

        X_train = np.asarray(fold["X_train"], dtype=np.float32)
        if len(X_train) > background_size:
            bg_idx = rng.choice(len(X_train), size=background_size, replace=False)
            background = X_train[bg_idx]
        else:
            background = X_train

        f_proba = ProtoPNetProbaWrapper(model, device=device)
        return shap.KernelExplainer(f_proba, background)

    raise ValueError(f"Unknown method: {method}")