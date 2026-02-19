import shap
from model_save_functions import save_model, load_model
from pathlib import Path
import numpy as np

base_dir = Path.cwd()/"trained_models"

def shap_attributions(explainer, model, X, y_pred):
    '''
    Returns SHAP attributions for the predicted class for each instance in X.
    :param explainer: SHAP explainer object
    :param model: Trained classifier object
    :param X: Test set features
    :param y_pred: Predicted class by trained classifier
    :return:
        SHAP attributions for test set with SHAP Explainer for predicted class by trained classifier.
    '''

    sv = explainer.shap_values(X)
    class_idx = np.array([np.where(model.classes_ == y)[0][0] for y in y_pred])
    sv_pred = sv[np.arange(len(X)), :, class_idx]
    return sv_pred


def CART_DT_5fold_shap(dataset, folds, *, random_state = 42, max_samples=None):
    '''
        This function produces SHAP feature attributions for test set with Decision Tree Classifier.
        :param folds: Dictionary of folds structured as:
            fold: Fold No.
            train_idx: Indices of training samples
            test_idx: Indices of testing samples
            X_train: Features of training samples
            X_test: Features of testing samples
            y_train: Labels of training samples
            y_test: Labels of testing samples
            scaler: Standard Scaler object for standardization
            model: Decision Tree classifier object
            performance_metrics: Dictionary of performance metrics and their corresponding values for the trained DT classifier

        :return: Dictionary of SHAP feature attributions for test set with Decision Tree Classifier
            ...
            explainer: SHAP explainer object
            feature_attribution_all_classes: Feature attributions for all classes for testing instances
            feature_attribution_pred_class: Feature attributions for predicted class via Decision Tree Classifier
            y_pred: Predicted class by Decision Tree Classifier
        '''

    post_hoc_path = base_dir/dataset/f"dt_fold_model_shap.joblib"
    DT_5fold_model_shap = []
    if post_hoc_path.exists():
        DT_5fold_model_shap = load_model(post_hoc_path)
    else:
        for fold_idx, fold in enumerate(folds):
            dt_model = fold['model']
            X_test = fold['X_test']
            y_test = fold['y_test']

            if max_samples is not None and len(X_test) > max_samples:
                X_test = X_test[:max_samples]

            explainer = shap.TreeExplainer(dt_model)
            sv = explainer.shap_values(X_test)

            y_pred = dt_model.predict(X_test)
            class_idx = np.array([np.where(dt_model.classes_==y)[0][0] for y in y_pred])
            sv_pred = sv[np.arange(len(X_test)), :, class_idx]
            DT_5fold_model_shap.append({
                "fold": fold['fold'],
                "train_idx": fold['train_idx'],
                "test_idx": fold['test_idx'],
                "X_train": fold['X_train'],
                "X_test": fold['X_test'],
                "y_train": fold['y_train'],
                "y_test": fold['y_test'],
                "scaler": fold['scaler'],
                "model": fold['model'],
                "performance_metrics": fold['performance_metrics'],
                # "explainer": explainer,
                "feature_attribution_all_classes": sv,
                "feature_attribution_pred_class": sv_pred,
                "y_pred": y_pred
            })
        save_model(DT_5fold_model_shap, post_hoc_path)
    return DT_5fold_model_shap

def XGB_5fold_shap(dataset, folds, *, random_state = 42, max_samples=None):
    '''
        This function produces SHAP feature attributions for test set with XGB Classifier.
        :param folds: Dictionary of folds structured as:
            fold: Fold No.
            train_idx: Indices of training samples
            test_idx: Indices of testing samples
            X_train: Features of training samples
            X_test: Features of testing samples
            y_train: Labels of training samples
            y_test: Labels of testing samples
            scaler: Standard Scaler object for standardization
            model: Decision Tree classifier object
            performance_metrics: Dictionary of performance metrics and their corresponding values for the trained XGB classifier

        :return: Dictionary of SHAP feature attributions for test set with XGB Classifier
            ...
            explainer: SHAP explainer object
            feature_attribution_all_classes: Feature attributions for all classes for testing instances
            feature_attribution_pred_class: Feature attributions for predicted class via XGB Classifier
            y_pred: Predicted class by XGB Classifier
        '''

    post_hoc_path = base_dir/dataset/f"xgb_fold_model_shap.joblib"
    XGB_5fold_model_shap = []
    if post_hoc_path.exists():
        XGB_5fold_model_shap = load_model(post_hoc_path)
    else:
        for fold_idx, fold in enumerate(folds):
            dt_model = fold['model']
            X_test = fold['X_test']

            if max_samples is not None and len(X_test) > max_samples:
                X_test = X_test[:max_samples]

            explainer = shap.TreeExplainer(dt_model)
            sv = explainer.shap_values(X_test)

            y_pred = dt_model.predict(X_test)
            class_idx = np.array([np.where(dt_model.classes_==y)[0][0] for y in y_pred])

            # Case A: multiclass often returns a list: [ (n, f), (n, f), ... ] length = n_classes
            if isinstance(sv, list):
                # stack to (n, f, c)
                sv_all = np.stack(sv, axis=2)
                sv_pred = sv_all[np.arange(len(X_test)), :, class_idx]

            # Case B: some setups return (n, f, c) already
            elif getattr(sv, "ndim", None) == 3:
                sv_all = sv
                sv_pred = sv_all[np.arange(len(X_test)), :, class_idx]

            # Case C: binary/single-output returns (n, f) only
            elif getattr(sv, "ndim", None) == 2:
                sv_all = sv
                # no per-class axis exists; best definition of "pred class" is just the same attribution
                sv_pred = sv_all

            XGB_5fold_model_shap.append({
                "fold": fold['fold'],
                "train_idx": fold['train_idx'],
                "test_idx": fold['test_idx'],
                "X_train": fold['X_train'],
                "X_test": fold['X_test'],
                "y_train": fold['y_train'],
                "y_test": fold['y_test'],
                "scaler": fold['scaler'],
                "model": fold['model'],
                "performance_metrics": fold['performance_metrics'],
                # "explainer": explainer,
                "feature_attribution_all_classes": sv,
                "feature_attribution_pred_class": sv_pred,
                "y_pred": y_pred
            })
        save_model(XGB_5fold_model_shap, post_hoc_path)
    return XGB_5fold_model_shap


class CBRProbaWrapper:
    """
    Pickle-friendly callable for SHAP KernelExplainer.

    Holds:
      - cbr_model: the trained CBR model
      - cat_cols: list of categorical column indices (encoded as ints)
      - round_categoricals: whether to round cat cols before calling predict_proba
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

        # IMPORTANT: ensure the class ordering matches self.cbr_model.classes_
        return self.cbr_model.predict_proba(X_in, classes=self.cbr_model.classes_, round_categoricals=False)



def CBR_5fold_shap(dataset, folds, *, random_state=42, max_samples=None, background_size=50, nsamples="auto"):
    post_hoc_path = base_dir / dataset / f"cbr_fold_model_shap.joblib"
    CBR_5fold_model_shap = []

    if post_hoc_path.exists():
        return load_model(post_hoc_path)

    rng = np.random.RandomState(random_state)

    for fold_idx, fold in enumerate(folds):
        cbr_model = fold['model']

        X_train_raw = np.asarray(fold["X_train"], dtype=object)
        X_test_raw = np.asarray(fold["X_test"], dtype=object)

        if max_samples is not None and len(X_test_raw) > max_samples:
            X_test_raw = X_test_raw[:max_samples]

        y_pred = cbr_model.predict(X_test_raw)

        cat_cols = sorted(list(getattr(cbr_model, "categorical_idx", [])))

        if len(cat_cols)>0:
            X_train_enc, X_test_enc, enc = cbr_model.encode_categoricals_for_xgb(X_train_raw, X_test_raw, cat_cols)
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

            f_proba=CBRProbaWrapper(cbr_model, cat_cols, round_categoricals=True)
            explainer = shap.KernelExplainer(f_proba, background)

            sv = explainer.shap_values(X_test_enc, nsamples=nsamples)
            classes = np.asarray(cbr_model.classes_)
            class_idx = np.array([np.where(classes == y)[0][0] for y in y_pred])

            if isinstance(sv, list):
                sv_pred = np.stack([sv[class_idx[i]][i, :] for i in range(len(X_test_enc))], axis=0)
            else:
                # fallback if SHAP returns array-like
                # expected shapes vary; safest is to index the last dim as class
                sv_pred = sv[np.arange(len(X_test_enc)), :, class_idx]

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

                # "explainer": None,
                "feature_attribution_all_classes": sv,
                "feature_attribution_pred_class": sv_pred,
                "y_pred": y_pred,

                # optional debug fields (handy)
                "categorical_idx": cat_cols,
                "encoder": enc,
                "X_test_encoded_used_for_shap": X_test_enc
            })

        finally:
            cbr_model.X_train_ = old_X_train

    save_model(CBR_5fold_model_shap, post_hoc_path)
    return CBR_5fold_model_shap



def rebuild_explainer_from_fold(method: str, fold: dict):
    """
    Rebuild a SHAP explainer from a saved fold dict.
    - method: "dt", "xgb", "cbr"
    """
    model = fold["model"]

    if method in ("dt", "xgb"):
        return shap.TreeExplainer(model)

    if method == "cbr":
        # reconstruct the KernelExplainer the same way you did in CBR_5fold_shap
        rng = np.random.RandomState(fold.get("shap_random_state", 42))
        background_size = int(fold.get("shap_background_size", 50))
        nsamples = fold.get("shap_nsamples", "auto")

        cbr_model = model
        X_train_raw = np.asarray(fold["X_train"], dtype=object)
        X_test_raw  = np.asarray(fold["X_test"], dtype=object)

        cat_cols = sorted(list(getattr(cbr_model, "categorical_idx", [])))

        if len(cat_cols) > 0:
            X_train_enc, X_test_enc, enc = cbr_model.encode_categoricals_for_xgb(
                X_train_raw, X_test_raw, cat_cols
            )
            X_train_enc = X_train_enc.astype(np.float32)
        else:
            X_train_enc = X_train_raw.astype(np.float32)

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
        finally:
            cbr_model.X_train_ = old_X_train

        return explainer

    raise ValueError(f"Unknown method: {method}")