from pathlib import Path
from typing import Any
from dataset_functions import stratified_5fold_standardize, load_dataset
from method_functions import CART_DT_5FOLD
import json
import joblib
import numpy as np
import math

def _to_json_safe(obj: Any):
    """Recursively convert numpy/xarray objects to JSON-safe Python types."""
    # --- numpy scalars/arrays ---
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not math.isfinite(x) else x

    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj

    # --- xarray (skexplain ALE/IAs outputs) ---
    try:
        import xarray as xr
        if isinstance(obj, (xr.Dataset, xr.DataArray)):
            return {
                "__xarray__": obj.__class__.__name__,
                "dims": dict(obj.sizes) if hasattr(obj, "sizes") else None,
                "data_vars": list(getattr(obj, "data_vars", {}).keys()),
                "coords": list(getattr(obj, "coords", {}).keys()),
            }
    except Exception:
        pass

    # --- dict/list recursion ---
    if isinstance(obj, dict):
        return {
            k: _to_json_safe(v)
            for k, v in obj.items()
            if k != "model"  # <-- DROP model objects
        }

    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]

    # --- sklearn objects ---
    if obj.__class__.__module__.startswith("sklearn."):
        return f"<{obj.__class__.__name__}>"

    return obj


def save_model(model: Any, output_path: Path) -> None:
    '''
    This function saves the model to disk.
    :param model: A model or a set of models to save.
    :param output_path: Location to save the model.

    :return: None
    '''

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

def load_model(model_path: Path) -> Any:
    '''
        This function saves the model to disk.
        :param model_path: Location to load the model from.
        :return: Loaded models.
        '''
    return joblib.load(model_path)

def save_json(file_path: Path, data_dict: dict) -> None:
    '''
        This function prepares a JSON file given a dictionary.
        :param data_dict: data_dictionary.
        :param file_path: Location to save the JSON.

        :return: None.
    '''
    json_safe_data = _to_json_safe(data_dict)
    with open(file_path, 'w') as outfile:
        json.dump(json_safe_data, outfile, indent=4)
    print(f"File {file_path} saved!")


if __name__ == "__main__":
    '''
    EXAMPLE:  TRAINED 5 FOLD CART-DT MODELS IRIS DATA SAVED AND THEN LOADED
    '''

    # X, y, feature_names = load_dataset('iris')
    # folds_stand = stratified_5fold_standardize(X, y, standardize=True)
    # DT_fold_model = CART_DT_5FOLD(folds_stand, random_state=42)
    # save_model(DT_fold_model, 'trained_models/iris/DT_fold_model.joblib')

    DT_fold_model = load_model("trained_models/iris/dt_fold_model.joblib")
    print(DT_fold_model[0])