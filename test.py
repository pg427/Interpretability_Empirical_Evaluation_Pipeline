from dataset_functions import load_dataset, stratified_5fold_standardize
from method_functions import CART_DT_5FOLD, XGB_5FOLD, CBR_5FOLD, PROTOPNET_5FOLD, MLP_5FOLD, DNN_8HL_5fold
from posthoc_functions import CART_DT_5fold_shap, XGB_5fold_shap, CBR_5fold_shap
from posthoc_measures_functions import identity_measure, separability_measure, similarity_measure, stability_measure
from model_save_functions import save_model, load_model, save_json
from pathlib import Path
import argparse
from direct_measures_functions import ria_measure, soc_all_methods_for_dataset, feature_synergy_all_methods_for_dataset, robustness_all_methods_for_dataset, mec_all_methods_for_datasets

ALL_DATASETS = ["iris", "wine", "breast_cancer"]
ALL_MODELS = ["dt", "xgb", "cbr", "proto", "mlp", "dnn"]

def parse_args():
    parser = argparse.ArgumentParser(description="Train AI methods on selected Datasets")
    parser.add_argument( "--datasets", nargs="+", choices=["iris", "wine", "breast_cancer", "all"],
                         default=["iris"],
                         help="Datasets to train on"
                         )
    parser.add_argument("--models", nargs="+", choices=["dt", "xgb", "cbr", "proto", "mlp", "dnn", "all"],
                        default=["dt"],
                        help="Models to train on"
                        )

    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing trained models")
    return parser.parse_args()

def normalize_datasets(ds):
    if "all" in ds:
        return ALL_DATASETS
    return ds

def normalize_models(ms):
    if "all" in ms:
        return ALL_MODELS
    return ms


def train_cli(args):
    datasets = normalize_datasets(args.datasets)
    models = normalize_models(args.models)

    base_dir = Path.cwd()/"trained_models"
    base_dir.mkdir(exist_ok=True)

    fold_models = {}
    fold_models_explanations = {}
    fold_models_explanations_measures = {}
    for ds in datasets:
        print(f"\n=== DATASET {ds.upper()} ===")
        X, y, feature_names = load_dataset(ds)

        # --- MEASURE 1: RIA ----
        ria_results = ria_measure((X, y, feature_names), ds)
        ria_json_path = (base_dir / ds) / f"{ds}_ria.json"
        save_json(ria_json_path, ria_results)

        folds_stand = stratified_5fold_standardize(X, y)
        folds_unstand = stratified_5fold_standardize(X, y, standardize=False)

        ds_dir = base_dir/ds
        ds_dir.mkdir(parents=True, exist_ok=True)

        fold_models[ds] = {}
        fold_models_explanations[ds] = {}
        fold_models_explanations_measures[ds] = {}
        for model_name in models:
            fold_models_explanations_measures[ds][model_name] = {}
            model_path = ds_dir/f"{model_name}_fold_model.joblib"

            if model_path.exists() and not args.overwrite:
                print(f"Model {model_name} already trained")
                fold_models[ds][model_name] = load_model(model_path)
                if model_name == "dt":
                    fold_models_explanations[ds][model_name] = CART_DT_5fold_shap(ds, fold_models[ds][model_name])
                elif model_name == "xgb":
                    fold_models_explanations[ds][model_name] = XGB_5fold_shap(ds, fold_models[ds][model_name])
                elif model_name == "cbr":
                    fold_models_explanations[ds][model_name] = CBR_5fold_shap(ds, fold_models[ds][model_name])

            else:
                print(f"Training Model {model_name} ")
                if model_name == "dt":
                    fold_models[ds][model_name] = CART_DT_5FOLD(folds_stand)
                    save_model(fold_models[ds][model_name], model_path)
                    fold_models_explanations[ds][model_name] = CART_DT_5fold_shap(ds, fold_models[ds][model_name])
                elif model_name == "xgb":
                    fold_models[ds][model_name] = XGB_5FOLD(folds_stand)
                    save_model(fold_models[ds][model_name], model_path)
                    fold_models_explanations[ds][model_name] = XGB_5fold_shap(ds, fold_models[ds][model_name])
                elif model_name == "cbr":
                    fold_models[ds][model_name] = CBR_5FOLD(folds_unstand)
                    save_model(fold_models[ds][model_name], model_path)
                    fold_models_explanations[ds][model_name] = CBR_5fold_shap(ds, fold_models[ds][model_name])
                elif model_name == "proto":
                    fold_models[ds][model_name] = PROTOPNET_5FOLD(folds_stand)
                    save_model(fold_models[ds][model_name], model_path)
                elif model_name == "mlp":
                    fold_models[ds][model_name] = MLP_5FOLD(folds_stand)
                    save_model(fold_models[ds][model_name], model_path)
                elif model_name == "dnn":
                    fold_models[ds][model_name] = DNN_8HL_5fold(folds_stand)
                    save_model(fold_models[ds][model_name], model_path)


            file_path_json = ds_dir / f"{model_name}_fold_model.json"
            save_json(file_path_json, fold_models[ds][model_name])

        # --- MEASURE 2: SOC ----
        soc_results = soc_all_methods_for_dataset(
            dataset_name=ds,
            method_fold_results=fold_models[ds],  # method -> folds
            distance_type="euclidean",
            force_recompute=args.overwrite,  # optional
        )
        soc_json_path = (base_dir / ds) / f"{ds}_soc_all_methods.json"
        save_json(soc_json_path, soc_results)

        # --- MEASURE 3: Feature Synergy ----
        fs_all = feature_synergy_all_methods_for_dataset(
            ds,
            fold_models[ds],
            n_generations=200,
            n_runs=10,
            include_accuracy_factor=True,
        )
        fs_json_path = (base_dir / ds) / f"{ds}_fs_all_methods.json"
        save_json(fs_json_path, fs_all)

        # --- MEASURE 4: Robustness ----
        rs_all = robustness_all_methods_for_dataset(
            ds,
            fold_models[ds],
            N=200,
            G=100
        )
        rs_json_path = (base_dir / ds) / f"{ds}_rs_all_methods.json"
        save_json(rs_json_path, rs_all)

        # --- MEASURE 5: No. of features, Interaction Strength, Main Effect Complexity ----
        mec_all = mec_all_methods_for_datasets(
            ds,
            fold_models[ds],
        )
        mec_json_path = (base_dir / ds) / f"{ds}_mec_all_methods.json"
        save_json(mec_json_path, mec_all)

    # print(fold_models_explanations["iris"]['dt'][0]['feature_attribution_pred_class'][0])
    # print(fold_models_explanations["iris"]['dt'][0]['y_pred'][0])
    # print(identity_measure(fold_models_explanations["iris"]['dt'])[0]["identity_total_test_avg"])

if __name__ == "__main__":
    args = parse_args()
    train_cli(args)



