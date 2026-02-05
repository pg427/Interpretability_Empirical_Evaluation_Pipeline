from dataset_functions import load_dataset, stratified_5fold_standardize
from method_functions import CART_DT_5FOLD, XGB_5FOLD, CBR_5FOLD, PROTOPNET_5FOLD
from posthoc_functions import CART_DT_5fold_shap, XGB_5fold_shap, CBR_5fold_shap
from posthoc_measures_functions import identity_measure, separability_measure, similarity_measure, stability_measure
from model_save_functions import save_model, load_model, save_json
from pathlib import Path
import argparse

ALL_DATASETS = ["iris", "wine", "breast_cancer"]
ALL_MODELS = ["dt", "xgb", "cbr", "proto"]

def parse_args():
    parser = argparse.ArgumentParser(description="Train AI methods on selected Datasets")
    parser.add_argument( "--datasets", nargs="+", choices=["iris", "wine", "breast_cancer", "all"],
                         default=["iris"],
                         help="Datasets to train on"
                         )
    parser.add_argument("--models", nargs="+", choices=["dt", "xgb", "cbr", "proto", "all"],
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
                    file_path_json = ds_dir / f"{model_name}_fold_model.json"
                    save_json(file_path_json, fold_models[ds][model_name])

            # fold_models_explanations_measures[ds][model_name]["shap"] = {}
            # fold_models_explanations_measures[ds][model_name]["shap"]["identity"] = identity_measure(ds, model_name, fold_models_explanations[ds][model_name]) #IDENTITY
            # fold_models_explanations_measures[ds][model_name]["shap"]["separability"] = separability_measure(ds, model_name, fold_models_explanations[ds][model_name]) # SEPARABILITY
            # fold_models_explanations_measures[ds][model_name]["shap"]["similarity"] = similarity_measure(ds,model_name,fold_models_explanations[ds][model_name])  # SIMILARITY
            # fold_models_explanations_measures[ds][model_name]["shap"]["stability"] = stability_measure(ds, model_name, fold_models_explanations[ds][model_name]) # STABILITY



            # file_path_json = ds_dir / f"{model_name}_fold_model_posthoc_measures.json"
            # save_json(file_path_json, fold_models_explanations_measures[ds][model_name])


    # print(fold_models_explanations["iris"]['dt'][0]['feature_attribution_pred_class'][0])
    # print(fold_models_explanations["iris"]['dt'][0]['y_pred'][0])
    # print(identity_measure(fold_models_explanations["iris"]['dt'])[0]["identity_total_test_avg"])

if __name__ == "__main__":
    args = parse_args()
    train_cli(args)



