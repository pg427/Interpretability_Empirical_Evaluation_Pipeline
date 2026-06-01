from dataset_functions import load_dataset, stratified_5fold_standardize
from method_functions import CART_DT_5FOLD, XGB_5FOLD, CBR_5FOLD, PROTOPNET_5FOLD, MLP_5FOLD, DNN_8HL_5fold
from posthoc_functions import CART_DT_5fold_shap, XGB_5fold_shap, CBR_5fold_shap, PROTOPNET_5fold_shap, DNN_8HL_5fold_shap, MLP_5fold_shap
from posthoc_measures_functions import  similarity_measure, stability_measure, neighborhood_fidelity_comprehensibility_stability_measures, parsimony_measure, faithfulness_measure
from model_save_functions import save_model, load_model, save_json
from pathlib import Path
import argparse
import numpy as np
from direct_measures_functions import ria_measure, soc_all_methods_for_dataset, feature_synergy_all_methods_for_dataset, robustness_all_methods_for_dataset, mec_all_methods_for_datasets
from statistical_measures import (spearman_features_vs_ria, spearman_classes_vs_ria, wilcoxon_h1_1_soc, wilcoxon_h1_2_soc, wilcoxon_h1_3_soc, wilcoxon_h1_1_feature_synergy, wilcoxon_h1_2_feature_synergy,
                                  wilcoxon_h1_3_feature_synergy, spearman_h2_1_feature_synergy, spearman_h2_2_feature_synergy, feature_synergy_structure_analysis,
                                  wilcoxon_h1_1_robustness, wilcoxon_h1_2_robustness, wilcoxon_h1_3_robustness, spearman_h2_1_robustness, spearman_h2_3_robustness, wilcoxon_h1_1_functional_complexity_all_measures,
                                  wilcoxon_functional_complexity_all_measures, spearman_h2_1_functional_complexity_all_measures, wilcoxon_shap_all_measures,
                                  spearman_features_vs_shap_measures, spearman_classes_vs_shap_measures, wilcoxon_surrogate_all_measures, spearman_surrogate_measures, run_cross_measure_consistency,
                                  run_direct_vs_posthoc_alignment, run_performance_interpretability_relationship, run_ranking_agreement_analysis)

import pandas as pd
ALL_DATASETS = ["iris", "wine", "breast_cancer", "german_credit", "darwin", "sepsis", "yeast"]
ALL_MODELS = ["dt", "xgb", "cbr", "proto", "mlp", "dnn"]

def parse_args():
    parser = argparse.ArgumentParser(description="Train AI methods on selected Datasets")
    parser.add_argument( "--datasets", nargs="+", choices=["iris", "wine", "breast_cancer", "german_credit", "darwin", "sepsis", "yeast","all"], # "arcene", "higgs", "isolet"
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
    # stats_dir = Path.cwd()/"statistical_calculations"
    # base_dir.mkdir(exist_ok=True)
    # stats_dir.mkdir(exist_ok=True)

    fold_models = {}
    fold_models_explanations = {}
    fold_models_explanations_measures = {}
    for ds in datasets:
        print(f"\n=== DATASET {ds.upper()} ===")
        X, y, feature_names = load_dataset(ds)

        # --- MEASURE 1: RIA ----
        # ria_results = ria_measure((X, y, feature_names), ds)
        # ria_json_path = (base_dir / ds) / f"{ds}_ria.json"
        # save_json(ria_json_path, ria_results)

        folds_stand = stratified_5fold_standardize(X, y)
        folds_unstand = stratified_5fold_standardize(X, y, standardize=False)

        ds_dir = base_dir/ds
        ds_dir.mkdir(parents=True, exist_ok=True)

        fold_models[ds] = {}
        fold_models_explanations[ds] = {}
        fold_models_explanations_measures[ds] = {}
        for model_name in models:
            fold_models_explanations[ds][model_name] = {}
            fold_models_explanations_measures[ds][model_name] = {}
            model_path = ds_dir/f"{model_name}_fold_model.joblib"

            if model_path.exists() and not args.overwrite:
                print(f"Model {model_name} already trained")
                fold_models[ds][model_name] = load_model(model_path)
    #
    #         else:
    #             print(f"Training Model {model_name} ")
    #             if model_name == "dt":
    #                 fold_models[ds][model_name] = CART_DT_5FOLD(folds_stand)
    #                 save_model(fold_models[ds][model_name], model_path)
    #             elif model_name == "xgb":
    #                 fold_models[ds][model_name] = XGB_5FOLD(folds_stand)
    #                 save_model(fold_models[ds][model_name], model_path)
    #             elif model_name == "cbr":
    #                 fold_models[ds][model_name] = CBR_5FOLD(folds_unstand)
    #                 save_model(fold_models[ds][model_name], model_path)
    #             elif model_name == "proto":
    #                 fold_models[ds][model_name] = PROTOPNET_5FOLD(folds_stand)
    #                 save_model(fold_models[ds][model_name], model_path)
    #             elif model_name == "mlp":
    #                 fold_models[ds][model_name] = MLP_5FOLD(folds_stand)
    #                 save_model(fold_models[ds][model_name], model_path)
    #             elif model_name == "dnn":
    #                 fold_models[ds][model_name] = DNN_8HL_5fold(folds_stand)
    #                 save_model(fold_models[ds][model_name], model_path)
    #
    #
    #         file_path_json = ds_dir / f"{model_name}_fold_model.json"
    #         save_json(file_path_json, fold_models[ds][model_name])

        # --- MEASURE 2: SOC ----
        # soc_results = soc_all_methods_for_dataset(
        #     dataset_name=ds,
        #     method_fold_results=fold_models[ds],  # method -> folds
        #     distance_type="euclidean",
        #     force_recompute=args.overwrite,  # optional
        # )
        # soc_json_path = (base_dir / ds) / f"{ds}_soc_all_methods.json"
        # save_json(soc_json_path, soc_results)
        #
        # # --- MEASURE 3: Feature Synergy ----
        # fs_all = feature_synergy_all_methods_for_dataset(
        #     ds,
        #     fold_models[ds],
        #     pop_size= 60,
        #     n_generations=30,
        #     n_runs=3,
        #     parent_count=30,
        #     include_accuracy_factor=False,
        # )
        # fs_json_path = (base_dir / ds) / f"{ds}_fs_all_methods.json"
        # save_json(fs_json_path, fs_all)

        # # --- MEASURE 4: Robustness ----
        # rs_all = robustness_all_methods_for_dataset(
        #     ds,
        #     fold_models[ds],
        #     k_bins=1,
        #     N=20,
        #     G=10,
        #     cf_only=False,
        #     max_instances_per_subpop=5,
        # )
        # rs_json_path = (base_dir / ds) / f"{ds}_rs_all_methods.json"
        # save_json(rs_json_path, rs_all)

        # # --- MEASURE 5: No. of features, Interaction Strength, Main Effect Complexity ----
        # mec_all = mec_all_methods_for_datasets(
        #     ds,
        #     fold_models[ds]
        # )
        #
        # mec_json_path = (base_dir / ds) / f"{ds}_mec_all_methods.json"
        # save_json(mec_json_path, mec_all)

        # # --- ALL SURROGATE MEASURES ----
        # surrogate_measures = {}
        # for model_name in models:
        #     surrogate_measures[model_name] = neighborhood_fidelity_comprehensibility_stability_measures(ds, model_name, fold_models[ds][model_name])
        # surrogate_json_path = (base_dir / ds) / f"{ds}_surrogate_all_measures.json"
        # save_json(surrogate_json_path, surrogate_measures)
        #
        # # # --- ALL SHAP MEASURES ----
        shap_explanations = {}
        for model_name in models:
            shap_explanations_measures = {}
            if model_name == "dt":
                shap_explanations[model_name] = CART_DT_5fold_shap(ds, fold_models[ds][model_name])
            elif model_name == "xgb":
                shap_explanations[model_name] = XGB_5fold_shap(ds, fold_models[ds][model_name])
            elif model_name == "cbr":
                shap_explanations[model_name] = CBR_5fold_shap(ds, fold_models[ds][model_name])
            elif model_name == "proto":
                shap_explanations[model_name] = PROTOPNET_5fold_shap(ds, fold_models[ds][model_name])
            elif model_name == "mlp":
                shap_explanations[model_name] = MLP_5fold_shap(ds, fold_models[ds][model_name])
            elif model_name == "dnn":
                shap_explanations[model_name] = DNN_8HL_5fold_shap(ds, fold_models[ds][model_name])

            shap_explanations_measures['similarity'] = similarity_measure(ds, model_name, shap_explanations[model_name])
            # shap_explanations_measures['stability'] = stability_measure(ds, model_name, shap_explanations[model_name])
            # shap_explanations_measures['parsimony'] = parsimony_measure(ds, model_name, shap_explanations[model_name])
            # shap_explanations_measures['faithfulness'] = faithfulness_measure(ds, model_name, shap_explanations[model_name])
        #
            # shap_json_path = (base_dir / ds) / f"{ds}_{model_name}_shap_explanations.json"
            # save_json(shap_json_path, shap_explanations[model_name])
            shap_measures_json_path = (base_dir / ds) / f"{ds}_{model_name}_shap_measures.json"
            save_json(shap_measures_json_path, shap_explanations_measures)

    # --- STATISTICAL ANALYSIS: RIA H2.1 ----
    # ria_h2_1_results = spearman_features_vs_ria(
    #     base_dir=base_dir,
    #     stats_dir=stats_dir,
    #     datasets=datasets,
    #     ria_metric="ARIA_mean",
    # )
    #
    # save_json(stats_dir / "ria_spearman.json", ria_h2_1_results)


    # --- STATISTICAL ANALYSIS: RIA H2.3 ----
    # ria_h2_3_results = spearman_classes_vs_ria(
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     datasets=datasets,
    #     ria_metric="ARIA_mean",
    # )

    # save_json(stats_dir / "ria_classes_spearman.json", ria_h2_3_results)

    # --- STATISTICAL ANALYSIS: SOC H1.1 ----
    # soc_h1_1_results = wilcoxon_h1_1_soc(
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     datasets=datasets,
    # )
    # save_json(stats_dir / "soc_h1_1_wilcoxon.json", soc_h1_1_results)

    # --- STATISTICAL ANALYSIS: SOC H1.2 ----
    # soc_h1_2_results = wilcoxon_h1_2_soc(
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     datasets=datasets,
    # )
    # save_json(stats_dir / "soc_h1_2_wilcoxon.json", soc_h1_2_results)

    # --- STATISTICAL ANALYSIS: SOC H1.2 ----
    # soc_h1_3_results = wilcoxon_h1_3_soc(
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     datasets=datasets,
    # )
    # save_json(stats_dir / "soc_h1_3_wilcoxon.json", soc_h1_3_results)

    # --- STATISTICAL ANALYSIS: FS H1.1 ----
    # fs_h1_1_results = wilcoxon_h1_1_feature_synergy(
    #         base_dir=base_dir,
    #         stats_dir=Path.cwd() / "statistical_calculations",
    #         datasets=datasets,
    # )
    # save_json(stats_dir / "feature_synergy_h1_1_wilcoxon.json", fs_h1_1_results)

    # --- STATISTICAL ANALYSIS: FS H1.2 ----
    # fs_h1_2_results = wilcoxon_h1_2_feature_synergy(
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     datasets=datasets,
    # )
    # save_json(stats_dir / "feature_synergy_h1_2_wilcoxon.json", fs_h1_2_results)

    # --- STATISTICAL ANALYSIS: FS H1.3 ----
    # fs_h1_3_results = wilcoxon_h1_3_feature_synergy(
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     datasets=datasets,
    # )
    # save_json(stats_dir / "feature_synergy_h1_3_wilcoxon.json", fs_h1_3_results)

    # --- STATISTICAL ANALYSIS: FS H2.1 ----
    # fs_h2_1_results = spearman_h2_1_feature_synergy(
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     datasets=datasets,
    # )
    # save_json(stats_dir / "feature_synergy_h2_1_spearman.json", fs_h2_1_results)

    # --- (ARCHIVE) STATISTICAL ANALYSIS: FS H2.2 ----
    # fs_h2_2_results = spearman_h2_2_feature_synergy(
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     datasets=datasets,
    # )
    # save_json(stats_dir / "(ARCHIVE) feature_synergy_h2_2_spearman.json", fs_h2_2_results)

    # --- STATISTICAL ANALYSIS: FS H2.2 ----
    # fs_h2_2_results = spearman_h2_2_feature_synergy(
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     datasets=datasets,
    # )
    # save_json(stats_dir / "feature_synergy_h2_2_spearman.json", fs_h2_2_results)

    # --- STATISTICAL ANALYSIS: FS Structure Analysis ----
    # fs_structure_results = feature_synergy_structure_analysis(
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     datasets=datasets,
    # )
    # save_json(stats_dir / "feature_synergy_structure_analysis.json", fs_structure_results)

    # --- STATISTICAL ANALYSIS: Robustness H1.1 ----
    # robustness_h1_1_results = wilcoxon_h1_1_robustness()
    # save_json(stats_dir / "robustness_h1_1_wilcoxon.json", robustness_h1_1_results)

    # --- STATISTICAL ANALYSIS: Robustness H1.2 ----
    # robustness_h1_2_results = wilcoxon_h1_2_robustness()
    # save_json(stats_dir / "robustness_h1_2_wilcoxon.json", robustness_h1_2_results)

    # --- STATISTICAL ANALYSIS: Robustness H1.3 ----
    # robustness_h1_3_results = wilcoxon_h1_3_robustness()
    # save_json(stats_dir / "robustness_h1_3_wilcoxon.json", robustness_h1_3_results)

    # --- STATISTICAL ANALYSIS: Robustness H2.1 ----
    # robustness_h2_1_results = spearman_h2_1_robustness(
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations"
    # )
    # save_json(stats_dir / "robustness_h2_1_spearman.json", robustness_h2_1_results)

    # --- STATISTICAL ANALYSIS: Robustness H2.3 ----
    # robustness_h2_3_results = spearman_h2_3_robustness(
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations"
    # )
    # save_json(stats_dir / "robustness_h2_3_spearman.json", robustness_h2_3_results)

    # --- STATISTICAL ANALYSIS: MEC H1.1 ----
    # functional_complexity_h1_1_results = wilcoxon_h1_1_functional_complexity_all_measures()
    # save_json(stats_dir / "functional_complexity_h1_1_wilcoxon.json", functional_complexity_h1_1_results)

    # --- STATISTICAL ANALYSIS: MEC H1.2 ----
    # functional_complexity_h1_2_results = wilcoxon_functional_complexity_all_measures(
    # hypothesis_id="H1.2",
    # description=(
    #     "Wilcoxon signed-rank tests comparing simpler and more complex "
    #     "variants within the same model family for NF, IAS, and MEC."
    # ),
    # expected_relationship="DT < XGB, CBR < ProtoPNet, MLP < DNN",
    # comparisons={
    #     "dt_vs_xgb": ("dt", "xgb"),
    #     "cbr_vs_proto": ("cbr", "proto"),
    #     "mlp_vs_dnn": ("mlp", "dnn"),
    # },
    # output_filename="functional_complexity_h1_2_wilcoxon.json",
# )
    # save_json(stats_dir / "functional_complexity_h1_2_wilcoxon.json", functional_complexity_h1_2_results)

# --- STATISTICAL ANALYSIS: MEC H1.3 ----
#     functional_complexity_h1_3_results =  wilcoxon_functional_complexity_all_measures(
#     hypothesis_id="H1.3",
#     description=(
#         "Wilcoxon signed-rank tests evaluating whether "
#         "ProtoPNet exhibits intermediate interpretability "
#         "between transparent and neural models for "
#         "NF, IAS, and MEC."
#     ),
#     expected_relationship=(
#         "DT/CBR < ProtoPNet < MLP/DNN"
#     ),
#     comparisons={
#         "dt_vs_proto": ("dt", "proto"),
#         "cbr_vs_proto": ("cbr", "proto"),
#         "proto_vs_mlp": ("proto", "mlp"),
#         "proto_vs_dnn": ("proto", "dnn"),
#     },
#     output_filename="functional_complexity_h1_3_wilcoxon.json",
# )
#     save_json(stats_dir / "functional_complexity_h1_3_wilcoxon.json", functional_complexity_h1_3_results)

    # --- STATISTICAL ANALYSIS: MEC H2.1 ----
    # functional_complexity_h2_1_results = spearman_h2_1_functional_complexity_all_measures()
    # save_json(stats_dir / "functional_complexity_h2_1_spearman.json", functional_complexity_h2_1_results)

    # --- STATISTICAL ANALYSIS: MEC H2.2 ----
    # functional_complexity_h2_2_results = spearman_h2_2_functional_complexity_all_measures()
    # save_json(stats_dir / "functional_complexity_h2_2_spearman.json", functional_complexity_h2_2_results)

    # --- STATISTICAL ANALYSIS: SHAP H1.1 ----
    # shap_h1_1_results = wilcoxon_shap_all_measures(
    #     hypothesis_id="H1.1",
    #     description=(
    #         "Wilcoxon signed-rank tests comparing SHAP explanation "
    #         "quality for transparent models (DT, CBR) against "
    #         "neural models (MLP, DNN)."
    #     ),
    #     expected_relationship="DT/CBR > MLP/DNN",
    #     comparisons={
    #         "dt_vs_mlp": ("dt", "mlp"),
    #         "dt_vs_dnn": ("dt", "dnn"),
    #         "cbr_vs_mlp": ("cbr", "mlp"),
    #         "cbr_vs_dnn": ("cbr", "dnn"),
    #     },
    #     output_filename="shap_h1_1_wilcoxon.json",
    # )
    # save_json(stats_dir / "shap_h1_1_wilcoxon.json", shap_h1_1_results)

    # --- STATISTICAL ANALYSIS: SHAP H1.2 ----
    # shap_h1_2_results = wilcoxon_shap_all_measures(
    #     hypothesis_id="H1.2",
    #     description=(
    #         "Wilcoxon signed-rank tests comparing SHAP explanation "
    #         "quality for simpler and more complex variants within the same "
    #         "model family."
    #     ),
    #     expected_relationship="DT > XGB, CBR > ProtoPNet, MLP > DNN",
    #     comparisons={
    #         "dt_vs_xgb": ("dt", "xgb"),
    #         "cbr_vs_proto": ("cbr", "proto"),
    #         "mlp_vs_dnn": ("mlp", "dnn"),
    #     },
    #     output_filename="shap_h1_2_wilcoxon.json",
    # )
    # save_json(stats_dir / "shap_h1_2_wilcoxon.json", shap_h1_2_results)

    # --- STATISTICAL ANALYSIS: SHAP H1.3 ----
    # shap_h1_3_results = wilcoxon_shap_all_measures(
    #     hypothesis_id="H1.3",
    #     description=(
    #         "Wilcoxon signed-rank tests evaluating whether SHAP explanation "
    #         "quality positions ProtoPNet between transparent models "
    #         "(DT, CBR) and neural models (MLP, DNN)."
    #     ),
    #     expected_relationship="DT/CBR > ProtoPNet > MLP/DNN",
    #     comparisons={
    #         "dt_vs_proto": ("dt", "proto"),
    #         "cbr_vs_proto": ("cbr", "proto"),
    #         "proto_vs_mlp": ("proto", "mlp"),
    #         "proto_vs_dnn": ("proto", "dnn"),
    #     },
    #     output_filename="shap_h1_3_wilcoxon.json",
    # )
    # save_json(stats_dir / "shap_h1_3_wilcoxon.json", shap_h1_3_results)

    # --- STATISTICAL ANALYSIS: SHAP H2.1 ----
    # shap_h2_1_results = spearman_features_vs_shap_measures(
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     output_filename="shap_h2_1_spearman.json",
    # )
    # save_json(stats_dir / "shap_h2_1_spearman.json", shap_h2_1_results)

    # --- STATISTICAL ANALYSIS: SHAP H2.2 ----
    # shap_h2_2_results = spearman_classes_vs_shap_measures(
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     output_filename="shap_h2_2_classes_spearman.json",
    # )
    # save_json(stats_dir / "shap_h2_2_classes_spearman.json", shap_h2_2_results)


    # --- STATISTICAL ANALYSIS: SURROGATE H1.1 ----
    # surrogate_h1_1_results = wilcoxon_surrogate_all_measures(
    #     hypothesis_id="H1.1",
    #     description=(
    #         "Wilcoxon signed-rank tests comparing surrogate "
    #         "explanation measures for transparent methods "
    #         "(DT, CBR) against neural models (MLP, DNN)."
    #     ),
    #     expected_relationship="DT/CBR > MLP/DNN",
    #     comparisons={
    #         "dt_vs_mlp": ("dt", "mlp"),
    #         "dt_vs_dnn": ("dt", "dnn"),
    #         "cbr_vs_mlp": ("cbr", "mlp"),
    #         "cbr_vs_dnn": ("cbr", "dnn"),
    #     },
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     output_filename="surrogate_h1_1_wilcoxon.json",
    # )
    # save_json(stats_dir / "surrogate_h1_1_wilcoxon.json", surrogate_h1_1_results)
    #
    # # --- STATISTICAL ANALYSIS: SURROGATE H1.2 ----
    # surrogate_h1_2_results = wilcoxon_surrogate_all_measures(
    #     hypothesis_id="H1.2",
    #     description=(
    #         "Wilcoxon signed-rank tests comparing surrogate "
    #         "explanation measures for simpler and more "
    #         "complex variants within the same model family."
    #     ),
    #     expected_relationship="DT > XGB, CBR > ProtoPNet, MLP > DNN",
    #     comparisons={
    #         "dt_vs_xgb": ("dt", "xgb"),
    #         "cbr_vs_proto": ("cbr", "proto"),
    #         "mlp_vs_dnn": ("mlp", "dnn"),
    #     },
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     output_filename="surrogate_h1_2_wilcoxon.json",
    # )
    # save_json(stats_dir / "surrogate_h1_2_wilcoxon.json", surrogate_h1_2_results)
    #
    # # --- STATISTICAL ANALYSIS: SURROGATE H1.3 ----
    # surrogate_h1_3_results = wilcoxon_surrogate_all_measures(
    #     hypothesis_id="H1.3",
    #     description=(
    #         "Wilcoxon signed-rank tests evaluating whether "
    #         "ProtoPNet exhibits intermediate surrogate "
    #         "explanation behavior between transparent "
    #         "methods and neural architectures."
    #     ),
    #     expected_relationship="DT/CBR > ProtoPNet > MLP/DNN",
    #     comparisons={
    #         "dt_vs_proto": ("dt", "proto"),
    #         "cbr_vs_proto": ("cbr", "proto"),
    #         "proto_vs_mlp": ("proto", "mlp"),
    #         "proto_vs_dnn": ("proto", "dnn"),
    #     },
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     output_filename="surrogate_h1_3_wilcoxon.json",
    # )
    # save_json(stats_dir / "surrogate_h1_3_wilcoxon.json", surrogate_h1_3_results)
    #
    # # --- STATISTICAL ANALYSIS: SURROGATE H2.1 ----
    # dataset_features = {
    #     "iris": 4,
    #     "wine": 13,
    #     "breast_cancer": 30,
    #     "german_credit": 20,
    #     "darwin": 451,
    #     "yeast": 8,
    #     "sepsis": 3,
    # }
    #
    # surrogate_h2_1_results = spearman_surrogate_measures(
    #     hypothesis_id="H2.1",
    #     description=(
    #         "Spearman correlation between dataset feature count "
    #         "and surrogate explanation measures."
    #     ),
    #     dataset_property_map=dataset_features,
    #     property_name="num_features",
    #     expected_direction="negative",
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     output_filename="surrogate_h2_1_spearman.json",
    # )
    # save_json(stats_dir / "surrogate_h2_1_spearman.json", surrogate_h2_1_results)
    #
    # # --- STATISTICAL ANALYSIS: SHAP H2.2 ----
    # dataset_classes = {
    #     "iris": 3,
    #     "wine": 3,
    #     "breast_cancer": 2,
    #     "german_credit": 2,
    #     "darwin": 2,
    #     "yeast": 10,
    #     "sepsis": 2,
    # }
    #
    # surrogate_h2_2_results = spearman_surrogate_measures(
    #     hypothesis_id="H2.2",
    #     description=(
    #         "Spearman correlation between dataset class count "
    #         "and surrogate explanation measures."
    #     ),
    #     dataset_property_map=dataset_classes,
    #     property_name="num_classes",
    #     expected_direction="negative",
    #     base_dir=base_dir,
    #     stats_dir=Path.cwd() / "statistical_calculations",
    #     output_filename="surrogate_h2_2_spearman.json",
    # )
    # save_json(stats_dir / "surrogate_h2_2_spearman.json", surrogate_h2_2_results)

    # run_cross_measure_consistency(
    #     trained_models_dir="trained_models",
    #     datasets=[
    #         "iris",
    #         "wine",
    #         "breast_cancer",
    #         "german_credit",
    #         "sepsis",
    #         "darwin",
    #         "yeast"
    #     ],
    #     output_json_path="statistical_calculations/cross_measure_consistency_results.json"
    # )

    # run_direct_vs_posthoc_alignment(
    #     trained_models_dir="trained_models",
    #     datasets=[
    #         "iris",
    #         "wine",
    #         "breast_cancer",
    #         "german_credit",
    #         "sepsis",
    #         "darwin",
    #         "yeast"
    #     ],
    #     output_json_path=(
    #         "statistical_calculations/"
    #         "direct_vs_posthoc_alignment.json"
    #     )
    # )

    # run_performance_interpretability_relationship(
    #     trained_models_dir="trained_models",
    #     datasets=[
    #         "iris",
    #         "wine",
    #         "breast_cancer",
    #         "german_credit",
    #         "sepsis",
    #         "darwin",
    #         "yeast"
    #     ],
    #     output_json_path=(
    #         "statistical_calculations/"
    #         "performance_interpretability_relationship.json"
    #     )
    # )

    # run_ranking_agreement_analysis(
    #     trained_models_dir="trained_models",
    #     datasets=[
    #         "iris",
    #         "wine",
    #         "breast_cancer",
    #         "german_credit",
    #         "sepsis",
    #         "darwin",
    #         "yeast"
    #     ],
    #     output_json_path="statistical_calculations/ranking_agreement_analysis.json"
    # )
if __name__ == "__main__":
    args = parse_args()
    train_cli(args)



