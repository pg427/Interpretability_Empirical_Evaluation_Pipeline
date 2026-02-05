from statistics import mean
import numpy as np
from pathlib import Path
from model_save_functions import save_model, load_model
from posthoc_functions import shap_attributions
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from posthoc_functions import rebuild_explainer_from_fold

base_dir = Path.cwd()/"trained_models"

def dist_calc(a,b, shap_dist:str="l2"):
    d = a - b
    if shap_dist == "l2":
        return float(np.linalg.norm(d, ord=2))
    if shap_dist == "l1":
        return float(np.linalg.norm(d, ord=1))
    if shap_dist == "linf":
        return float(np.linalg.norm(d, ord=np.inf))
    raise ValueError("shap_dist must be one of: 'l2', 'l1', 'linf'")

def identity_measure(dataset, method, folds, *, shap_dist:str="l2"):
    '''
    This function calculates identity values given a dataset, method and a fold structure of the type below.
    :param folds: Dictionary of folds structured as:
        fold: Fold No.
        train_idx: Indices of training samples
        test_idx: Indices of testing samples
        X_train: Features of training samples
        X_test: Features of testing samples
        y_train: Labels of training samples
        y_test: Labels of testing samples
        scaler: Standard Scaler object for standardization
        model:  Classifier object
        performance_metrics: Dictionary of performance metrics and their corresponding values for the trained classifier
        explainer: SHAP explainer object
        feature_attribution_all_classes: Feature attributions for all classes for testing instances
        feature_attribution_pred_class: Feature attributions for predicted class via Classifier
        y_pred: Predicted class by Classifier

    :return: Dictionary of SHAP feature attributions for test set with Classifier
        ...
        identity_cross_comparison: Produces a sub dictionary that includes:
            'test_instance': The test instance for which identity is going to be measured,
            'entry_num1': ID for one attribution amongst the five attributions created for the test instance,
            'attr1': First attribution created for test instance
            'entry_num2': ID for another attribution amongst the five attributions created for the test instance,
            'attr2': Second attribution created for test instance,
            'distance': Distance calculated between two attributions using dist_calc
        identity_per_test_avg: List of average distance calculated for feature attribution comparisons for one test instance.
        identity_total_test_avg: Averaged distance of SHAP explanation across all comparisons across all test instances.
    '''
    identity_path = base_dir/dataset/f"{method}_fold_model_shap_identity.joblib"
    fold_model_shap_identity = []
    if identity_path.exists():
        fold_model_shap_identity = load_model(identity_path)
    else:
        for fold_idx, fold in enumerate(folds):
            X_test = fold['X_test']
            y_pred_shap = fold['y_pred']

            model = fold['model']
            avg_dists_shap = []
            shap_dists_list_i = {}
            for idx, i in enumerate(X_test):
                shap_dists_list_i[f"test_instance-{idx}"] = []
                test_i = np.tile(i, (5, 1))
                y_pred_i = np.repeat(y_pred_shap[idx], 5)

                explainer = rebuild_explainer_from_fold(method,fold)

                sv = explainer.shap_values(test_i)
                class_idx = np.array([np.where(model.classes_ == y)[0][0] for y in y_pred_i])
                sv_pred = sv[np.arange(len(test_i)), :, class_idx]

                dists_shap_i = []
                for idj in range(len(sv_pred)):
                    for idk in range(idj + 1, len(sv_pred)):
                        j = sv_pred[idj]
                        k = sv_pred[idk]
                        expl_expl_dist = dist_calc(j, k, shap_dist=shap_dist)
                        dists_shap_i.append(expl_expl_dist)
                        shap_dists_list_i[f"test_instance-{idx}"].append({
                            'test_instance': i,
                            'entry_num1': idj,
                            'attr1': j,
                            'entry_num2': idk,
                            'attr2': k,
                            'distance': expl_expl_dist
                        })
                avg_dists_shap.append(mean(dists_shap_i))
            avg_dists_shap_all = mean(avg_dists_shap)
            fold_model_shap_identity.append({
                "fold": fold['fold'],
                "train_idx": fold['train_idx'],
                "test_idx": fold['test_idx'],
                "X_train": fold['X_train'],
                "X_test": fold['X_test'],
                "y_train": fold['y_train'],
                "y_test": fold['y_test'],
                "scaler": fold['scaler'],
                "performance_metrics": fold['performance_metrics'],
                "feature_attribution_all_classes": fold["feature_attribution_all_classes"],
                "feature_attribution_pred_class": fold["feature_attribution_pred_class"],
                "y_pred": fold["y_pred"],
                "identity_cross_comparison": shap_dists_list_i,
                "identity_per_test_avg": avg_dists_shap,
                "identity_total_test_avg": avg_dists_shap_all
            })
        save_model(fold_model_shap_identity, identity_path)
        # "model": fold['model'],
        # "explainer": fold['explainer'],
    return fold_model_shap_identity

def separability_measure(dataset, method, folds, *, shap_dist:str="l2", tol: float = 1e-12):
    '''
        This function calculates separability values given a dataset, method and a fold structure of the type below.
        :param folds: Dictionary of folds structured as:
            fold: Fold No.
            train_idx: Indices of training samples
            test_idx: Indices of testing samples
            X_train: Features of training samples
            X_test: Features of testing samples
            y_train: Labels of training samples
            y_test: Labels of testing samples
            scaler: Standard Scaler object for standardization
            model:  Classifier object
            performance_metrics: Dictionary of performance metrics and their corresponding values for the trained classifier
            explainer: SHAP explainer object
            feature_attribution_all_classes: Feature attributions for all classes for testing instances
            feature_attribution_pred_class: Feature attributions for predicted class via Classifier
            y_pred: Predicted class by Classifier

        :return: Dictionary of SHAP feature attributions + separability measure calculations  for test set with Classifier
            ...
            separability_cross_comparison_per_instance: Produces a sub dictionary that includes:
                'test_instance_1_ID': One test instance number ID,
                'test_instance_1': Test instance corresponding with first ID,
                'attr1': SHAP attribution created for first test instance,
                'test_instance_2_ID': Second test instance number ID,
                'test_instance_2': Test instance corresponding with second ID,
                'attr2': SHAP attribution created for second test instance,
                'distance': Distance calculated between two attributions using dist_calc
            separability_per_test_avg: List of average distance calculated for feature attribution comparisons for one test instance.
            separability_total_test_avg: Averaged distance of SHAP explanation across all comparisons across all test instances.
    '''

    separability_path = base_dir/dataset/f"{method}_fold_model_shap_separability.joblib"
    fold_model_shap_separability = []
    if separability_path.exists():
        fold_model_shap_separability = load_model(separability_path)
    else:
        for fold_idx, fold in enumerate(folds):
            X_test = fold['X_test']
            y_pred_shap = fold['y_pred']
            # explainer = fold['explainer']
            # model = fold['model']
            sv_pred = fold['feature_attribution_pred_class']

            # X_unique, unique_idx = np.unique(X_test, axis=0, return_index=True)
            # unique_idx = np.sort(unique_idx)
            # S = X_test[unique_idx]
            # y_pred_S = y_pred_shap[unique_idx]
            #
            # # attr = shap_attributions(explainer, model, S, y_pred_S)
            # sv = explainer.shap_values(S)
            # class_idx = np.array([np.where(model.classes_ == y)[0][0] for y in y_pred_S])
            # sv_pred = sv[np.arange(len(S)), :, class_idx]

            # ----- COMPARE EACH INTERPRETATION TO ALL OTHER INTERPRETATIONS -----
            per_instance = []
            cross = {}
            avg_sep_dists_shap = []
            avg_sep_dists_shap_total = []

            for idi, i in enumerate(X_test):
                key = f"test_instance-{i}"
                cross[key] = []
                dists_shap_i = []

                for idj, j in enumerate(X_test):
                    if idi == idj:
                        continue
                    d = dist_calc(sv_pred[idi], sv_pred[idj], shap_dist=shap_dist)
                    cross[key].append({
                        'test_instance_1_ID': idi,
                        'test_instance_1': i,
                        'attr1': sv_pred[idi],
                        'test_instance_2_ID': idj,
                        'test_instance_2': j,
                        'attr2': sv_pred[idj],
                        'distance': float(d)
                    })
                    if d<=tol:
                        dists_shap_i.append(0)
                    else:
                        dists_shap_i.append(d)
                avg_sep_dists_shap.append(mean(dists_shap_i))
            avg_sep_dists_shap_total = mean(avg_sep_dists_shap)

            fold_model_shap_separability.append({
                "fold": fold['fold'],
                "train_idx": fold['train_idx'],
                "test_idx": fold['test_idx'],
                "X_train": fold['X_train'],
                "X_test": fold['X_test'],
                "y_train": fold['y_train'],
                "y_test": fold['y_test'],
                "scaler": fold['scaler'],
                "performance_metrics": fold['performance_metrics'],
                "feature_attribution_all_classes": fold["feature_attribution_all_classes"],
                "feature_attribution_pred_class": fold["feature_attribution_pred_class"],
                "y_pred": fold["y_pred"],

                "separability_cross_comparison_per_instance": cross,
                "separability_per_test_avg": avg_sep_dists_shap,
                "separability_total_test_avg": avg_sep_dists_shap_total
            })
        save_model(fold_model_shap_separability, separability_path)
    return fold_model_shap_separability

def similarity_measure(dataset, method, folds, *, shap_dist:str="l2", tot: float=1e-12, dbscan_eps: float = 0.1,
    dbscan_min_samples: int = 5,):
    '''
        This function calculates similarity values given a dataset, method and a fold structure of the type below.
        :param folds: Dictionary of folds structured as:
            fold: Fold No.
            train_idx: Indices of training samples
            test_idx: Indices of testing samples
            X_train: Features of training samples
            X_test: Features of testing samples
            y_train: Labels of training samples
            y_test: Labels of testing samples
            scaler: Standard Scaler object for standardization
            model:  Classifier object
            performance_metrics: Dictionary of performance metrics and their corresponding values for the trained classifier
            explainer: SHAP explainer object
            feature_attribution_all_classes: Feature attributions for all classes for testing instances
            feature_attribution_pred_class: Feature attributions for predicted class via Classifier
            y_pred: Predicted class by Classifier

        :return: Dictionary of SHAP feature attributions + similarity measure calculations  for test set with Classifier
            ...

            similarity_total_test_avg: Averaged distance of SHAP explanation across all comparisons across all test instances.
    '''

    similarity_path = base_dir / dataset / f"{method}_fold_model_shap_similarity.joblib"
    fold_model_shap_similarity = []
    if similarity_path.exists():
        fold_model_shap_similarity = load_model(similarity_path)
    else:
        for fold_idx, fold in enumerate(folds):
            X_test = fold['X_test']
            sv_pred = fold['feature_attribution_pred_class']
            scaler = fold['scaler']

            # STEP 1: Normalize test instances and DBSCAN Clustering
            if scaler == None:
                scaler = MinMaxScaler()
                X_train = fold['X_train']
                scaler.fit(X_train)
            Xn = scaler.transform(X_test)
            db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            cluster_labels = db.fit_predict(Xn)

            # STEP 2: Mean Pairwise distances within each cluster
            clusters = {}
            cluster_means =[]
            for c in sorted(set(cluster_labels)):
                if c == -1:
                    continue
                idx = np.where(cluster_labels == c)[0]
                if len(idx) < 2:
                    continue

                dists = []
                for a in range(len(idx)):
                    for b in range(a + 1, len(idx)):
                        i = idx[a]
                        j = idx[b]
                        d = dist_calc(sv_pred[i], sv_pred[j], shap_dist)
                        dists.append(d)

                c_mean = float(np.mean(dists)) if len(dists) else float("nan")
                cluster_means.append(c_mean)
                clusters[f"cluster-{int(c)}"] = {
                    "indices": idx,
                    "size": int(len(idx)),
                    "mean_pairwise_euclidean_dist": c_mean,
                }
            similarity_score = float(np.mean(cluster_means)) if len(cluster_means) else float("nan")

            fold_model_shap_similarity.append({
                "fold": fold["fold"],
                "train_idx": fold["train_idx"],
                "test_idx": fold["test_idx"],
                "X_train": fold["X_train"],
                "X_test": fold["X_test"],
                "y_train": fold["y_train"],
                "y_test": fold["y_test"],
                "scaler": fold["scaler"],
                "performance_metrics": fold["performance_metrics"],
                "feature_attribution_all_classes": fold["feature_attribution_all_classes"],
                "feature_attribution_pred_class": fold["feature_attribution_pred_class"],
                "y_pred": fold["y_pred"],

                "similarity_clusters": clusters,
                "similarity_total_test_avg": similarity_score,
            })
        save_model(fold_model_shap_similarity, similarity_path)
    return fold_model_shap_similarity

def stability_measure(dataset, method, folds, *, random_state:int = 42):
    '''
        Measures whether instances of same class have comparable interpretations (Dobranska et al. 2023)
        :param folds: Dictionary of folds structured as:
            fold: Fold No.
            train_idx: Indices of training samples
            test_idx: Indices of testing samples
            X_train: Features of training samples
            X_test: Features of testing samples
            y_train: Labels of training samples
            y_test: Labels of testing samples
            scaler: Standard Scaler object for standardization
            model:  Classifier object
            performance_metrics: Dictionary of performance metrics and their corresponding values for the trained classifier
            explainer: SHAP explainer object
            feature_attribution_all_classes: Feature attributions for all classes for testing instances
            feature_attribution_pred_class: Feature attributions for predicted class via Classifier
            y_pred: Predicted class by Classifier

        :return: Dictionary of SHAP feature attributions + similarity measure calculations  for test set with Classifier
            ...

            stability_clusters - Stores what the explanation clusters look like:
                "indices" : Index of instances in the cluster.
                "size" : Number of instances in the cluster.
                "majority_class": Class label for the cluster
                "class_distribution": Size distribution of the cluster allotted to the specific cluster class

            stability_per_instance - Stores stability by instance i.e. if interpretation prediction label matches cluster label
                "test_instance_id" : ID of test instance.
                "y_pred" : Prediction Class of instance.
                "cluster_id": Cluster ID.
                "cluster_majority_class": Cluster majority class.
                "match": whether the cluster matched or not.

            stability_total_test_avg - Stores final averaged stability
    '''
    stability_path = base_dir / dataset / f"{method}_fold_model_shap_stability.joblib"
    fold_model_shap_stability = []

    if stability_path.exists():
        fold_model_shap_stability = load_model(stability_path)
    else:
        for fold_idx, fold in enumerate(folds):
            sv_pred = fold['feature_attribution_pred_class']
            y_pred = fold['y_pred']

            # Number of clusters = Number of labels in the data
            unique_labels = np.unique(y_pred)
            k = int(len(unique_labels))

            if k<=1:
                majority = int(unique_labels[0]) if len(unique_labels) else None
                stability_clusters = {
                    "cluster-0": {
                        "indices": np.arange(len(y_pred)),
                        "size": int(len(y_pred)),
                        "majority_class": majority,
                        "class_distribution": {majority: int(len(y_pred))} if majority is not None else {},
                    }
                }
                per_instance = [
                    {
                        "test_instance_id": int(i),
                        "y_pred": int(y_pred[i]),
                        "cluster_id": 0,
                        "cluster_majority_class": majority,
                        "match": True
                    }
                    for i in range(len(y_pred))
                ]
                stability_score = 1.0

            else:
                scaler = StandardScaler()
                sv_scaled = scaler.fit_transform(sv_pred)

                # Use K-MEANS; n_init runs KMeans that number of times with different centroid initializations - solution with lowest inertia kept
                km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                cluster_ids = km.fit_predict(sv_scaled)

                #map each cluster
                cluster_to_class = {}
                stability_clusters = {}
                for c in range(k):
                    idx = np.where(cluster_ids == c)[0]
                    if len(idx) == 0:
                        cluster_to_class[c] = None
                        stability_clusters[f"cluster-{c}"] = {
                            "indices": idx,
                            "size": 0,
                            "majority_class": None,
                            "class_distribution": {},
                        }
                        continue

                    classes, counts = np.unique(y_pred[idx], return_counts=True)
                    maj_class = classes[int(np.argmax(counts))]
                    cluster_to_class[c] = maj_class

                    stability_clusters[f"cluster-{c}"] = {
                        "indices": idx,
                        "size": int(len(idx)),
                        "majority_class": int(maj_class),
                        "class_distribution": {int(cls): int(cnt) for cls, cnt in zip(classes, counts)},
                    }

                # per-instance match + overall score (fraction of matches)
                per_instance = []
                matches = []
                for i in range(len(y_pred)):
                    c = int(cluster_ids[i])
                    maj = cluster_to_class.get(c, None)
                    ok = bool(maj == y_pred[i])
                    matches.append(ok)
                    per_instance.append({
                        "test_instance_id": int(i),
                        "y_pred": int(y_pred[i]),
                        "cluster_id": c,
                        "cluster_majority_class": int(maj) if maj is not None else None,
                        "match": ok
                    })

                stability_score = float(np.mean(matches)) if len(matches) else float("nan")

            fold_model_shap_stability.append(
                {
                    "fold": fold["fold"],
                    "train_idx": fold["train_idx"],
                    "test_idx": fold["test_idx"],
                    "X_train": fold["X_train"],
                    "X_test": fold["X_test"],
                    "y_train": fold["y_train"],
                    "y_test": fold["y_test"],
                    "scaler": fold["scaler"],
                    "performance_metrics": fold["performance_metrics"],
                    "feature_attribution_all_classes": fold["feature_attribution_all_classes"],
                    "feature_attribution_pred_class": fold["feature_attribution_pred_class"],
                    "y_pred": fold["y_pred"],

                    "stability_clusters": stability_clusters,
                    "stability_per_instance": per_instance,
                    "stability_total_test_avg": stability_score,
                }
            )

        save_model(fold_model_shap_stability, stability_path)

    return fold_model_shap_stability