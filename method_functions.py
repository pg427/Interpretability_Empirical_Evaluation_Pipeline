from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from dataset_functions import stratified_5fold_standardize, load_dataset
from typing import List, Dict, Any
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import OrdinalEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def CART_DT(X_train, y_train, *, random_state = 42):
    '''
    This function trains a single CART decision tree classifier (ONE TREE ONLY).
    :param X_train: Array of training data
    :param y_train: Array of training labels
    :param random_state: seed for random state for reproducibility (DEFAULT: 42)
    :return:
        model: CART decision tree classifier (ONE TREE ONLY)
    '''

    model = DecisionTreeClassifier(criterion="entropy", random_state=random_state)
    model.fit(X_train, y_train)
    return model

def CART_DT_5FOLD(folds, *, random_state = 42):
    '''
    This function trains a single CART decision tree classifier per fold
    :param folds: Dictionary of folds structured as:
        fold: Fold No.
        train_idx: Indices of training samples
        test_idx: Indices of testing samples
        X_train: Features of training samples
        X_test: Features of testing samples
        y_train: Labels of training samples
        y_test: Labels of testing samples
        scaler: Standard Scaler object for standardization
    :param random_state: Random seed for reproducibility (DEFAULT: 42)
    :return:
        DT_fold_model: Dictionary of CART decision tree classifiers per fold
            ...
            model: Saved & Trained DT Classifier model
            performance_metrics: Dictionary of performance metrics and their corresponding values for the trained DT classifier
                accuracy: calculates accuracy on X_test
                precision: calculated precision on X_test
                recall: calculated recall on X_test
                f1: calculated f1 on X_test
    '''
    DT_fold_model: List[Dict[str, Any]] = []
    for fold_idx, fold in enumerate(folds):
        X_train = fold['X_train']
        y_train = fold['y_train']
        X_test = fold['X_test']
        y_test = fold['y_test']
        model = DecisionTreeClassifier(criterion="entropy", random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted")
        }

        DT_fold_model.append(
            {
                "fold": fold['fold'],
                "train_idx": fold['train_idx'],
                "test_idx": fold['test_idx'],
                "X_train": fold['X_train'],
                "X_test": fold['X_test'],
                "y_train": fold['y_train'],
                "y_test": fold['y_test'],
                "scaler": fold['scaler'],
                "model": model,
                "performance_metrics": metrics
            }
        )
    return DT_fold_model

def XGB(X_train, y_train, *, random_state = 42):
    '''
    This function trains an XGBoost based Classifier.
    :param X_train: Array of training data
    :param y_train: Array of training labels
    :param random_state: seed for random state for reproducibility (DEFAULT: 42)
    :return:
        model: XGBoost based Classifier
    '''

    model = XGBClassifier(n_estimators = 50, random_state=random_state, objective = "multi:softprob")
    model.fit(X_train, y_train)
    return model

def XGB_5FOLD(folds, *, random_state = 42):
    '''
        This function trains an XGBoost based decision tree classifier with atleast 2 trees per fold
        :param folds: Dictionary of folds structured as:
            fold: Fold No.
            train_idx: Indices of training samples
            test_idx: Indices of testing samples
            X_train: Features of training samples
            X_test: Features of testing samples
            y_train: Labels of training samples
            y_test: Labels of testing samples
            scaler: Standard Scaler object for standardization
        :param random_state: Random seed for reproducibility (DEFAULT: 42)
        :return:
            XGB_fold_model: Dictionary of XGB decision tree classifiers per fold
            ...
            model: Saved & Trained XGB Classifier model
            performance_metrics: Dictionary of performance metrics and their corresponding values for the trained XGB classifier
                accuracy: calculates accuracy on X_test
                precision: calculated precision on X_test
                recall: calculated recall on X_test
                f1: calculated f1 on X_test
        '''
    XGB_fold_model: List[Dict[str, Any]] = []
    for fold_idx, fold in enumerate(folds):
        X_train = fold['X_train']
        y_train = fold['y_train']
        X_test = fold['X_test']
        y_test = fold['y_test']

        n_classes = np.unique(y_train).size
        if n_classes < 2:
            raise ValueError(
                f"Fold {fold.get('fold', fold_idx)}: degenerate labels (n_classes={n_classes}). "
                f"Unique={np.unique(y_train)}"
            )

        if n_classes == 2:
            model = XGBClassifier(
                n_estimators=50,
                early_stopping_rounds=1,
                random_state=random_state,
                objective="binary:logistic",
                eval_metric="logloss",
            )
        else:
            model = XGBClassifier(
                n_estimators=50,
                early_stopping_rounds=1,
                random_state=random_state,
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
            )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted")
        }

        XGB_fold_model.append(
            {
                "fold": fold['fold'],
                "train_idx": fold['train_idx'],
                "test_idx": fold['test_idx'],
                "X_train": fold['X_train'],
                "X_test": fold['X_test'],
                "y_train": fold['y_train'],
                "y_test": fold['y_test'],
                "scaler": fold['scaler'],
                "model": model,
                "performance_metrics": metrics
            }
        )
    return XGB_fold_model

class CBR:
    '''
    This class prepares CBR methodology with
    :param X_train: Array of training data
    :param y_train: Array of training labels
    :param random_state: seed for random state for reproducibility (DEFAULT: 42)
    :return:
        model: CBR model
    '''

    def __init__(self, categorical_idx=None, k=3, importance_type="gain", random_state=42, xgb_params=None):
        self.categorical_idx = categorical_idx
        self.k = k
        self.random_state = random_state
        self.xgb_params = xgb_params or {}

        self.X_train_ = None
        self.y_train_ = None
        self.weights_ = None
        self.importance_type = importance_type

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D.")
        if y_train.ndim != 1:
            raise ValueError("y_train must be 1D.")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have same number of rows.")

        if self.categorical_idx is None:
            self.categorical_idx = set(self.infer_categorical_features(X_train))
        else:
            self.categorical_idx = set(self.categorical_idx)

        self.X_train_ = X_train
        self.y_train_ = y_train
        self.classes_ = np.unique(y_train)

        n_features = X_train.shape[1]
        self.num_range_ = np.ones(n_features, dtype=float)

        for j in range(n_features):
            if j in self.categorical_idx:
                continue
            col = X_train[:, j].astype(float)
            r = float(np.max(col) - np.min(col))
            self.num_range_[j] = r if r>0 else 1

        X_train_enc, _, _ = self.encode_categoricals_for_xgb(
            X_train, X_train, self.categorical_idx
        )

        self.weights_ = self.learn_feature_weights(X_train_enc, y_train)
        return self

    def infer_categorical_features(self, X: np.ndarray, *, max_unique: int = 10, max_unique_ratio: float = 0.05):
        """
        A feature is categorical if its column contains strings. Otherwise, it is treated as numerical.

        Returns: categorical_idx : List[int]
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        categorical_idx = []

        for j in range(n_features):
            col = X[:, j]
            for v in col:
                if v is None:
                    continue
                if isinstance(v, str):
                    categorical_idx.append(j)
                    break

        return categorical_idx

    def encode_categoricals_for_xgb(self, X_train, X_test, categorical_idx):
        cat_cols = list(categorical_idx)

        if len(cat_cols) == 0:
            return (
                np.asarray(X_train, dtype=np.float32),
                np.asarray(X_test, dtype=np.float32),
                None
            )
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr = X_train.copy()
        Xte = X_test.copy()

        cat_cols = list(categorical_idx)
        Xtr[:, cat_cols] = enc.fit_transform(X_train[:, cat_cols])
        Xte[:, cat_cols] = enc.transform(X_test[:, cat_cols])

        return Xtr.astype(np.float32), Xte.astype(np.float32), enc

    def learn_feature_weights(self, X_train, y_train):
        params = dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=-1
        )

        # params.update(self.xgb_params)
        model = XGBClassifier(**params)
        model.fit(X_train,y_train)

        booster = model.get_booster()
        score = booster.get_score(importance_type=self.importance_type)

        n_features = X_train.shape[1]
        w = np.zeros(n_features, dtype=float)
        for j in range(n_features):
            w[j] = float(score.get(f"f{j}", 0.0))

        if np.allclose(w, 0.0):
            w[:] = 1.0

        w = w/np.sum(w)
        return w

    def similarities_to_train(self,x):
        X_train = self.X_train_
        w = self.weights_
        denom = float(np.sum(w)) if w is not None else 1

        sims = np.zeros(X_train.shape[0], dtype=float)

        for j in range(X_train.shape[1]):
            if w[j] == 0:
                continue

            if j in self.categorical_idx:
                sim_j = (X_train[:, j] == x[j]).astype(float)
            else:
                diff = np.abs(X_train[:, j].astype(float) - float(x[j]))
                sim_j  = 1 - (diff/self.num_range_[j])
                sim_j = np.clip(sim_j, 0, 1)

            sims += w[j]*sim_j

        return sims/denom if denom>0 else sims


    def predict(self, X_test):
        X_test = np.asarray(X_test)
        if X_test.ndim == 1:
            X_test = X_test[None, :]

        preds = []
        for x in X_test:
            sims = self.similarities_to_train(x)
            top_idx = np.argsort(-sims)[:self.k]
            top_labels = self.y_train_[top_idx]
            pred_label = mode(top_labels, keepdims=False).mode
            preds.append(pred_label)
        return np.asarray(preds)

    def predict_one(self, x):
        return self.predict(np.asarray(x))[0]

    def predict_proba(self, X, classes=None, round_categoricals=True):
        X = np.asarray(X)

        if X.ndim == 1:
            X = X[None, :]

        if classes is None:
            classes = self.classes_
        else:
            classes = np.asarray(classes)

        class_to_i = {c: i for i, c in enumerate(classes)}
        out = np.zeros((X.shape[0], len(classes)), dtype=float)

        cat_cols = sorted(list(self.categorical_idx)) if hasattr(self, "categorical_idx") else []

        for i in range(X.shape[0]):
            x = X[i]

            if round_categoricals and len(cat_cols) > 0:
                x = np.asarray(x).copy()
                x[cat_cols] = np.rint(x[cat_cols]).astype(np.int32)

            sims = self.similarities_to_train(x)
            top_idx = np.argsort(-sims)[: self.k]
            top_labels = self.y_train_[top_idx]

            for lab in top_labels:
                out[i, class_to_i[lab]] += 1.0

            out[i, :] /= float(self.k)

        return out

def CBR_5FOLD(folds, *, k: int =3, importance_type: str = "gain", random_state: int = 42) -> List[Dict[str, Any]]:
    CBR_fold_model: List[Dict[str, Any]] = []
    for fold_idx, fold in enumerate(folds):
        X_train = fold["X_train"]
        y_train = fold["y_train"]
        X_test = fold["X_test"]
        y_test = fold["y_test"]

        # Important: keep dtype=object so strings remain strings for similarity matching
        X_train = np.asarray(X_train, dtype=object)
        X_test = np.asarray(X_test, dtype=object)

        model = CBR(
            categorical_idx=None,  # auto-detect string columns
            k=k,
            importance_type=importance_type,
            random_state=random_state
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }

        CBR_fold_model.append(
            {
                "fold": fold["fold"],
                "train_idx": fold["train_idx"],
                "test_idx": fold["test_idx"],
                "X_train": fold["X_train"],  # keep original
                "X_test": fold["X_test"],
                "y_train": fold["y_train"],
                "y_test": fold["y_test"],
                "scaler": fold.get("scaler", None),  # will be None if you used standardize=False
                "model": model,
                "feature_weights": model.weights_,
                "performance_metrics": metrics
            }
        )

    return CBR_fold_model

class TabularProtoPNet(nn.Module):
    def __init__(self, input_dim, n_classes, n_prototypes_per_class=3):
        super().__init__()
        self.n_classes = n_classes
        self.n_prototypes_per_class = n_prototypes_per_class
        self.n_prototypes = n_classes * n_prototypes_per_class

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

        proto_class = []
        for c in range(n_classes):
            proto_class += [c] * n_prototypes_per_class
        self.register_buffer("prototypes_class_identity", torch.tensor(proto_class, dtype=torch.long))

        self.prototypes = nn.Parameter(
            torch.randn(self.n_prototypes, input_dim)
        )

        self.tau = nn.Parameter(torch.tensor(1.0))

    def forward(self, X):
        z = self.encoder(X) # Encode input
        dists = torch.cdist(z, self.prototypes, p=2) ** 2 # Distance to prototypes (latent space)
        tau = self.tau.abs() + 1e-6
        sim = torch.exp(-dists/tau) # Similarity (higher = closer)

        # logits built from class-wise max similarity (nearest prototype of each class)
        logits = []
        for c in range(self.n_classes):
            mask = (self.prototypes_class_identity == c).to(sim.device)  # [P]
            logits.append(sim[:, mask].max(dim=1).values)  # [B]
        logits = torch.stack(logits, dim=1)  # [B, C]
        return logits, sim, dists

    @torch.no_grad()
    def predict(self, X, *, batch_size: int = 4096, device: str | None = None) -> np.ndarray:
        """
        sklearn-like: returns hard class labels (argmax over logits).
        Accepts numpy arrays or torch tensors.
        """
        self.eval()

        if device is None:
            device = next(self.parameters()).device.type
        dev = torch.device(device)

        # Convert to tensor
        if isinstance(X, np.ndarray):
            X_t = torch.from_numpy(X.astype(np.float32, copy=False))
        else:
            X_t = X.detach() if torch.is_tensor(X) else torch.tensor(X, dtype=torch.float32)

        preds = []
        for start in range(0, X_t.shape[0], batch_size):
            xb = X_t[start:start + batch_size].to(dev, non_blocking=True)
            out = self(xb)  # (logits, sim, dists)
            logits = out[0]
            yb = torch.argmax(logits, dim=1)
            preds.append(yb.cpu())

        return torch.cat(preds, dim=0).numpy().astype(int)

    @torch.no_grad()
    def predict_proba(self, X, *, batch_size: int = 4096, device: str | None = None) -> np.ndarray:
        """
        sklearn-like: returns probabilities via softmax(logits).
        """
        self.eval()

        if device is None:
            device = next(self.parameters()).device.type
        dev = torch.device(device)

        if isinstance(X, np.ndarray):
            X_t = torch.from_numpy(X.astype(np.float32, copy=False))
        else:
            X_t = X.detach() if torch.is_tensor(X) else torch.tensor(X, dtype=torch.float32)

        probs = []
        for start in range(0, X_t.shape[0], batch_size):
            xb = X_t[start:start + batch_size].to(dev, non_blocking=True)
            logits, _, _ = self(xb)
            pb = torch.softmax(logits, dim=1)
            probs.append(pb.cpu())

        return torch.cat(probs, dim=0).numpy()


def PROTOPNET_5FOLD(folds, *, epochs=100, lr=0.001, n_prototypes_per_class=3, device=None, average="macro", batch_size = 32, use_amp=True, num_workers=0,weight_decay=1e-4, project_every=5, tau=1.0):
    torch.set_float32_matmul_precision("high")  # PyTorch 2.x
    torch.backends.cudnn.benchmark = True
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(use_amp and device.startswith("cuda"))

    print(f"[ProtoPNet] device={device}, amp={use_amp}, batch_size={batch_size}")

    proto_models = []
    for fold_id, fold in enumerate(folds):

        # -------------------------
        # Load fold data
        # -------------------------

        X_train = torch.tensor(fold["X_train"], dtype=torch.float32)
        y_train = torch.tensor(fold["y_train"], dtype=torch.long)

        X_test = torch.tensor(fold["X_test"], dtype=torch.float32).to(device)
        y_test = torch.tensor(fold["y_test"], dtype=torch.long).to(device)

        # DataLoader for minibatches
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_ds,
            batch_size=min(batch_size, len(train_ds)),
            shuffle=True,
            pin_memory=device.startswith("cuda"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )


        # -------------------------
        # Create model
        # -------------------------
        n_classes = int(torch.unique(y_train).numel())
        model = TabularProtoPNet(
            input_dim=X_train.shape[1],
            n_classes=n_classes,
            n_prototypes_per_class=n_prototypes_per_class,
        ).to(device)

        model.prototypes_class_identity = model.prototypes_class_identity.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

        # -------------------------
        # Train
        # -------------------------
        model.train()
        for epoch in range(1, epochs+1):
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits, _, _ = model(xb)
                    loss = criterion(logits, yb)

                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()

            # if project_every and (epoch%project_every == 0):
            #     project_prototypes_to_train(model, X_train, device, use_amp)
            #     model.train()

        # -------------------------
        # Evaluate
        # -------------------------
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, sim, dists = model(X_test)
            y_pred_logits = torch.argmax(logits, dim=1)

            # nearest_proto = dists.argmin(dim=1)
            # y_pred_nearest = model.prototypes_class_identity[nearest_proto]
            z_test = model.encoder(X_test)

        y_true_np = y_test.detach().cpu().numpy()
        y_pred_logits_np = y_pred_logits.detach().cpu().numpy()
        # y_pred_nearest_np = y_pred_nearest.detach().cpu().numpy()

        acc_logits = accuracy_score(y_true_np, y_pred_logits_np)
        prec_l, rec_l, f1_l, _ = precision_recall_fscore_support(
            y_true_np, y_pred_logits_np, average=average, zero_division=0
        )

        # acc_nearest = accuracy_score(y_true_np, y_pred_nearest_np)
        # prec_n, rec_n, f1_n, _ = precision_recall_fscore_support(
        #     y_true_np, y_pred_nearest_np, average=average, zero_division=0
        # )

        # -------------------------
        # Store fold results
        # -------------------------
        proto_models.append({
            "fold": fold_id,
            "train_idx": fold['train_idx'],
            "test_idx": fold['test_idx'],
            "X_train": fold['X_train'],
            "X_test": fold['X_test'],
            "y_train": fold['y_train'],
            "y_test": fold['y_test'],
            "scaler": fold.get("scaler", None),
            "model": model,
            "performance_metrics":{
                "accuracy_logits": float(acc_logits),
                "precision_logits": float(prec_l),
                "recall_logits": float(rec_l),
                "f1_logits": float(f1_l),

                # "accuracy_nearest_proto": float(acc_nearest),
                # "precision_nearest_proto": float(prec_n),
                # "recall_nearest_proto": float(rec_n),
                # "f1_nearest_proto": float(f1_n),
            },
            "proto_outputs": {
                "similarity": sim.detach().cpu().numpy(), # How similar each test instance is to each prototype
                "distances": dists.detach().cpu().numpy(), # How far each test instance is to each prototype
                "prototypes": model.prototypes.detach().cpu().numpy(), # The learned prototype vectors
                "prototypes_class_identity": model.prototypes_class_identity.detach().cpu().numpy(), # Which class prototype is intended to represent
                "test_instances_encoded": z_test.detach().cpu().numpy()
            },
        })

    return proto_models

def MLP_5FOLD(folds, *, hidden_units=64, activation="relu", solver="adam", alpha=0.0001, random_state=42):
    """
        Train and evaluate an MLP Classifier with ONE hidden layer using 5-fold CV.

        Parameters
        ----------
        dataset : str
            Dataset name (for bookkeeping/logging consistency)
        folds : dict
            Dictionary of folds structured as:
                fold_id:
                    train_idx
                    test_idx
                    X_train
                    X_test
                    y_train
                    y_test
        hidden_units : int
            Number of neurons in the single hidden layer
        activation : str
            Activation function ('relu', 'tanh', 'logistic')
        solver : str
            Optimization solver ('adam', 'sgd', 'lbfgs')
        alpha : float
            L2 regularization parameter
        max_iter : int
            Maximum training iterations
        random_state : int
            Random seed

        Returns
        -------
        dict
            Dictionary containing fold-wise models, predictions, and metrics
        """

    fold_results = []

    for fold_id, fold in enumerate(folds):
        X_train = fold["X_train"]
        X_test = fold["X_test"]
        y_train = fold["y_train"]
        y_test = fold["y_test"]

        # -----------------------------
        # MLP with ONE hidden layer
        # -----------------------------
        model = MLPClassifier(
            hidden_layer_sizes=(hidden_units,),
            activation=activation,
            solver="adam",
            alpha=alpha,
            learning_rate_init=1e-3,
            early_stopping=True,  # dynamic stop
            validation_fraction=0.1,  # hold-out from training fold
            n_iter_no_change=30,  # patience
            tol=1e-4,  # improvement threshold
            max_iter=5000,  # just a safety ceiling
            random_state=random_state
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # -----------------------------
        # Performance Metrics
        # -----------------------------
        performance_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        }

        # -----------------------------
        # Save fold results
        # -----------------------------
        fold_results.append({
            "fold": fold_id,
            "train_idx": fold['train_idx'],
            "test_idx": fold['test_idx'],
            "X_train": fold['X_train'],
            "X_test": fold['X_test'],
            "y_train": fold['y_train'],
            "y_test": fold['y_test'],
            "scaler": fold.get("scaler", None),
            "model": model,
            "y_pred": y_pred,
            "y_pred_proba": y_proba,
            "model_params": {
                "hidden_units": hidden_units,
                "activation": activation,
                "solver": solver,
                "alpha": alpha
            },
            "performance_metrics": performance_metrics
        })

    return fold_results

def DNN_8HL_5fold(folds,
    *,
    hidden_layers=(512, 512, 256, 256, 128, 128, 64, 64),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    max_iter=500,
    random_state=42
):
    """
        Train and evaluate a Deep Neural Network (8 hidden layers) using 5-fold CV.

        Hidden architecture:
            512 → 512 → 256 → 256 → 128 → 128 → 64 → 64

        Parameters
        ----------
        dataset : str
            Dataset name (for bookkeeping/logging consistency)
        folds : dict
            Dictionary of folds structured as:
                fold_id:
                    train_idx
                    test_idx
                    X_train
                    X_test
                    y_train
                    y_test
        hidden_layers : tuple
            Sizes of hidden layers
        activation : str
            Activation function ('relu', 'tanh', 'logistic')
        solver : str
            Optimization solver ('adam', 'sgd', 'lbfgs')
        alpha : float
            L2 regularization parameter
        max_iter : int
            Maximum training iterations
        random_state : int
            Random seed

        Returns
        -------
        dict
            Dictionary containing fold-wise models, predictions, and metrics
        """
    fold_results = []
    for fold_id, fold in enumerate(folds):
        X_train = fold["X_train"]
        X_test = fold["X_test"]
        y_train = fold["y_train"]
        y_test = fold["y_test"]

        # -----------------------------
        # Deep Neural Network
        # -----------------------------
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state
        )

        model.fit(X_train, y_train)

        # -----------------------------
        # Predictions
        # -----------------------------
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # -----------------------------
        # Performance Metrics
        # -----------------------------
        performance_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        }

        # -----------------------------
        # Save fold results
        # -----------------------------
        fold_results.append({
            "fold": fold_id,
            "train_idx": fold['train_idx'],
            "test_idx": fold['test_idx'],
            "X_train": fold['X_train'],
            "X_test": fold['X_test'],
            "y_train": fold['y_train'],
            "y_test": fold['y_test'],
            "scaler": fold.get("scaler", None),
            "model": model,
            "y_pred": y_pred,
            "y_pred_proba": y_proba,
            "performance_metrics": performance_metrics
        })

    return fold_results

if __name__ == "__main__":
    '''
    TRAINS A SAMPLE 5 FOLD DT ON IRIS DATASET
    '''
    X, y, feature_names = load_dataset('iris')
    folds_stand = stratified_5fold_standardize(X, y, standardize=True)
    DT_fold_model = CART_DT_5FOLD(folds_stand, random_state = 42)

    '''
        TRAINS A SAMPLE 5 FOLD XGB ON IRIS DATASET
    '''
    XGB_fold_model = CART_DT_5FOLD(folds_stand, random_state=42)
