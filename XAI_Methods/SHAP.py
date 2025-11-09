# SHAP (SHapley Additive exPlanations)
# -----------------------------------------
# What: SHAP uses Shapley values from cooperative game theory to assign each feature an importance value
#       for a particular prediction. It provides both global and local explanations.

# How: We can use TreeExplainer for tree-based models (e.g., Random Forest) for fast computation.
#       SHAP values show the contribution of each feature to the difference between the actual
#       prediction and the average prediction. shap.initjs() enables interactive plots in notebooks.

# Why: Model-agnostic via KernelExplainer, but also optimized for specific models (Tree, Deep, etc.).
#       Guarantees consistency (features contributing more receive larger values), making explanations reliable.


# shap_explainer.py
# -------------------------------------------------
# SHAP works for tree, linear, and deep learning models.
# This version handles model types automatically.




import pandas as pd
import shap
import numpy as np
import random

def explain_with_shap(model, X_train, X_test, model_type=None, num_samples=100, random_state=42):
    # Set deterministic seeds for SHAP internals relying on numpy/random
    np.random.seed(random_state)
    random.seed(random_state)

    # Detect model type
    if model_type is None:
        name = model.__class__.__name__.lower()
        if any(k in name for k in ["tree", "forest", "boost", "gbm", "xgb", "catboost", "lgbm"]):
            model_type = "tree"
        elif any(k in name for k in ["logistic", "regression", "linear"]):
            model_type = "linear"
        else:
            model_type = "kernel"

    # Ensure DataFrame with columns
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=X_train.columns)

    # Deterministic background sampling
    background = X_train.sample(min(100, len(X_train)), random_state=random_state)

    explainer = None
    try:
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, X_train)
        elif model_type == "kernel":
            def predict_fn(X):
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X, columns=background.columns)
                X = X[background.columns]
                return model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)
            explainer = shap.KernelExplainer(predict_fn, background)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    except Exception as e:
        def fallback_predict_fn(X):
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=background.columns)
            X = X[background.columns]
            return model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)
        print(f"Explainer init failed ({e}). Falling back to KernelExplainer.")
        explainer = shap.KernelExplainer(fallback_predict_fn, background)

    X_to_explain = X_test.iloc[:num_samples]
    # Seed again before SHAP computation to stabilize any internal randomness
    np.random.seed(random_state)
    shap_values = explainer.shap_values(X_to_explain)
    return shap_values, explainer





