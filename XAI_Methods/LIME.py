# LIME (Local Interpretable Model-agnostic Explanations)
# ---------------------------------------------------------
# What: LIME explains individual predictions by approximating the model locally with a simple interpretable model,
#       usually a linear model. It perturbs data around the instance to see how predictions change.
# How: LimeTabularExplainer is created with training data; explain_instance() fits a local linear model
#       around a specific data point by sampling (perturbing) it and querying the black-box model.
# Why: Model-agnostic and intuitive for single-instance explanations. Helps understand why the model made a
#       particular decision by showing feature weights in that local approximation.
#
# lime_explainer.py
# -------------------------------------------------
# LIME is model-agnostic. This works for classification or regression.



import numpy as np
import pandas as pd
from lime import lime_tabular
import random

def explain_with_lime(
    model,
    X_train,
    X_test=None,
    mode='classification',
    num_features=None,
    instance_idx=0,
    class_names=None,
    discretize_continuous=True,
    return_instance_exp=True,
    num_samples=5000,
    kernel_width=None,
    random_state=42
):
    # Global determinism for LIME sampling
    np.random.seed(random_state)
    random.seed(random_state)

    # Normalize inputs
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
        train_data = X_train.values
    else:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        train_data = np.asarray(X_train)

    if mode == 'classification' and class_names is None and hasattr(model, 'classes_'):
        class_names = [str(c) for c in model.classes_]

    # Deterministic kernel width if not provided
    if kernel_width is None:
        kernel_width = np.sqrt(len(feature_names)) * 0.75

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=train_data,
        feature_names=feature_names,
        class_names=class_names,
        mode=mode,
        discretize_continuous=discretize_continuous,
        kernel_width=kernel_width,
        random_state=random_state
    )

    if not return_instance_exp or X_test is None:
        return explainer, None

    if isinstance(X_test, pd.DataFrame):
        data_row = X_test.iloc[instance_idx].values
    else:
        data_row = np.asarray(X_test)[instance_idx]

    if num_features is None:
        num_features = len(feature_names)

    predict_fn = model.predict_proba if mode == 'classification' else model.predict

    exp = explainer.explain_instance(
        data_row=data_row.astype(np.double),
        predict_fn=lambda x: predict_fn(pd.DataFrame(x, columns=feature_names)),
        num_features=num_features,
        num_samples=num_samples
    )
    return explainer, exp



