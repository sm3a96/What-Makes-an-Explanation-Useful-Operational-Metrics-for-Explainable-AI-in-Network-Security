# 4.2 LIME (Local Interpretable Model-agnostic Explanations)
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











# # I uawd this experment for the Expolanability power 
# import numpy as np
# import pandas as pd
# from lime import lime_tabular

# def explain_with_lime(
#     model,
#     X_train,
#     X_test=None,
#     mode='classification',
#     num_features=None,
#     instance_idx=0,
#     class_names=None,
#     discretize_continuous=True,
#     return_instance_exp=True,
#     num_samples=5000,
#     kernel_width=None,
#     random_state=42
# ):
#     # Normalize inputs
#     if isinstance(X_train, pd.DataFrame):
#         feature_names = X_train.columns.tolist()
#         train_data = X_train.values
#     else:
#         feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
#         train_data = np.asarray(X_train)

#     if mode == 'classification' and class_names is None and hasattr(model, 'classes_'):
#         class_names = [str(c) for c in model.classes_]

#     explainer = lime_tabular.LimeTabularExplainer(
#         training_data=train_data,
#         feature_names=feature_names,
#         class_names=class_names,
#         mode=mode,
#         discretize_continuous=discretize_continuous,
#         kernel_width=kernel_width,
#         random_state=random_state
#     )

#     if not return_instance_exp or X_test is None:
#         return explainer, None

#     if isinstance(X_test, pd.DataFrame):
#         data_row = X_test.iloc[instance_idx].values
#     else:
#         data_row = np.asarray(X_test)[instance_idx]

#     if num_features is None:
#         num_features = len(feature_names)

#     predict_fn = model.predict_proba if mode == 'classification' else model.predict

#     exp = explainer.explain_instance(
#         data_row=data_row.astype(np.double),
#         predict_fn=lambda x: predict_fn(pd.DataFrame(x, columns=feature_names)),
#         num_features=num_features,
#         num_samples=num_samples
#     )
#     return explainer, exp






















# from lime import lime_tabular
# import numpy as np
# import pandas as pd

# def explain_with_lime(
#     model,
#     X_train,
#     X_test=None,
#     mode='classification',
#     num_features=None,
#     instance_idx=0,
#     class_names=None,
#     discretize_continuous=True,
#     return_instance_exp=True,
#     num_samples=5000,  # CRITICAL: Added for better approximation
#     kernel_width=None,  # For local neighborhood control
#     random_state=42     # For reproducibility
# ):
#     """
#     Create a LIME explainer and (optionally) explain a specific instance.

#     Args:
#         model: Trained ML model.
#         X_train: Training data (DataFrame or ndarray).
#         X_test: Test data (DataFrame or ndarray), optional.
#         mode: 'classification' or 'regression'.
#         num_features: Number of features to explain (default: all).
#         instance_idx: Index of the instance in X_test to explain.
#         class_names: List of class names (for classification).
#         discretize_continuous: Whether to discretize continuous features.
#         return_instance_exp: If True, also return explanation for one instance.
#         num_samples: Number of samples to generate for local model (default: 5000).
#                     CRITICAL: Higher values = better approximation but slower.
#                     Default LIME uses only 5000, but for complex models use more.
#         kernel_width: Width of the kernel for weighting samples (default: auto).
#                      Controls the "locality" of the explanation.
#         random_state: Random seed for reproducibility.

#     Returns:
#         explainer [, explanation]
#     """
#     # Ensure DataFrame for feature names
#     if isinstance(X_train, np.ndarray):
#         feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
#         X_train_df = pd.DataFrame(X_train, columns=feature_names)
#     else:
#         feature_names = X_train.columns.tolist()
#         X_train_df = X_train.copy()
#         X_train = X_train.values

#     # Auto-detect class names if not provided
#     if mode == 'classification' and class_names is None and hasattr(model, 'classes_'):
#         class_names = [str(c) for c in model.classes_]

#     # Calculate kernel width if not provided (use sqrt of feature count)
#     if kernel_width is None:
#         kernel_width = np.sqrt(len(feature_names)) * 0.75

#     explainer = lime_tabular.LimeTabularExplainer(
#         training_data=X_train,
#         feature_names=feature_names,
#         class_names=class_names,
#         mode=mode,
#         discretize_continuous=discretize_continuous,
#         kernel_width=kernel_width,
#         random_state=random_state
#     )

#     if return_instance_exp and X_test is not None:
#         # Prepare instance to explain
#         if isinstance(X_test, pd.DataFrame):
#             data_row = X_test.iloc[instance_idx].values
#         else:
#             data_row = X_test[instance_idx]
        
#         if num_features is None:
#             num_features = len(feature_names)
        
#         # CRITICAL FIX: Wrap predict function to handle DataFrame conversion
#         if mode == 'classification':
#             def predict_fn(x):
#                 """Wrapper to ensure DataFrame input to model"""
#                 if not isinstance(x, pd.DataFrame):
#                     x = pd.DataFrame(x, columns=feature_names)
#                 return model.predict_proba(x)
#         else:
#             def predict_fn(x):
#                 """Wrapper to ensure DataFrame input to model"""
#                 if not isinstance(x, pd.DataFrame):
#                     x = pd.DataFrame(x, columns=feature_names)
#                 return model.predict(x)
        
#         # CRITICAL: Add num_samples parameter
#         exp = explainer.explain_instance(
#             data_row=data_row,
#             predict_fn=predict_fn,
#             num_features=num_features,
#             num_samples=num_samples  # CRITICAL: This was missing!
#         )
        
#         return explainer, exp

#     return explainer




# from lime import lime_tabular
# import numpy as np
# import pandas as pd

# def explain_with_lime(
#     model,
#     X_train,
#     X_test=None,
#     mode='classification',
#     num_features=None,
#     instance_idx=0,
#     class_names=None,
#     discretize_continuous=True,
#     return_instance_exp=True
# ):
#     """
#     Create a LIME explainer and (optionally) explain a specific instance.

#     Args:
#         model: Trained ML model.
#         X_train: Training data (DataFrame or ndarray).
#         X_test: Test data (DataFrame or ndarray), optional.
#         mode: 'classification' or 'regression'.
#         num_features: Number of features to explain (default: all).
#         instance_idx: Index of the instance in X_test to explain.
#         class_names: List of class names (for classification).
#         discretize_continuous: Whether to discretize continuous features.
#         return_instance_exp: If True, also return explanation for one instance.

#     Returns:
#         explainer [, explanation]
#     """
#     # Ensure DataFrame for feature names
#     if isinstance(X_train, np.ndarray):
#         feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
#     else:
#         feature_names = X_train.columns.tolist()
#         X_train = X_train.values

#     # Auto-detect class names if not provided
#     if mode == 'classification' and class_names is None and hasattr(model, 'classes_'):
#         class_names = [str(c) for c in model.classes_]

#     explainer = lime_tabular.LimeTabularExplainer(
#         training_data=X_train,
#         feature_names=feature_names,
#         class_names=class_names,
#         mode=mode,
#         discretize_continuous=discretize_continuous
#     )

#     if return_instance_exp and X_test is not None:
#         # Prepare instance to explain
#         if isinstance(X_test, pd.DataFrame):
#             data_row = X_test.iloc[instance_idx].values
#         else:
#             data_row = X_test[instance_idx]
#         if num_features is None:
#             num_features = len(feature_names)
#         exp = explainer.explain_instance(
#             data_row=data_row,
#             predict_fn=model.predict_proba if mode == 'classification' else model.predict,
#             num_features=num_features
#         )
#         return explainer, exp

#     return explainer













# from lime import lime_tabular

# def explain_with_lime(model, X_train, X_test, mode='classification', num_features=10):
#     """
#     Generate LIME explanations.

#     Args:
#         model: Trained ML model.
#         X_train: Training data (pandas DataFrame).
#         X_test: Test data (pandas DataFrame).
#         mode: 'classification' or 'regression'.
#         num_features: Number of features to explain.

#     Returns:
#         explainer, explanation
#     """
#     explainer = lime_tabular.LimeTabularExplainer(
#         training_data=X_train.values,
#         feature_names=X_train.columns.tolist(),
#         class_names=None,
#         mode=mode,
#         discretize_continuous=True
#     )
#     exp = explainer.explain_instance(
#         data_row=X_test.iloc[0].values,
#         predict_fn=model.predict_proba if mode == 'classification' else model.predict,
#         num_features=num_features
#     )
#     return explainer, exp





# import lime
# import lime.lime_tabular
# import numpy as np

# def explain_model_lime(model, X_train, instance, mode="classification", feature_names=None, class_names=None):
#     """
#     LIME explanation for tabular models.
    
#     Parameters:
#     - model: Trained model with predict or predict_proba
#     - X_train: Training data used to fit the explainer
#     - instance: The row/sample you want to explain
#     - mode: "classification" or "regression"
#     - feature_names: List of column names (optional)
#     - class_names: List of class names (for multi-class)
#     """
#     explainer = lime.lime_tabular.LimeTabularExplainer(
#         training_data=np.array(X_train),
#         mode=mode,
#         feature_names=feature_names,
#         class_names=class_names,
#         discretize_continuous=True
#     )

#     explanation = explainer.explain_instance(
#         data_row=instance,
#         predict_fn=model.predict_proba if mode == "classification" else model.predict
#     )

#     return explanation




# # How to Use It
# # Example 1: Binary classification with feature names

# # feature_names = list(X_train.columns)
# # explanation = explain_model_lime(model, X_train, X_test[5], mode="classification", feature_names=feature_names)
# # ------------------------------------------------------------

# # Example 2: Multi-class classification
# # class_names = ['Class A', 'Class B', 'Class C']
# # feature_names = list(X_train.columns)
# # explanation = explain_model_lime(model, X_train, X_test[0], mode="classification", feature_names=feature_names, class_names=class_names)

# # ----------------------
# # Example 3: No names available
# # explanation = explain_model_lime(model, X_train.values, X_test.values[0])


# # Old Code
# # import lime
# # import lime.lime_tabular
# # import numpy as np

# # def explain_model_lime(model, X_train, instance, mode="classification"):
# #     """
# #     LIME explanation for tabular models.
# #     - `mode`: "classification" or "regression"
# #     - `instance`: single sample to explain (reshaped as 2D)
# #     """
# #     explainer = lime.lime_tabular.LimeTabularExplainer(
# #         training_data=np.array(X_train),
# #         mode=mode,
# #         feature_names=None,  # Add if available
# #         class_names=None,    # Add if available
# #         discretize_continuous=True
# #     )

# #     explanation = explainer.explain_instance(
# #         data_row=instance,
# #         predict_fn=model.predict_proba if mode == "classification" else model.predict
# #     )
# #     return explanation
