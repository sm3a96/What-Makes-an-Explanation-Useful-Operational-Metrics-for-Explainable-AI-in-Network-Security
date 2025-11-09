# 4.1 SHAP (SHapley Additive exPlanations)
# -----------------------------------------
# What: SHAP uses Shapley values from cooperative game theory to assign each feature an importance value
#       for a particular prediction. It provides both global and local explanations.
# How: We can use TreeExplainer for tree-based models (e.g., Random Forest) for fast computation.
#       SHAP values show the contribution of each feature to the difference between the actual
#       prediction and the average prediction. shap.initjs() enables interactive plots in notebooks.
# Why: Model-agnostic via KernelExplainer, but also optimized for specific models (Tree, Deep, etc.).
#       Guarantees consistency (features contributing more receive larger values), making explanations reliable.
#

# shap_explainer.py
# -------------------------------------------------
# SHAP works for tree, linear, and deep learning models.
# This version handles model types automatically.


# and i am using this for the Acctionability Experiments

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




# # I used this for Exp Power 
# import pandas as pd
# import shap

# def explain_with_shap(model, X_train, X_test, model_type=None, num_samples=100):
#     # Detect model type
#     if model_type is None:
#         name = model.__class__.__name__.lower()
#         if any(k in name for k in ["tree", "forest", "boost", "gbm", "xgb", "catboost", "lgbm"]):
#             model_type = "tree"
#         elif any(k in name for k in ["logistic", "regression", "linear"]):
#             model_type = "linear"
#         else:
#             model_type = "kernel"

#     # Ensure DataFrame with columns
#     if not isinstance(X_train, pd.DataFrame):
#         X_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
#     if not isinstance(X_test, pd.DataFrame):
#         X_test = pd.DataFrame(X_test, columns=X_train.columns)

#     background = X_train.sample(min(100, len(X_train)), random_state=0)

#     explainer = None
#     try:
#         if model_type == "tree":
#             explainer = shap.TreeExplainer(model)
#         elif model_type == "linear":
#             explainer = shap.LinearExplainer(model, X_train)
#         elif model_type == "kernel":
#             def predict_fn(X):
#                 if not isinstance(X, pd.DataFrame):
#                     X = pd.DataFrame(X, columns=background.columns)
#                 X = X[background.columns]
#                 return model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)
#             explainer = shap.KernelExplainer(predict_fn, background)
#         else:
#             raise ValueError(f"Unsupported model_type: {model_type}")
#     except Exception as e:
#         def fallback_predict_fn(X):
#             if not isinstance(X, pd.DataFrame):
#                 X = pd.DataFrame(X, columns=background.columns)
#             X = X[background.columns]
#             return model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)
#         print(f"Explainer init failed ({e}). Falling back to KernelExplainer.")
#         explainer = shap.KernelExplainer(fallback_predict_fn, background)

#     X_to_explain = X_test.iloc[:num_samples]
#     shap_values = explainer.shap_values(X_to_explain)
#     return shap_values, explainer































# def explain_with_shap(model, X_train, X_test, model_type=None, num_samples=100):
#     """
#     Generate SHAP explanations for a given model, with robust model type detection and fallback.

#     Args:
#         model: Trained machine learning model.
#         X_train: Training data.
#         X_test: Test data.
#         model_type: Optional ('tree', 'linear', 'deep', 'kernel'). If None, it tries to guess.
#         num_samples: Number of samples to explain.

#     Returns:
#         shap_values, explainer
#     """
#     import shap
#     import numpy as np
    
#     # Improved model type detection
#     if model_type is None:
#         model_name = model.__class__.__name__.lower()
#         if any(x in model_name for x in ['tree', 'forest', 'boost', 'gbm', 'xgb', 'catboost', 'lgbm']):
#             model_type = 'tree'
#         elif any(x in model_name for x in ['regression', 'logistic', 'linear']):
#             model_type = 'linear'
#         elif any(x in model_name for x in ['mlp', 'dnn', 'deep', 'sequential']):
#             model_type = 'deep'
#         else:
#             model_type = 'kernel'

#     # Prepare background data for KernelExplainer outside of try block
#     import pandas as pd
#     # Make sure X_train has columns to avoid feature warning
#     if not isinstance(X_train, pd.DataFrame) and hasattr(X_train, 'shape'):
#         # Try to infer column names if not provided
#         columns = [f"feature_{i}" for i in range(X_train.shape[1])] if hasattr(X_train, 'shape') else None
#         X_train_df = pd.DataFrame(X_train, columns=columns)
#     else:
#         X_train_df = X_train
        
#     # Sample background data for KernelExplainer
#     background = X_train_df.sample(min(100, len(X_train_df))) if hasattr(X_train_df, 'sample') and len(X_train_df) > 100 else X_train_df
    
#     # For test data preparation
#     if hasattr(X_test, 'iloc'):
#         X_to_explain = X_test.iloc[:num_samples]
#     else:
#         X_to_explain = X_test[:num_samples]
        
#     if not isinstance(X_to_explain, pd.DataFrame):
#         X_to_explain = pd.DataFrame(X_to_explain, columns=X_train_df.columns)
    
#     # Try TreeExplainer first for decision trees (fast)
#     if model_type == 'tree':
#         try:
#             tree_explainer = shap.TreeExplainer(model)
#             tree_values = tree_explainer.shap_values(X_to_explain)
            
#             # Check if values are all zeros (common issue with some datasets)
#             all_zeros = False
#             if isinstance(tree_values, list):
#                 all_zeros = all(np.all(values == 0) for values in tree_values)
#             else:
#                 all_zeros = np.all(tree_values == 0)
                
#             # If not all zeros, return these values
#             if not all_zeros:
#                 print("Using TreeExplainer")
#                 return tree_values, tree_explainer
#             else:
#                 print("TreeExplainer produced all zeros. Falling back to KernelExplainer.")
#                 # Fall through to KernelExplainer
#         except Exception as e:
#             print(f"TreeExplainer failed: {e}. Trying KernelExplainer.")
#             # Fall through to KernelExplainer
    
#     # Try LinearExplainer for linear models
#     elif model_type == 'linear':
#         try:
#             linear_explainer = shap.LinearExplainer(model, X_train_df)
#             linear_values = linear_explainer.shap_values(X_to_explain)
            
#             # Check if values are all zeros
#             all_zeros = False
#             if isinstance(linear_values, list):
#                 all_zeros = all(np.all(values == 0) for values in linear_values)
#             else:
#                 all_zeros = np.all(linear_values == 0)
                
#             if not all_zeros:
#                 print("Using LinearExplainer")
#                 return linear_values, linear_explainer
#             else:
#                 print("LinearExplainer produced all zeros. Falling back to KernelExplainer.")
#         except Exception as e:
#             print(f"LinearExplainer failed: {e}. Trying KernelExplainer.")
    
#     # Try DeepExplainer for neural networks
#     elif model_type == 'deep':
#         try:
#             deep_explainer = shap.DeepExplainer(model, X_train_df)
#             deep_values = deep_explainer.shap_values(X_to_explain)
            
#             # Check if values are all zeros
#             all_zeros = False
#             if isinstance(deep_values, list):
#                 all_zeros = all(np.all(values == 0) for values in deep_values)
#             else:
#                 all_zeros = np.all(deep_values == 0)
                
#             if not all_zeros:
#                 print("Using DeepExplainer")
#                 return deep_values, deep_explainer
#             else:
#                 print("DeepExplainer produced all zeros. Falling back to KernelExplainer.")
#         except Exception as e:
#             print(f"DeepExplainer failed: {e}. Trying KernelExplainer.")
    
#     # KernelExplainer as fallback (works for any model)
#     print("Using KernelExplainer")
    
#     # Create prediction function for KernelExplainer
#     def predict_fn(X):
#         if not isinstance(X, pd.DataFrame):
#             X = pd.DataFrame(X, columns=background.columns)
#         # Ensure X has the correct columns in the right order
#         X = X[background.columns]
#         if hasattr(model, "predict_proba"):
#             return model.predict_proba(X)
#         else:
#             return model.predict(X)
    
#     # Create KernelExplainer with carefully prepared background data
#     kernel_explainer = shap.KernelExplainer(predict_fn, background)
    
#     # Generate SHAP values with KernelExplainer
#     kernel_values = kernel_explainer.shap_values(X_to_explain)
    
#     # Final check that we have non-zero values
#     if isinstance(kernel_values, list):
#         if all(np.all(values == 0) for values in kernel_values):
#             print("WARNING: KernelExplainer also produced all zeros!")
#     elif np.all(kernel_values == 0):
#         print("WARNING: KernelExplainer also produced all zeros!")
    
#     return kernel_values, kernel_explainer
















# def explain_with_shap(model, X_train, X_test, model_type=None, num_samples=100):
#     """
#     Generate SHAP explanations for a given model, with robust model type detection and fallback.

#     Args:
#         model: Trained machine learning model.
#         X_train: Training data.
#         X_test: Test data.
#         model_type: Optional ('tree', 'linear', 'deep', 'kernel'). If None, it tries to guess.
#         num_samples: Number of samples to explain.

#     Returns:
#         shap_values, explainer
#     """
#     import shap
#     # Improved model type detection
#     if model_type is None:
#         model_name = model.__class__.__name__.lower()
#         if any(x in model_name for x in ['tree', 'forest', 'boost', 'gbm', 'xgb', 'catboost', 'lgbm']):
#             model_type = 'tree'
#         elif any(x in model_name for x in ['regression', 'logistic', 'linear']):
#             model_type = 'linear'
#         elif any(x in model_name for x in ['mlp', 'dnn', 'deep', 'sequential']):
#             model_type = 'deep'
#         else:
#             model_type = 'kernel'  # fallback to model-agnostic

#     # Prepare background data for KernelExplainer outside of try block
#     import pandas as pd
#     # Make sure X_train has columns to avoid feature warning
#     if not isinstance(X_train, pd.DataFrame) and hasattr(X_train, 'shape'):
#         # Try to infer column names if not provided
#         columns = [f"feature_{i}" for i in range(X_train.shape[1])] if hasattr(X_train, 'shape') else None
#         X_train_df = pd.DataFrame(X_train, columns=columns)
#     else:
#         X_train_df = X_train
        
#     # Sample background data for KernelExplainer
#     background = X_train_df.sample(100) if hasattr(X_train_df, 'sample') and len(X_train_df) > 100 else X_train_df
    
#     # Try to create the appropriate explainer
#     explainer = None
#     try:
#         if model_type == 'tree':
#             explainer = shap.TreeExplainer(model)
#         elif model_type == 'linear':
#             explainer = shap.LinearExplainer(model, X_train_df)
#         elif model_type == 'deep':
#             explainer = shap.DeepExplainer(model, X_train_df)
#         elif model_type == 'kernel':
#             # KernelExplainer is model-agnostic but slow
#             # Ensure input to predict functions is a DataFrame with correct feature names
#             def predict_fn(X):
#                 if not isinstance(X, pd.DataFrame):
#                     X = pd.DataFrame(X, columns=background.columns)
#                 # Ensure X is always a DataFrame with correct columns
#                 X = X[background.columns]
#                 if hasattr(model, "predict_proba"):
#                     return model.predict_proba(X)
#                 else:
#                     return model.predict(X)
#             explainer = shap.KernelExplainer(predict_fn, background)
#         else:
#             raise ValueError(f"Unsupported model type for SHAP: {model_type}")
#     except Exception as e:
#         # Fallback to KernelExplainer if other explainers fail
#         def fallback_predict_fn(X):
#             if not isinstance(X, pd.DataFrame):
#                 X = pd.DataFrame(X, columns=background.columns)
#             X = X[background.columns]
#             if hasattr(model, "predict_proba"):
#                 return model.predict_proba(X)
#             else:
#                 return model.predict(X)
#         print(f"Original explainer failed: {e}. Falling back to KernelExplainer.")
#         explainer = shap.KernelExplainer(fallback_predict_fn, background)
    
#     # Verify explainer is created before proceeding
#     # Calculate SHAP values for the test data
#     try:
#         # Prepare data for explanation
#         if hasattr(X_test, 'iloc'):
#             X_to_explain = X_test.iloc[:num_samples]
#         else:
#             X_to_explain = X_test[:num_samples]
            
#         if not isinstance(X_to_explain, pd.DataFrame):
#             X_to_explain = pd.DataFrame(X_to_explain, columns=background.columns)
            
#         # Calculate SHAP values
#         shap_values = explainer.shap_values(X_to_explain)
#     except Exception as e:
#         try:
#             # Last attempt with X_train.columns
#             if hasattr(X_test, 'iloc'):
#                 X_to_explain = X_test.iloc[:num_samples]
#             else:
#                 X_to_explain = X_test[:num_samples]
                
#             if not isinstance(X_to_explain, pd.DataFrame):
#                 X_to_explain = pd.DataFrame(X_to_explain, columns=X_train.columns)
                
#             shap_values = explainer.shap_values(X_to_explain)
#         except Exception as e2:
#             raise RuntimeError(f"SHAP explanation failed: {e2}")

#     return shap_values, explainer


