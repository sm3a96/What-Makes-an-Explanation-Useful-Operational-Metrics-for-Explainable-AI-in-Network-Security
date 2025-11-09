import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# ---------------------------- Helper Functions ----------------------------
def evaluate_model(y_true, y_pred, model_name):
    """Calculate and return performance metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\n{model_name} Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
    }

def remove_minority_classes(df, min_samples=1000):
    """
    Remove classes with fewer than `min_samples` instances.
    
    Args:
        df (pd.DataFrame): Input dataframe with a 'Label' column.
        min_samples (int): Minimum number of samples required to keep a class.
    
    Returns:
        pd.DataFrame: Filtered dataframe with minority classes removed.
    """
    # Get class counts
    label_counts = df['Label'].value_counts()
    
    # Identify classes to keep (those with enough samples)
    valid_classes = label_counts[label_counts >= min_samples].index
    
    # Filter the dataframe
    filtered_df = df[df['Label'].isin(valid_classes)]
    
    print(f"Original class distribution:\n{label_counts}\n")
    print(f"After removing minority classes (min_samples={min_samples}):")
    print(filtered_df['Label'].value_counts())
    
    return filtered_df

def plot_confusion_matrix(y_true, y_pred, classes, title):
    """Plot and display a confusion matrix."""
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def balance_dataset_by_downsampling(df, max_class_0=100_000):
    """Downsample class 0 (normal) and retain all other classes."""
    normal_class = df[df['Label'] == 0].sample(n=max_class_0, random_state=42)
    attack_classes = df[df['Label'] != 0]
    
    balanced_df = pd.concat([normal_class, attack_classes])
    print("New class distribution:\n", balanced_df['Label'].value_counts())
    return balanced_df

        
def data_preprocessing(min_samples=1000):
    """
    Modified preprocessing with class balancing (downsampling class 0).
    
    Args:
        min_samples (int): Minimum samples required to keep a class.
    """
    file_path = "/home/ibibers/XAI_Evalation_For_IDS_datasets/IDS_Datasets/Combined_datasets/CICIDS2017_combined_dataset.csv" 
    df = pd.read_csv(file_path)

    # Clean data
    df['Flow Bytes/s'] = df['Flow Bytes/s'].fillna(df['Flow Bytes/s'].mean())
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # Fix label inconsistencies
    label_mapping = {
        "Web Attack � XSS": "XSS",
        "Web Attack � Sql Injection": "Sql Injection",
        "Web Attack � Brute Force": "Brute Force"
    }
    df["Label"].replace(label_mapping, inplace=True)

    # Remove minority classes
    df = remove_minority_classes(df, min_samples=min_samples)

    # Initialize LabelEncoder before using it
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])

    # Balance dataset by downsampling class 0
    df = balance_dataset_by_downsampling(df, max_class_0=100_000)

    print("Balanced class distribution:")
    print(df['Label'].value_counts())

    # Split into features and target
    dropped_cols = ['Flow Packets/s', 'Flow Bytes/s']
    X = df.drop(columns=dropped_cols + ['Label'], axis=1, errors='ignore')
    y = df['Label']

    # Train-test split (stratified to maintain class ratios)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    return le, X, y, X_train, X_test, y_train, y_test, df


# def data_preprocessing():
#     """
#     A comprehensive data preprocessing pipeline:
#     - Loads the dataset
#     - Handles missing values, duplicates, and irrelevant columns
#     - Fixes label inconsistencies
#     - Splits the data into training and testing sets
#     """

#     file_path = "/home/ibibers/XAI_Evalation_For_IDS_datasets/IDS_Datasets/Combined_datasets/CICIDS2017_combined_dataset.csv" 
#     df = pd.read_csv(file_path)

#     # ------------------ Step 1: Initial Exploration and Cleaning ------------------
#     print("Dataset Info:")
#     df.info()

#     # Handle missing values
#     pd.options.mode.use_inf_as_na = True
#     print("Missing Values:")
#     print(df.loc[:, df.isnull().any()].isnull().sum())
#     df['Flow Bytes/s'] = df['Flow Bytes/s'].fillna(df['Flow Bytes/s'].mean())
#     df.dropna(inplace=True)

#     # Remove columns with all zero values
#     describe_info = df.describe()
#     all_zero_cols = describe_info.loc[:, (describe_info.iloc[1:] == 0).all()].columns
#     df.drop(columns=all_zero_cols, inplace=True)

#     # Remove duplicate rows
#     df.drop_duplicates(inplace=True)

#     # Strip leading spaces in column names
#     df.rename(columns=lambda x: x.strip(), inplace=True)

#     # -------------------- Step 2: Preprocess Labels --------------------
#     print("Unique Labels Before:", df["Label"].unique())

#     # Fix label inconsistencies
#     label_mapping = {
#         "Web Attack � XSS": "XSS",
#         "Web Attack � Sql Injection": "Sql Injection",
#         "Web Attack � Brute Force": "Brute Force"
#     }
#     df["Label"].replace(label_mapping, inplace=True)

#     print("Unique Labels After:", df["Label"].unique())

#     # Label encoding for the target variable
#     le = LabelEncoder()
#     df["Label"] = le.fit_transform(df["Label"])

#     # -------------------- Step 3: Split Data into Train/Test Sets --------------------
#     # Drop irrelevant columns before splitting
#     dropped_cols = ['Flow Packets/s', 'Flow Bytes/s']
#     X = df.drop(columns=dropped_cols + ['Label'], axis=1, errors='ignore')  # Avoid errors if columns are already dropped
#     y = df['Label']

#     # Split into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


#     return le, X, y, X_train, X_test, y_train, y_test, df



    # print("Training set size:", X_train.shape)
    # print("Test set size:", X_test.shape)

    # # ---------------------------- The Generated Top Features using different XAI Methods ----------------------------

    # # Extract top features from each method we used before
    # shap_importance = pd.read_csv("/home/ibibers/XAI_features_selection/TopFeatureUsingXAI_Methodes/shap_feature_importance.csv")
    # loco_importance = pd.read_csv("/home/ibibers/XAI_features_selection/TopFeatureUsingXAI_Methodes/loco_feature_importance.csv")
    # pfi_importance = pd.read_csv("/home/ibibers/XAI_features_selection/TopFeatureUsingXAI_Methodes/pfi_feature_importance.csv")
    # dalex_importance = pd.read_csv("/home/ibibers/XAI_features_selection/TopFeatureUsingXAI_Methodes/dalex_feature_importance.csv")
    # profweight_importance = pd.read_csv("/home/ibibers/XAI_features_selection/TopFeatureUsingXAI_Methodes/profweight_importance.csv")
    # feature_ensemble = pd.read_csv("/home/ibibers/XAI_features_selection/TopFeatureUsingXAI_Methodes/feature_ensemble_final_feature_importance_scores.csv")

    # # Extract top 5 and top 10 features for each method
    # shap_top_5_features = shap_importance['Feature'].head(5).tolist()
    # shap_top_10_features = shap_importance['Feature'].head(10).tolist()

    # loco_top_5_features = loco_importance['Feature'].head(5).tolist()
    # loco_top_10_features = loco_importance['Feature'].head(10).tolist()

    # pfi_top_5_features = pfi_importance['Feature'].head(5).tolist()
    # pfi_top_10_features = pfi_importance['Feature'].head(10).tolist()

    # dalex_top_5_features = dalex_importance['Feature'].head(5).tolist()
    # dalex_top_10_features = dalex_importance['Feature'].head(10).tolist()

    # profweight_top_5_features = profweight_importance['Feature'].head(5).tolist()
    # profweight_top_10_features = profweight_importance['Feature'].head(10).tolist()

    # ensemble_ALL_features = feature_ensemble['Feature'].tolist()
    # ensemble_top_5_features = feature_ensemble['Feature'].head(5).tolist()
    # ensemble_top_10_features = feature_ensemble['Feature'].head(10).tolist()

    # # Subset data based on feature selection
    # datasets = {
    #     "All Features": (X_train, X_test , y_train, y_test),

    #     "All Ensemble Features": (X_train[ensemble_ALL_features], X_test[ensemble_ALL_features], y_train, y_test),
    #     "Ensemble Top 5 Features": (X_train[ensemble_top_5_features], X_test[ensemble_top_5_features], y_train, y_test),
    #     "Ensemble Top 10 Features": (X_train[ensemble_top_10_features], X_test[ensemble_top_10_features], y_train, y_test),

    #     "SHAP Top 5 Features": (X_train[shap_top_5_features], X_test[shap_top_5_features], y_train, y_test),
    #     "SHAP Top 10 Features": (X_train[shap_top_10_features], X_test[shap_top_10_features], y_train, y_test),

    #     "LOCO Top 5 Features": (X_train[loco_top_5_features], X_test[loco_top_5_features], y_train, y_test),
    #     "LOCO Top 10 Features": (X_train[loco_top_10_features], X_test[loco_top_10_features], y_train, y_test),

    #     "PFI Top 5 Features": (X_train[pfi_top_5_features], X_test[pfi_top_5_features], y_train, y_test),
    #     "PFI Top 10 Features": (X_train[pfi_top_10_features], X_test[pfi_top_10_features], y_train, y_test),

    #     "Dalex Top 5 Features": (X_train[dalex_top_5_features], X_test[dalex_top_5_features], y_train, y_test),
    #     "Dalex Top 10 Features": (X_train[dalex_top_10_features], X_test[dalex_top_10_features], y_train, y_test),

    #     "ProfWeight Top 5 Features": (X_train[profweight_top_5_features], X_test[profweight_top_5_features], y_train, y_test),
    #     "ProfWeight Top 10 Features": (X_train[profweight_top_10_features], X_test[profweight_top_10_features], y_train, y_test),

    # }
    # return datasets , le

