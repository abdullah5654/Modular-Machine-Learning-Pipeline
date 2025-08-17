import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

def load_and_split_data(file_path, target_column):
    """Load dataset and split into train/test with stratified sampling if categorical target exists."""
    data = pd.read_csv(file_path)

    if target_column in data.columns and len(data[target_column].unique()) <= 10:
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(data, data[target_column]):
            train_set = data.loc[train_index]
            test_set = data.loc[test_index]
    else:
        train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    return train_set, test_set

def preprocess_data(train_set, target_column):
    """Handle missing values, scaling, and return processed data + pipeline."""
    X_train = train_set.drop(target_column, axis=1)
    y_train = train_set[target_column].copy()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    X_train_prepared = num_pipeline.fit_transform(X_train)

    # Save pipeline for test data preprocessing
    joblib.dump(num_pipeline, "artifacts/preprocessing_pipeline.joblib")

    return X_train_prepared, y_train
