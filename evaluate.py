import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import joblib

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and visualize results."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("Model Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Residual plot
    residuals = y_test - predictions
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residual Distribution")
    plt.show()

    return rmse, r2

def load_pipeline_and_preprocess(test_set, target_column):
    """Load preprocessing pipeline and prepare test set."""
    pipeline = joblib.load("artifacts/preprocessing_pipeline.joblib")
    X_test = test_set.drop(target_column, axis=1)
    y_test = test_set[target_column].copy()
    X_test_prepared = pipeline.transform(X_test)
    return X_test_prepared, y_test
