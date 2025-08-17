from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

def train_and_tune(X_train, y_train, model_type="RandomForest"):
    """Train and tune ML model with GridSearchCV."""
    if model_type == "LinearRegression":
        model = LinearRegression()
        param_grid = {}
    elif model_type == "DecisionTree":
        model = DecisionTreeRegressor(random_state=42)
        param_grid = {"max_depth": [2, 4, 6, None]}
    else:
        model = RandomForestRegressor(random_state=42)
        param_grid = {"n_estimators": [50, 100], "max_depth": [None, 10, 20]}

    grid_search = GridSearchCV(model, param_grid, cv=5,
                               scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Save trained model
    joblib.dump(best_model, f"artifacts/{model_type}_model.joblib")

    return best_model, grid_search.best_params_
