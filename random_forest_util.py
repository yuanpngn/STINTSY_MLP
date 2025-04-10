import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def tune_random_forest(X_train, y_train, X_test, y_test, verbose=2):
    # Validation checks
    if X_train is None or y_train is None or X_test is None or y_test is None:
        print("Training or test data is None.")
        return None
    if len(X_train) == 0 or len(y_train) == 0 or len(X_test) == 0 or len(y_test) == 0:
        print("Training or test data is empty.")
        return None
    if X_train.shape[0] != y_train.shape[0]:
        print("Mismatch between number of samples in X_train and y_train.")
        return None
    if X_train.shape[1] != X_test.shape[1]:
        print("Mismatch between number of features in X_train and X_test.")
        return None

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }

    # Initialize the RandomForestRegressor
    rf_regressor = RandomForestRegressor(random_state=42)

    # Initialize the GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf_regressor,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=verbose
    )

    try:
        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)

        print("Best Parameters:", grid_search.best_params_)
        print("Best Score (CV):", grid_search.best_score_)

        # Retrieve the best model
        if grid_search.best_estimator_:
            best_rf_regressor = grid_search.best_estimator_
            print(f"Best Estimator: {best_rf_regressor}")

            # Test set evaluation
            y_pred_best = best_rf_regressor.predict(X_test)
            test_mse_best = mean_squared_error(y_test, y_pred_best)
            test_rmse_best = np.sqrt(test_mse_best)
            test_r2_best = r2_score(y_test, y_pred_best)

            print(f"Test MSE (Best Model): {test_mse_best}")
            print(f"Test RMSE (Best Model): {test_rmse_best}")
            print(f"Test R² (Best Model): {test_r2_best}")

            return best_rf_regressor
        else:
            print("No valid model found after GridSearchCV.")
            return None

    except Exception as e:
        print(f"Error during GridSearchCV: {e}")
        return None


def print_RF():
    print("Random Forest Regressor TEST")
    pass



def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Plot learning curves for a given estimator.
    
    Parameters:
    - estimator: The machine learning model to evaluate.
    - title: Title of the plot.
    - X: Feature matrix.
    - y: Target vector.
    - cv: Cross-validation splitting strategy.
    - n_jobs: Number of jobs to run in parallel.
    - train_sizes: Relative or absolute numbers of training examples to use for generating the learning curve.
    """
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes, 
                     train_scores_mean - train_scores_std, 
                     train_scores_mean + train_scores_std, 
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, 
                     test_scores_mean - test_scores_std, 
                     test_scores_mean + test_scores_std, 
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    return plt