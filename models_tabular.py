"""
models_tabular.py
------------------
Tabular regression models that all share the same interface:
    model = build_xxx()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

IMPORTANT: When predicting raw price on a strongly trending stock, tree-based
models (Random Forest, XGBoost) and SVR CANNOT extrapolate beyond the price
range seen in training -- if training data tops out at $300 and the test
period reaches $400+, these models will flatten out near $300 and look
terrible. This is a structural limitation, not a bug.

The fix used in main.py: also train every model to predict the next-day
% RETURN instead of raw price. Returns don't have this extrapolation
problem (a +0.5% prediction is valid whether the stock is at $50 or $500),
so this is what makes tree models and SVR competitive.
"""

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


def build_linear_regression():
    return LinearRegression()


def build_random_forest(n_estimators=500, max_depth=12, min_samples_leaf=2, random_state=42):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )


def build_svr(kernel="rbf", C=1000, gamma="scale", epsilon=0.01):
    # NOTE: SVR is very sensitive to feature/target scale -- always scale
    # both X and y before use. It performs far better in "return" mode
    # (small, bounded values) than in raw "price" mode.
    return SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)


def build_xgboost(n_estimators=800, max_depth=4, learning_rate=0.03, random_state=42):
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        objective="reg:squarederror",
        early_stopping_rounds=30,
        n_jobs=-1,
    )


TABULAR_MODEL_BUILDERS = {
    "Linear Regression": build_linear_regression,
    "Random Forest": build_random_forest,
    "SVR": build_svr,
    "XGBoost": build_xgboost,
}


# ----------------------------------------------------------------------
# Regularized model set used by the DASHBOARD ensemble (dashboard_logic.py)
# ----------------------------------------------------------------------
# Daily returns are mostly noise, so flexible models overfit it easily.
# In a 5-stock benchmark (see README "Accuracy improvements") these
# heavily regularized settings beat the defaults above on held-out data,
# and SVR (C=1000, epsilon=0.01 -- as large as a typical daily return)
# was the least accurate model on every stock, so it's left out here.
# The originals above are kept unchanged for main.py's model comparison.

def build_ridge(alpha=10.0):
    # Linear regression with an L2 penalty: stable when features are correlated
    # (many moving-average features are), unlike plain LinearRegression.
    return Ridge(alpha=alpha)


def build_random_forest_regularized(random_state=42):
    # Shallow trees with big leaves = averages over many days, not single noisy days.
    return RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=50,
                                 random_state=random_state, n_jobs=-1)


def build_xgboost_regularized(random_state=42):
    return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
                        objective="reg:squarederror", random_state=random_state, n_jobs=-1)


ENSEMBLE_MODEL_BUILDERS = {
    "Ridge": build_ridge,
    "Random Forest": build_random_forest_regularized,
    "XGBoost": build_xgboost_regularized,
}
