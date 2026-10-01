"""
models_classification.py
--------------------------
Direction (up/down) classification models -- a second opinion
alongside price prediction. Predicting whether tomorrow's close will
be higher or lower is a fundamentally easier, more robust problem than
predicting the exact price, since it only needs the SIGN right, not
the magnitude.

Same shared interface as models_tabular.py:
    model = build_xxx()
    model.fit(X_train, y_train)          # y_train is 0/1
    probs = model.predict_proba(X_test)[:, 1]   # probability of "up"
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def build_logistic_regression():
    return LogisticRegression(max_iter=1000, C=1.0)


def build_random_forest_classifier(n_estimators=500, max_depth=8, min_samples_leaf=4, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )


def build_xgboost_classifier(n_estimators=500, max_depth=4, learning_rate=0.03, random_state=42):
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )


CLASSIFICATION_MODEL_BUILDERS = {
    "Logistic Regression": build_logistic_regression,
    "Random Forest Classifier": build_random_forest_classifier,
    "XGBoost Classifier": build_xgboost_classifier,
}
