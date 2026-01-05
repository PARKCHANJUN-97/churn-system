import os, json, joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils.features import add_derived_features, build_target, build_X

RANDOM_STATE = 42
DATA_PATH = "data/BankChurners.csv"
MODEL_DIR = "models"
METRICS_PATH = "metrics.json"

FILTER_CARD_CATEGORY = False
CARD_CATEGORIES = ["Silver", "Gold", "Platinum"]

def best_threshold_by_f1(y_true, y_prob) -> float:
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = (2*p*r)/(p+r+1e-12)
    f1 = f1[:-1]
    return float(t[int(np.argmax(f1))])

def eval_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if FILTER_CARD_CATEGORY:
        df = df[df["Card_Category"].isin(CARD_CATEGORIES)].copy()

    df = add_derived_features(df)
    y = build_target(df)
    X = build_X(df)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    metrics = {
        "data": {"path": DATA_PATH, "rows": int(len(df)), "filter_card_category": FILTER_CARD_CATEGORY},
        "feature_columns": feature_columns,
        "models": {}
    }

    scaler = StandardScaler()
    X_train_lr = scaler.fit_transform(X_train)
    X_test_lr = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    lr.fit(X_train_lr, y_train)
    lr_prob = lr.predict_proba(X_test_lr)[:, 1]
    lr_thr = best_threshold_by_f1(y_test.values, lr_prob)
    metrics["models"]["logistic_regression"] = eval_metrics(y_test.values, lr_prob, lr_thr)
    joblib.dump({"model": lr, "scaler": scaler, "feature_columns": feature_columns}, f"{MODEL_DIR}/logistic_regression.joblib")

    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_thr = best_threshold_by_f1(y_test.values, rf_prob)
    metrics["models"]["random_forest"] = eval_metrics(y_test.values, rf_prob, rf_thr)
    joblib.dump({"model": rf, "feature_columns": feature_columns}, f"{MODEL_DIR}/random_forest.joblib")

    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    xgb_thr = best_threshold_by_f1(y_test.values, xgb_prob)
    metrics["models"]["xgboost"] = eval_metrics(y_test.values, xgb_prob, xgb_thr)
    joblib.dump({"model": xgb, "feature_columns": feature_columns}, f"{MODEL_DIR}/xgboost.joblib")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("✅ DONE: models/ 와 metrics.json 생성 완료")

if __name__ == "__main__":
    main()
