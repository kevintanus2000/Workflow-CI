import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, learning_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score
)
from sklearn.utils import estimator_html_repr
import json

def save_estimator_html(model, filename='estimator.html'):
    html_content = estimator_html_repr(model)
    with open(filename, 'w') as f:
        f.write(html_content)
    return filename

def save_metric_info_json(model, filename='metric_info.json'):
    class_name = type(model).__name__
    metric_info = {
        f"{class_name}_score-2_X_test":  f"{class_name}.score(X=X_test, y=y_test)",
        f"{class_name}_score_X_train":   f"{class_name}.score(X=X_train, y=y_train)"
    }
    with open(filename, 'w') as f:
        json.dump(metric_info, f, indent=2)
    return filename

def setup_mlflow(tracking: str = "local"):
    if tracking.lower() in ("local", ""):
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("Iris_Modelling_Tuning")
    else:
        DAGSHUB_USERNAME = "kevintanus2000"
        REPO_NAME = "iris-mlflow-project"
        TOKEN = "9f2a913dc6316f9ffea25b302892fef310432484"
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = TOKEN
        os.environ['DAGSHUB_USER_TOKEN']        = TOKEN
        dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)
        mlflow.set_tracking_uri(
            f'https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow'
        )
        mlflow.set_experiment("Iris_Modelling_Tuning")

def save_confusion_matrix(y_test, y_pred, title, filename):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title); plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout(); plt.savefig(filename); plt.close()
    return filename

def save_feature_importance(model, feature_names, title, filename):
    fi = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=fi,
                hue='feature', palette='viridis', legend=False)
    plt.title(title); plt.tight_layout(); plt.savefig(filename); plt.close()
    return filename

def save_learning_curve(model, X, y, filename='learning_curve.png'):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Score')
    plt.fill_between(train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
    plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation Score')
    plt.fill_between(train_sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
    plt.xlabel('Training Size'); plt.ylabel('Accuracy')
    plt.title('Learning Curve - Random Forest Tuned')
    plt.legend(); plt.tight_layout(); plt.savefig(filename); plt.close()
    return filename

def save_hyperparam_heatmap(cv_results, filename='hyperparam_heatmap.png'):
    df_cv = pd.DataFrame(cv_results)
    pivot = df_cv.pivot_table(
        values='mean_test_score',
        index='param_n_estimators',
        columns='param_max_depth'
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('GridSearch: n_estimators vs max_depth')
    plt.tight_layout(); plt.savefig(filename); plt.close()
    return filename

def train_with_tuning():
    tracking = input("Tracking destination [local/dagshub] (default: local): ").strip().lower()
    setup_mlflow(tracking)
    df = pd.read_csv('iris_cleaned.csv')
    X  = df.drop('target', axis=1)
    y  = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = {
        'n_estimators':      [50, 100, 200, 300],
        'max_depth':         [2, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],
        'criterion':         ['gini', 'entropy'],
    }

    print("Menjalankan GridSearchCV")
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring='accuracy',
        n_jobs=-1, verbose=1, return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model  = grid_search.best_estimator_

    with mlflow.start_run(run_name="RandomForest_Tuned", nested=True):
        y_pred       = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)

        # Log Params
        mlflow.log_params(best_params)
        mlflow.log_param("random_state",   42)
        mlflow.log_param("cv_folds",       5)
        mlflow.log_param("tuning_method",  "GridSearchCV")
        mlflow.set_tag("estimator_name",   "RandomForestClassifier")

        # Log Metrics
        train_acc = accuracy_score(y_train, best_model.predict(X_train))
        test_acc  = accuracy_score(y_test, y_pred)

        mlflow.log_metric("training_score",      train_acc)
        mlflow.log_metric("test_score",          test_acc)
        mlflow.log_metric("accuracy",            test_acc)
        mlflow.log_metric("best_cv_score",       grid_search.best_score_)
        mlflow.log_metric("cv_mean_score",       cross_val_score(best_model, X, y, cv=5).mean())
        mlflow.log_metric("precision_weighted",
            precision_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("recall_weighted",
            recall_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("f1_weighted",
            f1_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("roc_auc_ovr",
            roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'))

        # Log Artifacts
        mlflow.log_artifact(save_confusion_matrix(
            y_test, y_pred, 'Confusion Matrix - Tuned', 'confusion_matrix_tuned.png'))
        mlflow.log_artifact(save_feature_importance(
            best_model, X.columns, 'Feature Importance - Tuned', 'feature_importance_tuned.png'))
        mlflow.log_artifact(save_learning_curve(best_model, X, y))
        mlflow.log_artifact(save_hyperparam_heatmap(grid_search.cv_results_))

        report = classification_report(y_test, y_pred)
        with open('classification_report_tuned.txt', 'w') as f:
            f.write(report)
        mlflow.log_artifact('classification_report_tuned.txt')

        # Log Model
        mlflow.sklearn.log_model(best_model, "model", input_example=X_train.iloc[:5])
        mlflow.log_artifact(save_estimator_html(best_model))
        mlflow.log_artifact(save_metric_info_json(best_model))

        print(f"Training : {train_acc:.4f} | Test : {test_acc:.4f}")
        print(f"Best CV  : {grid_search.best_score_:.4f}")

if __name__ == "__main__":
    train_with_tuning()
