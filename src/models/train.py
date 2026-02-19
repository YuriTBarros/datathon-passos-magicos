import pandas as pd
import numpy as np
from feast import FeatureStore
import mlflow
import mlflow.sklearn
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
from datetime import datetime, timezone

# Configurações de Caminho
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEAST_REPO_PATH = PROJECT_ROOT / "feature_repo"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_final.parquet"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "Passos_Magicos_Evasao_Otimizada"
LOCAL_MODELS_DIR = PROJECT_ROOT / "models_artifacts"
INFERENCE_MODEL_PATH = LOCAL_MODELS_DIR / "model_evasao_inferencia.joblib"

def get_training_data():
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    entity_df = pd.read_parquet(DATA_PATH)
    
    features = [
        "academic_features:inde", "academic_features:cg", "academic_features:cf",
        "academic_features:ct", "academic_features:mat", "academic_features:por",
        "academic_features:ing", "academic_features:ida", "academic_features:defasagem",
        "profile_behavior_features:idade", "profile_behavior_features:pedra_num",
        "profile_behavior_features:ipv", "profile_behavior_features:ian",
        "profile_behavior_features:genero_masculino"
    ]
    
    df = store.get_historical_features(entity_df=entity_df, features=features).to_df()
    return df.fillna(0)

def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "class_weight": "balanced"
    }

    clf = RandomForestClassifier(**params, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return f1_score(y_test, preds)

def setup_mlflow_experiment():
    """
    Configura tracking local e garante experimento com artifact_location acessível.
    Se o experimento original apontar para caminho inválido, cria um sufixado local.
    """
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_root_uri = MLFLOW_ARTIFACTS_DIR.resolve().as_uri()

    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_root_uri)
        return EXPERIMENT_NAME

    artifact_location = (exp.artifact_location or "").lower()
    if "/users/yuridebarros" in artifact_location:
        local_experiment_name = f"{EXPERIMENT_NAME}_local"
        local_exp = mlflow.get_experiment_by_name(local_experiment_name)
        if local_exp is None:
            mlflow.create_experiment(local_experiment_name, artifact_location=artifact_root_uri)
        return local_experiment_name

    return EXPERIMENT_NAME

def run_training():
    print("📥 Coletando dados da Feature Store...")
    df = get_training_data()
    
    X = df.drop(columns=['ra', 'event_timestamp', 'evadiu', 'ano'])
    y = df['evadiu']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    experiment_name = setup_mlflow_experiment()
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="RandomForest_Optuna_Final"):
        print(f"📊 Dados de Treino: {len(X_train)} | Dados de Teste: {len(X_test)}")
        
        print("🧪 Iniciando otimização de hiperparâmetros...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=30)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1_optuna", study.best_value)

        # Treinando o modelo final
        final_clf = RandomForestClassifier(**study.best_params, class_weight="balanced", random_state=42)
        final_clf.fit(X_train, y_train)
        
        y_pred = final_clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Logs de métricas
        mlflow.log_metric("final_test_f1", report['1.0']['f1-score'])
        mlflow.log_metric("final_test_recall", report['1.0']['recall'])
        mlflow.log_metric("final_test_precision", report['1.0']['precision'])
        
        # --- GERAÇÃO DE ARTEFATOS GRÁFICOS ---
        print("🎨 Gerando gráficos de performance...")
        
        # 1. Matriz de Confusão
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusão - Evasão')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        confusion_path = PROJECT_ROOT / "confusion_matrix.png"
        plt.savefig(confusion_path)
        mlflow.log_artifact(str(confusion_path))
        plt.close()

        # 2. Importância das Features
        
        plt.figure(figsize=(10, 8))
        importances = pd.Series(final_clf.feature_importances_, index=X.columns)
        importances.nlargest(10).sort_values().plot(kind='barh', color='skyblue')
        plt.title('Top 10 Features que mais influenciam a Evasão')
        plt.tight_layout()
        importance_path = PROJECT_ROOT / "feature_importance.png"
        plt.savefig(importance_path)
        mlflow.log_artifact(str(importance_path))
        plt.close()
        
        # Salvando o modelo
        mlflow.sklearn.log_model(final_clf, "model_evasao_final")

        # Salvando arquivo local para inferência (modelo + metadados)
        LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        inference_bundle = {
            "model": final_clf,
            "feature_columns": X.columns.tolist(),
            "target_column": "evadiu",
            "default_threshold": 0.5,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "best_params": study.best_params,
        }
        joblib.dump(inference_bundle, INFERENCE_MODEL_PATH)
        mlflow.log_artifact(str(INFERENCE_MODEL_PATH), artifact_path="inference_bundle")
        
        print(f"✅ Sucesso! Melhor F1 no Teste: {report['1.0']['f1-score']:.4f}")
        print(f"💾 Arquivo de inferência salvo em: {INFERENCE_MODEL_PATH}")
        print("🚀 Experimento completo registrado no MLflow.")

if __name__ == "__main__":
    run_training()
