import os
from feast import FeatureStore
import mlflow

def train():
    print("🚀 Iniciando pipeline de treinamento...")
    
    # Placeholder para conexão com a Feature Store
    # Em produção, o repo_path deve apontar para onde está o feature_store.yaml
    repo_path = "feature_repo"
    
    try:
        store = FeatureStore(repo_path=repo_path)
        print("✅ Conexão com a Feature Store estabelecida.")
    except Exception as e:
        print(f"⚠️ Aviso: Feature Store não detectada no ambiente de CI (esperado). Erro: {e}")

    # Configuração inicial do MLflow
    mlflow.set_experiment("Passos_Magicos_Evasao")
    
    with mlflow.start_run(run_name="Baseline_Skeleton"):
        mlflow.log_param("status", "skeleton_ready")
        print("✅ Log de smoke-test registrado no MLflow.")

if __name__ == "__main__":
    train()