import pandas as pd
from feast import FeatureStore
import mlflow.sklearn

# 1. Configurações
RUN_ID = "1301b8da7bf841eba799135a9cfa8b72" 
MODEL_URI = f"runs:/{RUN_ID}/model_evasao_final"
FEAST_REPO_PATH = "feature_repo"

def prever_risco(ra_aluno):
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    model = mlflow.sklearn.load_model(MODEL_URI)

    cols_modelo = list(model.feature_names_in_)
    
    features_set = set()
    for col in cols_modelo:
        base_name = col.replace("__", "")
        # Ignoramos colunas geradas por get_dummies (que contém _) para a busca no Feast
        if "_" in base_name and base_name not in ['pedra_num', 'genero_masculino', 'delta_defasagem']:
             continue
             
        if base_name in ['idade', 'pedra_num', 'ipv', 'ian', 'genero_masculino']:
            features_set.add(f"profile_behavior_features:{base_name}")
        else:
            features_set.add(f"academic_features:{base_name}")
    
    features_para_buscar = list(features_set)
    print(f"🔍 Buscando {len(features_para_buscar)} features únicas para o aluno {ra_aluno}...")
    
    # 5. Busca no Feast usando TRY para capturar erros de projeção
    try:
        online_features = store.get_online_features(
            features=features_para_buscar,
            entity_rows=[{"ra": ra_aluno}]
        ).to_dict()
        df_raw = pd.DataFrame.from_dict(online_features)
    except Exception as e:
        print(f"⚠️ Aviso: Algumas features não foram encontradas no Feast, usando valores padrão. ({e})")
        df_raw = pd.DataFrame([{"ra": ra_aluno}])

    # Limpeza dos nomes vindos do Feast
    df_clean = pd.DataFrame()
    for col in df_raw.columns:
        clean_name = col.split(":")[-1] if ":" in col else col
        df_clean[clean_name] = df_raw[col]

    # 6. Alinhamento Final com o Modelo
    X = pd.DataFrame(index=[0])
    for col in cols_modelo:
        clean_col = col.replace("__", "")
        # Se a coluna existe no Feast, usamos. Se não (como pv_Sim), usamos 0.
        if clean_col in df_clean.columns:
            X[col] = df_clean[clean_col].values[0]
        else:
            X[col] = 0 

    X = X[cols_modelo].fillna(0)

    # 7. Predição
    probabilidade = model.predict_proba(X)[0][1]
    risco = "⚠️ ALTO RISCO" if probabilidade > 0.5 else "✅ BAIXO RISCO"

    print(f"\n--- Diagnóstico para o Aluno RA: {ra_aluno} ---")
    print(f"Status: {risco} (Probabilidade: {probabilidade:.2%})")
    print("-" * 45)
    
    importancias = model.feature_importances_
    indices_top = importancias.argsort()[-5:][::-1]
    print("Principais Fatores de Influência:")
    for i in indices_top:
        print(f" • {cols_modelo[i].replace('__', '')}: {importancias[i]:.2%}")
    print("-" * 45)
if __name__ == "__main__":
    meu_ra_teste = "RA-1286" 
    try:
        explicar_risco(meu_ra_teste)
    except Exception as e:
        print(f"❌ Erro na predição: {e}")