import pandas as pd
import numpy as np
import os
import src.data.unify_dataset as unify
import src.data.preprocessing as prep

def main():
    print("Iniciando processamento dos dados da Passos Mágicos...")
    
    # 1. Ingestão e Unificação
    # O dirname(__file__) pega a pasta 'src/data'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Subimos dois níveis para chegar na raiz e entrar em 'data/raw'
    file_path = os.path.join(base_dir, '../../data/raw/PEDE_2022-2024.xlsx')
    
    if not os.path.exists(file_path):
        print(f"❌ Erro: Arquivo não encontrado em {os.path.abspath(file_path)}")
        return

    print("Lendo abas do Excel...")
    xls = pd.ExcelFile(file_path)
    df = pd.concat([
        unify.unificar_safra(pd.read_excel(xls, 'PEDE2022'), unify.MAP_2022, 2022),
        unify.unificar_safra(pd.read_excel(xls, 'PEDE2023'), unify.MAP_2023, 2023),
        unify.unificar_safra(pd.read_excel(xls, 'PEDE2024'), unify.MAP_2024, 2024)
    ], ignore_index=True)
    
    # 2. Processamento de Regras de Negócio e Limpeza Inicial
    print("Processando regras acadêmicas e rankings...")
    df = df[df['inde'].notnull()].copy()
    df = prep.pipeline_recomposicao_academica(df)
    df = prep.tratar_e_recalcular_rankings(df)
    df = prep.aplicar_calculo_idade(df)
    df = prep.calcular_atingiu_pv(df)
    
    # 3. Imputação de ML (Bolsa/Indicado)
    # Treinamos com 2022 para preencher 2023/24
    print("Iniciando treino do modelo de Bolsa (imputação)...")
    modelo, feats, f_map = prep.treinar_modelo_bolsa_v3(df)
    
    # SÓ EXECUTA O PREENCHIMENTO SE O TREINO DER CERTO (Evita TypeError: NoneType)
    if modelo is not None and f_map is not None:
        df = prep.preencher_coluna_indicado(df, modelo, feats, f_map)
        print("✅ Coluna 'indicado' preenchida via ML.")
    else:
        print("⚠️ Aviso: Pulando imputação de bolsa devido a falta de dados válidos em 2022.")
    
    # --- 🚀 NOVO: Geração de Targets para ML ---
    # Realizamos isso ANTES da limpeza final para aproveitar as colunas 'ra' e 'ano'
    print("Calculando indicadores de Evasão e Delta de Defasagem (Momentum)...")
    df = prep.rotular_evasao(df)
    df = prep.calcular_delta_defasagem(df)
    
    # 4. Limpeza Final, Normalização e Encoding
    # Remove colunas brutas e transforma chaves em números/dummies
    print("Executando limpeza final e encodings...")
    df_final = prep.pipeline_limpeza_final(df)
    
    # Ajuste final de tipos para garantir integridade
    if 'idade' in df_final.columns:
        df_final['idade'] = df_final['idade'].fillna(0).astype(int)
    
    # 5. Salvar o Produto Final
    output_path = os.path.join(base_dir, '../../data/processed/dataset_final.csv')
    
    # Garante que a pasta 'processed' existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_final.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"🚀 Dataset Processado com Sucesso em: {os.path.abspath(output_path)}")
    print(f"Total de registros: {len(df_final)}")
    print(f"Colunas geradas: {len(df_final.columns)}")
    print("-" * 30)

if __name__ == "__main__":
    main()