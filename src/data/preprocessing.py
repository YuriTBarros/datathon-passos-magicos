import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from src.data.maps import MAPA_PEDRAS

# --- 1. FUNÇÕES ACADÊMICAS E DE RANKING ---

def calcular_nota_reversa(row):
    notas = {'mat': row['mat'], 'por': row['por'], 'ing': row['ing']}
    ida_alvo = row['ida']
    nulos = [k for k, v in notas.items() if pd.isnull(v)]
    preenchidos = [v for k, v in notas.items() if pd.notnull(v)]
    
    if ida_alvo == 0:
        for k in nulos: notas[k] = 0.0
    elif len(nulos) == 1:
        valor_faltante = (ida_alvo * 3) - sum(preenchidos)
        notas[nulos[0]] = max(0.0, round(valor_faltante, 2))
    else:
        for k in nulos: notas[k] = 0.0
    return pd.Series(notas)

def pipeline_recomposicao_academica(df):
    df_temp = df.copy()
    cols_notas = ['mat', 'por', 'ing']
    mask = (df_temp['ida'].notnull()) & (df_temp[cols_notas].isnull().any(axis=1))
    if mask.any():
        df_temp.loc[mask, cols_notas] = df_temp[mask].apply(calcular_nota_reversa, axis=1)
    return df_temp

def tratar_e_recalcular_rankings(df):
    df_clean = df.copy()
    mask_ativo = (df_clean['inde'].notnull()) & (df_clean['inde'] > 0)
    df_clean.loc[mask_ativo, 'cg'] = df_clean[mask_ativo].groupby('ano')['inde'].rank(ascending=False, method='min')
    df_clean.loc[mask_ativo, 'cf'] = df_clean[mask_ativo].groupby(['ano', 'fase'])['inde'].rank(ascending=False, method='min')
    df_clean.loc[mask_ativo, 'ct'] = df_clean[mask_ativo].groupby(['ano', 'fase', 'turma'])['inde'].rank(ascending=False, method='min')
    return df_clean

def aplicar_calculo_idade(df):
    df_temp = df.copy()
    mask = (df_temp['idade'].isnull()) & (df_temp['ano_nascimento'].notnull())
    df_temp.loc[mask, 'idade'] = df_temp.loc[mask, 'ano'] - df_temp.loc[mask, 'ano_nascimento']
    return df_temp

def calcular_atingiu_pv(df):
    df_temp = df.copy()
    for ano in [2023, 2024]:
        mask_ano = df_temp['ano'] == ano
        if mask_ano.any():
            threshold = df_temp.loc[mask_ano, 'ipv'].mean() + df_temp.loc[mask_ano, 'ipv'].std()
            df_temp.loc[mask_ano, 'atingiu_pv'] = df_temp.loc[mask_ano, 'ipv'].apply(lambda x: 'Sim' if x >= threshold else 'Não')
    return df_temp

# --- 2. MODELAGEM DE BOLSA (INDICADO) ---

def treinar_modelo_bolsa_v3(df):
    # 1. Filtramos 2022 e GARANTIMOS que o target não seja nulo
    df_train = df[(df['ano'] == 2022) & (df['indicado'].notnull())].copy()
    
    # Se por algum motivo o filtro retornar vazio, precisamos tratar
    if df_train.empty:
        print("⚠️ Aviso: Nenhum dado válido encontrado para treinar o modelo de Bolsa em 2022.")
        return None, None, None

    mapear_escola = lambda v: 1 if 'PÚBLICA' in str(v).upper() else 0
    df_train['tipo_inst_num'] = df_train['tipo_instituicao'].apply(mapear_escola)
    
    features = ['inde', 'ieg', 'ida', 'ipv', 'ian', 'ipp', 'ips', 'cf', 'cg', 'ct', 'tipo_inst_num']
    
    # Preenchemos nulos nas colunas de entrada (X) com a média para o RF não reclamar
    X = df_train[features].fillna(df_train[features].mean())
    
    # Mapeamos o target e garantimos que seja inteiro
    y = df_train['indicado'].astype(str).str.strip().str.capitalize().map({'Sim': 1, 'Não': 0})
    
    # Se após o mapeamento sobrarem NaNs no y, dropamos
    valid_idx = y.notnull()
    X = X[valid_idx]
    y = y[valid_idx].astype(int)

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=42)
    rf.fit(X, y)
    
    print(f"✅ Modelo de Bolsa treinado com {len(y)} amostras de 2022.")
    return rf, features, mapear_escola

def preencher_coluna_indicado(df, modelo, features, func_map):
    df_result = df.copy()
    df_result['tipo_inst_num'] = df_result['tipo_instituicao'].apply(func_map)
    mask_vazio = (df_result['indicado'].isnull()) | (df_result['ano'].isin([2023, 2024]))
    if mask_vazio.any():
        X_imputar = df_result.loc[mask_vazio, features].fillna(df_result[features].mean())
        preds = modelo.predict(X_imputar)
        df_result.loc[mask_vazio, 'indicado'] = ['Sim' if p == 1 else 'Não' for p in preds]
    return df_result

# --- 3. NORMALIZAÇÕES E AUXILIARES ---

def consolidar_instituicao(valor):
    v = str(valor).strip().upper()
    if 'PÚBLICA' in v or 'PUBLICA' in v: return 'PÚBLICA'
    if any(x in v for x in ['PRIVADA', 'REDE DECISÃO', 'JP II']): return 'PRIVADA'
    if 'CONCLUÍDO' in v: return 'CONCLUÍDO'
    return 'OUTROS'

def normalizar_fases(valor):
    v = str(valor).strip().upper()
    if 'ALFA' in v: return 0
    match = re.search(r'\d', v)
    return int(match.group()) if match else None

# --- 4. NOVAS FUNÇÕES DE TARGET (EVASÃO E MOMENTUM) ---

def rotular_evasao(df):
    df_temp = df.copy()
    df_temp['ra'] = df_temp['ra'].astype(str)
    ras_por_ano = {ano: set(df_temp[df_temp['ano'] == ano]['ra']) for ano in df_temp['ano'].unique()}
    
    def check_evasao(row):
        proximo_ano = row['ano'] + 1
        if proximo_ano not in ras_por_ano: return np.nan
        evadiu = 1 if row['ra'] not in ras_por_ano[proximo_ano] else 0
        if evadiu == 1 and normalizar_fases(row['fase']) in [7, 8]: return 0
        return evadiu

    df_temp['evadiu'] = df_temp.apply(check_evasao, axis=1)
    return df_temp

def calcular_delta_defasagem(df):
    df_temp = df.copy()
    df_temp['ra'] = df_temp['ra'].astype(str)
    df_shift = df_temp[['ra', 'ano', 'defasagem']].copy()
    df_shift['ano_match'] = df_shift['ano'] + 1
    df_shift = df_shift.rename(columns={'defasagem': 'def_ant'})

    df_temp = df_temp.merge(
        df_shift[['ra', 'ano_match', 'def_ant']],
        left_on=['ra', 'ano'],
        right_on=['ra', 'ano_match'],
        how='left'
    ).drop(columns=['ano_match'])

    df_temp['delta_defasagem'] = df_temp['defasagem'] - df_temp['def_ant']
    return df_temp.drop(columns=['def_ant'])

# --- 5. LIMPEZA FINAL ---

def pipeline_limpeza_final(df):
    df_proc = df.copy()
    
    # 1. Criação do Timestamp para a Feature Store (Feast)
    # Simulamos o evento no último dia de cada ano letivo
    df_proc['event_timestamp'] = pd.to_datetime(df_proc['ano'].astype(str) + '-12-31')
    
    # 2. Normalizações Básicas
    df_proc['fase_num'] = df_proc['fase'].apply(normalizar_fases)
    df_proc['fase_ideal_num'] = df_proc['fase_ideal'].apply(normalizar_fases)
    df_proc['pedra'] = df_proc['pedra'].astype(str).str.strip().str.upper().replace('AGATA', 'ÁGATA')
    df_proc['pedra_num'] = df_proc['pedra'].map(MAPA_PEDRAS).fillna(0).astype(int)
    
    # 3. Consolidação de Instituição
    df_proc['tipo_inst_consolidado'] = df_proc['tipo_instituicao'].apply(consolidar_instituicao)
    df_proc['tipo_inst_num'] = df_proc['tipo_inst_consolidado'].map({
        'PÚBLICA': 2, 'OUTROS': 1, 'PRIVADA': 0, 'CONCLUÍDO': 0
    })
    
    # 4. Encoding de Variáveis Categóricas
    # Mantemos drop_first=True para evitar a armadilha da multicolinearidade (Dummy Variable Trap)
    df_proc = pd.get_dummies(df_proc, columns=['genero', 'atingiu_pv', 'indicado'], 
                             prefix=['genero', 'pv', 'indicado'], drop_first=True)
    
    # Converter booleanos resultantes do get_dummies para int (0/1)
    cols_dummies = [c for c in df_proc.columns if any(x in c for x in ['genero_', 'pv_', 'indicado_'])]
    df_proc[cols_dummies] = df_proc[cols_dummies].astype(int)
    
    # 5. Limpeza de Registros e Colunas
    df_proc = df_proc.drop(df_proc[df_proc['ra'] == 'RA-1519'].index, errors='ignore')
    
    # Definimos o que remover (colunas de texto sujo ou intermediárias)
    # IMPORTANTE: Note que 'ra', 'ano', 'evadiu' e 'delta_defasagem' NÃO estão nesta lista
    drops = [
        'destaque_ieg', 'destaque_ida', 'destaque_ipv', 'rec_psicologica', 
        'ano_nascimento', 'n_av', 'avaliadores', 'recomendacoes', 
        'situacao_ativo', 'fase', 'fase_ideal', 'pedra', 'turma', 
        'tipo_instituicao', 'tipo_inst_consolidado', 'tipo_inst_num_aux', 'nome'
    ]
    
    df_proc = df_proc.drop(columns=drops, errors='ignore')
    
    # 6. Garantia de Tipos para o Feast (Float32 e Int64 são preferíveis)
    # Isso evita erros de incompatibilidade no registry do Feast
    cols_float = df_proc.select_dtypes(include=['float64']).columns
    df_proc[cols_float] = df_proc[cols_float].astype('float32')
    
    return df_proc