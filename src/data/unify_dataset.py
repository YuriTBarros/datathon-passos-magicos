import pandas as pd
import numpy as np
from utils import to_numeric_safe, normalize_gender, extract_year, collect_list
from maps import MAP_2022, MAP_2023, MAP_2024

def calculate_ipp_2022(row):
    """Realiza a engenharia reversa do IPP para a safra 2022."""
    if str(row.get('Fase')).strip() == '8': return 0.0
    try:
        # Pesos oficiais: INDE = IAN*0.1 + IDA*0.2 + IEG*0.2 + IAA*0.1 + IPS*0.2 + IPP*0.1 + IPV*0.1
        inde = to_numeric_safe(row.get('INDE 22', 0))
        ian  = to_numeric_safe(row.get('IAN', 0))
        ida  = to_numeric_safe(row.get('IDA', 0))
        ieg  = to_numeric_safe(row.get('IEG', 0))
        iaa  = to_numeric_safe(row.get('IAA', 0))
        ips  = to_numeric_safe(row.get('IPS', 0))
        ipv  = to_numeric_safe(row.get('IPV', 0))
        
        soma_conhecida = (ian*0.1) + (ida*0.2) + (ieg*0.2) + (iaa*0.1) + (ips*0.2) + (ipv*0.1)
        ipp_calc = (inde - soma_conhecida) / 0.1
        return round(max(0, min(10, ipp_calc)), 3)
    except:
        return np.nan

def unificar_safra(df_raw, mapping, ano):
    """Padroniza e limpa os dados de uma safra específica para o Schema Master."""
    df_temp = df_raw.copy()
    
    # 1. Tratamentos Específicos por Safra
    if ano == 2022: 
        df_temp['IPP'] = df_temp.apply(calculate_ipp_2022, axis=1)
    
    if ano == 2023 and 'Destaque IPV.1' in df_temp.columns:
        df_temp['Destaque IPV'] = df_temp['Destaque IPV'].fillna('').astype(str) + " " + \
                                 df_temp['Destaque IPV.1'].fillna('').astype(str)
        df_temp['Destaque IPV'] = df_temp['Destaque IPV'].str.strip()

    # 2. Mapeamento de Colunas
    df_mapped = df_temp.rename(columns=mapping)
    
    # 3. Forçar RA como String (Crucial para o Join de Evasão/Delta)
    if 'ra' in df_mapped.columns:
        df_mapped['ra'] = df_mapped['ra'].astype(str).str.strip()

    # 4. Conversão Numérica em Massa
    cols_numericas = ['inde', 'mat', 'por', 'ing', 'iaa', 'ieg', 'ips', 'ipp', 'ipv', 'ian', 'idade', 'defasagem']
    for col in cols_numericas:
        if col in df_mapped.columns:
            df_mapped[col] = df_mapped[col].apply(to_numeric_safe)

    # 5. Normalizações e Metadados
    df_mapped['ano'] = ano
    df_mapped['genero'] = df_mapped['genero'].apply(normalize_gender)
    df_mapped['ano_nascimento'] = df_mapped['nasc_raw'].apply(extract_year)
    
    # 6. Agrupamento de Listas (Avaliadores/Recomendações)
    df_mapped['avaliadores'] = df_temp.apply(lambda r: collect_list(r, ['Avaliador']), axis=1)
    df_mapped['recomendacoes'] = df_temp.apply(lambda r: collect_list(r, ['Rec Av']), axis=1)
    
    # 7. Definição de Status
    df_mapped['situacao_ativo'] = df_mapped['inde'].apply(
        lambda x: 'Ativo' if pd.notnull(x) and x > 0 else 'Inativo'
    )
    
    # 8. Garantia do Target Schema (Colunas Finais)
    target_columns = [
        'ra', 'nome', 'ano', 'ano_nascimento', 'idade', 'genero', 'ano_ingresso', 'tipo_instituicao',
        'fase', 'turma', 'inde', 'pedra', 'cg', 'cf', 'ct', 'n_av', 'iaa', 'ieg', 'ips', 'ipp',
        'ida', 'mat', 'por', 'ing', 'indicado', 'atingiu_pv', 'ipv', 'ian', 'fase_ideal', 
        'defasagem', 'rec_psicologica', 'destaque_ieg', 'destaque_ida', 'destaque_ipv',
        'situacao_ativo', 'avaliadores', 'recomendacoes'
    ]
    
    # Adiciona colunas faltantes como NaN para manter o formato tabular
    for col in target_columns:
        if col not in df_mapped.columns: 
            df_mapped[col] = np.nan
            
    return df_mapped[target_columns]