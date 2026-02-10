import pandas as pd 
import numpy as np
from datetime import datetime

def to_numeric_safe(value):
    """Converte valores para float, tratando vírgulas e strings de forma robusta."""
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float)): return float(value)
    try:
        # Remove espaços e troca vírgula por ponto
        clean_val = str(value).strip().replace(',', '.')
        # Caso a string venha com algo tipo "8.5.0", pegamos apenas a primeira parte
        if clean_val.count('.') > 1:
            clean_val = clean_val.split('.')[0] + '.' + clean_val.split('.')[1]
        return float(clean_val)
    except (ValueError, TypeError):
        return np.nan

def extract_year(value):
    """Extrai o ano de forma robusta, priorizando anos isolados e datas brasileiras."""
    if pd.isna(value) or str(value).strip() == '':
        return np.nan
    
    # 1. Tenta tratar como ano direto (ex: 2005 ou "2005")
    try:
        val_numeric = int(float(str(value).replace(',', '.')))
        if 1900 <= val_numeric <= datetime.now().year:
            return val_numeric
    except (ValueError, TypeError):
        pass

    # 2. Tenta tratar como data completa (especificando dayfirst=True para formato BR)
    try:
        return pd.to_datetime(value, dayfirst=True, errors='coerce').year
    except:
        return np.nan

def normalize_gender(value):
    """Padroniza o gênero para 'feminino', 'masculino' ou NaN."""
    if pd.isna(value): return np.nan
    v = str(value).lower().strip()
    if any(x in v for x in ['menina', 'fem', 'f']): return 'feminino'
    if any(x in v for x in ['menino', 'mas', 'm']): return 'masculino'
    return np.nan

def collect_list(row, keywords):
    """Agrupa valores de colunas que contenham palavras-chave em uma lista."""
    cols = [c for c in row.index if any(k in c for k in keywords)]
    return [row[c] for c in cols if pd.notnull(row[c]) and str(row[c]).strip() != '']