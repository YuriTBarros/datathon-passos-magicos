from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    ValueType,
)
from feast.types import Float32, Int64, String

#1. Definição da Fonte de Dados (Parquet Processado)
path_parquet = "/Users/renanmelo/Documents/GitHub/datathon-passos-magicos/data/processed/dataset_final.parquet"

pede_source = FileSource(
    path=path_parquet,
    timestamp_field="event_timestamp",
)

# 2. Definição da Entidade (O Aluno)
aluno = Entity(
    name="ra", 
    join_keys=["ra"], 
    value_type=ValueType.STRING,
    description="RA do aluno (Identificador Único)",
)

# 3. View de Performance Acadêmica (Features Contínuas)
academic_features_view = FeatureView(
    name="academic_features",
    entities=[aluno],
    ttl=timedelta(days=730),
    schema=[
        Field(name="inde", dtype=Float32),
        Field(name="cg", dtype=Float32),
        Field(name="cf", dtype=Float32),
        Field(name="ct", dtype=Float32),
        Field(name="mat", dtype=Float32),
        Field(name="por", dtype=Float32),
        Field(name="ing", dtype=Float32),
        Field(name="ida", dtype=Float32),
        Field(name="defasagem", dtype=Float32),
    ],
    online=True,
    source=pede_source,
)

# 4. View de Comportamento e Perfil (Features Categóricas/Inteiras)
profile_behavior_view = FeatureView(
    name="profile_behavior_features",
    entities=[aluno],
    ttl=timedelta(days=730),
    schema=[
        Field(name="idade", dtype=Int64),
        Field(name="fase_num", dtype=Int64),
        Field(name="pedra_num", dtype=Int64),
        Field(name="tipo_inst_num", dtype=Int64),
        Field(name="genero_masculino", dtype=Int64),
        Field(name="pv_Sim", dtype=Int64),
        Field(name="ieg", dtype=Float32),
        Field(name="iaa", dtype=Float32),
        Field(name="ips", dtype=Float32),
        Field(name="ipp", dtype=Float32),
        Field(name="ipv", dtype=Float32),
        Field(name="ian", dtype=Float32),
    ],
    online=True,
    source=pede_source,
)

# 5. View de Alvos (Targets para Treinamento)
targets_view = FeatureView(
    name="targets_features",
    entities=[aluno],
    ttl=timedelta(days=730),
    schema=[
        Field(name="evadiu", dtype=Int64),
        Field(name="delta_defasagem", dtype=Float32),
    ],
    online=True,
    source=pede_source,
)