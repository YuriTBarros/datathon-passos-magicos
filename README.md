# Datathon Passos Magicos

Projeto com pipeline de ML para estimar risco de evasao/defasagem, usando:
- Feast (feature store)
- Scikit-learn + Optuna (treino)
- MLflow (tracking e artefatos)
- FastAPI (inferencia e ingestao para re-treino)
- Docker (execucao da API)

## Estrutura principal

- `src/models/train.py`: treino, tuning, log no MLflow e geracao de bundle local para inferencia.
- `src/api/main.py`: endpoints de healthcheck, inferencia, ingestao para re-treino/drift e disparo de treino.
- `feature_repo/`: configuracao do Feast (`feature_store.yaml`, feature views).
- `models_artifacts/model_evasao_inferencia.joblib`: arquivo local do modelo para API.

## Pre-requisitos

- Python 3.12
- Docker + Docker Compose
- Feast CLI instalado (vem do `requirements.txt`)
- (Opcional) `jq` para formatar saidas JSON de comandos `curl`

## Fluxo completo (do zero ate inferencia)

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Aplicar definicoes do Feast:

```bash
feast -c feature_repo apply
```

3. Materializar online store (dados para inferencia online por `ra`):

```bash
feast -c feature_repo materialize-incremental $(date +%Y-%m-%d)
```

4. Treinar modelo e gerar bundle de inferencia:

```bash
python3 src/models/train.py
```

Saidas esperadas do treino:
- Modelo no MLflow (`model_evasao_final`)
- Arquivo local para inferencia: `models_artifacts/model_evasao_inferencia.joblib`
- Graficos: `confusion_matrix.png` e `feature_importance.png`

5. Subir API em container:

```bash
docker compose up --build
```

6. Validar API:

```bash
curl -s http://localhost:8008/health | jq .
```

7. Fazer inferencia por aluno:

```bash
curl -s -X POST http://localhost:8008/inference \
  -H "Content-Type: application/json" \
  -d '{"ra":"RA-1519","threshold":0.6}' | jq .
```

8. Enviar novos dados para buffer de re-treino e checagem de drift:

```bash
curl -s -X POST http://localhost:8008/retrain/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source":"api_manual",
    "detect_drift": true,
    "records":[
      {
        "ra":"RA-DRIFT-001",
        "event_timestamp":"2026-02-19T12:00:00Z",
        "inde":8.4,
        "cg":120.0,
        "cf":30.0,
        "ct":15.0,
        "mat":9.2,
        "por":8.9,
        "ing":9.1,
        "ida":9.0,
        "defasagem":0.0,
        "idade":14,
        "pedra_num":4,
        "ipv":9.4,
        "ian":9.0,
        "genero_masculino":1
      }
    ]
  }' | jq .
```

9. Disparar treino pela API (opcional):

```bash
curl -s -X POST http://localhost:8008/train/run | jq .
```

10. Acompanhar status do treino disparado:

```bash
curl -s http://localhost:8008/train/status | jq .
```

11. Derrubar containers:

```bash
docker compose down
```

## Endpoints da API

Base URL (local): `http://localhost:8008`

### `GET /health`

Verifica status da API, modelo e repo do Feast.

Exemplo:

```bash
curl -s http://localhost:8008/health
```

### `POST /inference`

Predicao por `ra` (a API busca features no Feast online store).

Request:

```json
{
  "ra": "RA-1286",
  "threshold": 0.5
}
```

Exemplo:

```bash
curl -s -X POST http://localhost:8008/inference \
  -H "Content-Type: application/json" \
  -d '{"ra":"RA-1286","threshold":0.5}'
```

Response (exemplo):

```json
{
  "ra": "RA-1286",
  "probabilidade_evasao": 0.73,
  "predicao": 1,
  "classe": "alto_risco",
  "threshold_usado": 0.5,
  "missing_features": []
}
```

### `POST /retrain/ingest`

Recebe novos registros, salva em buffer local e calcula drift basico via PSI.

Exemplo:

```bash
curl -s -X POST http://localhost:8008/retrain/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source":"api_manual",
    "detect_drift": true,
    "records":[
      {
        "ra":"RA-DRIFT-001",
        "event_timestamp":"2026-02-19T12:00:00Z",
        "inde":8.4,
        "cg":120.0,
        "cf":30.0,
        "ct":15.0,
        "mat":9.2,
        "por":8.9,
        "ing":9.1,
        "ida":9.0,
        "defasagem":0.0,
        "idade":14,
        "pedra_num":4,
        "ipv":9.4,
        "ian":9.0,
        "genero_masculino":1
      }
    ]
  }'
```

Buffer salvo em:
- `data/processed/retrain_buffer.parquet`

### `POST /train/run`

Dispara o `src/models/train.py` em background pela API.

Exemplo:

```bash
curl -s -X POST http://localhost:8008/train/run | jq .
```

### `GET /train/status`

Consulta estado do ultimo treino iniciado via API (`idle`, `running`, `success`, `failed`), com timestamps e `return_code`.

Exemplo:

```bash
curl -s http://localhost:8008/train/status | jq .
```

Log do treino via API:
- `data/processed/train_api.log`

## O que significa manter a online store atualizada

Endpoint `/inference` por `ra` consulta o Feast online store.
Se a online store estiver desatualizada, a inferencia usa valores antigos ou ausentes.

Atualize com:

```bash
feast -c feature_repo materialize-incremental $(date +%Y-%m-%d)
```

## Data drift (regra atual)

A API calcula PSI por feature numerica comum entre:
- base de referencia: `data/processed/dataset_final.parquet`
- novo lote recebido em `/retrain/ingest`

Interpretacao pratica:
- PSI < 0.10: estavel
- 0.10 <= PSI < 0.20: alerta
- PSI >= 0.20: drift relevante

## Rodar API sem Docker

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8008 --reload
```

## Troubleshooting rapido

Erro de feature view inexistente no Feast:
1. `feast -c feature_repo apply`
2. `feast -c feature_repo materialize-incremental $(date +%Y-%m-%d)`

Erro de modelo nao encontrado na API:
1. `python3 src/models/train.py`
2. confirmar arquivo em `models_artifacts/model_evasao_inferencia.joblib`

Erro em comandos `curl ... | jq`:
- Instalar `jq` ou remover `| jq .` dos comandos.
