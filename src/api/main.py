from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import threading
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from feast import FeatureStore
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEAST_REPO_PATH = PROJECT_ROOT / "feature_repo"
MODEL_BUNDLE_PATH = PROJECT_ROOT / "models_artifacts" / "model_evasao_inferencia.joblib"
TRAIN_REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_final.parquet"
RETRAIN_BUFFER_PATH = PROJECT_ROOT / "data" / "processed" / "retrain_buffer.parquet"
TRAIN_SCRIPT_PATH = PROJECT_ROOT / "src" / "models" / "train.py"
TRAIN_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "train_api.log"

# Mesmo conjunto usado no treino (Feature Store)
FEATURE_REFS = [
    "academic_features:inde",
    "academic_features:cg",
    "academic_features:cf",
    "academic_features:ct",
    "academic_features:mat",
    "academic_features:por",
    "academic_features:ing",
    "academic_features:ida",
    "academic_features:defasagem",
    "profile_behavior_features:idade",
    "profile_behavior_features:pedra_num",
    "profile_behavior_features:ipv",
    "profile_behavior_features:ian",
    "profile_behavior_features:genero_masculino",
]


app = FastAPI(title="Passos Magicos Inference API", version="1.0.0")

MODEL_BUNDLE: dict[str, Any] | None = None
TRAIN_LOCK = threading.Lock()
TRAIN_STATE: dict[str, Any] = {
    "status": "idle",
    "started_at_utc": None,
    "finished_at_utc": None,
    "return_code": None,
    "message": "Nenhum treino executado via API.",
}


class InferenceRequest(BaseModel):
    ra: str = Field(..., description="Identificador do aluno")
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class InferenceResponse(BaseModel):
    ra: str
    probabilidade_evasao: float
    predicao: int
    classe: str
    threshold_usado: float
    missing_features: list[str]


class IngestRequest(BaseModel):
    source: str = Field("api", description="Origem dos dados")
    detect_drift: bool = True
    records: list[dict[str, Any]]


def _run_training_process() -> None:
    global MODEL_BUNDLE
    completed = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT_PATH)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with TRAIN_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n===== TRAIN RUN {timestamp} =====\n")
        f.write(completed.stdout or "")
        if completed.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(completed.stderr)

    with TRAIN_LOCK:
        TRAIN_STATE["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        TRAIN_STATE["return_code"] = completed.returncode
        if completed.returncode == 0:
            TRAIN_STATE["status"] = "success"
            TRAIN_STATE["message"] = "Treino finalizado com sucesso."
            try:
                MODEL_BUNDLE = _load_bundle()
            except Exception as exc:
                TRAIN_STATE["message"] = (
                    "Treino concluiu, mas falhou ao recarregar modelo na API: " f"{exc}"
                )
        else:
            TRAIN_STATE["status"] = "failed"
            TRAIN_STATE["message"] = "Treino finalizado com erro. Consulte o log."


def _start_training_background() -> None:
    try:
        _run_training_process()
    except Exception as exc:
        with TRAIN_LOCK:
            TRAIN_STATE["status"] = "failed"
            TRAIN_STATE["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            TRAIN_STATE["message"] = f"Erro inesperado no treino: {exc}"


def _load_bundle() -> dict[str, Any]:
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Modelo de inferencia nao encontrado em: {MODEL_BUNDLE_PATH}. Rode o treino primeiro."
        )
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    if "model" not in bundle or "feature_columns" not in bundle:
        raise ValueError("Bundle invalido: chaves 'model' e 'feature_columns' sao obrigatorias.")
    return bundle


def _fetch_features_for_ra(ra: str) -> dict[str, Any]:
    store = FeatureStore(repo_path=str(FEAST_REPO_PATH))
    online = store.get_online_features(features=FEATURE_REFS, entity_rows=[{"ra": ra}]).to_dict()

    row: dict[str, Any] = {}
    for col, values in online.items():
        clean_name = col.split(":")[-1] if ":" in col else col
        row[clean_name] = values[0] if values else np.nan
    return row


def _align_for_model(row: dict[str, Any], feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    missing = [col for col in feature_columns if col not in row or pd.isna(row[col])]
    x = pd.DataFrame([{col: row.get(col, 0) for col in feature_columns}]).fillna(0)
    return x, missing


def _calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float | None:
    ref = reference.replace([np.inf, -np.inf], np.nan).dropna()
    cur = current.replace([np.inf, -np.inf], np.nan).dropna()
    if ref.empty or cur.empty or ref.nunique() < 5 or cur.nunique() < 2:
        return None

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        return None

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.where(ref_counts == 0, 1e-6, ref_counts / ref_counts.sum())
    cur_pct = np.where(cur_counts == 0, 1e-6, cur_counts / cur_counts.sum())
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


@app.on_event("startup")
def startup_event() -> None:
    global MODEL_BUNDLE
    MODEL_BUNDLE = _load_bundle()


@app.get("/health")
def health() -> dict[str, Any]:
    model_ready = MODEL_BUNDLE is not None and MODEL_BUNDLE_PATH.exists()
    feast_ready = FEAST_REPO_PATH.joinpath("feature_store.yaml").exists()
    return {
        "status": "ok" if model_ready and feast_ready else "degraded",
        "model_ready": model_ready,
        "model_path": str(MODEL_BUNDLE_PATH),
        "feast_repo_ready": feast_ready,
        "feast_repo_path": str(FEAST_REPO_PATH),
        "utc_now": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/inference", response_model=InferenceResponse)
def inference(payload: InferenceRequest) -> InferenceResponse:
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=500, detail="Modelo nao carregado.")

    try:
        raw_row = _fetch_features_for_ra(payload.ra)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha ao buscar features no Feast: {exc}") from exc

    feature_columns = MODEL_BUNDLE["feature_columns"]
    model = MODEL_BUNDLE["model"]
    x, missing_features = _align_for_model(raw_row, feature_columns)

    try:
        prob = float(model.predict_proba(x)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha na inferencia: {exc}") from exc

    pred = int(prob >= payload.threshold)
    classe = "alto_risco" if pred == 1 else "baixo_risco"
    return InferenceResponse(
        ra=payload.ra,
        probabilidade_evasao=prob,
        predicao=pred,
        classe=classe,
        threshold_usado=payload.threshold,
        missing_features=missing_features,
    )


@app.post("/retrain/ingest")
def ingest_for_retraining(payload: IngestRequest) -> dict[str, Any]:
    if not payload.records:
        raise HTTPException(status_code=400, detail="Envie ao menos 1 registro em 'records'.")

    df_new = pd.DataFrame(payload.records)
    if "ra" not in df_new.columns:
        raise HTTPException(status_code=400, detail="Cada registro precisa conter 'ra'.")
    if "event_timestamp" not in df_new.columns:
        df_new["event_timestamp"] = datetime.now(timezone.utc).isoformat()
    df_new["ingest_source"] = payload.source
    df_new["ingested_at_utc"] = datetime.now(timezone.utc).isoformat()

    RETRAIN_BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RETRAIN_BUFFER_PATH.exists():
        df_old = pd.read_parquet(RETRAIN_BUFFER_PATH)
        df_buffer = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_buffer = df_new.copy()
    df_buffer.to_parquet(RETRAIN_BUFFER_PATH, index=False)

    drift_report: dict[str, Any] = {"enabled": payload.detect_drift, "feature_psi": {}}
    if payload.detect_drift and TRAIN_REFERENCE_PATH.exists():
        df_ref = pd.read_parquet(TRAIN_REFERENCE_PATH)
        numeric_cols = [c for c in df_new.columns if c in df_ref.columns]
        numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df_ref[c])]
        for col in numeric_cols:
            psi = _calculate_psi(df_ref[col], df_new[col])
            if psi is not None:
                drift_report["feature_psi"][col] = psi
        drift_report["drift_detected"] = any(v > 0.2 for v in drift_report["feature_psi"].values())
    else:
        drift_report["drift_detected"] = False

    return {
        "message": "Registros recebidos para re-treino.",
        "received_records": len(df_new),
        "buffer_total_records": len(df_buffer),
        "buffer_path": str(RETRAIN_BUFFER_PATH),
        "drift_report": drift_report,
    }


@app.post("/train/run")
def run_training_from_api() -> dict[str, Any]:
    if not TRAIN_SCRIPT_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Arquivo de treino nao encontrado: {TRAIN_SCRIPT_PATH}")

    with TRAIN_LOCK:
        if TRAIN_STATE["status"] == "running":
            raise HTTPException(status_code=409, detail="Ja existe um treino em execucao.")
        TRAIN_STATE["status"] = "running"
        TRAIN_STATE["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        TRAIN_STATE["finished_at_utc"] = None
        TRAIN_STATE["return_code"] = None
        TRAIN_STATE["message"] = "Treino iniciado pela API."

    thread = threading.Thread(target=_start_training_background, daemon=True)
    thread.start()

    return {
        "message": "Treino disparado com sucesso.",
        "status_endpoint": "/train/status",
        "log_path": str(TRAIN_LOG_PATH),
    }


@app.get("/train/status")
def get_training_status() -> dict[str, Any]:
    with TRAIN_LOCK:
        state = dict(TRAIN_STATE)
    state["log_path"] = str(TRAIN_LOG_PATH)
    state["train_script_path"] = str(TRAIN_SCRIPT_PATH)
    return state
