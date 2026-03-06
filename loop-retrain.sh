#!/bin/bash

while true; do
  echo "=============================="
  echo "Iniciando ciclo em: $(date)"
  echo "=============================="

  echo "1) Executando ingest..."
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
  echo
  sleep 5

  echo "2) Executando inference..."
  curl -s -X POST http://localhost:8008/inference \
    -H "Content-Type: application/json" \
    -d '{"ra":"RA-1519","threshold":0.6}' | jq .
  sleep 5

  echo "3) Executando train/run..."
  curl -s -X POST http://localhost:8008/train/run | jq .
  sleep 5

  echo "4) Consultando train/status..."
  curl -s http://localhost:8008/train/status | jq .
  sleep 5

done
