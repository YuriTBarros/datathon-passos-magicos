FROM python:3.11-slim

WORKDIR /app

RUN pip install prometheus_client

COPY mlflow_exporter.py /app/mlflow_exporter.py

CMD ["python", "mlflow_exporter.py"]