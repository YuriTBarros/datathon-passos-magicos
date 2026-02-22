import sqlite3
import time
import os
from datetime import datetime
from prometheus_client import start_http_server, Gauge

DB_PATH = "/app/mlflow.db"

# ===============================
# 🔹 RUN STATUS METRICS
# ===============================

mlflow_total_runs = Gauge("mlflow_total_runs", "Total MLflow runs")
mlflow_finished_runs = Gauge("mlflow_finished_runs", "Finished runs")
mlflow_failed_runs = Gauge("mlflow_failed_runs", "Failed runs")
mlflow_running_runs = Gauge("mlflow_running_runs", "Running runs")

mlflow_runs_by_status = Gauge(
    "mlflow_runs_by_status",
    "Runs grouped by status",
    ["status"]
)

# ===============================
# 🔹 EXPERIMENT METRICS
# ===============================

mlflow_runs_per_experiment = Gauge(
    "mlflow_runs_per_experiment",
    "Runs per experiment",
    ["experiment"]
)

mlflow_runs_by_experiment_status = Gauge(
    "mlflow_runs_by_experiment_status",
    "Runs per experiment grouped by status",
    ["experiment", "status"]
)

mlflow_best_metric_per_experiment = Gauge(
    "mlflow_best_metric_per_experiment",
    "Best metric value per experiment",
    ["experiment", "metric"]
)

mlflow_best_accuracy = Gauge(
    "mlflow_best_accuracy",
    "Best accuracy per experiment",
    ["experiment"]
)

# ===============================
# 🔹 TIME SERIES METRICS
# ===============================

mlflow_runs_per_day = Gauge(
    "mlflow_runs_per_day",
    "Runs per day",
    ["date"]
)

# ===============================
# 🔹 DATABASE METRICS
# ===============================

mlflow_db_size_bytes = Gauge(
    "mlflow_db_size_bytes",
    "MLflow SQLite database size in bytes"
)


def collect():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # ===============================
        # 🔹 GLOBAL STATUS COUNTS
        # ===============================

        cursor.execute("SELECT COUNT(*) FROM runs")
        mlflow_total_runs.set(cursor.fetchone()[0])

        cursor.execute("SELECT COUNT(*) FROM runs WHERE status='FINISHED'")
        mlflow_finished_runs.set(cursor.fetchone()[0])

        cursor.execute("SELECT COUNT(*) FROM runs WHERE status='FAILED'")
        mlflow_failed_runs.set(cursor.fetchone()[0])

        cursor.execute("SELECT COUNT(*) FROM runs WHERE status='RUNNING'")
        mlflow_running_runs.set(cursor.fetchone()[0])

        mlflow_runs_by_status.clear()
        cursor.execute("""
            SELECT status, COUNT(*)
            FROM runs
            GROUP BY status
        """)
        for status, count in cursor.fetchall():
            mlflow_runs_by_status.labels(status).set(count)

        # ===============================
        # 🔹 RUNS POR EXPERIMENTO
        # ===============================

        mlflow_runs_per_experiment.clear()
        mlflow_runs_by_experiment_status.clear()

        cursor.execute("""
            SELECT e.name, COUNT(r.run_uuid)
            FROM experiments e
            LEFT JOIN runs r ON e.experiment_id = r.experiment_id
            GROUP BY e.name
        """)
        for exp, count in cursor.fetchall():
            mlflow_runs_per_experiment.labels(exp).set(count)

        cursor.execute("""
            SELECT e.name, r.status, COUNT(r.run_uuid)
            FROM experiments e
            JOIN runs r ON e.experiment_id = r.experiment_id
            GROUP BY e.name, r.status
        """)
        for exp, status, count in cursor.fetchall():
            mlflow_runs_by_experiment_status.labels(exp, status).set(count)

        # ===============================
        # 🔹 MELHORES MÉTRICAS
        # ===============================

        mlflow_best_metric_per_experiment.clear()
        mlflow_best_accuracy.clear()

        cursor.execute("""
            SELECT e.name, m.key, MAX(m.value)
            FROM experiments e
            JOIN runs r ON e.experiment_id = r.experiment_id
            JOIN metrics m ON r.run_uuid = m.run_uuid
            GROUP BY e.name, m.key
        """)
        for exp, metric, value in cursor.fetchall():
            mlflow_best_metric_per_experiment.labels(exp, metric).set(value)

            if metric.lower() == "accuracy":
                mlflow_best_accuracy.labels(exp).set(value)

        # ===============================
        # 🔹 RUNS POR DIA (TIME SERIES)
        # ===============================

        mlflow_runs_per_day.clear()

        cursor.execute("""
            SELECT DATE(start_time/1000, 'unixepoch') as run_date,
                   COUNT(*)
            FROM runs
            GROUP BY run_date
        """)
        for date, count in cursor.fetchall():
            if date:
                mlflow_runs_per_day.labels(date).set(count)

        conn.close()

        # ===============================
        # 🔹 DB SIZE
        # ===============================

        if os.path.exists(DB_PATH):
            mlflow_db_size_bytes.set(os.path.getsize(DB_PATH))

    except Exception as e:
        print(f"Exporter error: {e}")


if __name__ == "__main__":
    print("Starting MLflow exporter on :8001")
    start_http_server(8001)

    while True:
        collect()
        time.sleep(15)