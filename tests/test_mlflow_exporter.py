import unittest
from unittest.mock import patch
import sys
from pathlib import Path

# root no path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests._stubs import ensure_stub_modules
ensure_stub_modules()

# IMPORTANTE: ajuste esse import para o caminho real do arquivo no seu repo
# Exemplo esperado:
from monitoring.exporters import mlflow_exporter as exp


class FakeCursor:
    """
    Cursor fake que retorna resultados diferentes dependendo da última query executada.
    """
    def __init__(self):
        self.last_sql = None

    def execute(self, sql):
        self.last_sql = " ".join(sql.split()).strip()  # normaliza espaços
        return None

    def fetchone(self):
        sql = self.last_sql

        if sql == "SELECT COUNT(*) FROM runs":
            return (10,)
        if sql == "SELECT COUNT(*) FROM runs WHERE status='FINISHED'":
            return (7,)
        if sql == "SELECT COUNT(*) FROM runs WHERE status='FAILED'":
            return (2,)
        if sql == "SELECT COUNT(*) FROM runs WHERE status='RUNNING'":
            return (1,)

        raise AssertionError(f"fetchone() não esperado para SQL: {sql}")

    def fetchall(self):
        sql = self.last_sql

        if "FROM runs GROUP BY status" in sql:
            return [("FINISHED", 7), ("FAILED", 2), ("RUNNING", 1)]

        if "LEFT JOIN runs r ON e.experiment_id = r.experiment_id" in sql:
            return [("Exp A", 5), ("Exp B", 5)]

        if "GROUP BY e.name, r.status" in sql:
            return [("Exp A", "FINISHED", 4), ("Exp A", "FAILED", 1), ("Exp B", "FINISHED", 3), ("Exp B", "RUNNING", 2)]

        if "MAX(m.value)" in sql:
            return [("Exp A", "accuracy", 0.91), ("Exp A", "f1", 0.77), ("Exp B", "accuracy", 0.88)]

        if "FROM runs GROUP BY run_date" in sql:
            return [("2026-02-20", 3), ("2026-02-21", 7), (None, 99)]  # None deve ser ignorado

        raise AssertionError(f"fetchall() não esperado para SQL: {sql}")


class FakeConn:
    def __init__(self):
        self._cursor = FakeCursor()

    def cursor(self):
        return self._cursor

    def close(self):
        return None


class TestMLflowExporterCollect(unittest.TestCase):

    def setUp(self):
        # limpa gauges antes de cada teste
        exp.mlflow_runs_by_status.clear()
        exp.mlflow_runs_per_experiment.clear()
        exp.mlflow_runs_by_experiment_status.clear()
        exp.mlflow_best_metric_per_experiment.clear()
        exp.mlflow_best_accuracy.clear()
        exp.mlflow_runs_per_day.clear()

        # zera valores
        exp.mlflow_total_runs.set(None)
        exp.mlflow_finished_runs.set(None)
        exp.mlflow_failed_runs.set(None)
        exp.mlflow_running_runs.set(None)
        exp.mlflow_db_size_bytes.set(None)

    @patch("monitoring.exporters.mlflow_exporter.os.path.getsize", return_value=12345)
    @patch("monitoring.exporters.mlflow_exporter.os.path.exists", return_value=True)
    @patch("monitoring.exporters.mlflow_exporter.sqlite3.connect", return_value=FakeConn())
    def test_collect_updates_gauges(self, mock_connect, mock_exists, mock_getsize):
        exp.collect()

        # globais
        self.assertEqual(exp.mlflow_total_runs._value, 10)
        self.assertEqual(exp.mlflow_finished_runs._value, 7)
        self.assertEqual(exp.mlflow_failed_runs._value, 2)
        self.assertEqual(exp.mlflow_running_runs._value, 1)

        # by status (labels)
        self.assertEqual(exp.mlflow_runs_by_status._labeled[(("FINISHED",))], 7)
        self.assertEqual(exp.mlflow_runs_by_status._labeled[(("FAILED",))], 2)
        self.assertEqual(exp.mlflow_runs_by_status._labeled[(("RUNNING",))], 1)

        # runs por experimento
        self.assertEqual(exp.mlflow_runs_per_experiment._labeled[(("Exp A",))], 5)
        self.assertEqual(exp.mlflow_runs_per_experiment._labeled[(("Exp B",))], 5)

        # runs por experimento + status
        self.assertEqual(exp.mlflow_runs_by_experiment_status._labeled[(("Exp A", "FINISHED"))], 4)
        self.assertEqual(exp.mlflow_runs_by_experiment_status._labeled[(("Exp A", "FAILED"))], 1)

        # melhores métricas (labels exp, metric)
        self.assertEqual(exp.mlflow_best_metric_per_experiment._labeled[(("Exp A", "accuracy"))], 0.91)
        self.assertEqual(exp.mlflow_best_metric_per_experiment._labeled[(("Exp A", "f1"))], 0.77)

        # best accuracy por experimento
        self.assertEqual(exp.mlflow_best_accuracy._labeled[(("Exp A",))], 0.91)
        self.assertEqual(exp.mlflow_best_accuracy._labeled[(("Exp B",))], 0.88)

        # runs por dia (None ignorado)
        self.assertEqual(exp.mlflow_runs_per_day._labeled[(("2026-02-20",))], 3)
        self.assertEqual(exp.mlflow_runs_per_day._labeled[(("2026-02-21",))], 7)
        self.assertFalse(((None,),) in exp.mlflow_runs_per_day._labeled)

        # db size
        self.assertEqual(exp.mlflow_db_size_bytes._value, 12345)

    @patch("monitoring.exporters.mlflow_exporter.sqlite3.connect", side_effect=Exception("boom"))
    def test_collect_handles_exceptions(self, mock_connect):
        # Não deve levantar exceção (o exporter faz try/except e printa)
        exp.collect()


if __name__ == "__main__":
    unittest.main()