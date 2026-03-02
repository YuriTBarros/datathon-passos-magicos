import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# --- adiciona root no sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# --- garante stubs antes de importar main ---
from tests._stubs import ensure_stub_modules
ensure_stub_modules()

from src.api import main as main_mod


class TestLoadBundle(unittest.TestCase):

    @patch("src.api.main.MODEL_BUNDLE_PATH")
    def test_load_bundle_raises_when_missing_file(self, mock_model_path):
        mock_model_path.exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            main_mod._load_bundle()

    @patch("src.api.main.joblib.load")
    @patch("src.api.main.MODEL_BUNDLE_PATH")
    def test_load_bundle_validates_keys(self, mock_model_path, mock_joblib_load):
        mock_model_path.exists.return_value = True
        mock_joblib_load.return_value = {"feature_columns": ["a"]}  # faltou "model"

        with self.assertRaises(ValueError):
            main_mod._load_bundle()

    @patch("src.api.main.joblib.load")
    @patch("src.api.main.MODEL_BUNDLE_PATH")
    def test_load_bundle_success(self, mock_model_path, mock_joblib_load):
        mock_model_path.exists.return_value = True
        bundle = {"model": object(), "feature_columns": ["a", "b"]}
        mock_joblib_load.return_value = bundle

        out = main_mod._load_bundle()
        self.assertEqual(out, bundle)


class TestAlignForModel(unittest.TestCase):

    def test_align_for_model_returns_df_and_missing(self):
        row = {"a": 1, "b": np.nan}  # b está NaN -> missing
        feature_columns = ["a", "b", "c"]  # c não existe -> missing

        x, missing = main_mod._align_for_model(row, feature_columns)

        self.assertIsInstance(x, pd.DataFrame)
        self.assertEqual(list(x.columns), feature_columns)
        self.assertEqual(missing, ["b", "c"])
        # preenchimento com 0
        self.assertEqual(float(x.loc[0, "b"]), 0.0)
        self.assertEqual(float(x.loc[0, "c"]), 0.0)


class TestCalculatePSI(unittest.TestCase):

    def test_calculate_psi_returns_none_when_not_enough_data(self):
        ref = pd.Series([1, 1, 1, 1])   # nunique < 5
        cur = pd.Series([1, 2])
        self.assertIsNone(main_mod._calculate_psi(ref, cur))

    def test_calculate_psi_returns_float(self):
        ref = pd.Series(np.linspace(0, 100, 200))
        cur = pd.Series(np.linspace(10, 90, 200))
        psi = main_mod._calculate_psi(ref, cur)
        self.assertIsInstance(psi, float)


class TestHealth(unittest.TestCase):

    @patch("src.api.main.FEAST_REPO_PATH")
    @patch("src.api.main.MODEL_BUNDLE_PATH")
    def test_health_degraded_when_not_ready(self, mock_model_path, mock_feast_repo):
        # MODEL_BUNDLE None por padrão
        mock_model_path.exists.return_value = False
        mock_feast_repo.joinpath.return_value.exists.return_value = False

        out = main_mod.health()
        self.assertEqual(out["status"], "degraded")
        self.assertFalse(out["model_ready"])
        self.assertFalse(out["feast_repo_ready"])

    @patch("src.api.main.FEAST_REPO_PATH")
    @patch("src.api.main.MODEL_BUNDLE_PATH")
    def test_health_ok_when_ready(self, mock_model_path, mock_feast_repo):
        main_mod.MODEL_BUNDLE = {"model": object(), "feature_columns": ["a"]}
        mock_model_path.exists.return_value = True
        mock_feast_repo.joinpath.return_value.exists.return_value = True

        out = main_mod.health()
        self.assertEqual(out["status"], "ok")
        self.assertTrue(out["model_ready"])
        self.assertTrue(out["feast_repo_ready"])


class TestInference(unittest.TestCase):

    @patch("src.api.main._fetch_features_for_ra")
    def test_inference_returns_response(self, mock_fetch):
        class FakeModel:
            def predict_proba(self, x):
                return np.array([[0.2, 0.8]])

        main_mod.MODEL_BUNDLE = {
            "model": FakeModel(),
            "feature_columns": ["inde", "cg"],
        }

        # retorna linha com 1 feature faltando pra exercitar missing_features
        mock_fetch.return_value = {"inde": 1.0, "cg": np.nan}

        payload = main_mod.InferenceRequest(ra="RA-1", threshold=0.6)
        resp = main_mod.inference(payload)

        self.assertEqual(resp.ra, "RA-1")
        self.assertAlmostEqual(resp.probabilidade_evasao, 0.8, places=6)
        self.assertEqual(resp.predicao, 1)  # 0.8 >= 0.6
        self.assertEqual(resp.classe, "alto_risco")
        self.assertEqual(resp.threshold_usado, 0.6)
        self.assertEqual(resp.missing_features, ["cg"])

    def test_inference_raises_when_model_not_loaded(self):
        main_mod.MODEL_BUNDLE = None
        payload = main_mod.InferenceRequest(ra="RA-1", threshold=0.5)
        with self.assertRaises(main_mod.HTTPException) as ctx:
            main_mod.inference(payload)
        self.assertEqual(ctx.exception.status_code, 500)


class TestIngestForRetraining(unittest.TestCase):

    @patch("src.api.main.RETRAIN_BUFFER_PATH")
    def test_ingest_requires_records(self, mock_buf_path):
        payload = main_mod.IngestRequest(source="api", detect_drift=True, records=[])
        with self.assertRaises(main_mod.HTTPException) as ctx:
            main_mod.ingest_for_retraining(payload)
        self.assertEqual(ctx.exception.status_code, 400)

    @patch("src.api.main.TRAIN_REFERENCE_PATH")
    @patch("src.api.main.RETRAIN_BUFFER_PATH")
    @patch("src.api.main.pd.DataFrame.to_parquet")
    @patch("src.api.main.pd.read_parquet")
    def test_ingest_writes_buffer_without_ref(
        self,
        mock_read_parquet,
        mock_to_parquet,
        mock_buf_path,
        mock_ref_path
    ):
        # simula buffer ainda não existe
        mock_buf_path.exists.return_value = False
        mock_buf_path.parent.mkdir = MagicMock()

        # sem referência -> drift_detected False
        mock_ref_path.exists.return_value = False

        payload = main_mod.IngestRequest(
            source="api",
            detect_drift=True,
            records=[{"ra": "RA-1", "x": 1}]
        )

        out = main_mod.ingest_for_retraining(payload)
        self.assertEqual(out["received_records"], 1)
        self.assertEqual(out["drift_report"]["drift_detected"], False)
        self.assertTrue(mock_to_parquet.called)


class TestTrainRunAndStatus(unittest.TestCase):

    @patch("src.api.main.threading.Thread")
    @patch("src.api.main.TRAIN_SCRIPT_PATH")
    def test_run_training_from_api_sets_running_and_starts_thread(self, mock_train_path, mock_thread):
        mock_train_path.exists.return_value = True
        thread_instance = MagicMock()
        mock_thread.return_value = thread_instance

        # garante estado inicial
        with main_mod.TRAIN_LOCK:
            main_mod.TRAIN_STATE["status"] = "idle"

        out = main_mod.run_training_from_api()

        self.assertIn("status_endpoint", out)
        self.assertEqual(out["status_endpoint"], "/train/status")
        thread_instance.start.assert_called_once()

        with main_mod.TRAIN_LOCK:
            self.assertEqual(main_mod.TRAIN_STATE["status"], "running")
            self.assertIsNotNone(main_mod.TRAIN_STATE["started_at_utc"])

    @patch("src.api.main.TRAIN_SCRIPT_PATH")
    def test_run_training_from_api_conflict_when_already_running(self, mock_train_path):
        mock_train_path.exists.return_value = True

        with main_mod.TRAIN_LOCK:
            main_mod.TRAIN_STATE["status"] = "running"

        with self.assertRaises(main_mod.HTTPException) as ctx:
            main_mod.run_training_from_api()

        self.assertEqual(ctx.exception.status_code, 409)

        with main_mod.TRAIN_LOCK:
            main_mod.TRAIN_STATE["status"] = "idle"

    def test_get_training_status_has_paths(self):
        out = main_mod.get_training_status()
        self.assertIn("log_path", out)
        self.assertIn("train_script_path", out)
        self.assertIn("status", out)


if __name__ == "__main__":
    unittest.main()