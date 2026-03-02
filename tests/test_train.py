import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests._stubs import ensure_stub_modules
ensure_stub_modules()

from src.models import train as train_mod


# ==========================================================
# TESTE get_training_data
# ==========================================================

class TestGetTrainingData(unittest.TestCase):

    @patch("src.models.train.FeatureStore")
    @patch("src.models.train.pd.read_parquet")
    def test_get_training_data_fills_na(self, mock_read_parquet, mock_feature_store):

        entity_df = pd.DataFrame({
            "ra": ["RA-1"],
            "event_timestamp": ["2026-02-01T00:00:00Z"]
        })

        feast_df = pd.DataFrame({
            "ra": ["RA-1"],
            "event_timestamp": ["2026-02-01T00:00:00Z"],
            "inde": [None],
            "evadiu": [0],
            "ano": [2025]
        })

        mock_read_parquet.return_value = entity_df

        mock_store_instance = MagicMock()
        mock_feature_store.return_value = mock_store_instance

        mock_hist = MagicMock()
        mock_hist.to_df.return_value = feast_df
        mock_store_instance.get_historical_features.return_value = mock_hist

        result = train_mod.get_training_data()

        self.assertEqual(result["inde"].iloc[0], 0)
        mock_feature_store.assert_called_once_with(repo_path=train_mod.FEAST_REPO_PATH)


# ==========================================================
# TESTE objective
# ==========================================================

class TestObjective(unittest.TestCase):

    @patch("src.models.train.RandomForestClassifier")
    @patch("src.models.train.f1_score")
    def test_objective_returns_f1(self, mock_f1, mock_rf):

        class FakeTrial:
            def suggest_int(self, name, low, high):
                return low

        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 1, 1]
        mock_rf.return_value = mock_model

        mock_f1.return_value = 0.85

        trial = FakeTrial()

        X_train = np.zeros((5, 2))
        y_train = np.array([0, 1, 0, 1, 1])
        X_test = np.zeros((3, 2))
        y_test = np.array([1, 0, 1])

        score = train_mod.objective(trial, X_train, y_train, X_test, y_test)

        self.assertEqual(score, 0.85)
        mock_model.fit.assert_called_once()


# ==========================================================
# TESTE setup_mlflow_experiment
# ==========================================================

class TestMLflowSetup(unittest.TestCase):

    @patch("src.models.train.mlflow.create_experiment")
    @patch("src.models.train.mlflow.get_experiment_by_name")
    @patch("src.models.train.mlflow.set_tracking_uri")
    def test_create_experiment_if_not_exists(
        self,
        mock_set_tracking,
        mock_get_exp,
        mock_create_exp
    ):
        mock_get_exp.return_value = None

        exp_name = train_mod.setup_mlflow_experiment()

        mock_create_exp.assert_called_once()
        self.assertEqual(exp_name, train_mod.EXPERIMENT_NAME)


# ==========================================================
# TESTE run_training (orquestração)
# ==========================================================

class TestRunTraining(unittest.TestCase):

    @patch("src.models.train.joblib.dump")
    @patch("src.models.train.mlflow.sklearn.log_model")
    @patch("src.models.train.mlflow.log_metric")
    @patch("src.models.train.mlflow.log_params")
    @patch("src.models.train.mlflow.set_experiment")
    @patch("src.models.train.setup_mlflow_experiment")
    @patch("src.models.train.optuna.create_study")
    @patch("src.models.train.RandomForestClassifier")
    @patch("src.models.train.classification_report")
    @patch("src.models.train.confusion_matrix")
    @patch("src.models.train.sns.heatmap")
    @patch("src.models.train.plt.savefig")
    @patch("src.models.train.train_test_split")
    @patch("src.models.train.get_training_data")
    def test_run_training_orchestration(
        self,
        mock_get_training_data,
        mock_train_test_split,
        mock_savefig,
        mock_heatmap,
        mock_confusion_matrix,
        mock_classification_report,
        mock_rf,
        mock_create_study,
        mock_setup_mlflow_exp,
        mock_set_experiment,
        mock_log_params,
        mock_log_metric,
        mock_log_model,
        mock_joblib_dump
    ):

        # -------- dataset fake --------
        df = pd.DataFrame({
            "ra": ["RA-1","RA-2","RA-3","RA-4","RA-5"],
            "event_timestamp": [1,2,3,4,5],
            "ano": [2025]*5,
            "evadiu": [0,1,0,1,1],
            "f1": [10,11,12,13,14],
            "f2": [0,1,0,1,0],
        })
        mock_get_training_data.return_value = df

        # -------- split fake --------
        def fake_split(X, y, test_size, random_state, stratify):
            self.assertTrue(stratify is y)
            return X.iloc[:3], X.iloc[3:], y.iloc[:3], y.iloc[3:]
        mock_train_test_split.side_effect = fake_split

        # -------- optuna fake --------
        fake_study = MagicMock()
        fake_study.best_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        }
        fake_study.best_value = 0.88
        mock_create_study.return_value = fake_study

        # -------- modelo fake --------
        model = MagicMock()
        model.predict.return_value = [1, 1]
        model.feature_importances_ = [0.2, 0.8]
        mock_rf.return_value = model

        mock_classification_report.return_value = {
            "1.0": {"f1-score": 0.66, "recall": 0.55, "precision": 0.77}
        }
        mock_confusion_matrix.return_value = [[1, 0], [0, 1]]
        mock_setup_mlflow_exp.return_value = "EXP_TEST"

        # -------- executa --------
        train_mod.run_training()

        # -------- asserts --------
        mock_get_training_data.assert_called_once()
        mock_create_study.assert_called_once_with(direction="maximize")
        fake_study.optimize.assert_called_once()
        mock_log_params.assert_called_once_with(fake_study.best_params)

        self.assertTrue(mock_joblib_dump.called)
        dumped_obj = mock_joblib_dump.call_args[0][0]

        self.assertIn("model", dumped_obj)
        self.assertIn("feature_columns", dumped_obj)
        self.assertEqual(dumped_obj["target_column"], "evadiu")
        self.assertEqual(dumped_obj["default_threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()