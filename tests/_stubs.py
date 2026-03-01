import sys
from types import ModuleType

def ensure_stub_modules():
    # =========================
    # feast stub
    # =========================
    if "feast" not in sys.modules:
        fake_feast = ModuleType("feast")

        class FeatureStore:
            def __init__(self, *args, **kwargs):
                pass

        fake_feast.FeatureStore = FeatureStore
        sys.modules["feast"] = fake_feast

    # =========================
    # mlflow stub (inclui mlflow.sklearn)
    # =========================
    if "mlflow" not in sys.modules:
        fake_mlflow = ModuleType("mlflow")

        fake_mlflow.set_tracking_uri = lambda *a, **k: None
        fake_mlflow.get_experiment_by_name = lambda *a, **k: None
        fake_mlflow.create_experiment = lambda *a, **k: None
        fake_mlflow.set_experiment = lambda *a, **k: None
        fake_mlflow.log_params = lambda *a, **k: None
        fake_mlflow.log_metric = lambda *a, **k: None
        fake_mlflow.log_artifact = lambda *a, **k: None

        class _Run:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb): return False

        fake_mlflow.start_run = lambda *a, **k: _Run()

        fake_mlflow_sklearn = ModuleType("mlflow.sklearn")
        fake_mlflow_sklearn.log_model = lambda *a, **k: None

        fake_mlflow.sklearn = fake_mlflow_sklearn
        sys.modules["mlflow"] = fake_mlflow
        sys.modules["mlflow.sklearn"] = fake_mlflow_sklearn

    # =========================
    # optuna stub
    # =========================
    if "optuna" not in sys.modules:
        fake_optuna = ModuleType("optuna")

        class _Study:
            best_params = {}
            best_value = 0.0
            def optimize(self, *a, **k): return None

        fake_optuna.create_study = lambda *a, **k: _Study()
        sys.modules["optuna"] = fake_optuna

    # =========================
    # fastapi stub (FastAPI + HTTPException)
    # =========================
    if "fastapi" not in sys.modules:
        fake_fastapi = ModuleType("fastapi")

        class HTTPException(Exception):
            def __init__(self, status_code: int, detail: str):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        class FastAPI:
            def __init__(self, *args, **kwargs):
                pass

            # decorators que retornam a func original
            def get(self, *args, **kwargs):
                def deco(fn): return fn
                return deco

            def post(self, *args, **kwargs):
                def deco(fn): return fn
                return deco

            def on_event(self, *args, **kwargs):
                def deco(fn): return fn
                return deco

        fake_fastapi.FastAPI = FastAPI
        fake_fastapi.HTTPException = HTTPException
        sys.modules["fastapi"] = fake_fastapi

    # =========================
    # pydantic stub (BaseModel + Field)
    # =========================
    if "pydantic" not in sys.modules:
        fake_pydantic = ModuleType("pydantic")

        class BaseModel:
            def __init__(self, **data):
                for k, v in data.items():
                    setattr(self, k, v)

        def Field(default, **kwargs):
            # Em tests unitários, não precisamos validar constraints;
            # o FastAPI/Pydantic real faria isso.
            return default

        fake_pydantic.BaseModel = BaseModel
        fake_pydantic.Field = Field
        sys.modules["pydantic"] = fake_pydantic

    # =========================
    # prometheus_fastapi_instrumentator stub
    # =========================
    if "prometheus_fastapi_instrumentator" not in sys.modules:
        fake_prom = ModuleType("prometheus_fastapi_instrumentator")

        class Instrumentator:
            def instrument(self, app):
                return self
            def expose(self, app, endpoint="/metrics"):
                return None

        fake_prom.Instrumentator = Instrumentator
        sys.modules["prometheus_fastapi_instrumentator"] = fake_prom

    # =========================
    # prometheus_client stub (Gauge + start_http_server)
    # =========================
    if "prometheus_client" not in sys.modules:
        fake_prom = ModuleType("prometheus_client")

        class _GaugeChild:
            def __init__(self, parent, label_values):
                self._parent = parent
                self._label_values = tuple(label_values)

            def set(self, value):
                self._parent._labeled[self._label_values] = value

        class Gauge:
            def __init__(self, name, documentation, labelnames=None):
                self.name = name
                self.documentation = documentation
                self.labelnames = list(labelnames) if labelnames else []
                self._value = None
                self._labeled = {}  # {(label_values...): value}

            def set(self, value):
                self._value = value

            def labels(self, *label_values):
                return _GaugeChild(self, label_values)

            def clear(self):
                self._labeled.clear()

        fake_prom.Gauge = Gauge
        fake_prom.start_http_server = lambda *a, **k: None

        sys.modules["prometheus_client"] = fake_prom