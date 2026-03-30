import joblib
import pandas as pd
from abc import ABC, abstractmethod

class PredictionStrategy(ABC):
    """
    Patrón Strategy para la predicción del modelo.
    Permite intercambiar fácilmente el modelo subyacente o la librería (por ejemplo, de sklearn a xgboost).
    """
    @abstractmethod
    def predict(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def predict_proba(self, data: pd.DataFrame):
        pass

class JoblibModelStrategy(PredictionStrategy):
    """
    Implementación concreta para modelos de scikit-learn guardados con joblib.
    """
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, data: pd.DataFrame):
        return self.model.predict(data)

    def predict_proba(self, data: pd.DataFrame):
        return self.model.predict_proba(data)
