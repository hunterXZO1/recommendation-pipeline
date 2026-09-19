from abc import ABC, abstractmethod
from pathlib import Path


class ModelTrainer(ABC):

    @abstractmethod
    def train(self, data):
        """
        Train a challenger model using recent interaction data.
        """
        pass

    @abstractmethod
    def save(self, model, path: str | Path):
        """
        Save a trained model.
        """
        pass

    @abstractmethod
    def load(self, path: str | Path):
        """
        Load a trained model.
        """
        pass