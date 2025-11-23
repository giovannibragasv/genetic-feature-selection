import numpy as np
from typing import Literal
from collections import Counter


class KNNClassifier:
    """
    Classificador K-Nearest Neighbors implementado do zero.
    Suporta múltiplas métricas de distância conforme metodologia do artigo.
    """

    def __init__(
        self,
        k: int = 7,
        distance_metric: Literal["euclidean", "manhattan", "chebyshev"] = "euclidean",
    ):
        """
        Inicializa o classificador KNN.

        Args:
            k (int): Número de vizinhos (padrão: 7, ótimo conforme artigo).
            distance_metric (str): Métrica de distância.
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """
        Treina o classificador (armazena dados de treino).

        Args:
            X (np.ndarray): Features de treino.
            y (np.ndarray): Labels de treino.

        Returns:
            KNNClassifier: Self para encadeamento.
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        return self
