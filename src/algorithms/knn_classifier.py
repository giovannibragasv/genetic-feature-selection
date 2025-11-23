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

    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calcula distância entre dois pontos.

        Args:
            x1 (np.ndarray): Ponto 1.
            x2 (np.ndarray): Ponto 2.

        Returns:
            float: Distância calculada.
        """
        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == "chebyshev":
            return np.max(np.abs(x1 - x2))
        else:
            raise ValueError(f"Métrica desconhecida: {self.distance_metric}")

    def _predict_single(self, x: np.ndarray) -> int:
        """
        Prediz classe para uma única amostra.

        Args:
            x (np.ndarray): Amostra.

        Returns:
            int: Classe predita.
        """
        distances = np.array([self._compute_distance(x, x_train) for x_train in self.X_train])

        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = self.y_train[k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz classes para múltiplas amostras.

        Args:
            X (np.ndarray): Features de teste.

        Returns:
            np.ndarray: Classes preditas.
        """
        if self.X_train is None:
            raise ValueError("Modelo não treinado. Chame fit() primeiro.")

        return np.array([self._predict_single(x) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula acurácia no conjunto de teste.

        Args:
            X (np.ndarray): Features de teste.
            y (np.ndarray): Labels verdadeiras.

        Returns:
            float: Acurácia (0 a 1).
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna probabilidades de classe (proporção nos k vizinhos).

        Args:
            X (np.ndarray): Features de teste.

        Returns:
            np.ndarray: Matriz de probabilidades (n_samples, n_classes).
        """
        if self.X_train is None:
            raise ValueError("Modelo não treinado. Chame fit() primeiro.")

        n_classes = len(np.unique(self.y_train))
        probas = []

        for x in X:
            distances = np.array([self._compute_distance(x, x_train) for x_train in self.X_train])

            k_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = self.y_train[k_indices]

            class_counts = np.bincount(k_nearest_labels, minlength=n_classes)
            class_probs = class_counts / self.k
            probas.append(class_probs)

        return np.array(probas)
