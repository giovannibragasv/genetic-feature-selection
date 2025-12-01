import numpy as np
from typing import Literal
from collections import Counter


class KNNClassifier:
    """
    Classificador K-Nearest Neighbors implementado do zero.
    Versão otimizada com operações vetorizadas numpy.
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
        self._n_classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """
        Treina o classificador (armazena dados de treino).

        Args:
            X (np.ndarray): Features de treino.
            y (np.ndarray): Labels de treino.

        Returns:
            KNNClassifier: Self para encadeamento.
        """
        self.X_train = np.ascontiguousarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.int32)
        self._n_classes = len(np.unique(self.y_train))
        return self

    def _compute_distances_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula distâncias de todas as amostras de teste para todas de treino.
        Versão vetorizada usando broadcasting.

        Args:
            X (np.ndarray): Amostras de teste (n_test, n_features).

        Returns:
            np.ndarray: Matriz de distâncias (n_test, n_train).
        """
        if self.distance_metric == "euclidean":
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            # Mais eficiente que calcular diferenças diretamente
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (n_test, 1)
            train_sq = np.sum(self.X_train ** 2, axis=1)  # (n_train,)
            cross = X @ self.X_train.T  # (n_test, n_train)
            
            distances_sq = X_sq + train_sq - 2 * cross
            # Corrigir pequenos valores negativos por erro numérico
            distances_sq = np.maximum(distances_sq, 0)
            return np.sqrt(distances_sq)
        
        elif self.distance_metric == "manhattan":
            # Usa broadcasting: (n_test, 1, n_features) - (n_train, n_features)
            diff = np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :])
            return np.sum(diff, axis=2)
        
        elif self.distance_metric == "chebyshev":
            diff = np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :])
            return np.max(diff, axis=2)
        
        else:
            raise ValueError(f"Métrica desconhecida: {self.distance_metric}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz classes para múltiplas amostras.
        Versão vetorizada.

        Args:
            X (np.ndarray): Features de teste (n_samples, n_features).

        Returns:
            np.ndarray: Classes preditas (n_samples,).
        """
        if self.X_train is None:
            raise ValueError("Modelo não treinado. Chame fit() primeiro.")

        X = np.asarray(X, dtype=np.float64)
        
        # Calcular todas as distâncias de uma vez
        distances = self._compute_distances_batch(X)  # (n_test, n_train)
        
        # Pegar índices dos k menores (argpartition é O(n) vs O(n log n) do argsort)
        # Para datasets pequenos, argsort pode ser mais rápido
        if self.X_train.shape[0] > 100:
            k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        else:
            k_indices = np.argsort(distances, axis=1)[:, :self.k]
        
        # Pegar labels dos k vizinhos mais próximos
        k_nearest_labels = self.y_train[k_indices]  # (n_test, k)
        
        # Votação majoritária vetorizada
        predictions = np.array([
            np.bincount(labels, minlength=self._n_classes).argmax()
            for labels in k_nearest_labels
        ])
        
        return predictions

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
        return np.mean(y_pred == y)

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

        X = np.asarray(X, dtype=np.float64)
        
        # Calcular distâncias
        distances = self._compute_distances_batch(X)
        
        # Pegar índices dos k menores
        if self.X_train.shape[0] > 100:
            k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        else:
            k_indices = np.argsort(distances, axis=1)[:, :self.k]
        
        # Pegar labels dos k vizinhos
        k_nearest_labels = self.y_train[k_indices]
        
        # Calcular probabilidades
        probas = np.array([
            np.bincount(labels, minlength=self._n_classes) / self.k
            for labels in k_nearest_labels
        ])
        
        return probas

    # Manter compatibilidade com código antigo
    def _compute_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calcula distância entre dois pontos (versão escalar para compatibilidade)."""
        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == "chebyshev":
            return np.max(np.abs(x1 - x2))
        else:
            raise ValueError(f"Métrica desconhecida: {self.distance_metric}")


def find_optimal_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_range: range = range(1, 19),
    distance_metric: str = "euclidean",
) -> tuple:
    """
    Encontra o valor ótimo de K testando múltiplos valores.
    Conforme Figure 9 do artigo (testa k de 1 a 18).

    Args:
        X_train (np.ndarray): Features de treino.
        y_train (np.ndarray): Labels de treino.
        X_test (np.ndarray): Features de teste.
        y_test (np.ndarray): Labels de teste.
        k_range (range): Range de valores de k para testar.
        distance_metric (str): Métrica de distância.

    Returns:
        tuple: (k_ótimo, acurácias, k_values)
    """
    accuracies = []
    k_values = list(k_range)

    for k in k_values:
        knn = KNNClassifier(k=k, distance_metric=distance_metric)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        accuracies.append(acc)
        print(f"k={k}: Acurácia={acc:.4f}")

    best_idx = np.argmax(accuracies)
    best_k = k_values[best_idx]
    best_acc = accuracies[best_idx]

    print(f"\n✓ Melhor k={best_k} com acurácia={best_acc:.4f}")

    return best_k, accuracies, k_values
