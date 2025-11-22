import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class DataPreprocessor:
    """
    Preprocessador de dados para feature selection com GA.
    Implementa normalização e preparação de dados conforme artigo original.
    """

    def __init__(self, normalization: str = "standard", random_state: int = 42):
        """
        Inicializa o preprocessador.

        Args:
            normalization (str): Tipo de normalização ('standard', 'minmax', 'none').
            random_state (int): Semente para reprodutibilidade.
        """
        self.normalization = normalization
        self.random_state = random_state
        self.scaler = None

        if normalization == "standard":
            self.scaler = StandardScaler()
        elif normalization == "minmax":
            self.scaler = MinMaxScaler()

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ajusta o preprocessador e transforma os dados.

        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features normalizadas e labels.
        """
        if self.scaler is not None:
            X_normalized = self.scaler.fit_transform(X)
        else:
            X_normalized = X.copy()

        print(f"Dados normalizados com {self.normalization}")
        print(
            f"Shape: {X_normalized.shape}, Min: {X_normalized.min():.4f}, Max: {X_normalized.max():.4f}"
        )

        return X_normalized, y

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforma novos dados usando scaler já ajustado.

        Args:
            X (np.ndarray): Features.

        Returns:
            np.ndarray: Features normalizadas.
        """
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X.copy()

    def split_data(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.3, stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide dados em treino e teste conforme metodologia do artigo.

        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.
            test_size (float): Proporção do conjunto de teste (padrão: 0.3 = 30%).
            stratify (bool): Se True, mantém proporção de classes.

        Returns:
            Tuple: X_train, X_test, y_train, y_test.
        """
        stratify_param = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify_param
        )

        print(f"Divisão treino/teste: {int((1-test_size)*100)}/{int(test_size*100)}")
        print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
        print(f"Distribuição treino: {np.bincount(y_train)}")
        print(f"Distribuição teste: {np.bincount(y_test)}")

        return X_train, X_test, y_train, y_test

    def preprocess(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.3, stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Pipeline completo de preprocessamento: normaliza e divide dados.

        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.
            test_size (float): Proporção do conjunto de teste.
            stratify (bool): Se True, mantém proporção de classes.

        Returns:
            Tuple: X_train, X_test, y_train, y_test (já normalizados).
        """
        X_normalized, y = self.fit_transform(X, y)
        X_train, X_test, y_train, y_test = self.split_data(X_normalized, y, test_size, stratify)

        return X_train, X_test, y_train, y_test
