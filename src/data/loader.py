import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, data_root: str = "data/raw"):
        """
        Inicializa o DataLoader com o diretório raiz dos dados.

        Args:
            data_root (str): Caminho para o diretório raiz dos dados.
        """
        self.data_root = Path(data_root)

    def load_fashion_mnist(
        self, train_samples: int = 80000, test_samples: int = 20000, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carrega o conjunto de dados Fashion-MNIST a partir de arquivos CSV dado especificações no artigo origina.

        Args:
            train_samples (int): Número de amostras para o conjunto de treinamento (80000 como no artigo original).
            test_samples (int): Número de amostras para o conjunto de teste (20000 como no artigo original).
            random_state (int): Semente para reprodução dos resultados.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays numpy contendo os dados de treinamento e teste.
        """
        print(
            f"Carregando Fashion-MNIST com {train_samples} amostras de treino e {test_samples} amostras de teste."
        )

        train_path = self.data_root / "FASHION_MNIST" / "fashion-mnist_train.csv"
        test_path = self.data_root / "FASHION_MNIST" / "fashion-mnist_test.csv"

        train_df = pd.read_csv(train_path)
        y_train = train_df.iloc[:, 0].values
        X_train = train_df.iloc[:, 1:].values

        test_df = pd.read_csv(test_path)
        y_test = test_df.iloc[:, 0].values
        X_test = test_df.iloc[:, 1:].values

        if len(X_test) > test_samples:
            X_test, _, y_test, _ = train_test_split(
                X_test,
                y_test,
                test_size=(len(X_test) - test_samples),
                random_state=random_state,
                stratify=y_test,
            )

        print(
            f"Fashion-MNIST carregado com sucesso. X_train: {X_train.shape}, X_test: {X_test.shape}"
        )
        return X_train, y_train, X_test, y_test
