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
        self, train_samples: int = 60000, test_samples: int = 10000, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carrega o conjunto de dados Fashion-MNIST a partir de arquivos CSV dado especificações no artigo original.
        Nota: no artigo, o autor utiliza 80 mil amostras, mas o dataset original possui 60000 amostras de treino e 10000 de teste.

        Args:
            train_samples (int): Número de amostras para o conjunto de treinamento.
            test_samples (int): Número de amostras para o conjunto de teste.
            random_state (int): Semente para reprodução dos resultados.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays numpy contendo os dados de treinamento e teste.
        """
        print(
            f"Carregando Fashion-MNIST com {train_samples} amostras de treino e {test_samples} amostras de teste."
        )

        train_path = self.data_root / "FASHION-MNIST" / "fashion-mnist_train.csv"
        test_path = self.data_root / "FASHION-MNIST" / "fashion-mnist_test.csv"

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

    def load_microarray_dataset(
        self, dataset_name: str, test_size: float = 0.3, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carrega um conjunto de dados de microarray específico a partir de arquivos .x e .dbc.

        Args:
            dataset_name (str): Nome da pasta do conjunto de dados.
            test_size (float): Proporção do conjunto de teste.
            random_state (int): Semente para reprodução dos resultados.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays numpy contendo os dados de treinamento e teste.
        """
        print(f"Carregando conjunto de dados de microarray: {dataset_name}")

        dataset_path = self.data_root / dataset_name

        x_files = sorted(list(dataset_path.glob("*.x")))
        dbc_files = sorted(list(dataset_path.glob("*.dbc")))

        if not x_files:
            raise FileNotFoundError(f"Arquivos .x não encontrados em {dataset_path}")
        if not dbc_files:
            raise FileNotFoundError(f"Arquivos .dbc não encontrados em {dataset_path}")

        train_x = [f for f in x_files if "train" in f.name.lower()]
        test_x = [f for f in x_files if "test" in f.name.lower()]

        if train_x and test_x:
            print("Dataset já dividido em treino/teste")
            X_train = self._load_x_file(train_x[0])
            X_test = self._load_x_file(test_x[0])

            train_dbc = [f for f in dbc_files if "train" in f.name.lower()]
            test_dbc = [f for f in dbc_files if "test" in f.name.lower()]

            if train_dbc and test_dbc:
                y_train = self._load_dbc_file(train_dbc[0])
                y_test = self._load_dbc_file(test_dbc[0])
            else:
                raise FileNotFoundError("Arquivos .dbc de treino/teste não encontrados")
        else:
            print("Dividindo dataset em treino/teste")
            X = self._load_x_file(x_files[0])
            y = self._load_dbc_file(dbc_files[0])

            if len(X) != len(y):
                print(
                    f"Aviso: O número de amostras em X ({len(X)}) e y ({len(y)}) não coincide. Ajustando para o menor tamanho."
                )
                min_len = min(len(X), len(y))
                X, y = X[:min_len], y[:min_len]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

        print(
            f"Conjunto de dados {dataset_name} carregado com sucesso. X_train: {X_train.shape}, X_test: {X_test.shape}"
        )
        print(
            f"Features: {X_train.shape[1]}, Samples: {len(X_train) + len(X_test)}, Classes: {len(np.unique(y_train))}"
        )
        return X_train, y_train, X_test, y_test

    def _load_x_file(self, filepath: Path) -> np.ndarray:
        """
        Carrega um arquivo .x e retorna os dados como um array numpy.
        Suporta formatos texto (tab/espaço) e binário (float32/float64).

        Args:
            filepath (Path): Caminho para o arquivo .x.

        Returns:
            np.ndarray: Dados carregados do arquivo .x.
        """
        try:
            df = pd.read_csv(filepath, sep="\t", header=None)
            if df.shape[1] > 1:
                print(f"Arquivo {filepath.name} carregado em formato texto (tab)")
                return df.values
        except:
            pass

        try:
            df = pd.read_csv(filepath, sep=r"\s+", header=None)
            if df.shape[1] > 1:
                print(f"Arquivo {filepath.name} carregado em formato texto (espaço)")
                return df.values
        except:
            pass

        try:
            with open(filepath, "rb") as f:
                content = f.read()

            for dtype in [np.float32, np.float64, np.int32]:
                try:
                    data = np.frombuffer(content, dtype=dtype)
                    if len(data) > 0:
                        print(
                            f"Arquivo {filepath.name} carregado em formato binário ({dtype.__name__})"
                        )
                        return data.reshape(-1, 1)
                except:
                    continue

            raise ValueError(f"Não foi possível determinar o formato do arquivo {filepath.name}")

        except Exception as e:
            print(f"Erro carregando {filepath}: {e}")
            raise

    def _load_dbc_file(self, filepath: Path) -> np.ndarray:
        """
        Carrega um arquivo .dbc e retorna os rótulos como um array numpy.
        Suporta formatos texto e binário (int32/float32).

        Args:
            filepath (Path): Caminho para o arquivo .dbc.

        Returns:
            np.ndarray: Rótulos carregados do arquivo .dbc.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]

            if labels:
                try:
                    y = np.array([int(label) for label in labels])
                    print(f"Arquivo {filepath.name} carregado com labels numéricos")
                    return y
                except ValueError:
                    unique_labels = sorted(set(labels))
                    label_map = {label: idx for idx, label in enumerate(unique_labels)}
                    y = np.array([label_map[label] for label in labels])
                    print(
                        f"Arquivo {filepath.name} carregado com labels categóricos ({len(unique_labels)} classes)"
                    )
                    return y
        except UnicodeDecodeError:
            pass
        except Exception as e:
            if "codec can't decode" not in str(e):
                print(f"Erro carregando {filepath}: {e}")

        try:
            with open(filepath, "rb") as f:
                content = f.read()

            for dtype in [np.int32, np.float32, np.int16]:
                try:
                    y = np.frombuffer(content, dtype=dtype)
                    if len(y) > 0:
                        y = y.astype(int)
                        print(
                            f"Arquivo {filepath.name} carregado em formato binário ({dtype.__name__})"
                        )
                        return y
                except:
                    continue

            raise ValueError(f"Não foi possível determinar o formato do arquivo {filepath.name}")

        except Exception as e:
            print(f"Erro carregando {filepath}: {e}")
            raise

    def load_colon(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Carrega dataset de Câncer de Cólon."""
        return self.load_microarray_dataset("COLON-TUMOR", **kwargs)

    def load_leukemia_allaml(
        self, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Carrega dataset de Leucemia ALL-AML."""
        return self.load_microarray_dataset("ALL-AML_LEUKEMIA", **kwargs)

    def load_cns(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Carrega dataset do Sistema Nervoso Central."""
        return self.load_microarray_dataset("NERVOUS-SYSTEM", **kwargs)

    def load_mll(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Carrega dataset de Leucemia MLL."""
        return self.load_microarray_dataset("MLL-LEUKEMIA", **kwargs)

    def load_ovarian(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Carrega dataset de Câncer de Ovário."""
        return self.load_microarray_dataset("OVARIAN-PBSII-061902", **kwargs)


def load_dataset(
    name: str, data_root: str = "data/raw", **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega qualquer dataset por nome.

    Args:
        name (str): Nome do dataset ('fashion-mnist', 'colon', 'leukemia', 'cns', 'mll', 'ovarian').
        data_root (str): Diretório raiz dos dados.
        **kwargs: Argumentos adicionais passados ao loader.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test.
    """
    loader = DataLoader(data_root)

    name = name.lower()

    if name in ["fashion-mnist", "fashion_mnist"]:
        return loader.load_fashion_mnist(**kwargs)
    elif name == "colon":
        return loader.load_colon(**kwargs)
    elif name in ["leukemia", "leukemia_allaml"]:
        return loader.load_leukemia_allaml(**kwargs)
    elif name == "cns":
        return loader.load_cns(**kwargs)
    elif name == "mll":
        return loader.load_mll(**kwargs)
    elif name == "ovarian":
        return loader.load_ovarian(**kwargs)
    else:
        raise ValueError(f"Dataset desconhecido: {name}")
