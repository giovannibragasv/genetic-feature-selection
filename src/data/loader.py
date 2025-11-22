import os
import re
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

    def _parse_elvira_dbc(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parser completo para arquivos .dbc em formato Elvira.
        Extrai dados da seção 'cases' e metadados dos nós.

        Args:
            filepath (Path): Caminho para arquivo .dbc.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features (X) e labels (y).
        """
        print(f"Parseando arquivo Elvira: {filepath.name}")

        with open(filepath, "r", encoding="latin1") as f:
            content = f.read()

        n_cases_match = re.search(r"number-of-cases\s*=\s*(\d+)", content)
        if n_cases_match:
            n_cases = int(n_cases_match.group(1))
            print(f"Número de casos esperados: {n_cases}")

        nodes = []
        node_pattern = r"node\s+(\w+)\s*\([^)]+\)\s*\{"
        for match in re.finditer(node_pattern, content):
            node_name = match.group(1)
            nodes.append(node_name)

        print(f"Encontrados {len(nodes)} nós/variáveis")

        cases_match = re.search(r"cases\s*=\s*\((.*?)\)\s*;", content, re.DOTALL)

        if not cases_match:
            raise ValueError(f"Seção 'cases' não encontrada em {filepath.name}")

        cases_text = cases_match.group(1)

        case_rows = []

        for line in cases_text.split("["):
            if not line.strip() or line.strip().startswith("//"):
                continue

            line = line.split("]")[0].strip()

            if not line:
                continue

            values_str = re.sub(r"\s+", " ", line).strip()

            tokens = []
            for token in values_str.split(","):
                token = token.strip()
                if token:
                    tokens.append(token)

            if tokens:
                case_rows.append(tokens)

        print(f"Extraídos {len(case_rows)} casos")

        if not case_rows:
            raise ValueError("Nenhum dado foi extraído da seção cases")

        label_node = nodes[0] if nodes else None
        print(f"Nó de label: {label_node}")

        processed_data = []
        labels = []

        for row in case_rows:
            if not row:
                continue

            label_value = row[0]
            if label_value in ["positive", "negative", "normal", "tumor"]:
                labels.append(0 if label_value in ["positive", "tumor"] else 1)
            else:
                try:
                    labels.append(int(label_value))
                except:
                    labels.append(0)

            features = []
            for val in row[1:]:
                try:
                    features.append(float(val))
                except:
                    features.append(0.0)

            processed_data.append(features)

        X = np.array(processed_data, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        print(f"Matriz de dados: X={X.shape}, y={y.shape}")
        print(f"Classes únicas em y: {np.unique(y)}")

        return X, y

    def _load_java_serialized(self, filepath: Path) -> np.ndarray:
        """
        Carrega arquivo .x serializado em Java (formato Elvira).

        Args:
            filepath (Path): Caminho para arquivo .x.

        Returns:
            np.ndarray: Matriz de features.
        """
        print(f"Tentando desserializar arquivo Java: {filepath.name}")

        try:
            import javaobj

            with open(filepath, "rb") as f:
                obj = javaobj.loads(f.read())

            print("Objeto Java deserializado com sucesso")
            return self._extract_data_from_java_object(obj)

        except ImportError:
            print("Biblioteca javaobj não disponível, tentando parser alternativo")
            return self._parse_java_serialized_alternative(filepath)

    def _parse_java_serialized_alternative(self, filepath: Path) -> np.ndarray:
        """
        Parser alternativo para arquivos Java serializados sem biblioteca externa.
        """
        with open(filepath, "rb") as f:
            content = f.read()

        try:
            text_parts = []
            i = 0
            while i < len(content):
                if content[i] >= 32 and content[i] < 127:
                    text_parts.append(chr(content[i]))
                i += 1

            text = "".join(text_parts)
            numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)

            if numbers:
                float_values = [float(n) for n in numbers if "." in n or "e" in n.lower()]
                if len(float_values) > 100:
                    print(f"Extraídos {len(float_values)} valores numéricos")
                    return np.array(float_values).reshape(-1, 1)

        except Exception as e:
            print(f"Parser alternativo falhou: {e}")

        raise ValueError(f"Não foi possível deserializar {filepath.name}")

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
        Carrega um conjunto de dados de microarray específico a partir de arquivos .dbc (formato Elvira).

        Args:
            dataset_name (str): Nome da pasta do conjunto de dados.
            test_size (float): Proporção do conjunto de teste.
            random_state (int): Semente para reprodução dos resultados.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays numpy contendo os dados de treinamento e teste.
        """
        print(f"Carregando conjunto de dados de microarray: {dataset_name}")

        dataset_path = self.data_root / dataset_name
        dbc_files = sorted(list(dataset_path.glob("*.dbc")))

        if not dbc_files:
            raise FileNotFoundError(f"Arquivos .dbc não encontrados em {dataset_path}")

        dbc_file = dbc_files[0]

        with open(dbc_file, "rb") as f:
            header = f.read(100)

        is_elvira = b"Elvira" in header or b"data-base" in header

        if is_elvira:
            print("Formato Elvira detectado")
            X, y = self._parse_elvira_dbc(dbc_file)
        else:
            raise ValueError(f"Formato não reconhecido para {dbc_file.name}")

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
        self, name: str, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carrega qualquer dataset por nome.

        Args:
            name (str): Nome do dataset ('fashion-mnist', 'colon', 'leukemia', 'cns', 'mll', 'ovarian').
            **kwargs: Argumentos adicionais passados ao loader.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test.
        """
        name = name.lower()

        if name in ["fashion-mnist", "fashion_mnist"]:
            return self.load_fashion_mnist(**kwargs)
        elif name == "colon":
            return self.load_colon(**kwargs)
        elif name in ["leukemia", "leukemia_allaml"]:
            return self.load_leukemia_allaml(**kwargs)
        elif name == "cns":
            return self.load_cns(**kwargs)
        elif name == "mll":
            return self.load_mll(**kwargs)
        elif name == "ovarian":
            return self.load_ovarian(**kwargs)
        else:
            raise ValueError(f"Dataset desconhecido: {name}")
