"""
Loader para Fashion-MNIST dataset.
Usado para replicar Table 3 do paper Feng 2024.

Fashion-MNIST:
- 784 features (28x28 pixels)
- 70000 samples (60000 train + 10000 test)
- 10 classes (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
"""

import os
import gzip
import numpy as np
from pathlib import Path
from typing import Tuple
import urllib.request


class FashionMNISTLoader:
    """Loader para Fashion-MNIST dataset."""
    
    BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    
    FILES = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    CLASS_NAMES = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    def __init__(self, data_root: str = "data/raw/fashion_mnist"):
        """
        Inicializa o loader.
        
        Args:
            data_root: Diretório para salvar/carregar os dados
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
    
    def _download_file(self, filename: str) -> Path:
        """Baixa arquivo se não existir."""
        filepath = self.data_root / filename
        
        if not filepath.exists():
            url = self.BASE_URL + filename
            print(f"Baixando {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"  Salvo em {filepath}")
        
        return filepath
    
    def _read_images(self, filepath: Path) -> np.ndarray:
        """Lê arquivo de imagens IDX."""
        with gzip.open(filepath, 'rb') as f:
            # Magic number, num images, rows, cols
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            
            # Ler dados
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_images, rows * cols)
        
        return data.astype(np.float64)
    
    def _read_labels(self, filepath: Path) -> np.ndarray:
        """Lê arquivo de labels IDX."""
        with gzip.open(filepath, 'rb') as f:
            # Magic number, num labels
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            
            # Ler dados
            data = np.frombuffer(f.read(), dtype=np.uint8)
        
        return data
    
    def download(self):
        """Baixa todos os arquivos necessários."""
        print("Baixando Fashion-MNIST...")
        for name, filename in self.FILES.items():
            self._download_file(filename)
        print("Download completo!")
    
    def load(
        self,
        n_train: int = None,
        n_test: int = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carrega o dataset.
        
        Args:
            n_train: Número de amostras de treino (None = todas 60000)
            n_test: Número de amostras de teste (None = todas 10000)
            normalize: Se True, normaliza pixels para [0, 1]
            
        Returns:
            Tuple: (X_train, y_train, X_test, y_test)
        """
        # Baixar se necessário
        for filename in self.FILES.values():
            if not (self.data_root / filename).exists():
                self.download()
                break
        
        print("Carregando Fashion-MNIST...")
        
        # Carregar dados
        X_train = self._read_images(self.data_root / self.FILES['train_images'])
        y_train = self._read_labels(self.data_root / self.FILES['train_labels'])
        X_test = self._read_images(self.data_root / self.FILES['test_images'])
        y_test = self._read_labels(self.data_root / self.FILES['test_labels'])
        
        # Normalizar
        if normalize:
            X_train = X_train / 255.0
            X_test = X_test / 255.0
        
        # Subamostrar se necessário
        if n_train is not None and n_train < len(X_train):
            indices = np.random.choice(len(X_train), n_train, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
        
        if n_test is not None and n_test < len(X_test):
            indices = np.random.choice(len(X_test), n_test, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]
        
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
        print(f"  Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
        
        return X_train, y_train, X_test, y_test
    
    def load_subset(
        self,
        n_samples: int = 10000,
        test_size: float = 0.3,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carrega um subconjunto menor (mais rápido para experimentos).
        
        Paper usa 80000 train + 15000 test, mas para experimentos
        menores podemos usar menos.
        
        Args:
            n_samples: Total de amostras
            test_size: Proporção de teste
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tuple: (X_train, y_train, X_test, y_test)
        """
        # Carregar tudo
        X_train_full, y_train_full, X_test_full, y_test_full = self.load(normalize=True)
        
        # Combinar
        X_all = np.vstack([X_train_full, X_test_full])
        y_all = np.concatenate([y_train_full, y_test_full])
        
        # Subamostrar
        np.random.seed(random_state)
        indices = np.random.choice(len(X_all), min(n_samples, len(X_all)), replace=False)
        X = X_all[indices]
        y = y_all[indices]
        
        # Split
        n_test = int(len(X) * test_size)
        n_train = len(X) - n_test
        
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        
        print(f"Subset carregado: {n_train} train, {n_test} test")
        
        return X_train, y_train, X_test, y_test


def test_loader():
    """Testa o loader."""
    loader = FashionMNISTLoader()
    
    # Testar com subset pequeno
    X_train, y_train, X_test, y_test = loader.load_subset(n_samples=1000)
    
    print(f"\nTeste OK!")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train unique: {np.unique(y_train)}")


if __name__ == "__main__":
    test_loader()
