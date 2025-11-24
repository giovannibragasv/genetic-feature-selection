import numpy as np
from typing import Optional, TYPE_CHECKING
from ..algorithms.knn_classifier import KNNClassifier

if TYPE_CHECKING:
    from ..encoding import BaseEncoding


class AccuracyFitness:
    """
    Função de fitness baseada em acurácia do KNN.
    
    Avalia qualidade de um chromosome (seleção de features) usando KNN.
    Suporta múltiplos encodings através do parâmetro encoding.
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        k: int = 7,
        metric: str = 'euclidean',
        penalty_weight: float = 0.0,
        encoding: Optional['BaseEncoding'] = None
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.k = k
        self.metric = metric
        self.penalty_weight = penalty_weight
        self.encoding = encoding
        
        self.n_features = X_train.shape[1]
        self.evaluation_count = 0
    
    def __call__(self, chromosome: np.ndarray) -> float:
        """
        Avalia fitness de um chromosome.
        
        Args:
            chromosome: Chromosome codificado (será decodificado se encoding fornecido)
            
        Returns:
            Fitness score (acurácia, possivelmente penalizada)
        """
        self.evaluation_count += 1
        
        if self.encoding is not None:
            binary = self.encoding.decode(chromosome)
        else:
            binary = chromosome
        
        selected_features = binary == 1
        n_selected = np.sum(selected_features)
        
        if n_selected == 0:
            return 0.0
        
        X_train_selected = self.X_train[:, selected_features]
        X_test_selected = self.X_test[:, selected_features]
        
        knn = KNNClassifier(k=self.k, distance_metric=self.metric)
        knn.fit(X_train_selected, self.y_train)
        accuracy = knn.score(X_test_selected, self.y_test)
        
        if self.penalty_weight > 0:
            feature_ratio = n_selected / self.n_features
            fitness = accuracy - (self.penalty_weight * feature_ratio)
        else:
            fitness = accuracy
        
        return fitness
    
    def reset_counter(self):
        """Reseta contador de avaliações."""
        self.evaluation_count = 0
    
    def get_evaluation_count(self) -> int:
        """Retorna número de avaliações realizadas."""
        return self.evaluation_count


def create_fitness_function(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int = 7,
    metric: str = 'euclidean',
    penalty_weight: float = 0.0,
    encoding: Optional['BaseEncoding'] = None
) -> AccuracyFitness:
    """
    Factory function para criar função de fitness.
    
    Args:
        X_train: Features de treino
        y_train: Labels de treino
        X_test: Features de teste
        y_test: Labels de teste
        k: Número de vizinhos para KNN
        metric: Métrica de distância
        penalty_weight: Peso da penalização por número de features (0-1)
        encoding: Encoding para decodificar chromosomes (opcional)
        
    Returns:
        Função de fitness configurada
    """
    return AccuracyFitness(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        k=k,
        metric=metric,
        penalty_weight=penalty_weight,
        encoding=encoding
    )
