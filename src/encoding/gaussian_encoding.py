import numpy as np
from typing import Tuple
from .base_encoding import BaseEncoding


class GaussianEncoding(BaseEncoding):
    """
    Encoding Gaussiano: genes amostrados de distribuição normal.
    """
    
    def __init__(
        self, 
        n_features: int, 
        mean: float = 0.5, 
        std: float = 0.2, 
        threshold: float = 0.5,
        initial_feature_ratio: float = 0.1
    ):
        super().__init__(n_features)
        self.mean = mean
        self.std = std
        self.threshold = threshold
        self.initial_feature_ratio = initial_feature_ratio
    
    def initialize_chromosome(self) -> np.ndarray:
        """
        Inicializa com distribuição esparsa.
        """
        chromosome = np.random.random(size=self.n_features) * self.threshold * 0.8
        
        n_to_select = max(1, int(self.n_features * self.initial_feature_ratio))
        n_to_select = np.random.randint(
            max(1, n_to_select // 2), 
            min(self.n_features, n_to_select * 2)
        )
        
        indices = np.random.choice(self.n_features, size=n_to_select, replace=False)
        chromosome[indices] = np.clip(
            np.random.normal(self.mean + 0.2, self.std, size=n_to_select),
            self.threshold,
            1.0
        )
        
        return chromosome
    
    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        """Decodifica para binário usando threshold."""
        binary = (chromosome >= self.threshold).astype(int)
        
        if np.sum(binary) == 0:
            max_idx = np.argmax(chromosome)
            binary[max_idx] = 1
        
        return binary
    
    def mutate(self, chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Mutação gaussiana."""
        mutated = chromosome.copy()
        
        for i in range(self.n_features):
            if np.random.random() < mutation_rate:
                noise = np.random.normal(0, self.std * 0.5)
                mutated[i] = np.clip(mutated[i] + noise, 0, 1)
        
        return mutated
    
    def crossover(
        self, 
        parent1: np.ndarray, 
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover aritmético ponderado."""
        weights = np.clip(
            np.random.normal(0.5, 0.2, size=self.n_features),
            0, 1
        )
        
        offspring1 = weights * parent1 + (1 - weights) * parent2
        offspring2 = (1 - weights) * parent1 + weights * parent2
        
        offspring1 = np.clip(offspring1, 0, 1)
        offspring2 = np.clip(offspring2, 0, 1)
        
        return offspring1, offspring2
    
    def set_threshold(self, threshold: float):
        """Permite ajustar threshold."""
        self.threshold = np.clip(threshold, 0, 1)
    
    def get_distribution_stats(self) -> dict:
        """Retorna parâmetros da distribuição."""
        return {
            'mean': self.mean,
            'std': self.std,
            'threshold': self.threshold
        }