import numpy as np
from typing import Tuple
from .base_encoding import BaseEncoding


class DecimalEncoding(BaseEncoding):
    """
    Encoding decimal: cada gene é um inteiro [0, max_value].
    Threshold define quais features são selecionadas.
    """
    
    def __init__(
        self, 
        n_features: int, 
        max_value: int = 10, 
        threshold: float = 0.5,
        initial_feature_ratio: float = 0.1
    ):
        super().__init__(n_features)
        self.max_value = max_value
        self.threshold = threshold
        self.initial_feature_ratio = initial_feature_ratio
    
    def initialize_chromosome(self) -> np.ndarray:
        """
        Inicializa com distribuição esparsa.
        Maioria zeros, alguns valores altos.
        """
        threshold_value = int(self.threshold * self.max_value)
        
        chromosome = np.random.randint(0, threshold_value, size=self.n_features)
        
        n_to_select = max(1, int(self.n_features * self.initial_feature_ratio))
        n_to_select = np.random.randint(
            max(1, n_to_select // 2), 
            min(self.n_features, n_to_select * 2)
        )
        
        indices = np.random.choice(self.n_features, size=n_to_select, replace=False)
        chromosome[indices] = np.random.randint(threshold_value, self.max_value + 1, size=n_to_select)
        
        return chromosome
    
    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Decodifica para binário.
        Feature selecionada se valor normalizado >= threshold.
        """
        normalized = chromosome / self.max_value
        binary = (normalized >= self.threshold).astype(int)
        
        if np.sum(binary) == 0:
            max_idx = np.argmax(chromosome)
            binary[max_idx] = 1
        
        return binary
    
    def mutate(self, chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Mutação por incremento/decremento."""
        mutated = chromosome.copy()
        
        for i in range(self.n_features):
            if np.random.random() < mutation_rate:
                change = np.random.choice([-2, -1, 1, 2])
                mutated[i] = np.clip(mutated[i] + change, 0, self.max_value)
        
        return mutated
    
    def crossover(
        self, 
        parent1: np.ndarray, 
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Two-point crossover."""
        points = sorted(np.random.choice(self.n_features, size=2, replace=False))
        p1, p2 = points
        
        offspring1 = np.concatenate([
            parent1[:p1],
            parent2[p1:p2],
            parent1[p2:]
        ])
        
        offspring2 = np.concatenate([
            parent2[:p1],
            parent1[p1:p2],
            parent2[p2:]
        ])
        
        return offspring1, offspring2