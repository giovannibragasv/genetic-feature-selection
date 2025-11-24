import numpy as np
from typing import Tuple
from .base_encoding import BaseEncoding


class RealEncoding(BaseEncoding):
    """
    Encoding real: cada gene é um float [0, 1].
    Threshold define quais features são selecionadas.
    """
    
    def __init__(self, n_features: int, threshold: float = 0.5):
        super().__init__(n_features)
        self.threshold = threshold
    
    def initialize_chromosome(self) -> np.ndarray:
        """Inicializa chromosome com valores reais aleatórios [0, 1]."""
        chromosome = np.random.random(size=self.n_features)
        return chromosome
    
    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Decodifica para binário usando threshold.
        Feature selecionada se valor >= threshold.
        """
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
                noise = np.random.normal(0, 0.1)
                mutated[i] = np.clip(mutated[i] + noise, 0, 1)
        
        return mutated
    
    def crossover(
        self, 
        parent1: np.ndarray, 
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Blend crossover (BLX-α)."""
        alpha = 0.5
        
        min_vals = np.minimum(parent1, parent2)
        max_vals = np.maximum(parent1, parent2)
        range_vals = max_vals - min_vals
        
        offspring1 = min_vals - alpha * range_vals + np.random.random(self.n_features) * (range_vals * (1 + 2 * alpha))
        offspring2 = min_vals - alpha * range_vals + np.random.random(self.n_features) * (range_vals * (1 + 2 * alpha))
        
        offspring1 = np.clip(offspring1, 0, 1)
        offspring2 = np.clip(offspring2, 0, 1)
        
        return offspring1, offspring2