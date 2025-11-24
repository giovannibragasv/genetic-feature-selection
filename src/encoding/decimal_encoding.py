import numpy as np
from typing import Tuple
from .base_encoding import BaseEncoding


class DecimalEncoding(BaseEncoding):
    """
    Encoding decimal: cada gene é um inteiro [0, max_value].
    Threshold define quais features são selecionadas.
    """
    
    def __init__(self, n_features: int, max_value: int = 10, threshold: float = 0.5):
        super().__init__(n_features)
        self.max_value = max_value
        self.threshold = threshold
    
    def initialize_chromosome(self) -> np.ndarray:
        """Inicializa chromosome com valores inteiros aleatórios."""
        chromosome = np.random.randint(0, self.max_value + 1, size=self.n_features)
        return chromosome.astype(float)
    
    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Decodifica para binário usando threshold.
        Feature selecionada se valor >= threshold * max_value.
        """
        threshold_value = self.threshold * self.max_value
        binary = (chromosome >= threshold_value).astype(int)
        
        if np.sum(binary) == 0:
            max_idx = np.argmax(chromosome)
            binary[max_idx] = 1
        
        return binary
    
    def mutate(self, chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Mutação: incremento/decremento aleatório."""
        mutated = chromosome.copy()
        
        for i in range(self.n_features):
            if np.random.random() < mutation_rate:
                delta = np.random.choice([-1, 1])
                mutated[i] = np.clip(mutated[i] + delta, 0, self.max_value)
        
        return mutated
    
    def crossover(
        self, 
        parent1: np.ndarray, 
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Arithmetic crossover."""
        alpha = np.random.random()
        
        offspring1 = alpha * parent1 + (1 - alpha) * parent2
        offspring2 = (1 - alpha) * parent1 + alpha * parent2
        
        offspring1 = np.clip(np.round(offspring1), 0, self.max_value)
        offspring2 = np.clip(np.round(offspring2), 0, self.max_value)
        
        return offspring1, offspring2