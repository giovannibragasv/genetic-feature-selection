import numpy as np
from typing import Tuple
from .base_encoding import BaseEncoding


class BinaryEncoding(BaseEncoding):
    """
    Encoding binário: cada gene é 0 ou 1.
    1 = feature selecionada, 0 = feature não selecionada.
    """
    
    def initialize_chromosome(self) -> np.ndarray:
        """Inicializa chromosome binário aleatório."""
        chromosome = np.random.randint(0, 2, size=self.n_features)
        
        if np.sum(chromosome) == 0:
            random_indices = np.random.choice(
                self.n_features,
                size=max(1, self.n_features // 10),
                replace=False
            )
            chromosome[random_indices] = 1
        
        return chromosome
    
    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        """Retorna o próprio chromosome (já é binário)."""
        return chromosome.astype(int)
    
    def mutate(self, chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Mutação bit-flip."""
        mutated = chromosome.copy()
        
        for i in range(self.n_features):
            if np.random.random() < mutation_rate:
                mutated[i] = 1 - mutated[i]
        
        if np.sum(mutated) == 0:
            random_idx = np.random.randint(0, self.n_features)
            mutated[random_idx] = 1
        
        return mutated
    
    def crossover(
        self, 
        parent1: np.ndarray, 
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover."""
        crossover_point = np.random.randint(1, self.n_features)
        
        offspring1 = np.concatenate([
            parent1[:crossover_point],
            parent2[crossover_point:]
        ])
        
        offspring2 = np.concatenate([
            parent2[:crossover_point],
            parent1[crossover_point:]
        ])
        
        return offspring1, offspring2