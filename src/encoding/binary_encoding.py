import numpy as np
from typing import Tuple
from .base_encoding import BaseEncoding


class BinaryEncoding(BaseEncoding):
    """
    Encoding binário: cada gene é 0 ou 1.
    1 = feature selecionada, 0 = feature não selecionada.
    """
    
    def __init__(self, n_features: int, initial_feature_ratio: float = 0.1):
        """
        Args:
            n_features: Número total de features
            initial_feature_ratio: Fração de features selecionadas inicialmente (default 10%)
        """
        super().__init__(n_features)
        self.initial_feature_ratio = initial_feature_ratio
    
    def initialize_chromosome(self) -> np.ndarray:
        """Inicializa chromosome binário ESPARSO."""
        chromosome = np.zeros(self.n_features, dtype=int)
        
        n_to_select = max(1, int(self.n_features * self.initial_feature_ratio))
        n_to_select = np.random.randint(
            max(1, n_to_select // 2),
            min(self.n_features, n_to_select * 2)
        )
        
        indices = np.random.choice(self.n_features, size=n_to_select, replace=False)
        chromosome[indices] = 1
        
        return chromosome
    
    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        """Retorna o próprio chromosome (já é binário)."""
        return chromosome.astype(int)
    
    def mutate(self, chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Mutação bit-flip com bias para esparsidade."""
        mutated = chromosome.copy()
        
        for i in range(self.n_features):
            if np.random.random() < mutation_rate:
                if mutated[i] == 1:
                    if np.random.random() < 0.7:
                        mutated[i] = 0
                else:
                    if np.random.random() < 0.3:
                        mutated[i] = 1
        
        if np.sum(mutated) == 0:
            random_idx = np.random.randint(0, self.n_features)
            mutated[random_idx] = 1
        
        return mutated
    
    def crossover(
        self, 
        parent1: np.ndarray, 
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover (melhor para feature selection)."""
        mask = np.random.randint(0, 2, size=self.n_features)
        
        offspring1 = np.where(mask, parent1, parent2)
        offspring2 = np.where(mask, parent2, parent1)
        
        if np.sum(offspring1) == 0:
            offspring1[np.random.randint(0, self.n_features)] = 1
        if np.sum(offspring2) == 0:
            offspring2[np.random.randint(0, self.n_features)] = 1
        
        return offspring1, offspring2