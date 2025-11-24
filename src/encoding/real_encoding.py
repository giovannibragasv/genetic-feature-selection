import numpy as np
from typing import Tuple
from .base_encoding import BaseEncoding


class RealEncoding(BaseEncoding):
    """
    Encoding real: cada gene é um float [0, 1].
    Threshold define quais features são selecionadas.
    """
    
    def __init__(
        self, 
        n_features: int, 
        threshold: float = 0.5,
        initial_feature_ratio: float = 0.1
    ):
        """
        Args:
            n_features: Número total de features
            threshold: Valor >= threshold seleciona a feature
            initial_feature_ratio: Fração de features a selecionar inicialmente (default 10%)
        """
        super().__init__(n_features)
        self.threshold = threshold
        self.initial_feature_ratio = initial_feature_ratio
    
    def initialize_chromosome(self) -> np.ndarray:
        """
        Inicializa chromosome com distribuição esparsa.
        Maioria dos valores baixos, poucos acima do threshold.
        """
        chromosome = np.random.random(size=self.n_features) * self.threshold * 0.8
        
        n_to_select = max(1, int(self.n_features * self.initial_feature_ratio))
        n_to_select = np.random.randint(
            max(1, n_to_select // 2), 
            min(self.n_features, n_to_select * 2)
        )
        
        indices = np.random.choice(self.n_features, size=n_to_select, replace=False)
        chromosome[indices] = self.threshold + np.random.random(n_to_select) * (1 - self.threshold)
        
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