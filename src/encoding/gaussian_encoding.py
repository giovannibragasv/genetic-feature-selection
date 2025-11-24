import numpy as np
from typing import Tuple
from .base_encoding import BaseEncoding


class GaussianEncoding(BaseEncoding):
    """
    Encoding gaussiano: genes são valores amostrados de distribuição normal.
    
    Genes representam "relevância" das features, inicializados com distribuição
    gaussiana centrada em mean com desvio std. Decode usa threshold para
    converter em seleção binária.
    
    Referência: Feng (2024), seção 4.2.4
    """
    
    def __init__(
        self, 
        n_features: int, 
        mean: float = 0.5,
        std: float = 0.2,
        threshold: float = 0.5
    ):
        super().__init__(n_features)
        self.mean = mean
        self.std = std
        self.threshold = threshold
    
    def initialize_chromosome(self) -> np.ndarray:
        """
        Inicializa chromosome com valores amostrados de distribuição gaussiana.
        Valores são clipped para [0, 1].
        """
        chromosome = np.random.normal(self.mean, self.std, size=self.n_features)
        chromosome = np.clip(chromosome, 0, 1)
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
        """
        Mutação gaussiana: adiciona ruído normal aos genes selecionados.
        O desvio do ruído é proporcional ao std do encoding.
        """
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
        """
        Crossover aritmético ponderado com peso gaussiano.
        Peso é amostrado de distribuição normal para cada gene.
        """
        weights = np.clip(np.random.normal(0.5, 0.2, size=self.n_features), 0, 1)
        
        offspring1 = weights * parent1 + (1 - weights) * parent2
        offspring2 = (1 - weights) * parent1 + weights * parent2
        
        offspring1 = np.clip(offspring1, 0, 1)
        offspring2 = np.clip(offspring2, 0, 1)
        
        return offspring1, offspring2
    
    def set_threshold(self, threshold: float) -> None:
        """
        Define novo threshold para decodificação.
        
        Args:
            threshold: Novo valor de threshold [0, 1]
        """
        self.threshold = np.clip(threshold, 0.05, 0.95)
    
    def get_distribution_stats(self, chromosome: np.ndarray) -> dict:
        """
        Retorna estatísticas da distribuição do chromosome.
        
        Args:
            chromosome: Chromosome para análise
            
        Returns:
            dict: Estatísticas (mean, std, min, max)
        """
        return {
            'mean': float(np.mean(chromosome)),
            'std': float(np.std(chromosome)),
            'min': float(np.min(chromosome)),
            'max': float(np.max(chromosome)),
            'n_selected': int(np.sum(self.decode(chromosome)))
        }
