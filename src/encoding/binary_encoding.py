import numpy as np
from typing import Tuple
from .base_encoding import BaseEncoding


class BinaryEncoding(BaseEncoding):
    """
    Encoding binário: cada gene é 0 ou 1.
    1 = feature selecionada, 0 = feature não selecionada.
    
    Operadores otimizados para feature selection com bias para esparsidade.
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
        # Variação de 50% a 150% do target
        n_to_select = np.random.randint(
            max(1, n_to_select // 2),
            min(self.n_features, int(n_to_select * 1.5) + 1)
        )
        
        indices = np.random.choice(self.n_features, size=n_to_select, replace=False)
        chromosome[indices] = 1
        
        return chromosome
    
    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        """Retorna o próprio chromosome (já é binário)."""
        return chromosome.astype(int)
    
    def mutate(self, chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        """
        Mutação bit-flip com FORTE bias para esparsidade.
        
        - Features ativas (1) têm 80% chance de serem desligadas quando mutadas
        - Features inativas (0) têm 20% chance de serem ligadas quando mutadas
        """
        mutated = chromosome.copy()
        
        for i in range(self.n_features):
            if np.random.random() < mutation_rate:
                if mutated[i] == 1:
                    # 80% chance de desligar feature ativa
                    if np.random.random() < 0.8:
                        mutated[i] = 0
                else:
                    # 20% chance de ligar feature inativa
                    if np.random.random() < 0.2:
                        mutated[i] = 1
        
        # Garantir pelo menos uma feature
        if np.sum(mutated) == 0:
            random_idx = np.random.randint(0, self.n_features)
            mutated[random_idx] = 1
        
        return mutated
    
    def crossover(
        self, 
        parent1: np.ndarray, 
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crossover AND/OR com bias para esparsidade.
        
        - offspring1: tende a usar AND (interseção) - menos features
        - offspring2: tende a usar OR (união) - mais features
        
        Isso cria diversidade: um filho mais esparso, outro mais denso.
        """
        # Probabilidade de usar AND vs herdar do pai
        and_prob = 0.6  # 60% AND, 40% herdar
        
        offspring1 = np.zeros(self.n_features, dtype=int)
        offspring2 = np.zeros(self.n_features, dtype=int)
        
        for i in range(self.n_features):
            if parent1[i] == 1 and parent2[i] == 1:
                # Ambos pais têm a feature - alta chance de manter
                offspring1[i] = 1
                offspring2[i] = 1
            elif parent1[i] == 1 or parent2[i] == 1:
                # Apenas um pai tem a feature
                if np.random.random() < and_prob:
                    # offspring1 mais esparso (AND)
                    offspring1[i] = 0
                    # offspring2 mais denso (OR)
                    offspring2[i] = 1
                else:
                    # Herdar aleatoriamente
                    offspring1[i] = np.random.choice([parent1[i], parent2[i]])
                    offspring2[i] = np.random.choice([parent1[i], parent2[i]])
            # else: ambos 0, mantém 0
        
        # Garantir pelo menos uma feature
        if np.sum(offspring1) == 0:
            # Copiar uma feature ativa de qualquer pai
            active_indices = np.where((parent1 == 1) | (parent2 == 1))[0]
            if len(active_indices) > 0:
                offspring1[np.random.choice(active_indices)] = 1
            else:
                offspring1[np.random.randint(0, self.n_features)] = 1
                
        if np.sum(offspring2) == 0:
            active_indices = np.where((parent1 == 1) | (parent2 == 1))[0]
            if len(active_indices) > 0:
                offspring2[np.random.choice(active_indices)] = 1
            else:
                offspring2[np.random.randint(0, self.n_features)] = 1
        
        return offspring1, offspring2
