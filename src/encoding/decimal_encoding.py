import numpy as np
from typing import Tuple
from .base_encoding import BaseEncoding


class DecimalEncoding(BaseEncoding):
    """
    Encoding decimal: cada gene é um inteiro [0, max_value].
    Threshold define quais features são selecionadas.
    
    Operadores otimizados para feature selection com bias para esparsidade.
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
        self.threshold_value = int(self.threshold * self.max_value)
    
    def initialize_chromosome(self) -> np.ndarray:
        """Inicializa com distribuição esparsa."""
        # Maioria dos valores baixos (abaixo do threshold)
        chromosome = np.random.randint(0, self.threshold_value, size=self.n_features)
        
        # Selecionar ~initial_feature_ratio para ficar acima do threshold
        n_to_select = max(1, int(self.n_features * self.initial_feature_ratio))
        n_to_select = np.random.randint(
            max(1, n_to_select // 2), 
            min(self.n_features, int(n_to_select * 1.5) + 1)
        )
        
        indices = np.random.choice(self.n_features, size=n_to_select, replace=False)
        chromosome[indices] = np.random.randint(
            self.threshold_value, 
            self.max_value + 1, 
            size=n_to_select
        )
        
        return chromosome
    
    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        """Decodifica para binário usando threshold."""
        normalized = chromosome / self.max_value
        binary = (normalized >= self.threshold).astype(int)
        
        if np.sum(binary) == 0:
            max_idx = np.argmax(chromosome)
            binary[max_idx] = 1
        
        return binary
    
    def mutate(self, chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        """
        Mutação com bias para reduzir valores.
        
        - Valores altos (>=threshold) têm tendência a diminuir
        - Valores baixos (<threshold) têm pequena chance de aumentar
        """
        mutated = chromosome.copy()
        
        for i in range(self.n_features):
            if np.random.random() < mutation_rate:
                normalized = mutated[i] / self.max_value
                
                if normalized >= self.threshold:
                    # Feature ativa: 70% chance de diminuir, 30% de aumentar
                    if np.random.random() < 0.7:
                        change = -np.random.randint(1, 4)  # diminui 1-3
                    else:
                        change = np.random.randint(1, 3)   # aumenta 1-2
                else:
                    # Feature inativa: 30% chance de aumentar significativamente
                    if np.random.random() < 0.3:
                        change = np.random.randint(2, 5)   # aumenta 2-4
                    else:
                        change = np.random.choice([-1, 0, 1])  # pequena mudança
                
                mutated[i] = np.clip(mutated[i] + change, 0, self.max_value)
        
        return mutated
    
    def crossover(
        self, 
        parent1: np.ndarray, 
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crossover que favorece valores menores (esparsidade).
        
        - offspring1: tende a pegar o MENOR valor entre os pais
        - offspring2: média aritmética dos pais
        """
        offspring1 = np.zeros(self.n_features, dtype=int)
        offspring2 = np.zeros(self.n_features, dtype=int)
        
        for i in range(self.n_features):
            p1_active = parent1[i] >= self.threshold_value
            p2_active = parent2[i] >= self.threshold_value
            
            if p1_active and p2_active:
                # Ambos ativos: offspring1 pega menor, offspring2 pega média
                offspring1[i] = min(parent1[i], parent2[i])
                offspring2[i] = (parent1[i] + parent2[i]) // 2
            elif p1_active or p2_active:
                # Apenas um ativo: 60% chance de offspring1 pegar o menor (inativo)
                if np.random.random() < 0.6:
                    offspring1[i] = min(parent1[i], parent2[i])
                    offspring2[i] = max(parent1[i], parent2[i])
                else:
                    offspring1[i] = np.random.choice([parent1[i], parent2[i]])
                    offspring2[i] = np.random.choice([parent1[i], parent2[i]])
            else:
                # Ambos inativos: pequena chance de ativar
                offspring1[i] = min(parent1[i], parent2[i])
                if np.random.random() < 0.1:
                    offspring2[i] = max(parent1[i], parent2[i]) + np.random.randint(0, 3)
                else:
                    offspring2[i] = (parent1[i] + parent2[i]) // 2
        
        # Garantir pelo menos uma feature ativa
        offspring1 = np.clip(offspring1, 0, self.max_value)
        offspring2 = np.clip(offspring2, 0, self.max_value)
        
        if np.sum(self.decode(offspring1)) == 0:
            # Ativar a feature com maior valor
            max_idx = np.argmax(offspring1)
            offspring1[max_idx] = self.threshold_value + 1
            
        if np.sum(self.decode(offspring2)) == 0:
            max_idx = np.argmax(offspring2)
            offspring2[max_idx] = self.threshold_value + 1
        
        return offspring1, offspring2
