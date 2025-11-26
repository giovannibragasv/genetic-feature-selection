import numpy as np
from typing import Tuple
from .base_encoding import BaseEncoding


class RealEncoding(BaseEncoding):
    """
    Encoding real: cada gene é um float [0, 1].
    Threshold define quais features são selecionadas.
    
    Operadores otimizados para feature selection com bias para esparsidade.
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
            initial_feature_ratio: Fração de features a selecionar inicialmente
        """
        super().__init__(n_features)
        self.threshold = threshold
        self.initial_feature_ratio = initial_feature_ratio
    
    def initialize_chromosome(self) -> np.ndarray:
        """Inicializa chromosome com distribuição esparsa."""
        # Maioria dos valores baixos (abaixo do threshold)
        chromosome = np.random.random(size=self.n_features) * self.threshold * 0.8
        
        # Selecionar ~initial_feature_ratio para ficar acima do threshold
        n_to_select = max(1, int(self.n_features * self.initial_feature_ratio))
        n_to_select = np.random.randint(
            max(1, n_to_select // 2), 
            min(self.n_features, int(n_to_select * 1.5) + 1)
        )
        
        indices = np.random.choice(self.n_features, size=n_to_select, replace=False)
        chromosome[indices] = self.threshold + np.random.random(n_to_select) * (1 - self.threshold)
        
        return chromosome
    
    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        """Decodifica para binário usando threshold."""
        binary = (chromosome >= self.threshold).astype(int)
        
        if np.sum(binary) == 0:
            max_idx = np.argmax(chromosome)
            binary[max_idx] = 1
        
        return binary
    
    def mutate(self, chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        """
        Mutação gaussiana com bias para reduzir valores.
        
        - Valores altos (>=threshold) têm tendência a diminuir
        - Valores baixos (<threshold) têm pequena chance de aumentar
        """
        mutated = chromosome.copy()
        
        for i in range(self.n_features):
            if np.random.random() < mutation_rate:
                if mutated[i] >= self.threshold:
                    # Feature ativa: 70% chance de empurrar para baixo
                    if np.random.random() < 0.7:
                        # Empurra para baixo do threshold
                        noise = -np.random.uniform(0.1, 0.3)
                    else:
                        # Pequena perturbação
                        noise = np.random.normal(0, 0.05)
                else:
                    # Feature inativa: 25% chance de empurrar para cima
                    if np.random.random() < 0.25:
                        # Empurra para cima do threshold
                        noise = np.random.uniform(0.2, 0.4)
                    else:
                        # Pequena perturbação
                        noise = np.random.normal(0, 0.05)
                
                mutated[i] = np.clip(mutated[i] + noise, 0, 1)
        
        return mutated
    
    def crossover(
        self, 
        parent1: np.ndarray, 
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crossover aritmético com bias para valores menores.
        
        Similar ao Gaussian, mas com peso maior para o menor valor.
        - offspring1: weighted blend favorecendo valor menor
        - offspring2: arithmetic mean padrão
        """
        offspring1 = np.zeros(self.n_features)
        offspring2 = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            
            p1_active = parent1[i] >= self.threshold
            p2_active = parent2[i] >= self.threshold
            
            if p1_active and p2_active:
                # Ambos ativos: offspring1 tende ao menor, offspring2 média
                weight = np.random.uniform(0.6, 0.9)  # peso maior pro menor
                offspring1[i] = weight * min_val + (1 - weight) * max_val
                offspring2[i] = 0.5 * parent1[i] + 0.5 * parent2[i]
                
            elif p1_active or p2_active:
                # Um ativo: offspring1 tende a desativar
                if np.random.random() < 0.6:
                    offspring1[i] = min_val * np.random.uniform(0.7, 1.0)
                    offspring2[i] = max_val * np.random.uniform(0.9, 1.1)
                else:
                    # Blend normal
                    alpha = np.random.uniform(0.3, 0.7)
                    offspring1[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
                    offspring2[i] = (1 - alpha) * parent1[i] + alpha * parent2[i]
                    
            else:
                # Ambos inativos: manter baixo, pequena chance de explorar
                offspring1[i] = min_val
                if np.random.random() < 0.1:
                    # Pequena chance de explorar
                    offspring2[i] = max_val + np.random.uniform(0, 0.2)
                else:
                    offspring2[i] = (parent1[i] + parent2[i]) / 2
        
        offspring1 = np.clip(offspring1, 0, 1)
        offspring2 = np.clip(offspring2, 0, 1)
        
        # Garantir pelo menos uma feature
        if np.sum(self.decode(offspring1)) == 0:
            max_idx = np.argmax(offspring1)
            offspring1[max_idx] = self.threshold + 0.1
            
        if np.sum(self.decode(offspring2)) == 0:
            max_idx = np.argmax(offspring2)
            offspring2[max_idx] = self.threshold + 0.1
        
        return offspring1, offspring2
