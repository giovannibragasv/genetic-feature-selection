import numpy as np
from typing import Tuple, Optional
from .base_encoding import BaseEncoding


class AdaptiveEncoding(BaseEncoding):
    """
    Encoding adaptativo: threshold ajusta dinamicamente baseado no fitness.
    
    Genes são valores reais [0, 1]. O threshold que define seleção de features
    é ajustado durante a evolução:
    - Fitness melhora → threshold aumenta (mais restritivo, menos features)
    - Fitness piora → threshold diminui (menos restritivo, mais features)
    """
    
    def __init__(
        self, 
        n_features: int, 
        initial_threshold: float = 0.5,
        learning_rate: float = 0.01,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9
    ):
        super().__init__(n_features)
        self.threshold = initial_threshold
        self.learning_rate = learning_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        self._best_fitness: Optional[float] = None
        self._previous_fitness: Optional[float] = None
        self._threshold_history: list[float] = [initial_threshold]
        self._fitness_history: list[float] = []
        self._stagnation_counter: int = 0
        self._stagnation_limit: int = 5
    
    def initialize_chromosome(self) -> np.ndarray:
        """Inicializa chromosome com valores reais aleatórios [0, 1]."""
        return np.random.random(size=self.n_features)
    
    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Decodifica para binário usando threshold adaptativo.
        Feature selecionada se valor >= threshold.
        """
        binary = (chromosome >= self.threshold).astype(int)
        
        if np.sum(binary) == 0:
            max_idx = np.argmax(chromosome)
            binary[max_idx] = 1
        
        return binary
    
    def mutate(self, chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Mutação gaussiana com desvio adaptativo."""
        mutated = chromosome.copy()
        
        adaptive_std = 0.1 * (1.0 - self.threshold + 0.5)
        
        for i in range(self.n_features):
            if np.random.random() < mutation_rate:
                noise = np.random.normal(0, adaptive_std)
                mutated[i] = np.clip(mutated[i] + noise, 0, 1)
        
        return mutated
    
    def crossover(
        self, 
        parent1: np.ndarray, 
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Blend crossover (BLX-α) com alpha adaptativo."""
        alpha = 0.3 + 0.4 * (1.0 - self.threshold)
        
        min_vals = np.minimum(parent1, parent2)
        max_vals = np.maximum(parent1, parent2)
        range_vals = max_vals - min_vals
        
        lower = min_vals - alpha * range_vals
        upper = max_vals + alpha * range_vals
        
        offspring1 = lower + np.random.random(self.n_features) * (upper - lower)
        offspring2 = lower + np.random.random(self.n_features) * (upper - lower)
        
        offspring1 = np.clip(offspring1, 0, 1)
        offspring2 = np.clip(offspring2, 0, 1)
        
        return offspring1, offspring2
    
    def update_threshold(self, current_fitness: float) -> None:
        """
        Atualiza threshold baseado no fitness atual.
        
        Args:
            current_fitness: Fitness da melhor solução na geração atual
        """
        self._fitness_history.append(current_fitness)
        
        if self._best_fitness is None:
            self._best_fitness = current_fitness
            self._previous_fitness = current_fitness
            return
        
        improvement = current_fitness - self._previous_fitness
        
        if current_fitness > self._best_fitness:
            self._best_fitness = current_fitness
            self._stagnation_counter = 0
            
            adjustment = self.learning_rate * (1.0 + improvement)
            self.threshold = min(self.threshold + adjustment, self.max_threshold)
        
        elif improvement < 0:
            self._stagnation_counter += 1
            
            adjustment = self.learning_rate * 0.5
            self.threshold = max(self.threshold - adjustment, self.min_threshold)
        
        else:
            self._stagnation_counter += 1
        
        if self._stagnation_counter >= self._stagnation_limit:
            self._handle_stagnation()
        
        self._previous_fitness = current_fitness
        self._threshold_history.append(self.threshold)
    
    def _handle_stagnation(self) -> None:
        """Lida com estagnação reduzindo threshold para explorar mais."""
        adjustment = self.learning_rate * 2.0
        self.threshold = max(self.threshold - adjustment, self.min_threshold)
        self._stagnation_counter = 0
    
    def reset(self, initial_threshold: Optional[float] = None) -> None:
        """
        Reseta estado do encoding para nova execução.
        
        Args:
            initial_threshold: Novo threshold inicial (opcional)
        """
        if initial_threshold is not None:
            self.threshold = initial_threshold
        else:
            self.threshold = self._threshold_history[0]
        
        self._best_fitness = None
        self._previous_fitness = None
        self._threshold_history = [self.threshold]
        self._fitness_history = []
        self._stagnation_counter = 0
    
    def get_threshold_history(self) -> list[float]:
        """Retorna histórico de thresholds."""
        return self._threshold_history.copy()
    
    def get_fitness_history(self) -> list[float]:
        """Retorna histórico de fitness."""
        return self._fitness_history.copy()
    
    def get_stats(self) -> dict:
        """Retorna estatísticas do encoding."""
        return {
            'current_threshold': self.threshold,
            'best_fitness': self._best_fitness,
            'n_updates': len(self._threshold_history) - 1,
            'threshold_range': (min(self._threshold_history), max(self._threshold_history)),
            'stagnation_counter': self._stagnation_counter
        }
