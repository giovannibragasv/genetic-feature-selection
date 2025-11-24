import numpy as np
from typing import Tuple, List
from .base_encoding import BaseEncoding


class AdaptiveEncoding(BaseEncoding):
    """
    Encoding Adaptativo: threshold ajusta dinamicamente baseado no fitness.
    
    - Fitness melhora → threshold aumenta (mais restritivo, menos features)
    - Fitness piora/estagna → threshold diminui (menos restritivo, mais features)
    """
    
    def __init__(
        self, 
        n_features: int,
        initial_threshold: float = 0.5,
        learning_rate: float = 0.01,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        initial_feature_ratio: float = 0.1
    ):
        super().__init__(n_features)
        self.initial_threshold = initial_threshold
        self.threshold = initial_threshold
        self.learning_rate = learning_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.initial_feature_ratio = initial_feature_ratio
        
        self.best_fitness = -np.inf
        self.fitness_history: List[float] = []
        self.threshold_history: List[float] = [initial_threshold]
        self.stagnation_counter = 0
        self.stagnation_threshold = 5
        self.n_updates = 0
    
    def initialize_chromosome(self) -> np.ndarray:
        """
        Inicializa com distribuição esparsa.
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
        """Decodifica usando threshold atual."""
        binary = (chromosome >= self.threshold).astype(int)
        
        if np.sum(binary) == 0:
            max_idx = np.argmax(chromosome)
            binary[max_idx] = 1
        
        return binary
    
    def update_threshold(self, current_fitness: float):
      """
      Atualiza threshold baseado no fitness atual.
      
      Lógica INVERTIDA para feature selection:
      - Fitness melhora → threshold DIMINUI (permite explorar com mais features)
      - Fitness estagna → threshold AUMENTA (força redução de features)
      """
      self.n_updates += 1
      self.fitness_history.append(current_fitness)
      
      if current_fitness > self.best_fitness + 0.001:
          self.best_fitness = current_fitness
          self.threshold -= self.learning_rate * 0.5
          self.stagnation_counter = 0
      else:
          self.stagnation_counter += 1
          
          if self.stagnation_counter >= self.stagnation_threshold:
              self.threshold += self.learning_rate * 2
              self.stagnation_counter = 0
          else:
              self.threshold += self.learning_rate
      
      self.threshold = np.clip(self.threshold, self.min_threshold, self.max_threshold)
      self.threshold_history.append(self.threshold)
      
    def mutate(self, chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Mutação gaussiana adaptativa."""
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
        """BLX-α crossover com alpha adaptativo."""
        alpha = 0.3 + 0.4 * (1.0 - self.threshold)
        
        min_vals = np.minimum(parent1, parent2)
        max_vals = np.maximum(parent1, parent2)
        range_vals = max_vals - min_vals
        
        offspring1 = min_vals - alpha * range_vals + np.random.random(self.n_features) * (range_vals * (1 + 2 * alpha))
        offspring2 = min_vals - alpha * range_vals + np.random.random(self.n_features) * (range_vals * (1 + 2 * alpha))
        
        offspring1 = np.clip(offspring1, 0, 1)
        offspring2 = np.clip(offspring2, 0, 1)
        
        return offspring1, offspring2
    
    def reset(self):
        """Reseta o encoding para estado inicial."""
        self.threshold = self.initial_threshold
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.threshold_history = [self.initial_threshold]
        self.stagnation_counter = 0
        self.n_updates = 0
    
    def get_threshold_history(self) -> List[float]:
        """Retorna histórico de thresholds."""
        return self.threshold_history
    
    def get_fitness_history(self) -> List[float]:
        """Retorna histórico de fitness."""
        return self.fitness_history
    
    def get_stats(self) -> dict:
        """Retorna estatísticas do encoding."""
        return {
            'current_threshold': self.threshold,
            'best_fitness': self.best_fitness,
            'n_updates': self.n_updates,
            'threshold_range': (min(self.threshold_history), max(self.threshold_history)),
            'stagnation_counter': self.stagnation_counter
        }