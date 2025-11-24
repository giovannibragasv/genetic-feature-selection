import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class BaseEncoding(ABC):
    """
    Classe base abstrata para encodings de chromosomes.
    Define interface comum para todos os encodings.
    """
    
    def __init__(self, n_features: int):
        self.n_features = n_features
    
    @abstractmethod
    def initialize_chromosome(self) -> np.ndarray:
        """
        Inicializa um chromosome aleatório.
        
        Returns:
            np.ndarray: Chromosome codificado
        """
        pass
    
    @abstractmethod
    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Decodifica chromosome para seleção binária de features.
        
        Args:
            chromosome: Chromosome codificado
            
        Returns:
            np.ndarray: Array binário (1=feature selecionada, 0=não selecionada)
        """
        pass
    
    @abstractmethod
    def mutate(self, chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        """
        Aplica mutação ao chromosome.
        
        Args:
            chromosome: Chromosome a ser mutado
            mutation_rate: Taxa de mutação
            
        Returns:
            np.ndarray: Chromosome mutado
        """
        pass
    
    @abstractmethod
    def crossover(
        self, 
        parent1: np.ndarray, 
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza crossover entre dois parents.
        
        Args:
            parent1: Primeiro parent
            parent2: Segundo parent
            
        Returns:
            Tuple: (offspring1, offspring2)
        """
        pass
    
    def get_n_selected_features(self, chromosome: np.ndarray) -> int:
        """
        Retorna número de features selecionadas.
        
        Args:
            chromosome: Chromosome codificado
            
        Returns:
            int: Número de features selecionadas
        """
        binary = self.decode(chromosome)
        return np.sum(binary)