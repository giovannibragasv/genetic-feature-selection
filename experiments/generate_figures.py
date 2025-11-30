#!/usr/bin/env python3
"""
Gera figuras do paper Feng 2024.

Figuras:
- Figure 9: K-value analysis
- Convergence plots por dataset/encoding
- Feature comparison (antes/depois)
- Box plots de distribuição (50 runs)
- Threshold evolution (adaptive encoding)
"""

import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import DataLoader, DataPreprocessor
from algorithms import KNNClassifier, find_optimal_k, GeneticAlgorithm
from encoding import (
    BinaryEncoding, DecimalEncoding, RealEncoding,
    GaussianEncoding, AdaptiveEncoding
)
from fitness import AccuracyFitness
from utils import (
    plot_k_value_analysis,
    plot_multi_encoding_convergence,
    plot_accuracy_comparison,
    plot_boxplot_results,
    plot_feature_comparison,
    plot_threshold_evolution
)


class FigureGenerator:
    """Gera figuras do paper."""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.results_path = self.base_path.parent / "results"
        self.figures_path = self.results_path / "figures"
        self.figures_path.mkdir(exist_ok=True)
        
        self.data_loader = DataLoader(str(self.base_path.parent / "data" / "raw"))
        
        # Carregar config
        with open(self.base_path / "config" / "base.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
    
    def generate_figure9_k_analysis(
        self,
        dataset_name: str = 'colon',
        k_range: range = range(1, 19),
        save: bool = True
    ):
        """
        Gera Figure 9: análise do valor K do KNN.
        
        Testa k de 1 a 18 e mostra que k=7 é ótimo.
        """
        print(f"\nGerando Figure 9: K-value analysis ({dataset_name})...")
        
        # Carregar dados
        X_train, y_train, X_test, y_test = self.data_loader.load_dataset(
            dataset_name, test_size=0.3, random_state=42
        )
        
        # Normalizar
        preprocessor = DataPreprocessor(normalization='standard')
        X_train, _ = preprocessor.fit_transform(X_train, y_train)
        X_test = preprocessor.transform(X_test)
        
        # Testar diferentes K
        k_values = list(k_range)
        accuracies = []
        
        for k in k_values:
            knn = KNNClassifier(k=k, distance_metric='euclidean')
            knn.fit(X_train, y_train)
            acc = knn.score(X_test, y_test)
            accuracies.append(acc)
            print(f"  k={k}: accuracy={acc:.4f}")
        
        # Encontrar k ótimo
        best_idx = np.argmax(accuracies)
        best_k = k_values[best_idx]
        print(f"\n  Melhor k={best_k} com accuracy={accuracies[best_idx]:.4f}")
        
        # Plotar
        save_path = self.figures_path / f"figure9_k_analysis_{dataset_name}.png" if save else None
        
        fig = plot_k_value_analysis(
            k_values=k_values,
            accuracies=accuracies,
            optimal_k=7,  # Paper usa k=7
            title=f"Figure 9: K-Value Analysis ({dataset_name.upper()})",
            save_path=str(save_path) if save_path else None,
            show=False
        )
        
        plt.close(fig)
        return k_values, accuracies
    
    def generate_convergence_plots(
        self,
        dataset_name: str = 'colon',
        save: bool = True
    ):
        """
        Gera plots de convergência para todos encodings em um dataset.
        """
        print(f"\nGerando convergence plots ({dataset_name})...")
        
        # Carregar dados
        X_train, y_train, X_test, y_test = self.data_loader.load_dataset(
            dataset_name, test_size=0.3, random_state=42
        )
        
        preprocessor = DataPreprocessor(normalization='standard')
        X_train, _ = preprocessor.fit_transform(X_train, y_train)
        X_test = preprocessor.transform(X_test)
        
        n_features = X_train.shape[1]
        ga_params = self.config['ga_params']
        knn_params = self.config['knn_params']
        
        # Rodar GA para cada encoding
        results = {}
        encodings = {
            'Binary': BinaryEncoding(n_features, initial_feature_ratio=0.1),
            'Decimal': DecimalEncoding(n_features, initial_feature_ratio=0.1),
            'Real': RealEncoding(n_features, initial_feature_ratio=0.1),
            'Gaussian': GaussianEncoding(n_features, initial_feature_ratio=0.1),
            'Adaptive': AdaptiveEncoding(n_features, initial_feature_ratio=0.1)
        }
        
        for name, encoding in encodings.items():
            print(f"  Rodando {name}...", end=" ", flush=True)
            
            if hasattr(encoding, 'reset'):
                encoding.reset()
            
            fitness_fn = AccuracyFitness(
                X_train, y_train, X_test, y_test,
                k=knn_params['k'],
                encoding=encoding
            )
            
            ga = GeneticAlgorithm(
                population_size=ga_params['population_size'],
                generations=ga_params['generations'],
                crossover_rate=ga_params['crossover_rate'],
                mutation_rate=ga_params['mutation_rate'],
                encoding=encoding,
                random_state=42
            )
            
            ga.fit(n_features, fitness_fn, verbose=False)
            results[name] = ga.get_fitness_history()
            
            print(f"Fitness={ga.best_fitness:.4f}")
        
        # Plotar
        save_path = self.figures_path / f"convergence_{dataset_name}.png" if save else None
        
        fig = plot_multi_encoding_convergence(
            results=results,
            title=f"Convergence Comparison ({dataset_name.upper()})",
            save_path=str(save_path) if save_path else None,
            show=False
        )
        
        plt.close(fig)
        return results
    
    def generate_accuracy_comparison(
        self,
        results_csv: Optional[str] = None,
        save: bool = True
    ):
        """
        Gera gráfico de comparação de accuracy entre encodings.
        Usa resultados de experimentos anteriores se disponíveis.
        """
        print("\nGerando accuracy comparison...")
        
        if results_csv:
            df = pd.read_csv(results_csv)
        else:
            # Procurar último arquivo de resultados
            tables_path = self.results_path / "tables"
            csv_files = list(tables_path.glob("results_full_*.csv"))
            
            if not csv_files:
                print("  Nenhum arquivo de resultados encontrado. Execute run_experiments.py primeiro.")
                return None
            
            latest = max(csv_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest)
            print(f"  Usando: {latest.name}")
        
        # Calcular médias e stds por encoding
        df_valid = df[df['accuracy'].notna()]
        
        summary = df_valid.groupby('encoding').agg({
            'accuracy': ['mean', 'std']
        }).round(4)
        
        encoding_names = summary.index.tolist()
        means = summary[('accuracy', 'mean')].tolist()
        stds = summary[('accuracy', 'std')].tolist()
        
        # Plotar
        save_path = self.figures_path / "accuracy_comparison.png" if save else None
        
        fig = plot_accuracy_comparison(
            encoding_names=encoding_names,
            accuracies=means,
            stds=stds,
            title="Accuracy Comparison by Encoding Method",
            save_path=str(save_path) if save_path else None,
            show=False
        )
        
        plt.close(fig)
        return summary
    
    def generate_boxplots(
        self,
        results_csv: Optional[str] = None,
        save: bool = True
    ):
        """
        Gera boxplot de distribuição dos resultados.
        """
        print("\nGerando boxplots...")
        
        if results_csv:
            df = pd.read_csv(results_csv)
        else:
            tables_path = self.results_path / "tables"
            csv_files = list(tables_path.glob("results_full_*.csv"))
            
            if not csv_files:
                print("  Nenhum arquivo de resultados encontrado.")
                return None
            
            latest = max(csv_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest)
        
        df_valid = df[df['accuracy'].notna()]
        
        # Agrupar por encoding
        results = {}
        for encoding in df_valid['encoding'].unique():
            results[encoding] = df_valid[df_valid['encoding'] == encoding]['accuracy'].tolist()
        
        # Plotar
        save_path = self.figures_path / "boxplot_results.png" if save else None
        
        fig = plot_boxplot_results(
            results=results,
            title="Distribution of Accuracy Results (Multiple Runs)",
            save_path=str(save_path) if save_path else None,
            show=False
        )
        
        plt.close(fig)
        return results
    
    def generate_feature_comparison(
        self,
        results_csv: Optional[str] = None,
        save: bool = True
    ):
        """
        Gera gráfico de comparação de features antes/depois.
        """
        print("\nGerando feature comparison...")
        
        if results_csv:
            df = pd.read_csv(results_csv)
        else:
            tables_path = self.results_path / "tables"
            csv_files = list(tables_path.glob("results_full_*.csv"))
            
            if not csv_files:
                print("  Nenhum arquivo de resultados encontrado.")
                return None
            
            latest = max(csv_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest)
        
        df_valid = df[df['accuracy'].notna()]
        
        # Calcular médias
        summary = df_valid.groupby('encoding').agg({
            'n_features_original': 'first',
            'n_features_selected': 'mean'
        }).round(0)
        
        encoding_names = summary.index.tolist()
        features_before = summary['n_features_original'].tolist()
        features_after = summary['n_features_selected'].tolist()
        
        # Plotar
        save_path = self.figures_path / "feature_comparison.png" if save else None
        
        fig = plot_feature_comparison(
            encoding_names=encoding_names,
            features_before=[int(f) for f in features_before],
            features_after=[int(f) for f in features_after],
            title="Feature Selection Comparison",
            save_path=str(save_path) if save_path else None,
            show=False
        )
        
        plt.close(fig)
        return summary
    
    def generate_all_figures(self, dataset: str = 'colon'):
        """
        Gera todas as figuras.
        """
        print("\n" + "="*60)
        print("GERANDO TODAS AS FIGURAS")
        print("="*60)
        
        # Figure 9: K-value analysis
        self.generate_figure9_k_analysis(dataset)
        
        # Convergence plots
        self.generate_convergence_plots(dataset)
        
        # Usar resultados existentes se disponíveis
        self.generate_accuracy_comparison()
        self.generate_boxplots()
        self.generate_feature_comparison()
        
        print("\n" + "="*60)
        print(f"Figuras salvas em: {self.figures_path}")
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Gera figuras do paper")
    parser.add_argument('--dataset', '-d', default='colon', help='Dataset para análises')
    parser.add_argument('--results', '-r', default=None, help='Arquivo CSV de resultados')
    parser.add_argument('--figure', '-f', default='all', 
                       choices=['all', 'k_analysis', 'convergence', 'accuracy', 'boxplot', 'features'],
                       help='Figura específica para gerar')
    
    args = parser.parse_args()
    
    generator = FigureGenerator()
    
    if args.figure == 'all':
        generator.generate_all_figures(args.dataset)
    elif args.figure == 'k_analysis':
        generator.generate_figure9_k_analysis(args.dataset)
    elif args.figure == 'convergence':
        generator.generate_convergence_plots(args.dataset)
    elif args.figure == 'accuracy':
        generator.generate_accuracy_comparison(args.results)
    elif args.figure == 'boxplot':
        generator.generate_boxplots(args.results)
    elif args.figure == 'features':
        generator.generate_feature_comparison(args.results)


if __name__ == "__main__":
    main()
