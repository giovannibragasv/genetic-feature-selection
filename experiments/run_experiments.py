#!/usr/bin/env python3
"""
Script principal para rodar experimentos de feature selection.
Replica metodologia do paper Feng 2024 (PLOS ONE).

Uso:
    python run_experiments.py                    # Roda todos datasets e encodings
    python run_experiments.py --dataset colon   # Apenas dataset colon
    python run_experiments.py --encoding binary # Apenas encoding binary
    python run_experiments.py --n_runs 10       # Apenas 10 runs (teste rápido)
"""

import os
import sys
import yaml
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import DataLoader, DataPreprocessor
from algorithms import GeneticAlgorithm, MatrixGeneticAlgorithm
from encoding import (
    BinaryEncoding, DecimalEncoding, RealEncoding,
    GaussianEncoding, AdaptiveEncoding
)
from fitness import AccuracyFitness
from utils import compute_statistics, format_result


class ExperimentRunner:
    """Classe para executar experimentos de feature selection."""
    
    def __init__(self, config_path: str = "config/base.yaml"):
        """
        Inicializa o runner com configurações.
        
        Args:
            config_path: Caminho para arquivo de configuração base
        """
        self.base_path = Path(__file__).parent
        self.config = self._load_config(config_path)
        self.results_path = self.base_path.parent / "results"
        self.results_path.mkdir(exist_ok=True)
        (self.results_path / "tables").mkdir(exist_ok=True)
        (self.results_path / "figures").mkdir(exist_ok=True)
        (self.results_path / "logs").mkdir(exist_ok=True)
        
        self.data_loader = DataLoader(str(self.base_path.parent / "data" / "raw"))
        
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configuração YAML."""
        full_path = self.base_path / config_path
        with open(full_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_dataset_config(self, dataset_name: str) -> Dict:
        """Carrega configuração específica do dataset."""
        config_path = self.base_path / "config" / f"{dataset_name}.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _create_encoding(self, encoding_name: str, n_features: int):
        """Cria instância do encoding especificado."""
        params = self.config.get('encoding_params', {}).get(encoding_name, {})
        
        if encoding_name == 'binary':
            return BinaryEncoding(n_features, **params)
        elif encoding_name == 'decimal':
            return DecimalEncoding(n_features, **params)
        elif encoding_name == 'real':
            return RealEncoding(n_features, **params)
        elif encoding_name == 'gaussian':
            return GaussianEncoding(n_features, **params)
        elif encoding_name == 'adaptive':
            return AdaptiveEncoding(n_features, **params)
        else:
            raise ValueError(f"Encoding desconhecido: {encoding_name}")
    
    def run_single_experiment(
        self,
        dataset_name: str,
        encoding_name: str,
        random_seed: int,
        verbose: bool = False
    ) -> Dict:
        """
        Executa um único experimento.
        
        Args:
            dataset_name: Nome do dataset
            encoding_name: Nome do encoding
            random_seed: Seed para reprodutibilidade
            verbose: Se True, imprime progresso
            
        Returns:
            Dicionário com resultados
        """
        np.random.seed(random_seed)
        
        # Carregar dados
        ga_params = self.config['ga_params']
        knn_params = self.config['knn_params']
        exp_params = self.config['experiment']
        
        # Carregar dataset
        X_train, y_train, X_test, y_test = self.data_loader.load_dataset(
            dataset_name,
            test_size=exp_params['test_size'],
            random_state=random_seed
        )
        
        # Normalização z-score
        preprocessor = DataPreprocessor(normalization='standard', random_state=random_seed)
        X_train, y_train = preprocessor.fit_transform(X_train, y_train)
        X_test = preprocessor.transform(X_test)
        
        n_features = X_train.shape[1]
        
        # Criar encoding
        encoding = self._create_encoding(encoding_name, n_features)
        
        # Resetar encoding se adaptativo
        if hasattr(encoding, 'reset'):
            encoding.reset()
        
        # Criar função de fitness
        fitness_fn = AccuracyFitness(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            k=knn_params['k'],
            metric=knn_params['metric'],
            penalty_weight=0.0,  # Paper não usa penalty
            encoding=encoding
        )
        
        # Criar e executar GA
        ga = GeneticAlgorithm(
            population_size=ga_params['population_size'],
            generations=ga_params['generations'],
            crossover_rate=ga_params['crossover_rate'],
            mutation_rate=ga_params['mutation_rate'],
            tournament_size=ga_params['tournament_size'],
            elitism=ga_params['elitism'],
            encoding=encoding,
            random_state=random_seed
        )
        
        start_time = time.time()
        ga.fit(n_features, fitness_fn, verbose=verbose)
        elapsed_time = time.time() - start_time
        
        # Coletar resultados
        selected_features = ga.get_selected_features()
        n_selected = len(selected_features)
        
        result = {
            'dataset': dataset_name,
            'encoding': encoding_name,
            'seed': random_seed,
            'accuracy': ga.best_fitness,
            'n_features_original': n_features,
            'n_features_selected': n_selected,
            'feature_reduction': 1 - (n_selected / n_features),
            'time_seconds': elapsed_time,
            'generations': ga_params['generations'],
            'selected_features': selected_features.tolist()
        }
        
        # Adicionar stats do encoding adaptativo
        if encoding_name == 'adaptive':
            stats = encoding.get_stats()
            result['final_threshold'] = stats['current_threshold']
            result['threshold_range'] = stats['threshold_range']
        
        return result
    
    def run_dataset_encoding(
        self,
        dataset_name: str,
        encoding_name: str,
        n_runs: Optional[int] = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Executa múltiplos runs para um par dataset/encoding.
        
        Args:
            dataset_name: Nome do dataset
            encoding_name: Nome do encoding
            n_runs: Número de runs (default: config)
            verbose: Se True, imprime progresso
            
        Returns:
            Lista de resultados
        """
        if n_runs is None:
            n_runs = self.config['experiment']['n_runs']
        
        seeds = self.config['random_seeds'][:n_runs]
        results = []
        
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name.upper()} | Encoding: {encoding_name.upper()}")
        print(f"Runs: {n_runs}")
        print(f"{'='*60}")
        
        for i, seed in enumerate(seeds):
            print(f"  Run {i+1}/{n_runs} (seed={seed})...", end=" ", flush=True)
            
            try:
                result = self.run_single_experiment(
                    dataset_name, encoding_name, seed, verbose=False
                )
                results.append(result)
                print(f"Accuracy: {result['accuracy']:.4f}, "
                      f"Features: {result['n_features_selected']}, "
                      f"Time: {result['time_seconds']:.1f}s")
            except Exception as e:
                print(f"ERRO: {e}")
                results.append({
                    'dataset': dataset_name,
                    'encoding': encoding_name,
                    'seed': seed,
                    'accuracy': None,
                    'error': str(e)
                })
        
        # Estatísticas
        valid_results = [r for r in results if r.get('accuracy') is not None]
        if valid_results:
            accuracies = [r['accuracy'] for r in valid_results]
            features = [r['n_features_selected'] for r in valid_results]
            times = [r['time_seconds'] for r in valid_results]
            
            stats = compute_statistics(accuracies)
            print(f"\n  Resumo: {format_result(stats['mean'], stats['std'])}")
            print(f"  Features: {np.mean(features):.1f} ± {np.std(features):.1f}")
            print(f"  Tempo médio: {np.mean(times):.1f}s")
        
        return results
    
    def run_all_experiments(
        self,
        datasets: Optional[List[str]] = None,
        encodings: Optional[List[str]] = None,
        n_runs: Optional[int] = None,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Executa todos os experimentos.
        
        Args:
            datasets: Lista de datasets (default: todos)
            encodings: Lista de encodings (default: todos)
            n_runs: Número de runs por experimento
            save_intermediate: Se True, salva resultados parciais
            
        Returns:
            DataFrame com todos os resultados
        """
        if datasets is None:
            datasets = ['colon', 'leukemia', 'cns', 'mll', 'ovarian']
        
        if encodings is None:
            encodings = self.config['encodings']
        
        all_results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        total_experiments = len(datasets) * len(encodings)
        current = 0
        
        print("\n" + "="*70)
        print("EXPERIMENTOS DE FEATURE SELECTION - REPLICAÇÃO FENG 2024")
        print("="*70)
        print(f"Datasets: {datasets}")
        print(f"Encodings: {encodings}")
        print(f"Runs por experimento: {n_runs or self.config['experiment']['n_runs']}")
        print(f"Total de configurações: {total_experiments}")
        print("="*70)
        
        for dataset in datasets:
            for encoding in encodings:
                current += 1
                print(f"\n[{current}/{total_experiments}] ", end="")
                
                results = self.run_dataset_encoding(
                    dataset, encoding, n_runs=n_runs
                )
                all_results.extend(results)
                
                # Salvar resultados intermediários
                if save_intermediate:
                    df_temp = pd.DataFrame(all_results)
                    df_temp.to_csv(
                        self.results_path / "tables" / f"results_partial_{timestamp}.csv",
                        index=False
                    )
        
        # Criar DataFrame final
        df = pd.DataFrame(all_results)
        
        # Salvar resultados finais
        output_path = self.results_path / "tables" / f"results_full_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n\nResultados salvos em: {output_path}")
        
        # Gerar tabela resumo
        self._generate_summary_table(df, timestamp)
        
        return df
    
    def _generate_summary_table(self, df: pd.DataFrame, timestamp: str):
        """Gera tabela resumo no formato do paper (Table 5)."""
        # Filtrar resultados válidos
        df_valid = df[df['accuracy'].notna()].copy()
        
        # Agrupar por dataset e encoding
        summary = df_valid.groupby(['dataset', 'encoding']).agg({
            'accuracy': ['mean', 'std', 'max'],
            'n_features_selected': ['mean', 'std'],
            'time_seconds': 'mean'
        }).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        # Criar tabela pivotada (formato Table 5)
        pivot_mean = df_valid.pivot_table(
            values='accuracy',
            index='dataset',
            columns='encoding',
            aggfunc='mean'
        ).round(4)
        
        pivot_std = df_valid.pivot_table(
            values='accuracy',
            index='dataset',
            columns='encoding',
            aggfunc='std'
        ).round(4)
        
        # Salvar
        summary.to_csv(
            self.results_path / "tables" / f"summary_{timestamp}.csv",
            index=False
        )
        
        pivot_mean.to_csv(
            self.results_path / "tables" / f"table5_mean_{timestamp}.csv"
        )
        
        # Criar tabela formatada (mean ± std)
        table5_formatted = pivot_mean.copy()
        for col in table5_formatted.columns:
            for idx in table5_formatted.index:
                mean = pivot_mean.loc[idx, col]
                std = pivot_std.loc[idx, col]
                table5_formatted.loc[idx, col] = f"{mean:.4f} ± {std:.4f}"
        
        table5_formatted.to_csv(
            self.results_path / "tables" / f"table5_formatted_{timestamp}.csv"
        )
        
        print("\n" + "="*70)
        print("TABLE 5: ACCURACY COMPARISON (Mean ± Std)")
        print("="*70)
        print(table5_formatted.to_string())
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Executa experimentos de feature selection (Feng 2024)"
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=None,
        help='Dataset específico (colon, leukemia, cns, mll, ovarian)'
    )
    parser.add_argument(
        '--encoding', '-e',
        type=str,
        default=None,
        help='Encoding específico (binary, decimal, real, gaussian, adaptive)'
    )
    parser.add_argument(
        '--n_runs', '-n',
        type=int,
        default=None,
        help='Número de runs por experimento (default: 50)'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Modo rápido: 5 runs apenas'
    )
    
    args = parser.parse_args()
    
    # Configurar
    runner = ExperimentRunner()
    
    datasets = [args.dataset] if args.dataset else None
    encodings = [args.encoding] if args.encoding else None
    n_runs = 5 if args.quick else args.n_runs
    
    # Executar
    df = runner.run_all_experiments(
        datasets=datasets,
        encodings=encodings,
        n_runs=n_runs
    )
    
    print("\nExperimentos concluídos!")
    print(f"Total de runs: {len(df)}")
    print(f"Runs com sucesso: {df['accuracy'].notna().sum()}")


if __name__ == "__main__":
    main()
