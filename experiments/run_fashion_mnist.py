#!/usr/bin/env python3
"""
Experimento Fashion-MNIST para replicar Table 3 do paper Feng 2024.

Paper usa:
- 80000 train + 15000 test
- Mesmos parâmetros GA
- 5 encodings

Para execução mais rápida, usamos subset menor (configurável).
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.fashion_mnist_loader import FashionMNISTLoader
from data.preprocessor import DataPreprocessor
from algorithms import GeneticAlgorithm
from encoding import (
    BinaryEncoding, DecimalEncoding, RealEncoding,
    GaussianEncoding, AdaptiveEncoding
)
from fitness import AccuracyFitness
from utils import compute_statistics, format_result


def run_fashion_mnist_experiment(
    n_samples: int = 10000,
    n_runs: int = 10,
    save_results: bool = True
):
    """
    Executa experimento Fashion-MNIST.
    
    Args:
        n_samples: Número total de amostras (default: 10000 para rapidez)
        n_runs: Número de runs por encoding
        save_results: Se True, salva CSV
    """
    print("="*70)
    print("EXPERIMENTO FASHION-MNIST - REPLICAÇÃO TABLE 3 FENG 2024")
    print("="*70)
    
    # Parâmetros do paper
    ga_params = {
        'population_size': 600,
        'generations': 55,
        'crossover_rate': 1.0,
        'mutation_rate': 0.2,
        'tournament_size': 3,
        'elitism': 2
    }
    
    knn_k = 7
    test_size = 0.3
    
    seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021][:n_runs]
    
    print(f"Samples: {n_samples}")
    print(f"Runs: {n_runs}")
    print(f"GA params: pop={ga_params['population_size']}, gen={ga_params['generations']}")
    print("="*70)
    
    # Carregar dados
    loader = FashionMNISTLoader(
        data_root=str(Path(__file__).parent.parent / "data" / "raw" / "fashion_mnist")
    )
    
    all_results = []
    
    encodings = ['binary', 'decimal', 'real', 'gaussian', 'adaptive']
    
    for enc_name in encodings:
        print(f"\n{'='*60}")
        print(f"Encoding: {enc_name.upper()}")
        print(f"{'='*60}")
        
        for i, seed in enumerate(seeds):
            print(f"  Run {i+1}/{n_runs} (seed={seed})...", end=" ", flush=True)
            
            np.random.seed(seed)
            
            # Carregar subset
            X_train, y_train, X_test, y_test = loader.load_subset(
                n_samples=n_samples,
                test_size=test_size,
                random_state=seed
            )
            
            # Normalizar (z-score adicional, já está em [0,1])
            preprocessor = DataPreprocessor(normalization='standard', random_state=seed)
            X_train, _ = preprocessor.fit_transform(X_train, y_train)
            X_test = preprocessor.transform(X_test)
            
            n_features = X_train.shape[1]  # 784
            
            # Criar encoding
            if enc_name == 'binary':
                encoding = BinaryEncoding(n_features, initial_feature_ratio=0.5)
            elif enc_name == 'decimal':
                encoding = DecimalEncoding(n_features, initial_feature_ratio=0.5)
            elif enc_name == 'real':
                encoding = RealEncoding(n_features, initial_feature_ratio=0.5)
            elif enc_name == 'gaussian':
                encoding = GaussianEncoding(n_features, initial_feature_ratio=0.5)
            elif enc_name == 'adaptive':
                encoding = AdaptiveEncoding(n_features, initial_feature_ratio=0.5)
            
            if hasattr(encoding, 'reset'):
                encoding.reset()
            
            # Fitness
            fitness_fn = AccuracyFitness(
                X_train, y_train, X_test, y_test,
                k=knn_k,
                encoding=encoding
            )
            
            # GA
            ga = GeneticAlgorithm(
                population_size=ga_params['population_size'],
                generations=ga_params['generations'],
                crossover_rate=ga_params['crossover_rate'],
                mutation_rate=ga_params['mutation_rate'],
                tournament_size=ga_params['tournament_size'],
                elitism=ga_params['elitism'],
                encoding=encoding,
                random_state=seed
            )
            
            start = time.time()
            ga.fit(n_features, fitness_fn, verbose=False)
            elapsed = time.time() - start
            
            selected = ga.get_selected_features()
            
            result = {
                'dataset': 'fashion_mnist',
                'encoding': enc_name,
                'seed': seed,
                'accuracy': ga.best_fitness,
                'n_features_original': n_features,
                'n_features_selected': len(selected),
                'feature_reduction': 1 - (len(selected) / n_features),
                'time_seconds': elapsed
            }
            
            all_results.append(result)
            
            print(f"Acc={result['accuracy']:.4f}, "
                  f"Features={result['n_features_selected']}/{n_features}, "
                  f"Time={elapsed:.1f}s")
    
    # Criar DataFrame
    df = pd.DataFrame(all_results)
    
    # Resumo (Table 3 format)
    print("\n" + "="*70)
    print("TABLE 3: FASHION-MNIST RESULTS")
    print("="*70)
    
    summary = df.groupby('encoding').agg({
        'n_features_original': 'first',
        'n_features_selected': 'mean',
        'accuracy': ['mean', 'std']
    }).round(4)
    
    print(f"{'Encoding':<12} {'Before':<10} {'After':<10} {'Accuracy':<20}")
    print("-"*52)
    
    for enc in encodings:
        enc_data = df[df['encoding'] == enc]
        before = int(enc_data['n_features_original'].iloc[0])
        after = int(enc_data['n_features_selected'].mean())
        acc_mean = enc_data['accuracy'].mean()
        acc_std = enc_data['accuracy'].std()
        print(f"{enc.capitalize():<12} {before:<10} {after:<10} {acc_mean:.4f} ± {acc_std:.4f}")
    
    print("="*70)
    
    # Salvar
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(__file__).parent.parent / "results" / "tables"
        results_path.mkdir(parents=True, exist_ok=True)
        
        output_file = results_path / f"fashion_mnist_results_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResultados salvos em: {output_file}")
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fashion-MNIST experiment")
    parser.add_argument('--samples', '-s', type=int, default=10000,
                       help='Número de amostras (default: 10000)')
    parser.add_argument('--runs', '-n', type=int, default=10,
                       help='Número de runs (default: 10)')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Modo rápido: 1000 samples, 3 runs')
    
    args = parser.parse_args()
    
    if args.quick:
        run_fashion_mnist_experiment(n_samples=1000, n_runs=3)
    else:
        run_fashion_mnist_experiment(n_samples=args.samples, n_runs=args.runs)


if __name__ == "__main__":
    main()
