#!/usr/bin/env python3
"""
Teste rápido para verificar se todos os componentes estão funcionando.
Executa um mini-experimento com poucos runs para validação.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_imports():
    """Testa se todos os imports funcionam."""
    print("="*60)
    print("TESTANDO IMPORTS")
    print("="*60)
    
    try:
        from data import DataLoader, DataPreprocessor
        print("✓ data.DataLoader, DataPreprocessor")
        
        from algorithms import KNNClassifier, GeneticAlgorithm, MatrixGeneticAlgorithm
        print("✓ algorithms.KNNClassifier, GeneticAlgorithm, MatrixGeneticAlgorithm")
        
        from encoding import (
            BinaryEncoding, DecimalEncoding, RealEncoding,
            GaussianEncoding, AdaptiveEncoding
        )
        print("✓ encoding.* (5 encodings)")
        
        from fitness import AccuracyFitness
        print("✓ fitness.AccuracyFitness")
        
        from utils import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, compute_statistics, format_result,
            plot_convergence, plot_accuracy_comparison
        )
        print("✓ utils.metrics")
        print("✓ utils.visualization")
        
        return True
    except Exception as e:
        print(f"✗ ERRO: {e}")
        return False


def test_data_loading():
    """Testa carregamento de dados."""
    print("\n" + "="*60)
    print("TESTANDO CARREGAMENTO DE DADOS")
    print("="*60)
    
    try:
        from data import DataLoader
        
        loader = DataLoader(str(Path(__file__).parent.parent / "data" / "raw"))
        
        datasets = ['colon', 'leukemia', 'cns', 'mll', 'ovarian']
        
        for ds in datasets:
            try:
                X_train, y_train, X_test, y_test = loader.load_dataset(
                    ds, test_size=0.3, random_state=42
                )
                n_features = X_train.shape[1]
                n_samples = X_train.shape[0] + X_test.shape[0]
                n_classes = len(np.unique(y_train))
                print(f"✓ {ds}: {n_samples} samples, {n_features} features, {n_classes} classes")
            except Exception as e:
                print(f"✗ {ds}: {e}")
        
        return True
    except Exception as e:
        print(f"✗ ERRO: {e}")
        return False


def test_single_encoding(encoding_name: str, n_features: int = 100):
    """Testa um encoding específico."""
    from encoding import (
        BinaryEncoding, DecimalEncoding, RealEncoding,
        GaussianEncoding, AdaptiveEncoding
    )
    
    encodings = {
        'binary': BinaryEncoding,
        'decimal': DecimalEncoding,
        'real': RealEncoding,
        'gaussian': GaussianEncoding,
        'adaptive': AdaptiveEncoding
    }
    
    EncodingClass = encodings[encoding_name]
    encoding = EncodingClass(n_features, initial_feature_ratio=0.1)
    
    # Testar inicialização
    chromosome = encoding.initialize_chromosome()
    assert len(chromosome) == n_features, "Tamanho incorreto"
    
    # Testar decode
    binary = encoding.decode(chromosome)
    assert len(binary) == n_features, "Decode incorreto"
    assert binary.dtype == int or binary.dtype == np.int64, "Tipo incorreto"
    
    # Testar mutação
    mutated = encoding.mutate(chromosome, 0.2)
    assert len(mutated) == n_features, "Mutação incorreta"
    
    # Testar crossover
    parent1 = encoding.initialize_chromosome()
    parent2 = encoding.initialize_chromosome()
    off1, off2 = encoding.crossover(parent1, parent2)
    assert len(off1) == n_features, "Crossover incorreto"
    
    return True


def test_all_encodings():
    """Testa todos os encodings."""
    print("\n" + "="*60)
    print("TESTANDO ENCODINGS")
    print("="*60)
    
    encodings = ['binary', 'decimal', 'real', 'gaussian', 'adaptive']
    
    for enc in encodings:
        try:
            test_single_encoding(enc)
            print(f"✓ {enc}")
        except Exception as e:
            print(f"✗ {enc}: {e}")
    
    return True


def test_quick_experiment():
    """Executa um experimento rápido para validação."""
    print("\n" + "="*60)
    print("TESTANDO EXPERIMENTO RÁPIDO")
    print("="*60)
    
    try:
        from data import DataLoader, DataPreprocessor
        from algorithms import GeneticAlgorithm
        from encoding import AdaptiveEncoding
        from fitness import AccuracyFitness
        
        print("Carregando dataset colon...")
        loader = DataLoader(str(Path(__file__).parent.parent / "data" / "raw"))
        X_train, y_train, X_test, y_test = loader.load_dataset(
            'colon', test_size=0.3, random_state=42
        )
        
        print("Normalizando dados...")
        preprocessor = DataPreprocessor(normalization='standard')
        X_train, _ = preprocessor.fit_transform(X_train, y_train)
        X_test = preprocessor.transform(X_test)
        
        n_features = X_train.shape[1]
        print(f"Features: {n_features}")
        
        print("Criando encoding adaptativo...")
        encoding = AdaptiveEncoding(n_features, initial_feature_ratio=0.1)
        
        print("Criando função de fitness...")
        fitness_fn = AccuracyFitness(
            X_train, y_train, X_test, y_test,
            k=7,
            encoding=encoding
        )
        
        print("Executando GA (pop=50, gen=10)...")
        ga = GeneticAlgorithm(
            population_size=50,
            generations=10,
            crossover_rate=1.0,
            mutation_rate=0.2,
            encoding=encoding,
            random_state=42
        )
        
        start = time.time()
        ga.fit(n_features, fitness_fn, verbose=True)
        elapsed = time.time() - start
        
        selected = ga.get_selected_features()
        
        print(f"\n✓ Experimento concluído em {elapsed:.1f}s")
        print(f"  - Accuracy: {ga.best_fitness:.4f}")
        print(f"  - Features: {len(selected)}/{n_features}")
        print(f"  - Redução: {(1 - len(selected)/n_features)*100:.1f}%")
        
        return True
    except Exception as e:
        print(f"✗ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Testa funções de métricas."""
    print("\n" + "="*60)
    print("TESTANDO MÉTRICAS")
    print("="*60)
    
    try:
        from utils import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, compute_statistics, format_result
        )
        
        # Dados de teste
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"✓ accuracy_score: {acc:.4f}")
        print(f"✓ precision_score: {prec:.4f}")
        print(f"✓ recall_score: {rec:.4f}")
        print(f"✓ f1_score: {f1:.4f}")
        print(f"✓ confusion_matrix shape: {cm.shape}")
        
        # Testar compute_statistics
        values = [0.85, 0.87, 0.86, 0.88, 0.84]
        stats = compute_statistics(values)
        formatted = format_result(stats['mean'], stats['std'])
        print(f"✓ compute_statistics: {formatted}")
        
        return True
    except Exception as e:
        print(f"✗ ERRO: {e}")
        return False


def main():
    """Executa todos os testes."""
    print("\n" + "="*60)
    print("TESTE DE VALIDAÇÃO DO SISTEMA")
    print("Feng 2024 Feature Selection Replication")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Encodings", test_all_encodings()))
    results.append(("Metrics", test_metrics()))
    results.append(("Quick Experiment", test_quick_experiment()))
    
    print("\n" + "="*60)
    print("RESUMO DOS TESTES")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ TODOS OS TESTES PASSARAM!")
        print("\nPróximos passos:")
        print("  1. Execute: python run_experiments.py --quick  (teste rápido)")
        print("  2. Execute: python run_experiments.py          (experimento completo)")
        print("  3. Execute: python generate_tables.py          (gerar tabelas)")
        print("  4. Execute: python generate_figures.py         (gerar figuras)")
    else:
        print("\n✗ ALGUNS TESTES FALHARAM!")
        print("Verifique os erros acima antes de prosseguir.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
