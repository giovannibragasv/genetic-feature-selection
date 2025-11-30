#!/usr/bin/env python3
"""
Gera tabelas do paper Feng 2024.

Tabelas:
- Table 3: Features e recognition rate por encoding (Fashion-MNIST)
- Table 5: Accuracy comparison por dataset e encoding
- Table 6: Comparação com outros métodos (se dados disponíveis)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils import compute_statistics, format_result


class TableGenerator:
    """Gera tabelas do paper."""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.results_path = self.base_path.parent / "results"
        self.tables_path = self.results_path / "tables"
        self.tables_path.mkdir(exist_ok=True)
    
    def _find_latest_results(self) -> Optional[Path]:
        """Encontra arquivo de resultados mais recente."""
        csv_files = list(self.tables_path.glob("results_full_*.csv"))
        if not csv_files:
            return None
        return max(csv_files, key=lambda x: x.stat().st_mtime)
    
    def generate_table3(
        self,
        results_csv: Optional[str] = None,
        output_name: str = "table3_features_recognition"
    ) -> pd.DataFrame:
        """
        Gera Table 3: número de features e taxa de reconhecimento.
        
        Formato:
        | Encoding | Before FS | Recognition Rate | After FS | Recognition Rate |
        """
        print("\nGerando Table 3: Features e Recognition Rate...")
        
        if results_csv:
            df = pd.read_csv(results_csv)
        else:
            latest = self._find_latest_results()
            if latest is None:
                print("  Nenhum resultado encontrado. Execute run_experiments.py primeiro.")
                return None
            df = pd.read_csv(latest)
            print(f"  Usando: {latest.name}")
        
        df_valid = df[df['accuracy'].notna()]
        
        # Agrupar por encoding
        summary = df_valid.groupby('encoding').agg({
            'n_features_original': 'first',
            'n_features_selected': 'mean',
            'accuracy': 'mean'
        }).round(4)
        
        # Criar tabela no formato do paper
        table = pd.DataFrame({
            'Encoding Method': summary.index,
            'Before Feature Selection': summary['n_features_original'].astype(int),
            'Correct Recognition Rate (Before)': '-',  # Não temos essa info antes do GA
            'After Feature Selection': summary['n_features_selected'].round(0).astype(int),
            'Correct Recognition Rate (After)': summary['accuracy'].round(4)
        })
        
        # Salvar
        output_path = self.tables_path / f"{output_name}.csv"
        table.to_csv(output_path, index=False)
        
        # Também salvar em formato markdown
        md_path = self.tables_path / f"{output_name}.md"
        with open(md_path, 'w') as f:
            f.write("# Table 3: Number of Features and Classifier Recognition Rate\n\n")
            f.write(table.to_markdown(index=False))
        
        print(f"  Salvo em: {output_path}")
        print("\n" + table.to_string(index=False))
        
        return table
    
    def generate_table5(
        self,
        results_csv: Optional[str] = None,
        output_name: str = "table5_accuracy_comparison"
    ) -> pd.DataFrame:
        """
        Gera Table 5: comparação de accuracy por dataset e encoding.
        
        Formato:
        | Dataset | Binary | Decimal | Real | Gaussian | Adaptive |
        """
        print("\nGerando Table 5: Accuracy Comparison...")
        
        if results_csv:
            df = pd.read_csv(results_csv)
        else:
            latest = self._find_latest_results()
            if latest is None:
                print("  Nenhum resultado encontrado.")
                return None
            df = pd.read_csv(latest)
            print(f"  Usando: {latest.name}")
        
        df_valid = df[df['accuracy'].notna()]
        
        # Criar pivot tables
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
        
        # Criar tabela formatada (mean ± std)
        table_formatted = pd.DataFrame(index=pivot_mean.index)
        
        # Ordem dos encodings conforme paper
        encoding_order = ['binary', 'decimal', 'real', 'gaussian', 'adaptive']
        
        for enc in encoding_order:
            if enc in pivot_mean.columns:
                col_data = []
                for idx in pivot_mean.index:
                    mean = pivot_mean.loc[idx, enc]
                    std = pivot_std.loc[idx, enc]
                    col_data.append(f"{mean:.4f} ± {std:.4f}")
                table_formatted[enc.capitalize()] = col_data
        
        # Destacar melhor resultado por linha
        table_formatted_highlight = table_formatted.copy()
        
        # Salvar versões
        output_path = self.tables_path / f"{output_name}.csv"
        table_formatted.to_csv(output_path)
        
        # Versão numérica
        pivot_mean.to_csv(self.tables_path / f"{output_name}_mean.csv")
        pivot_std.to_csv(self.tables_path / f"{output_name}_std.csv")
        
        # Markdown
        md_path = self.tables_path / f"{output_name}.md"
        with open(md_path, 'w') as f:
            f.write("# Table 5: Accuracy Comparison by Dataset and Encoding\n\n")
            f.write("Values shown as: Mean ± Standard Deviation\n\n")
            f.write(table_formatted.to_markdown())
            
            # Adicionar resumo
            f.write("\n\n## Summary Statistics\n\n")
            overall_mean = df_valid.groupby('encoding')['accuracy'].mean()
            overall_std = df_valid.groupby('encoding')['accuracy'].std()
            
            f.write("### Overall Performance by Encoding\n\n")
            for enc in encoding_order:
                if enc in overall_mean.index:
                    f.write(f"- **{enc.capitalize()}**: {overall_mean[enc]:.4f} ± {overall_std[enc]:.4f}\n")
        
        print(f"  Salvo em: {output_path}")
        print("\n" + "="*80)
        print("TABLE 5: ACCURACY COMPARISON")
        print("="*80)
        print(table_formatted.to_string())
        print("="*80)
        
        # Estatísticas adicionais
        print("\nMELHORES RESULTADOS POR DATASET:")
        for dataset in pivot_mean.index:
            best_enc = pivot_mean.loc[dataset].idxmax()
            best_val = pivot_mean.loc[dataset].max()
            print(f"  {dataset}: {best_enc} ({best_val:.4f})")
        
        return table_formatted
    
    def generate_table_features(
        self,
        results_csv: Optional[str] = None,
        output_name: str = "table_features_selected"
    ) -> pd.DataFrame:
        """
        Gera tabela de features selecionadas por dataset/encoding.
        """
        print("\nGerando tabela de features selecionadas...")
        
        if results_csv:
            df = pd.read_csv(results_csv)
        else:
            latest = self._find_latest_results()
            if latest is None:
                print("  Nenhum resultado encontrado.")
                return None
            df = pd.read_csv(latest)
        
        df_valid = df[df['accuracy'].notna()]
        
        # Pivot tables
        pivot_mean = df_valid.pivot_table(
            values='n_features_selected',
            index='dataset',
            columns='encoding',
            aggfunc='mean'
        ).round(1)
        
        pivot_std = df_valid.pivot_table(
            values='n_features_selected',
            index='dataset',
            columns='encoding',
            aggfunc='std'
        ).round(1)
        
        # Criar tabela formatada
        table_formatted = pd.DataFrame(index=pivot_mean.index)
        
        encoding_order = ['binary', 'decimal', 'real', 'gaussian', 'adaptive']
        
        for enc in encoding_order:
            if enc in pivot_mean.columns:
                col_data = []
                for idx in pivot_mean.index:
                    mean = pivot_mean.loc[idx, enc]
                    std = pivot_std.loc[idx, enc]
                    col_data.append(f"{mean:.1f} ± {std:.1f}")
                table_formatted[enc.capitalize()] = col_data
        
        # Adicionar coluna de features originais
        original_features = df_valid.groupby('dataset')['n_features_original'].first()
        table_formatted.insert(0, 'Original Features', original_features)
        
        # Salvar
        output_path = self.tables_path / f"{output_name}.csv"
        table_formatted.to_csv(output_path)
        
        md_path = self.tables_path / f"{output_name}.md"
        with open(md_path, 'w') as f:
            f.write("# Features Selected by Dataset and Encoding\n\n")
            f.write("Values shown as: Mean ± Standard Deviation\n\n")
            f.write(table_formatted.to_markdown())
        
        print(f"  Salvo em: {output_path}")
        print("\n" + table_formatted.to_string())
        
        return table_formatted
    
    def generate_summary_report(
        self,
        results_csv: Optional[str] = None,
        output_name: str = "experiment_summary"
    ):
        """
        Gera relatório resumido dos experimentos.
        """
        print("\nGerando relatório resumido...")
        
        if results_csv:
            df = pd.read_csv(results_csv)
        else:
            latest = self._find_latest_results()
            if latest is None:
                print("  Nenhum resultado encontrado.")
                return None
            df = pd.read_csv(latest)
        
        df_valid = df[df['accuracy'].notna()]
        
        report = []
        report.append("="*80)
        report.append("EXPERIMENT SUMMARY REPORT")
        report.append("Replication of Feng 2024 (PLOS ONE)")
        report.append("="*80)
        report.append("")
        
        # Estatísticas gerais
        report.append("## OVERALL STATISTICS")
        report.append(f"Total experiments: {len(df)}")
        report.append(f"Successful runs: {len(df_valid)}")
        report.append(f"Datasets: {df_valid['dataset'].nunique()}")
        report.append(f"Encodings: {df_valid['encoding'].nunique()}")
        report.append(f"Runs per configuration: {df_valid.groupby(['dataset', 'encoding']).size().mean():.0f}")
        report.append("")
        
        # Performance por encoding
        report.append("## PERFORMANCE BY ENCODING")
        encoding_stats = df_valid.groupby('encoding').agg({
            'accuracy': ['mean', 'std', 'max'],
            'n_features_selected': 'mean',
            'time_seconds': 'mean'
        }).round(4)
        
        for enc in encoding_stats.index:
            acc_mean = encoding_stats.loc[enc, ('accuracy', 'mean')]
            acc_std = encoding_stats.loc[enc, ('accuracy', 'std')]
            acc_max = encoding_stats.loc[enc, ('accuracy', 'max')]
            n_feat = encoding_stats.loc[enc, ('n_features_selected', 'mean')]
            time_s = encoding_stats.loc[enc, ('time_seconds', 'mean')]
            
            report.append(f"\n### {enc.upper()}")
            report.append(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f} (max: {acc_max:.4f})")
            report.append(f"  Features selected: {n_feat:.1f}")
            report.append(f"  Avg time: {time_s:.1f}s")
        
        report.append("")
        
        # Performance por dataset
        report.append("## PERFORMANCE BY DATASET")
        dataset_stats = df_valid.groupby('dataset').agg({
            'accuracy': ['mean', 'std', 'max'],
            'n_features_original': 'first',
            'n_features_selected': 'mean'
        }).round(4)
        
        for ds in dataset_stats.index:
            acc_mean = dataset_stats.loc[ds, ('accuracy', 'mean')]
            acc_std = dataset_stats.loc[ds, ('accuracy', 'std')]
            acc_max = dataset_stats.loc[ds, ('accuracy', 'max')]
            n_orig = dataset_stats.loc[ds, ('n_features_original', 'first')]
            n_sel = dataset_stats.loc[ds, ('n_features_selected', 'mean')]
            reduction = (1 - n_sel/n_orig) * 100
            
            report.append(f"\n### {ds.upper()}")
            report.append(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f} (max: {acc_max:.4f})")
            report.append(f"  Features: {int(n_orig)} → {n_sel:.1f} ({reduction:.1f}% reduction)")
        
        report.append("")
        
        # Melhores resultados
        report.append("## BEST RESULTS")
        best_per_dataset = df_valid.loc[df_valid.groupby('dataset')['accuracy'].idxmax()]
        
        for _, row in best_per_dataset.iterrows():
            report.append(f"  {row['dataset']}: {row['encoding']} = {row['accuracy']:.4f}")
        
        report.append("")
        report.append("="*80)
        
        # Salvar
        report_text = "\n".join(report)
        
        output_path = self.tables_path / f"{output_name}.txt"
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        md_path = self.tables_path / f"{output_name}.md"
        with open(md_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nRelatório salvo em: {output_path}")
        
        return report_text
    
    def generate_all_tables(self, results_csv: Optional[str] = None):
        """
        Gera todas as tabelas.
        """
        print("\n" + "="*60)
        print("GERANDO TODAS AS TABELAS")
        print("="*60)
        
        self.generate_table3(results_csv)
        self.generate_table5(results_csv)
        self.generate_table_features(results_csv)
        self.generate_summary_report(results_csv)
        
        print("\n" + "="*60)
        print(f"Tabelas salvas em: {self.tables_path}")
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Gera tabelas do paper")
    parser.add_argument('--results', '-r', default=None, help='Arquivo CSV de resultados')
    parser.add_argument('--table', '-t', default='all',
                       choices=['all', 'table3', 'table5', 'features', 'summary'],
                       help='Tabela específica para gerar')
    
    args = parser.parse_args()
    
    generator = TableGenerator()
    
    if args.table == 'all':
        generator.generate_all_tables(args.results)
    elif args.table == 'table3':
        generator.generate_table3(args.results)
    elif args.table == 'table5':
        generator.generate_table5(args.results)
    elif args.table == 'features':
        generator.generate_table_features(args.results)
    elif args.table == 'summary':
        generator.generate_summary_report(args.results)


if __name__ == "__main__":
    main()
