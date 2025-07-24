#!/usr/bin/env python3
"""
Aggregate LRA benchmark results from multiple task evaluations
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_task_results(results_dir: Path, task_name: str) -> dict:
    """Load results for a specific task"""
    task_dir = results_dir / task_name
    results_file = task_dir / "evaluation_results.json"
    
    if not results_file.exists():
        print(f"Warning: Results file not found for task {task_name}")
        return {}
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results


def aggregate_lra_results(results_dir: str, output_file: str):
    """Aggregate results from all LRA tasks"""
    results_dir = Path(results_dir)
    
    # LRA tasks
    tasks = ['listops', 'text', 'retrieval', 'pathfinder']
    
    # Aggregate results
    aggregated = {
        'summary': {},
        'detailed': {},
        'performance': {}
    }
    
    # Collect results for each task
    all_accuracies = []
    task_results = {}
    
    for task in tasks:
        print(f"Loading results for {task}...")
        results = load_task_results(results_dir, task)
        
        if not results:
            continue
            
        task_results[task] = results
        
        # Extract key metrics
        if 'lra_results' in results and task in results['lra_results']:
            task_data = results['lra_results'][task]
            accuracy = task_data.get('accuracy', 0.0)
            loss = task_data.get('loss', float('inf'))
            
            all_accuracies.append(accuracy)
            
            # Store in summary
            aggregated['summary'][task] = {
                'accuracy': accuracy,
                'loss': loss,
                'num_samples': task_data.get('num_samples', 0)
            }
            
            # Add F1 scores if available
            if 'macro_f1' in task_data:
                aggregated['summary'][task]['macro_f1'] = task_data['macro_f1']
            if 'weighted_f1' in task_data:
                aggregated['summary'][task]['weighted_f1'] = task_data['weighted_f1']
        
        # Extract performance metrics
        if 'performance' in results:
            perf_data = results['performance']
            
            aggregated['performance'][task] = {}
            
            if 'throughput' in perf_data:
                aggregated['performance'][task]['tokens_per_second'] = perf_data['throughput']['tokens_per_second']
            
            if 'flops' in perf_data and 'gflops' in perf_data['flops']:
                aggregated['performance'][task]['gflops'] = perf_data['flops']['gflops']
            
            if 'memory' in perf_data:
                # Get memory usage for batch size 32 (typical)
                if 'batch_32' in perf_data['memory']:
                    mem_data = perf_data['memory']['batch_32']
                    if 'peak_memory_gb' in mem_data:
                        aggregated['performance'][task]['peak_memory_gb'] = mem_data['peak_memory_gb']
    
    # Calculate overall statistics
    if all_accuracies:
        aggregated['overall'] = {
            'mean_accuracy': np.mean(all_accuracies),
            'std_accuracy': np.std(all_accuracies),
            'min_accuracy': np.min(all_accuracies),
            'max_accuracy': np.max(all_accuracies),
            'num_tasks': len(all_accuracies)
        }
    
    # Store detailed results
    aggregated['detailed'] = task_results
    
    # Performance summary
    if aggregated['performance']:
        perf_summary = {}
        
        # Average throughput
        throughputs = [p.get('tokens_per_second', 0) for p in aggregated['performance'].values()]
        if throughputs and any(t > 0 for t in throughputs):
            perf_summary['avg_throughput'] = np.mean([t for t in throughputs if t > 0])
        
        # Average memory
        memories = [p.get('peak_memory_gb', 0) for p in aggregated['performance'].values()]
        if memories and any(m > 0 for m in memories):
            perf_summary['avg_memory_gb'] = np.mean([m for m in memories if m > 0])
        
        # Average GFLOPs
        gflops = [p.get('gflops', 0) for p in aggregated['performance'].values()]
        if gflops and any(g > 0 for g in gflops):
            perf_summary['avg_gflops'] = np.mean([g for g in gflops if g > 0])
        
        aggregated['performance_summary'] = perf_summary
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Aggregated results saved to {output_path}")
    
    # Print summary
    print_summary(aggregated)
    
    # Create CSV summary for easy viewing
    create_csv_summary(aggregated, output_path.parent / "lra_summary.csv")


def print_summary(results: dict):
    """Print a formatted summary of results"""
    print("\n" + "="*60)
    print("LRA BENCHMARK SUMMARY")
    print("="*60)
    
    if 'summary' in results:
        print("\nTask Results:")
        print("-" * 40)
        print(f"{'Task':<12} {'Accuracy':<10} {'Loss':<10} {'Samples':<8}")
        print("-" * 40)
        
        for task, metrics in results['summary'].items():
            acc = metrics.get('accuracy', 0.0)
            loss = metrics.get('loss', 0.0)
            samples = metrics.get('num_samples', 0)
            print(f"{task:<12} {acc:<10.4f} {loss:<10.4f} {samples:<8}")
    
    if 'overall' in results:
        print("\nOverall Statistics:")
        print("-" * 40)
        overall = results['overall']
        print(f"Mean Accuracy: {overall['mean_accuracy']:.4f} ± {overall['std_accuracy']:.4f}")
        print(f"Range: {overall['min_accuracy']:.4f} - {overall['max_accuracy']:.4f}")
        print(f"Tasks Completed: {overall['num_tasks']}")
    
    if 'performance_summary' in results:
        print("\nPerformance Summary:")
        print("-" * 40)
        perf = results['performance_summary']
        
        if 'avg_throughput' in perf:
            print(f"Average Throughput: {perf['avg_throughput']:.0f} tokens/sec")
        if 'avg_memory_gb' in perf:
            print(f"Average Memory: {perf['avg_memory_gb']:.2f} GB")
        if 'avg_gflops' in perf:
            print(f"Average GFLOPs: {perf['avg_gflops']:.2f}")
    
    print("\n" + "="*60)


def create_csv_summary(results: dict, output_file: Path):
    """Create a CSV summary of results"""
    if 'summary' not in results:
        return
    
    # Create DataFrame
    data = []
    for task, metrics in results['summary'].items():
        row = {
            'task': task,
            'accuracy': metrics.get('accuracy', 0.0),
            'loss': metrics.get('loss', 0.0),
            'num_samples': metrics.get('num_samples', 0),
            'macro_f1': metrics.get('macro_f1', 0.0),
            'weighted_f1': metrics.get('weighted_f1', 0.0),
        }
        
        # Add performance metrics if available
        if 'performance' in results and task in results['performance']:
            perf = results['performance'][task]
            row.update({
                'tokens_per_second': perf.get('tokens_per_second', 0.0),
                'gflops': perf.get('gflops', 0.0),
                'peak_memory_gb': perf.get('peak_memory_gb', 0.0),
            })
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"CSV summary saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate LRA benchmark results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing task result folders")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output file for aggregated results")
    
    args = parser.parse_args()
    
    aggregate_lra_results(args.results_dir, args.output_file)


if __name__ == "__main__":
    main()