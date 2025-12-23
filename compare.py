"""
Example usage of the experiment framework.

# After training:
python compare.py --dataset maze

# Or in Python:
from experiments import find_runs, plot_comparison, plot_accuracy_comparison
from experiments import evaluate_model

runs = find_runs("results", dataset="maze")
plot_comparison(runs, metric="loss")
"""
import argparse
from experiments import find_runs, plot_training, plot_comparison, plot_accuracy_comparison
from experiments import evaluate_on_dataset


def main(args):
    runs = find_runs(args.results_dir, dataset=args.dataset, algorithm=args.algorithm)
    
    if not runs:
        print(f"No runs found in {args.results_dir}")
        return
    
    print(f"Found {len(runs)} runs:")
    for r in runs:
        print(f"  - {r}")
    
    if args.plot_training:
        for run in runs:
            plot_training(run)
    
    if args.plot_comparison:
        plot_comparison(runs, metric=args.metric)
    
    if args.evaluate:
        test_path = args.test_path or f"data/{args.dataset}/test.parquet"
        results = {}
        for run in runs:
            config_path = f"{run}/config.json"
            import json
            with open(config_path) as f:
                config = json.load(f)
            algo = config["algorithm"].upper()
            
            print(f"Evaluating {algo}...")
            res = evaluate_on_dataset(run, test_path, args.dataset)
            results[algo] = {args.dataset: {"accuracy": res["accuracy"]}}
            print(f"  Accuracy: {res['accuracy']:.2%}")
        
        plot_accuracy_comparison(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--metric", type=str, default="loss")
    parser.add_argument("--plot-training", action="store_true")
    parser.add_argument("--plot-comparison", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--test-path", type=str, default=None)
    args = parser.parse_args()
    main(args)
