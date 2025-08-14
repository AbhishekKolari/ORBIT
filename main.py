#!/usr/bin/env python3
import argparse
import os
import json

from opensource import run_opensource
from closedsource import run_closedsource
from utils import compute_metrics_from_results

def main():
    parser = argparse.ArgumentParser(description="VLM Benchmark runner (open & closed source)")
    parser.add_argument('--mode', choices=['opensource', 'closedsource'], required=True,
                        help='Choose "opensource" or "closedsource"')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name or HF repo id (for opensource) or API model id (for closedsource)')
    parser.add_argument('--processor_path', type=str, default=None,
                        help='Processor path or HF repo id (for opensource). If omitted, model_name is used.')
    parser.add_argument('--benchmark_json', type=str, required=True, help='Path to benchmark.json')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with images (paths in benchmark.json are relative to this)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to write results JSON')
    parser.add_argument('--batch_size', type=int, default=360, help='How many images to process (default=360)')
    parser.add_argument('--analyze', action='store_true', help='Compute analysis metrics on the saved results after run')
    parser.add_argument('--off_by_n', type=int, default=1, help='Tolerance for off-by-n analysis (analysis only)')
    args = parser.parse_args()

    if args.mode == 'opensource':
        run_opensource(
            model_name=args.model_name,
            processor_path=(args.processor_path or args.model_name),
            benchmark_json=args.benchmark_json,
            data_dir=args.data_dir,
            output_file=args.output_file,
            batch_size=args.batch_size
        )
    else:
        # closedsource: API keys are read from environment variables inside closedsource.py
        run_closedsource(
            model_name=args.model_name,
            benchmark_json=args.benchmark_json,
            data_dir=args.data_dir,
            output_file=args.output_file,
            batch_size=args.batch_size
        )

    if args.analyze:
        print("\nRunning analysis on results...")
        metrics = compute_metrics_from_results(args.output_file, off_by_n=args.off_by_n, by_category=True)
        print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
