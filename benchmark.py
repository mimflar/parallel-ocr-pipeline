from main import load_config, run_pipeline

if __name__ == '__main__':
    image_dir, base_workers = load_config("config.txt")
    worker_counts_to_test = sorted(list(set([1, 2, 4, 8] + [base_workers])))
        
    print(f"Starting OCR Benchmarks on directory: {image_dir}")
    print("-" * 50)
    
    results = {}
    t1_time = None
    
    for n in worker_counts_to_test:
        output_filename = f"ocr_results_{n}_workers.csv"
        print(f"Running pipeline with {n} worker(s)...")
        
        # calls the pipeline function from main.py
        elapsed = run_pipeline(image_dir, n, output_file=output_filename)
        results[n] = elapsed
        
        if n == 1:
            t1_time = elapsed
            
        speedup = t1_time / elapsed if t1_time else 1.0
        print(f"Completed in {elapsed:.2f} seconds. Speedup (S): {speedup:.2f}x")
        print("-" * 50)
        
    print("\nBenchmark Summary:")
    print(f"{'Workers (n)':<15} | {'Time (Tn)':<15} | {'Speedup (T1/Tn)':<15}")
    for n, elapsed in results.items():
        speedup = t1_time / elapsed
        print(f"{n:<15} | {elapsed:<15.2f} | {speedup:<15.2f}x")