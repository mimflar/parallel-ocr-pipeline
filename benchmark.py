import os
os.environ["OMP_THREAD_LIMIT"] = "1"

from main import load_config, run_pipeline

N_RUNS = 3
WORKER_COUNTS = [1, 2, 4, 8]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image_dir, _ = load_config("config.txt")

    print(f"Starting OCR Benchmarks on directory: {image_dir}")
    print(f"{N_RUNS} runs for each worker count: {WORKER_COUNTS}")
    print("-" * 50)

    print("Warm-up run (not counted)...")
    run_pipeline(image_dir, max(WORKER_COUNTS), output_file="warmup.csv")
    os.remove("warmup.csv")
    print("-" * 50)

    times = {n: [] for n in WORKER_COUNTS}
    speedups = {n: [] for n in WORKER_COUNTS}

    for run in range(1, N_RUNS + 1):
        print(f"Run {run} of {N_RUNS}")
        t1 = None

        for n in WORKER_COUNTS:
            # calls the pipeline function from main.py
            elapsed = run_pipeline(image_dir, n, output_file=f"ocr_results_{n}_workers.csv")

            if n == 1:
                t1 = elapsed

            speedup = t1 / elapsed
            times[n].append(elapsed)
            speedups[n].append(speedup)

            print(f"  {n} worker(s): {elapsed:.2f} s, speedup {speedup:.2f}x")

        print("-" * 50)

    avg_time = [sum(times[n]) / N_RUNS for n in WORKER_COUNTS]
    avg_speedup = [sum(speedups[n]) / N_RUNS for n in WORKER_COUNTS]

    serial_fractions = {
        n: (1 / s - 1 / n) / (1 - 1 / n)
        for n, s in zip(WORKER_COUNTS, avg_speedup) if n > 1
    }
    avg_e = sum(serial_fractions.values()) / len(serial_fractions)

    print("\nBenchmark Summary (averages)")
    print(f"{'Workers':<10} | {'Avg Time (s)':<14} | {'Avg Speedup':<12} | {'Serial Fraction (e)':<20}")
    for n, t, s in zip(WORKER_COUNTS, avg_time, avg_speedup):
        e = f"{serial_fractions[n]:.3f}" if n in serial_fractions else "-"
        print(f"{n:<10} | {t:<14.2f} | {s:<12.2f} | {e:<20}")
    print(f"\nAverage serial fraction (e): {avg_e:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(WORKER_COUNTS, avg_time, marker='o', color='black')
    ax1.set_title("Average time")
    ax1.set_xlabel("Workers")
    ax1.set_ylabel("Seconds")

    ax2.plot(WORKER_COUNTS, avg_speedup, marker='o', color='black', label="Measured")
    ax2.plot(WORKER_COUNTS, WORKER_COUNTS, linestyle='--', color='gray', label="Ideal")
    amdahl_x = list(range(1, max(WORKER_COUNTS) + 1))
    amdahl_y = [1 / (avg_e + (1 - avg_e) / n) for n in amdahl_x]
    ax2.plot(amdahl_x, amdahl_y, linestyle=':', color='black', label=f"Amdahl (e = {avg_e:.3f})")
    ax2.set_title("Average speedup")
    ax2.set_xlabel("Workers")
    ax2.set_ylabel("Speedup (T1 / Tn)")
    ax2.legend()

    for ax in (ax1, ax2):
        ax.set_xticks(WORKER_COUNTS)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("benchmark.png", dpi=150)
    print("\nGraph saved to benchmark.png")
    plt.show()