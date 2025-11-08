#!/usr/bin/env python3
import subprocess
import re
import csv
from datetime import datetime
import os

import VGG11, VGG16, AlexNet, LeNet, pad_conv
import shutil

# ==============================================================
# Configuration section — modify these lists to define experiments
# ==============================================================

# MODELS = ["LeNet", "AlexNet", "VGG11"]

MODELS = ["LeNet", "AlexNet", "VGG11"]
# MODELS = ["VGG11"]
THREADS = [1, 2, 4, 8]
BATCH_SIZES = [4, 8, 16]
ITERATIONS = [1, 2, 4, 8]
# THREADS = [8]
# BATCH_SIZES = [8]
# ITERATIONS = [1]
BENCH_BIN = "./build/bench"
OUTDIR = "./logs"
# ==============================================================


def run_command(cmd, logfile):
    """Run command and capture stdout+stderr into a logfile, return full text."""
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        out, _ = proc.communicate()
        with open(logfile, "w") as f:
            f.write(out)
    return out


def parse_output(output: str, model:str):
    """Extract interesting metrics from program output."""
    res = {}
    succ = True

    m = re.search(r'num_vars\s*=\s*(\d+)', output)
    res['num_vars'] = int(m.group(1)) if m else None
    if not m:
        succ = True

    m = re.findall(r'\[prove .* total\]\s*cost:\s*([\d.]+)\s*s', output)
    res['prover_time_s'] = sum(map(float, m)) if m else None

    m = re.findall(r'\[ligero commit\]\s*cost:\s*([\d.]+)\s*s', output)
    res['commit_s'] = sum(map(float, m)) if m else None

    m = re.findall(r'\[ligero open\]\s*cost:\s*([\d.]+)\s*s', output)
    res['open_s'] = sum(map(float, m)) if m else None

    m = re.findall(r'\[prove layers\]\s*cost:\s*([\d.]+)\s*s', output)
    res['prove_layers_s'] = sum(map(float, m)) if m else None

    m = re.findall(r'\[final logup\]\s*cost:\s*([\d.]+)\s*s', output)
    res['logup_s'] = sum(map(float, m)) if m else None

    m = re.findall(r'\[final map\]\s*cost:\s*([\d.]+)\s*s', output)
    res['map_s'] = sum(map(float, m)) if m else None

    pat = rf'{re.escape(model)}:\s*([0-9.eE+-]+)MB'
    m = re.search(pat, output)
    res['proof_size_MB'] = float(m.group(1)) if m else None

    m = re.findall(r'\[verifier\]\s*cost:\s*([\d.]+)\s*s', output)
    res['verifier_time_s'] = sum(map(float, m)) if m else None

    m = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', output)
    res['max_memory_GB'] = float(m.group(1)) / 1e6 if m else None

    return res, succ


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # We will save results to separate CSV files per model, under OUTDIR.
    fieldnames = [
        "model", "threads", "batch", "iters",
        "num_vars", "prover_time_s", 'commit_s', 'open_s', 'prove_layers_s', 'logup_s', 'map_s',
        "proof_size_MB",
        "verifier_time_s", "max_memory_GB", "logfile"
    ]

    per_model_csv = {}

    try:
        for model in MODELS:
            model_csv_path = os.path.join(OUTDIR, f"{model}_benchmark_results.csv")
            write_header = not os.path.exists(model_csv_path)

            # Open one CSV per model and keep the writer during that model's sweep
            csvfile = open(model_csv_path, "a", newline="")
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            per_model_csv[model] = (csvfile, writer, model_csv_path)

            for threads in THREADS:
                for batch in BATCH_SIZES:
                    iteraions = ITERATIONS
                    while len(iteraions) > 0:
                        iters = iteraions[0]
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        logname = f"{model}_t{threads}_b{batch}_i{iters}_{ts}.log"
                        logfile = os.path.join(OUTDIR, logname)

                        print(f"\n=== Running {model}, threads={threads}, batch={batch}, iters={iters} ===")

                        # Step 1: Preprocessing
                        print(f"[*] Preprocessing data...")

                        shutil.rmtree(f"/training_trace/{model}", ignore_errors=True)
                        if model == "AlexNet":
                            AlexNet.train_manual(batch, iters)
                            pad_conv.pad_AlexNet()
                        elif model == "VGG11":
                            VGG11.train_manual(batch, iters)
                            pad_conv.pad_VGG11()
                        elif model == "VGG16":
                            VGG16.train_manual(batch, iters)
                            pad_conv.pad_VGG16()
                        elif model == "LeNet":
                            LeNet.train_manual(batch, iters)
                            pad_conv.pad_LeNet()
                        else:
                            raise ValueError(f"Unknown model: {model}")

                        # Step 2: Benchmark
                        print(f"[*] Running benchmark...")
                        cmd = ["/usr/bin/time", "-v", BENCH_BIN, str(threads), model]
                        output = run_command(cmd, logfile)

                        # Step 3: Parse
                        results, succ = parse_output(output, model)

                        if succ:
                            iteraions = iteraions[1:]
                        else:
                            print(f"[!] Benchmark failed for {model} with threads={threads}, batch={batch}, iters={iters}. Check {logfile} for details.")
                            # Do not proceed to next iteration count; retry the same one
                            continue

                        results.update({
                            "model": model,
                            "threads": threads,
                            "batch": batch,
                            "iters": iters,
                            "logfile": logfile
                        })

                        # Step 4: Record to this model's CSV
                        writer.writerow(results)
                        csvfile.flush()

                        # Step 5: Print summary
                        summary = ", ".join(
                            f"{k}={v}" for k, v in results.items() if k not in ("timestamp", "logfile")
                        )
                        print(f"[+] Done: {summary}")
                        print(f"[+] Log saved to {logfile}")
    finally:
        # Ensure all per-model CSV files are closed properly
        for model, (csvfile, _writer, _path) in per_model_csv.items():
            try:
                csvfile.close()
            except Exception:
                pass

    # Print a final summary of where results were saved
    print("\nAll benchmarks complete. Per-model results saved to:")
    for model, (_csvfile, _writer, path) in per_model_csv.items():
        print(f" - {path}")


if __name__ == "__main__":
    main()
