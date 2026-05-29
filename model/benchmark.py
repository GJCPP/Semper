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

NUMBER_RE = r'([0-9.eE+-]+)'


def parse_list_env(name, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def parse_bool_env(name, default=False):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def normalize_backend(raw):
    backend = (raw or "orion").strip().lower()
    if backend not in ("orion", "ligero"):
        raise ValueError(f"Unknown ZKCNN_PCS backend: {backend}")
    return backend


PCS_BACKEND = normalize_backend(os.environ.get("ZKCNN_PCS") or os.environ.get("ZKCNN_PCS_BACKEND"))
MODELS = parse_list_env("ZKCNN_MODELS", ["LeNet", "AlexNet", "VGG11"])
# MODELS = ["VGG11"]
THREADS = parse_list_env("ZKCNN_THREADS", [1], int)
BATCH_SIZES = parse_list_env("ZKCNN_BATCH_SIZES", [4, 8, 16], int)
ITERATIONS = parse_list_env("ZKCNN_ITERATIONS", [1, 2, 4, 8], int)
# THREADS = [8]
# BATCH_SIZES = [8]
# ITERATIONS = [1]
BENCH_BIN = os.environ.get("ZKCNN_BENCH_BIN", "./build/bench")
OUTDIR = os.environ.get("ZKCNN_BENCH_OUTDIR", os.path.join("./logs", PCS_BACKEND))
KEEP_TRACES = parse_bool_env("ZKCNN_KEEP_TRACES", False)
RERUN_COMPLETED = parse_bool_env("ZKCNN_RERUN_COMPLETED", False)
# ==============================================================


def model_trace_path(model):
    return os.path.join("training_trace", model)


def remove_model_trace(model):
    shutil.rmtree(model_trace_path(model), ignore_errors=True)


def case_key(model, threads, batch, iters):
    return (model, PCS_BACKEND, int(threads), int(batch), int(iters))


def has_number(row, name):
    value = row.get(name)
    if value is None or value == "":
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def row_is_complete(row):
    backend = row.get("pcs_backend") or PCS_BACKEND
    try:
        backend = normalize_backend(backend)
    except ValueError:
        return False
    if backend != PCS_BACKEND:
        return False

    returncode = row.get("returncode")
    if returncode not in (None, "", "0"):
        return False

    required = ["prover_time_s", "commit_s", "open_s", "proof_size_MB"]
    return all(has_number(row, key) for key in required)


def load_completed_cases(csv_path, model):
    completed = set()
    if RERUN_COMPLETED or not os.path.exists(csv_path):
        return completed

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row_is_complete(row):
                continue
            if row.get("model") not in (None, "", model):
                continue
            try:
                completed.add(case_key(model, row["threads"], row["batch"], row["iters"]))
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def run_command(cmd, logfile, env):
    """Run command and capture stdout+stderr into a logfile."""
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as proc:
        out, _ = proc.communicate()
        with open(logfile, "w") as f:
            f.write(out)
    return out, proc.returncode


def sum_timer(output, label):
    pattern = rf'\[{re.escape(label)}\]\s*cost:\s*{NUMBER_RE}\s*s'
    values = [float(x) for x in re.findall(pattern, output)]
    return sum(values) if values else None


def parse_output(output: str, model:str):
    """Extract interesting metrics from program output."""
    res = {}
    succ = True

    m = re.search(r'num_vars\s*=\s*(\d+)', output)
    res['num_vars'] = int(m.group(1)) if m else None

    m = re.search(r'PCS backend:\s*([A-Za-z0-9_+-]+)', output)
    parsed_backend = normalize_backend(m.group(1)) if m else PCS_BACKEND
    res['pcs_backend'] = parsed_backend

    m = re.findall(rf'\[prove .* total\]\s*cost:\s*{NUMBER_RE}\s*s', output)
    res['prover_time_s'] = sum(map(float, m)) if m else None

    res['orion_commit_s'] = sum_timer(output, "orion commit")
    res['orion_open_s'] = sum_timer(output, "orion open")
    res['ligero_commit_s'] = sum_timer(output, "ligero commit")
    res['ligero_open_s'] = sum_timer(output, "ligero open")

    if parsed_backend == "orion":
        res['commit_s'] = res['orion_commit_s']
        res['open_s'] = res['orion_open_s']
    else:
        res['commit_s'] = res['ligero_commit_s']
        res['open_s'] = res['ligero_open_s']

    res['prove_layers_s'] = sum_timer(output, "prove layers")
    res['logup_s'] = sum_timer(output, "final logup")
    res['map_s'] = sum_timer(output, "final map")

    pat = rf'{re.escape(model)}:\s*{NUMBER_RE}MB'
    m = re.search(pat, output)
    res['proof_size_MB'] = float(m.group(1)) if m else None

    res['verifier_time_s'] = sum_timer(output, "verifier")

    m = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', output)
    res['max_memory_GB'] = float(m.group(1)) / 1e6 if m else None

    if "terminate called" in output or "❌" in output:
        succ = False
    required = ["prover_time_s", "commit_s", "open_s", "proof_size_MB"]
    if any(res.get(key) is None for key in required):
        succ = False

    return res, succ


def main():
    os.environ["ZKCNN_PCS"] = PCS_BACKEND
    os.makedirs(OUTDIR, exist_ok=True)
    # We will save results to separate CSV files per model, under OUTDIR.
    fieldnames = [
        "model", "pcs_backend", "threads", "batch", "iters",
        "num_vars", "prover_time_s", "commit_s", "open_s",
        "orion_commit_s", "orion_open_s", "ligero_commit_s", "ligero_open_s",
        "prove_layers_s", "logup_s", "map_s",
        "proof_size_MB",
        "verifier_time_s", "max_memory_GB", "returncode", "logfile"
    ]

    per_model_csv = {}
    bench_env = os.environ.copy()
    bench_env["ZKCNN_PCS"] = PCS_BACKEND
    print(f"PCS backend: {PCS_BACKEND}")
    print(f"Results directory: {OUTDIR}")
    print(f"Keep training traces: {KEEP_TRACES}")
    print(f"Rerun completed cases: {RERUN_COMPLETED}")

    try:
        for model in MODELS:
            model_csv_path = os.path.join(OUTDIR, f"{model}_benchmark_results.csv")
            write_header = not os.path.exists(model_csv_path)
            completed_cases = load_completed_cases(model_csv_path, model)
            if completed_cases:
                print(f"Resume: found {len(completed_cases)} completed {model} cases in {model_csv_path}")

            # Open one CSV per model and keep the writer during that model's sweep
            csvfile = open(model_csv_path, "a", newline="")
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            per_model_csv[model] = (csvfile, writer, model_csv_path)

            for threads in THREADS:
                for batch in BATCH_SIZES:
                    remaining_iterations = list(ITERATIONS)
                    while len(remaining_iterations) > 0:
                        iters = remaining_iterations[0]
                        current_case = case_key(model, threads, batch, iters)
                        if current_case in completed_cases:
                            print(f"[=] Skipping completed {model}, threads={threads}, batch={batch}, iters={iters}")
                            remaining_iterations = remaining_iterations[1:]
                            continue

                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        logname = f"{model}_{PCS_BACKEND}_t{threads}_b{batch}_i{iters}_{ts}.log"
                        logfile = os.path.join(OUTDIR, logname)

                        print(f"\n=== Running {model}, threads={threads}, batch={batch}, iters={iters} ===")

                        # Step 1: Preprocessing
                        print(f"[*] Preprocessing data...")

                        remove_model_trace(model)
                        try:
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
                            output, returncode = run_command(cmd, logfile, bench_env)

                            # Step 3: Parse
                            results, succ = parse_output(output, model)
                            results["returncode"] = returncode
                            if returncode != 0:
                                succ = False

                            if succ:
                                remaining_iterations = remaining_iterations[1:]
                            else:
                                print(f"[!] Benchmark failed for {model} with threads={threads}, batch={batch}, iters={iters}. Check {logfile} for details.")
                                break

                            results.update({
                                "model": model,
                                "pcs_backend": results.get("pcs_backend", PCS_BACKEND),
                                "threads": threads,
                                "batch": batch,
                                "iters": iters,
                                "logfile": logfile
                            })

                            # Step 4: Record to this model's CSV
                            writer.writerow(results)
                            csvfile.flush()
                            completed_cases.add(current_case)

                            # Step 5: Print summary
                            summary = ", ".join(
                                f"{k}={v}" for k, v in results.items() if k not in ("timestamp", "logfile")
                            )
                            print(f"[+] Done: {summary}")
                            print(f"[+] Log saved to {logfile}")
                        finally:
                            if not KEEP_TRACES:
                                remove_model_trace(model)
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
