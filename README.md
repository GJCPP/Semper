# Semper

A proof system for proving CNN training.

## Requirement

- [Goldilocks](https://github.com/0xPolygonHermez/goldilocks.git), which supports Goldilocks field.

- [cnpy](https://github.com/rogersce/cnpy), for loading CNN data and witness.

- [OpenMP](https://www.openmp.org), for parallization.

Note that Goldilocks is already included in the repo, and cnpy will be fetched automatically upon build.

## Build & Benchmark

Run `sh ./setup.sh` command, which executes the following script

```sh
cmake -S . -B build -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run `sh ./benchmark.sh` for benchmarking, which executes

```sh
sh ./setup.sh
python3 model/benchmark.py
```

The default PCS backend is Orion. Use `ZKCNN_PCS=ligero sh ./benchmark.sh` to run the Ligero backend instead.

Benchmarks run single-threaded by default. Use `ZKCNN_THREADS=1,2,4,8 sh ./benchmark.sh` only when a multi-thread sweep is needed.

The results of benchmarking can be find under `./logs/<pcs_backend>` directory.
Generated traces under `./training_trace/<model>` are removed after each benchmark case by default to reduce disk pressure. Use `ZKCNN_KEEP_TRACES=1 sh ./benchmark.sh` when traces need to be kept for debugging.
Benchmark resume is based on completed rows in `./logs/<pcs_backend>/*_benchmark_results.csv`, not on trace files. Use `ZKCNN_RERUN_COMPLETED=1 sh ./benchmark.sh` to ignore existing CSV rows and rerun everything.

## File Structure

In directory `./model` we implement CNN networks, includeing LeNet, AlexNet, VGG11 and VGG16. These scripts output training trace and witness to directory `./training_trace` for generating proof.

In directory `./src` and `./include` we implement our proof system, Semper.

## Citation

TODO
