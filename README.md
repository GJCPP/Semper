# Semper

A proof system for proving CNN training and inference.

## Requirement

- [Goldilocks](https://github.com/0xPolygonHermez/goldilocks.git), which supports Goldilocks field.

- [cnpy](https://github.com/rogersce/cnpy), for loading CNN data and witness.

- [OpenMP](https://www.openmp.org), for parallization.

Note that Goldilocks is already included in the repo, and cnpy will be fetched automatically upon build.

## Build & Benchmark

Run `sh ./setup.sh` command, which executes the following script

```sh
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_BUILD_TYPE=Release ..
make -j
```

Run `sh ./benchmark.sh` for benchmarking, which executes

```sh
sh ./setup.sh
python3 model/benchmark.py
```

The results of benchmarking can be find under `./logs` directory.

## File Structure

In directory `./model` we implement CNN networks, includeing LeNet, AlexNet, VGG11 and VGG16. These scripts output training trace and witness to directory `./training_trace` for generating proof.

In directory `./src` and `./include` we implement our proof system, Semper.

## Citation

TODO
