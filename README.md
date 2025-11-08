# Zenith

A proof system for proving CNN training and inference.

## Requirement

- [Goldilocks](https://github.com/0xPolygonHermez/goldilocks.git), which supports Goldilocks field.

- [cnpy](https://github.com/rogersce/cnpy), for loading CNN data and witness.

- [OpenMP](https://www.openmp.org), for parallization.

Note that Goldilocks is already included in the repo, and cnpy will be fetched automatically upon build.

## Build & Benchmark

Run `sh setup.sh` command, which executes the following script

```sh
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_BUILD_TYPE=Release ..
make -j
```

Run `sh benchmark.sh` for benchmarking, which executes

```sh
sh setup.sh
python3 model/benchmark.py
```

The results of benchmarking can be find under `logs` directory.

## File Structure

In directory `model` we implement CNN networks, includeing LeNet, AlexNet, VGG11 and VGG16. These scripts output training trace and witness to directory `training_trace` for generating proof.

In directory `src` and `include` we implement our proof system, Zenith.

## Design

- `oracle.h/cpp` includes the declaration of an abstract base class, `oracle`. This class abstracts the concept of oracle (which can be queried in black-box manner), and is inherited by class `MLE`, `ligeropcs_base`, and `ligeropcs_ext`.

- `ligero.h/cpp` implements the Ligero PCS.

- `xxx_check.h/cpp` implements the corresbonding protocol for the chceck.

- `CNN.h/cpp` implements the API for loading, committing, and proving a CNN training process. `CNN_check.h/cpp` checks the training trace for debugging purpose, and `CNN_proof.h/cpp` proves the training.

- `LeNet.h/cpp`, `AlexNet.h/cpp`, etc. specify the loading procedures for the corresponding models.

## Citation

TODO
