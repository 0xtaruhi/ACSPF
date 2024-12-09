# ACSPF

![License](https://img.shields.io/badge/License-LGPL_v2.1-blue)
![Language](https://img.shields.io/badge/Language-C++-blue)
![Platform](https://img.shields.io/badge/Platform-Linux-green)

Analog Cat S-parameter Fitting. This is a tool for fitting S-parameters of a circuit to a rational function, 
which is the result of the 2024 EDA Challenge Contest.

Analog Cat S参数拟合工具。这是一个用于将电路的S参数拟合为有理函数的工具，该仓库是2024 EDA设计精英挑战赛的比赛成果。

## Table of Contents

- [ACSPF](#acspf)
  - [Table of Contents](#table-of-contents)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)
    - [Prerequisites](#prerequisites)
    - [Installing](#installing)
  - [Usage ](#usage-)

## About <a name = "about"></a>

ACSPF implements the Vector Fitting algorithm and the ORA algorithm in C++. 

ACSPF 实现了向量拟合算法和ORA算法，并使用C++实现。

## Getting Started <a name = "getting_started"></a>

### Prerequisites

- CMake
- C++17 compiler
- spdlog
- Eigen3
- jemalloc
- oneapi-mkl

### Building

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DONEAPI_ROOT=<path to oneapi> -DUSE_MKL=ON 
make -j
```

You could also use `-DUSE_INTELCC=ON` to enable Intel C++ compiler.

你也可以使用 `-DUSE_INTELCC=ON` 来启用Intel C++编译器，使用`-DGEN_PYTHON_LIB=ON`来生成Python绑定库。

### Installing

```bash
cmake --install . --prefix <path to install>
```

## Usage <a name = "usage"></a>

```
Usage: vecfit [--help] [--method VAR] [--pole VAR] [--eval] [--write-ref] [--max-iters VAR] [--no-exact-dc] [--no-reduced-columns] [--threads VAR] [--verbose] [--version] file output

Positional arguments:
  file                  input file, TouchStone format [required]
  output                output file [nargs=0..1] [default: "eda240709_{case}.dat"]

Optional arguments:
  -h, --help            shows help message and exits 
  -m, --method          method to use, ORA or VF [nargs=0..1] [default: "ORA"]
  -p, --pole            number of poles, -1 for auto [nargs=0..1] [default: -1]
  -e, --eval            evaluate the model after fitting 
  --write-ref           write reference file, only available when --eval is set 
  --max-iters           maximum number of iterations, default 35 for VF, 8 for ORA 
  --no-exact-dc         do not use exact DC 
  --no-reduced-columns  use all data while pole fitting 
  -t, --threads         maximum number of threads, -1 for hardware concurrency [nargs=0..1] [default: 8]
  -v, --verbose         print verbose information 
  -V, --version         print version 
```

For example

```
vecfit -e -v -p 10 examples/symind.s2p
```

This will fit the S-parameters in `examples/symind.s2p` to a rational function with 10 poles and evaluate the model. The output will be written to `eda240709_symind.dat`.

该命令将使用向量拟合算法，使用10个极点，对`examples/symind.s2p`中的S参数进行拟合，并评估拟合结果。输出文件为`eda240709_symind.dat`。
