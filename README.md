# Differentiable Material Point Method (Diff-MPM)

MPM simulations are applied in various fields such as computer graphics, geotechnical engineering, computational mechanics and more. `diffmpm` is a differentiable MPM simulation library written entirely in JAX which means it also has all the niceties that come with JAX. It is a highly parallel, Just-In-Time compiled code that can run on CPUs, GPUs or TPUs. It aims to be a fast solver that can be used in various problems like optimization and inverse problems. Having a differentiable MPM simulation opens up several advantages - 
- **Efficient Gradient-based Optimization:** Since the entire simulation model is differentiable, it can be used in conjunction with various gradient-based optimization techniques such as stochastic gradient descent (SGD), ADAM etc.
- **Inverse Problems:** It also enables us to solve inverse problems to determine material properties by formulating an inverse problem as an optimization task.
- **Integration with Deep Learning:** It can be seamlessly integrated with other Neural Network models to enable training physics-informed neural networks.

## Installation
`diffmpm` can be installed directly from PyPI using `pip`

``` shell
pip install diffmpm
```

#### ToDo
Add separate installation commands for CPU/GPU.

## Usage
Once installed, `diffmpm` can be used as a CLI tool or can be imported as a library in Python. Example input files can be found in the `benchmarks/` directory.

```
Usage: mpm [OPTIONS]

  CLI utility for DiffMPM.

Options:
  -f, --file TEXT  Input TOML file  [required]
  --version        Show the version and exit.
  --help           Show this message and exit.
```

Further documentation about the input file can be found in the documentation _[INSERT LINK HERE]_. `diffmpm` can write the output to various file types like `.npz`, `.vtk` etc. that can then be used to visualize the output of the simulations.

## Examples
