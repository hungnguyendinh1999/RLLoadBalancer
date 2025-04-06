<h1 align="center"><b>RL/LB</b></h1>
<h4 align="center">Using Reinforcement Learning to solve load balancing</h4>

## Build
- Create a build directory using: `mkdir build; cd build`
- Generating build files: `cmake ..`
- Build using **make**: `make`

## Install LibTorch for Build:
You can find the instructions here: [Installing C++ Distributions of PyTorch](https://pytorch.org/cppdocs/installing.html)

Or if you already have torch installed, you can redirect the `CMAKE_PREFIX_PATH` by compiling with:
```bash
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/torch/share/cmake ..
```

Or if you used `conda`, here is a specific way to call cmake:
```bash
cmake -DCMAKE_PREFIX_PATH=$(conda activate example_env; python -c 'import torch;print(torch.utils.cmake_prefix_path)')
```