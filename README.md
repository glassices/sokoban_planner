# Sokoban Planner Based on Self-Curriculum Learning

This is a CPP/PyTorch implementation of the sSokoban Planner

**Solving Hard AI Planning Instances Using Curriculum-Driven Deep Reinforcement Learning (https://arxiv.org/abs/2006.02689).**

**Cite as:**

*Dieqiao Feng, Carla Gomes, Bart Selman. Solving Hard AI Planning Instances Using Curriculum-Driven Deep Reinforcement Learning. 29th International Joint Conference on Artificial Intelligence Main track. (IJCAI 2020).*

## Requirement
Pytorch >= 1.8.0 and GCC compiler that supports C++17

## Running the Solver

To run the Sokoban solver simply run `python3 main.py`. Use `--resume` flag to resume from the last saved checkpoint. The solver automatically detects maximum usable GPU and CPU resources on the node and makes full use of them. To limit resource usage please add `CUDA_VISIBLE_DEVICES` and `OMP_NUM_THREADS` environment variables before running `python3 main.py`. We use “just in time” CPP extension compiling via `torch.utils.cpp_extension.load()` and the first-time compiling will be slow. Due to different argparse formats between python and cpp, please change the constant variable `mc::map_path` of `cpp/mcts_constants.hpp` if you want to run another Sokoban input instance. For Sokoban level format please check http://sokobano.de/wiki/index.php?title=Level_format.

## Models

The Sokoban planner has following components:

1. A heuristic predictor that outputs policy and value estimation of the current board using ResNet structure.
2. A multi-threaded multi-GPU MCTS component that searchs for solutions and generate training data as byproduct.
3. A replay buffer that maintains training data for each iterations.

The multi-threaded network predictor structure implemented in `cpp/nn.hpp` and `cpp/nn.cpp` is highly optimized and fine-tuned and can be used for other frameworks with minimum changes. The structure can automatically detects the CPU-GPU balance on the node and distributes computational jobs accordingly. To generalize to other tasks and for further use, just simply change the JIT module path variable `_module_path` in the constructor of class `NN`.
