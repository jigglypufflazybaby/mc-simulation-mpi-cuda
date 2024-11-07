# mc-simulation-mpi-cuda
 A parallelized Monte Carlo simulation using OpenMPI for distributed computing and CUDA for GPU acceleration, modeling systems like the Ising model for statistical physics. This repository demonstrates efficient high-performance computing (HPC) techniques to simulate physical systems and perform large-scale computations.

# Monte Carlo Simulation using OpenMPI and CUDA

## Overview

This project implements a **Monte Carlo simulation** using the **Ising model**, which is a classic model in statistical physics to simulate ferromagnetic systems. The simulation uses **OpenMPI** for parallel computation and **CUDA** for GPU acceleration, enabling efficient large-scale simulations on modern high-performance computing (HPC) systems.

### Key Features:
- **Parallelism with OpenMPI**: Distributes the Monte Carlo computation across multiple processors to handle large grids.
- **GPU Acceleration with CUDA**: Offloads computation-intensive steps to the GPU for faster performance.
- **Ising Model**: Models particle spins in a 2D grid with nearest-neighbor interactions using the Metropolis criterion.
- **Scalable**: Suitable for running on multiple CPUs/GPUs for large-scale simulations.

## Installation

### Prerequisites:
Before running this project, make sure the following software is installed:

- **CUDA Toolkit** (for GPU computation): [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
- **OpenMPI** (for parallel computing): 
  - On **Ubuntu/Debian**:
    ```bash
    sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
    ```
  - On **macOS** (using Homebrew):
    ```bash
    brew install open-mpi
    ```

- **C++ Compiler** (e.g., `g++` or `mpic++` for OpenMPI)

### Clone the Repository:
To clone the repository, run:

```bash
git clone https://github.com/your-username/mc-simulation-mpi-cuda.git
cd mc-simulation-mpi-cuda
```

### Usage

Compile the Code:
To compile the Monte Carlo simulation code, use the following command:

```bash
mpic++ -o mc_simulation mc_simulation.cpp -lcuda
```
This will create an executable file mc_simulation that is ready to run.

### Run the Simulation:

To run the simulation, use mpirun to launch the program across multiple processes. For example, to run with 4 processes:
```bash
mpirun -np 4 ./mc_simulation
```

## Input and Output:

- **Input**: The simulation starts with a randomly initialized grid of spins (either **+1** or **-1**). The grid can be of size **N x N**, with **N** specified in the code.
- **Output**: The simulation will output the updated grid configurations periodically (every **100 steps**). You can also modify the code to output other simulation properties, such as **energy**, **magnetization**, or **pressure**.

## File Formats:

- **.cpp**: C++ source code file containing the Monte Carlo simulation logic.
- **.xyz**: A simple text format for storing atomic coordinates and simulation data (if applicable).
- **.dat**: A generic data format used to store simulation results like **energy** and **magnetization**.
- **.txt**: Text files for logging results and statistics.

## Contributing

Feel free to fork this repository, report issues, or submit pull requests to contribute to the project. Contributions are welcome, especially in the following areas:

- **Optimizing** the simulation for larger systems
- Implementing more complex models (e.g., **3D Ising model**)
- Improving **GPU utilization**
- Enhancing **parallelization strategies**

## License

This project is licensed under the **MIT License**. See the **LICENSE** file for details.


