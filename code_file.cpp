#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

#define N 1024 // Size of the grid (N x N)
#define STEPS 1000 // Number of Monte Carlo steps

// CUDA kernel to perform the Ising model simulation step
__global__ void ising_step(int* grid, int* new_grid, int size, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int i = idx / N;
        int j = idx % N;

        int up = (i - 1 + N) % N;
        int down = (i + 1) % N;
        int left = (j - 1 + N) % N;
        int right = (j + 1) % N;

        int sum_neighbors = grid[up * N + j] + grid[down * N + j] + grid[i * N + left] + grid[i * N + right];
        int delta_energy = 2 * grid[i * N + j] * sum_neighbors;

        // Metropolis acceptance criterion
        if (delta_energy < 0 || (rand() / (float)RAND_MAX) < exp(-beta * delta_energy)) {
            new_grid[i * N + j] = -grid[i * N + j]; // Flip spin
        } else {
            new_grid[i * N + j] = grid[i * N + j]; // No flip
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize grid
    int grid[N * N];
    int new_grid[N * N];
    float beta = 0.5; // Inverse temperature

    // Initialize the grid with random values (+1 or -1)
    if (rank == 0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 1);

        for (int i = 0; i < N * N; i++) {
            grid[i] = dis(gen) * 2 - 1;  // Random -1 or +1
        }
    }

    // Broadcast the initial grid to all processes
    MPI_Bcast(grid, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory on the GPU
    int* d_grid;
    int* d_new_grid;
    cudaMalloc(&d_grid, N * N * sizeof(int));
    cudaMalloc(&d_new_grid, N * N * sizeof(int));

    cudaMemcpy(d_grid, grid, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Perform Monte Carlo steps
    for (int step = 0; step < STEPS; step++) {
        ising_step<<<(N * N + 255) / 256, 256>>>(d_grid, d_new_grid, N * N, beta);

        // Copy the new grid back to the host
        cudaMemcpy(new_grid, d_new_grid, N * N * sizeof(int), cudaMemcpyDeviceToHost);

        // Update the grid (for visualization or further calculations)
        if (rank == 0 && step % 100 == 0) {  // Print grid state every 100 steps
            std::cout << "Step " << step << ":\n";
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    std::cout << new_grid[i * N + j] << " ";
                }
                std::cout << "\n";
            }
        }

        // Swap grids for the next step
        std::swap(grid, new_grid);
        cudaMemcpy(d_grid, grid, N * N * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Free GPU memory
    cudaFree(d_grid);
    cudaFree(d_new_grid);

    MPI_Finalize();
    return 0;
}
