/*
Matrix-Matrix Multiplication using CUDA and OpenMP
Hirokatsu Suzuki
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

// fill in matrix with factor input (float)
void fill_matrix(int *A, float fac, int m, int n) {
  int i, j;
  for (i=0; i<m;i++)
  {
    for (j=0;j<n;j++)
    {
      A[i*n+j] = i+j*fac;
    }
  }
}

// print matrix from GPU
void print_matrix(int *A, int m, int n) {

  int i, j;

  for (i=0; i<m;i++)
  {
    for (j=0;j<n;j++)
    {
      printf("mat[%d, %d] = %d\n", i, j, A[i*n+j]);
    }
  }
}

// matrix-matrix multiplication - CPU
void perform_operation(int *A, int *B, int *C, int m, int n) {

  int i, j;

  for (i=0; i<m;i++)
  {
    for (j=0;j<n;j++)
    {
      C[i*n+j] = A[i*n+j]*B[i*n+j];
      //printf("C[%d, %d] = %d\n", i, j, C[i*n+j]);
    }
  }
}

// matrix-matrix multiplication - CPU
__global__ void perform_operation_cuda(int *A, int *B, int *C, int m, int n) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i<m)
  {
    if (j<n)
    {
      C[i*n+j] = A[i*n+j]*B[i*n+j];
    }
  }
}

// construct device id
struct aux_device {
  int gpuid; //gpu id
  int *A_d; // device array A on the gpu gpuid
};

int main (int argc, char **argv) {

  // define variables
  const int N = 300;
  const int M = 300;
  int A[N*M], B[N*M], C[N*M];
  int f, nf=3, sum = 0;
  int *B_d, *C_d;

  // set grid and block dimensions
  //dim3 dimBlock(N*M, N*M);
  dim3 dimGrid(4, 4);
  dim3 dimBlock(200, 200); //change this

  // set gpu, cpu, thread ids
  int num_gpus = -1, num_cpus=-1;
  int gpuid = -1;
  aux_device *dev_mem;
  unsigned int cpu_thread_id = -1;

  // fill in matrices A, B, C
  fill_matrix(A, 3.0, N, M);
  fill_matrix(B, 1.0, N, M);
  fill_matrix(C, 0.0, N, M);

  // set GPU cores, Threads
  cudaGetDeviceCount(&num_gpus);
  num_cpus = omp_get_max_threads();
  dev_mem = (aux_device *)malloc(num_gpus*sizeof(aux_device));

  // Set the threads to each GPUs
#pragma omp parallel shared(num_gpus, dev_mem) private(cpu_thread_id, gpuid)
{
#pragma omp critical
  {
  cpu_thread_id = omp_get_thread_num();
  cudaSetDevice(cpu_thread_id % num_gpus);
  cudaGetDevice(&gpuid);
  dev_mem[gpuid].gpuid = cpu_thread_id;
  }
}

  //allocate and copy the array A ones by GPU
#pragma omp parallel shared(A, dev_mem, num_gpus) private(f, cpu_thread_id, gpuid)
{
#pragma omp critical
  {
    cpu_thread_id = omp_get_thread_num();
    for (f = 0;f<num_gpus;f++)
    {
      if (cpu_thread_id == dev_mem[f].gpuid)
      {
        cudaMalloc( (void **)&dev_mem[f].A_d, sizeof(int) * N*M);
        cudaMemcpy( dev_mem[f].A_d, A, sizeof(int) * N*M, cudaMemcpyHostToDevice);
        cudaGetDevice(&gpuid);
        break;
      }
    }
  }
}

  // matrix-matrix multiplication on GPU with OpenMP
  sum = 0;
#pragma omp parallel shared(dimGrid, dimBlock, nf, sum, dev_mem) private(f, B, B_d, C_d, C, gpuid, cpu_thread_id)
{

  cudaMalloc( (void **)&B_d, sizeof(int) * N*M);
  cudaMalloc( (void **)&C_d, sizeof(int) * N*M);


  cudaGetDevice(&gpuid);
  #pragma omp for reduction(+:sum)
  for (f=0; f<nf; f++)
  {

    // fill in matrix B and allocate memory
    fill_matrix(B, f+1.0, N, M);
    cudaMemcpy( B_d, B, sizeof(int) * N*M, cudaMemcpyHostToDevice);

    // begin time
    struct timespec startGPU, endGPU;
	  double elapsedGPU;
	  clock_gettime(CLOCK_REALTIME, &startGPU);

    // call function
    perform_operation_cuda<<<dimGrid, dimBlock>>>(dev_mem[gpuid].A_d, B_d, C_d, N, M);

    // end time
    clock_gettime(CLOCK_REALTIME, &endGPU);
	  elapsedGPU = (endGPU.tv_sec - startGPU.tv_sec) + (endGPU.tv_nsec - startGPU.tv_nsec) / 1000000000.0;
	  printf("Time taken (GPU): %lf s\n", elapsedGPU);
    // printf("=============================\n")
    // print_matrix(C, N, M)

    // copy data to host 
    cudaMemcpy( C, C_d, sizeof(int) * N*M, cudaMemcpyDeviceToHost);
  }

  // free memory
  cudaFree(B_d);
  cudaFree(C_d);
}

  for (f = 0;f<num_gpus;f++)
  {
    cudaFree(dev_mem[f].A_d);
  }

  // reset cuda
  cudaDeviceReset();

  // cpu matrix-multiplication
  printf("================================\n");

  // begin time
  struct timespec startCPU, endCPU;
  double elapsedCPU;
  clock_gettime(CLOCK_REALTIME, &startCPU);

  // call function
  perform_operation(A, B, C, N, M);

  // end time
  clock_gettime(CLOCK_REALTIME, &endCPU);
  elapsedCPU = (endCPU.tv_sec - startCPU.tv_sec) + (endCPU.tv_nsec - startCPU.tv_nsec) / 1000000000.0;
  printf("Time taken (CPU): %lf s\n", elapsedCPU);

  return 0;
}