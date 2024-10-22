#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mpi.h>
#include <string>

static void HandleError(cudaError_t err, const char *file, int line){
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__))


int main(int argc, char** argv){

    MPI_Init(&argc, &argv);

    int size,rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(argc < 2){
      if(rank == 0){
	printf("Usage: mpiexec -n 2 ./bw_and_leak_check <H|D>\n");
      }
    }

    bool host_transfer = std::string(argv[1])=="H";

    if(rank == 0){
      if(host_transfer){
	printf("Doing data transfers on host\n");
      } else {
	printf("Doing data transfers on device\n");
      }
    }
    
    MPI_Status stat;

    if (size !=2) {
        if (rank == 0) {
            printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
        }
        MPI_Finalize();
        return 0;
    }
    HANDLE_ERROR(cudaSetDevice(rank));

    if(rank == 0){
        printf("%20s %20s %20s %20s \n", "Trans size (B)", "Time (s)", "BW (GB/s)", "G-Mem Avail (KB)");
    }

    // Loop 1 GB
    for (int i=0; i<=5; i++) {
        long int N;
        N = 1 << 27; // minus 3 because double is 2^3 bytes

        // Alocate memory for A on CPU
        auto *A = (double*)malloc(N*sizeof(double));

        // Initialize all elements of A to 0.0
        for (int j=0; j<N; j++) {
            A[j] = 0.0;
        }

        double *d_A;
	if(host_transfer){
	  d_A = new double[N];
	  for(int j=0; j<N; j++) d_A[j] = A[j];
	} else {
	  HANDLE_ERROR(cudaMalloc(&d_A, N*sizeof(double)));
	  HANDLE_ERROR(cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice));
	}

        int tag1 = 10;
        int tag2 = 11;

        int loop_count = 20;

        double start_time, stop_time, elapsed_time;
        start_time = MPI_Wtime();

        for (int j=1; j<=loop_count; j++) {
            if(rank == 0) {
                MPI_Send(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
                MPI_Recv(d_A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
            }
            else if(rank == 1) {
                MPI_Recv(d_A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
                MPI_Send(d_A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
            }
        }

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;

        long int num_B = 8*N;
        long int B_in_GB = 1 << 30;
        double num_GB = (double)num_B /(double)B_in_GB;
        double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

        size_t amem, tmem;
        HANDLE_ERROR(cudaMemGetInfo(&amem, &tmem));

        if(rank == 0) printf("%20li %20.9f %20.9f %20zu \n",
                             num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer, amem>>10);

	if(host_transfer){
	  delete[] d_A;
	} else {
	  HANDLE_ERROR(cudaFree(d_A));
	}
        free(A);
    }

    MPI_Finalize();
    return 0;
}
