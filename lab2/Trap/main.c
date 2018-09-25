#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

double f(double x) {
    return sin(x);
}

double Trap(double a, double b, int n, double h) {
    double sum = (f(a) + f(b)) / 2.0;
    int i;
    for (i = 1; i < n; ++i) {
        a += h;
        sum += f(a);
    }
    sum *= h;
    return sum;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    if (argc >= 4) {
        int rank, size;
        int n = atoi(argv[1]);
        double a = atof(argv[2]), b = atof(argv[3]);

        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int local_n = n / size;
        int remain = n % size;
        double h = (b - a) / n;
        double local_a, local_b;
        if (rank < remain) {
            local_n++;
            local_a = a + (rank * local_n) * h;
        } else {
            local_a = a + (remain * (local_n + 1) + (rank - remain) * local_n) * h;
        }
        local_b = local_a + local_n * h;

        double local_int, total_int;
        local_int = Trap(local_a, local_b, local_n, h);
        MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("With n = %d trapezoids, our estimate of the integral from %g to %g is %g\n",
                   n, a, b, total_int);
        }
        MPI_Finalize();
    }
    return 0;
}