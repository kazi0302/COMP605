/*

Hirokatsu Suzuki
COMP605 - HW2 part b. approximate pi by randomly choosing points

*/

#include<iostream>
#include<omp.h>

#define n 100000000 //number of darts

int main(int arvc, char* argv[]) {

    //Initialize variables
    double x_coord = 0;
    int count = 0;
    double y_coord = 0;
    double r = 0;
    double pi = 0;

    //Begin time
    double dtime = omp_get_wtime();

    //Parallelize loop
#   pragma omp parallel for reduction(+:count)

    //generate random numbers
    for (unsigned int i = 0; i < n; i++) {
        x_coord = ((double) rand_r(&i) / (RAND_MAX));
        y_coord = ((double) rand_r(&i) / (RAND_MAX));

    //Calculate the radius (must be less than 1)
        r = (x_coord*x_coord + y_coord*y_coord);
        if (r <= 1) {
            count++;
        };
    };

    //Calculate runtime
    dtime = omp_get_wtime() - dtime;

    //Compute pi
    pi = (double)count/ n*4;

    //print result
    printf("# of trials= %d , estimate of pi is %g - %f sec\n", n, pi, dtime);

    return 0;
}
