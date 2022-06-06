/*

Hirokatsu Suzuki
COMP605 - HW2 part c. approximate pi by integral

*/

#include<iostream>
#include<omp.h>

#define n 5 //number of trapezoids

//Define the function
double f(float x)
{
    return (4.0 / (1 + (x*x)));
};

int main(int argc, char* argv[]){

    //Define, initialize variables
    float h, a, b, sum, integral;
    a = 0.0;
    b = 1.0;
    h = (b - a) / n;
    sum = f(a) + f(b);

    //Begin time
    double dtime = omp_get_wtime();

    //Parallelize loop
#   pragma omp parallel for reduction(+:sum)

    //Trapezoidal rule - summation of the terms
    for (int i = 1; i < n; i++) {
        sum = sum + (2*f(a + (i*h)));
    };

    //Calculate runtime
    dtime = omp_get_wtime() - dtime;

    //Eavluate integral
    integral = (h / 2) * sum;

    //print result
    printf("Integral: %f with %f sec\n", integral, dtime);

    return 0;
}

