/*

Hirokatsu Suzuki
COMP605 - HW2 part a. matrix multiplication

*/

#include<iostream>
#include<omp.h>

#define row1 3000 //define row
#define col1 3000 //define column
#define row2 3000 //define row
#define col2 3000 //define column

int main(int argc, char* argv[]){

    //Initialize the size of matrices
    int** mat1 = new int* [row1];
    for (int i = 0; i < row1; i++) {
         mat1[i] = new int[col1];
    };

    int** mat2 = new int*[row2];
    for (int i = 0; i < row2; i++) {
        mat2[i] = new int[col2];
    };

    int** mat3 = new int*[row1];
    for (int i = 0; i < row1; i++) {
        mat3[i] = new int[col2];
    }

    //Fill random numbers into matrices
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col1; j++) {
            mat1[i][j] = rand() % 9;
        };
    };

    for (int i = 0; i < row2; i++) {
        for (int j = 0; j < col2; j++) {
            mat2[i][j] = rand() % 9;
        };
    };

    //start time
    double dtime = omp_get_wtime();

    //parallelize loop
#   pragma omp parallel
        //int i, j, k;
       #pragma omp for
       //ijk-form matrix multipication
        for (int i = 0; i < row1; i++) {
            for (int j = 0; j < col2; j++) {
                mat3[i][j] = 0;
                for (int k = 0; k < col1; k++) {
                    mat3[i][j] = mat3[i][j] + mat1[i][k] * mat2[k][j];
                };
            };
        };

    //calculate runtime
    dtime = omp_get_wtime() - dtime;
    printf("%f\n", dtime);

    //Delete matrices
    for (int i = 0; i < row1; i++) {
        delete[] mat1[i];
    };
    delete[] mat1;

    for (int i = 0; i < row2; i++) {
        delete[] mat2[i];
    };
    delete[] mat2;

    for (int i = 0; i < row1; i++) {
        delete[] mat3[i];
    };
    delete[] mat3;



    return 0;
}