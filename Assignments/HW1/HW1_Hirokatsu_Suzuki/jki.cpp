/*

Hirokatsu Suzuki
COMP605 - HW1 (Serial Matrix Multiplication: jki form)

*/


#include <iostream>
#include <sys/time.h>
#define row1 400
#define col1 400
#define row2 400
#define col2 400

int main() {

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

    //Matrix multiplication (jki form)
    int r = 0;
    struct timeval start,end;
    gettimeofday(&start,0);
    for (int j = 0; j < col2; j++) {
        for (int k = 0; k < col1; k++) {
            r = mat2[k][j];
            for (int i = 0; i < row1; i++) {
                mat3[i][j] = mat3[i][j] + mat1[i][k] * mat2[k][j];
            };
        };
    };
    gettimeofday(&end, 0);

    //Calculate the runtime
    long sec = end.tv_sec - start.tv_sec;
    long microsec = end.tv_usec - start.tv_usec;
    double ellapse = sec + (microsec * 0.000001);
    std::cout << ellapse << std::endl;


    // Print matrix3
    // for (int x=0; x<row1;x++) {
    //     for (int y=0;y<col2;y++) {
    //         std::cout<<mat3[x][y];
    //         std::cout<<',';
    //     };
    // std::cout << std::endl;
    // };
    // std::cout << std::endl;

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
};