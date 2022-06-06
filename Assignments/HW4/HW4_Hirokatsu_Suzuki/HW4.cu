/*
RGB to Grayscale Conversion using CUDA
Hirokatsu (Hiro) Suzuki
*/

#include <stdio.h>
#include <string>
#include <math.h>
#include <time.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

unsigned char *h_rgb_image; //store rgb data

size_t loadimg(unsigned char *grey_image, const std::string &input, int *rows, int *cols) 
{
    //opencv Mat object
	cv::Mat img_data; 

	// read image data into img_data Mat object
	img_data = cv::imread(input.c_str());
	*rows = img_data.rows;
	*cols = img_data.cols;

	// allocate memory
	h_rgb_image = (unsigned char*) malloc(*rows * *cols * sizeof(unsigned char) * 3);
	unsigned char* rgb_image = (unsigned char*)img_data.data;

	//populate host's rgb data array
	int x = 0;
	for (x = 0; x < *rows * *cols * 3; x++)
	{
		h_rgb_image[x] = rgb_image[x];
	}
	
    // calculate total pixels
	size_t pixel = img_data.rows * img_data.cols;
	
	return pixel;
}

// RGB to grey GPU
__global__ void RGB2Grey(unsigned char *rgb, unsigned char *grey, int rows, int cols) {

    // get row and column numbers
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

    // RGB to Grayscale conversion
	if (col < cols && row < rows) {
		int offset_1 = row * cols + col;
		int offset = offset_1 * 3;
	
        // separate RGB
    	unsigned char red = rgb[offset + 0];
	    unsigned char green = rgb[offset + 1];
	    unsigned char blue = rgb[offset + 2];
	
        // luminosity method
	    grey[offset_1] = red * 0.3f + green * 0.59f + blue * 0.11f;
    }
}

int main(int argc, char **argv) 
{
	std::string input;
	std::string output;

	// take input jpg file and name for output jpg file
    input = std::string(argv[1]);
    output = "new.jpg";
	
	unsigned char *d_rgb_image; // store rgb array pointer
	unsigned char *h_grey_image, *d_grey_image; // host and local's grey data array pointers
	int rows; //number of rows
	int cols; //number of columns
	
	// load image and get pixels number
	const size_t total_pixels = loadimg(h_grey_image, input, &rows, &cols);

	// allocate memory of host's grey data array
	h_grey_image = (unsigned char *)malloc(sizeof(unsigned char*)* total_pixels);

	// allocate memory on local
	cudaMalloc(&d_rgb_image, sizeof(unsigned char) * total_pixels * 3);
	cudaMalloc(&d_grey_image, sizeof(unsigned char) * total_pixels);

    // set device memory to a value
	cudaMemset(d_grey_image, 0, sizeof(unsigned char) * total_pixels);
	
	// copy host rgb data array to device rgb data array
	cudaMemcpy(d_rgb_image, h_rgb_image, sizeof(unsigned char) * total_pixels * 3, cudaMemcpyHostToDevice);

	// define block and grid dimensions
	// const dim3 dimGrid(200, 100);
    const dim3 dimGrid((int)ceil((cols)/16), (int)ceil((rows)/16)); // 152 106
	const dim3 dimBlock(32, 32);
	
	struct timespec start, end;
	double elapsed;
	clock_gettime(CLOCK_REALTIME, &start);
	
	// call RGB2Gray
	RGB2Grey<<<dimGrid, dimBlock>>>(d_rgb_image, d_grey_image, rows, cols);

	clock_gettime(CLOCK_REALTIME, &end);
	elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Time taken: %lf s\n", elapsed);

	// copy gray image from device to host
	cudaMemcpy(h_grey_image, d_grey_image, sizeof(unsigned char) * total_pixels, cudaMemcpyDeviceToHost);

	// put grey image data into cv Mat object
    cv::Mat greyData(rows, cols, CV_8UC1,(void *) h_grey_image);

	// save gray image as new jpg file
	cv::imwrite(output.c_str(), greyData);

    // free memories
	cudaFree(d_rgb_image);
	cudaFree(d_grey_image);
	return 0;
}