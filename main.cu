
// For the CUDA runtime routines ( prefixed with "cuda_" )
#include <cuda_runtime.h>

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <string>

#include "timer.hpp"

// #define _DISP_IMAGE_ARRAY

// host func to create array ( image frame )
__host__ void create_imageframe ( double *(*frame), const int width, const int height, const int channels )
{
	// generate frame
	for ( int i=0; i<height; i++ ) for ( int j=0; j<width; j++ ) for ( int c=0; c<channels; c++ )
	{
		int idx = ( i*width+j )*channels + c;
		(*frame)[idx] = (float)( i+j+c );
	}
}

// kernel function to convert image
__global__ void cvtImg_cuda ( double *pcd, const double *image, const int size )
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int cidx = idx % 3; // channels=3
	if ( idx < size )
	{
		// convert G and B into 0
		if ( cidx == 1 || cidx == 2 )
			pcd[idx] = 0.0;
		else
			pcd[idx] = image[idx];
	}
}

__host__ void cvtImg ( double *pcd, const double *image, const int size )
{
	for ( int idx=0; idx<size; idx++ )
	{
		int cidx = idx % 3; // channels=3
		// convert G and B into 0
		if ( cidx == 1 || cidx == 2 )
			pcd[idx] = 0.0;
		else
			pcd[idx] = image[idx];
	}
}

// disp image
__host__ void disp_image ( const double *image, const int width, const int height, const int channels, const bool mode_debug=true )
{
	if ( !mode_debug ) return;
	for ( int i=0; i<height; i++ )
	{
		for ( int j=0; j<width; j++ )
		{
			int idx = (i*width+j)*channels;
			std::cout << " (" << image[idx+0];
			if ( channels > 1 )
			{
				for ( int c=1; c<channels; c++ )
					std::cout << "," << image[idx+c];
			}
			std::cout << ")";
		}
		std::cout << std::endl;
	}
}

__host__ void logging ( std::string msg, int *logcount=nullptr, const bool mode_debug=true )
{
	if ( !mode_debug ) return;
	std::cout << "[TEST]";
	if ( logcount )
	{
		(*logcount) ++;
		std::cout << "(" << *logcount << ")";
	}
	std::cout << ": " << msg << std::endl;
}

__host__ void get_args ( bool *mode_debug, bool *mode_cuda, bool *mode_cuda_um,
						 int *width, int *height, int *channels,
						 int *dim_block,
						 const int argc, const char **argv )
{
	if ( argc == 1 )
	{
		std::cout << std::endl;
		std::cout << "options:" << std::endl;
		std::cout << "    -h | --help           : Help options." << std::endl;
		std::cout << "    -v | --verbose        : Debug logs." << std::endl;
		std::cout << "    -c | --cuda           : (on/off) Use CUDA computing." << std::endl;
		std::cout << "    -u | --unified-memory : (on/off) Use unified memory system." << std::endl;
		std::cout << "    -r | --resolution     : (QQVGA/QVGA/VGA/XGA/HD/FHD/2K/3K/4K/5K/6K/8K/10K/16K)" << std::endl;
		std::cout << "    -W | --width          : (int value) Image width." << std::endl;
		std::cout << "    -H | --height         : (int value) Image height." << std::endl;
		std::cout << "    -C | --channels       : (int value) Number of image color channels." << std::endl;
		std::cout << "    -B | --blocks         : (int value) Number of GPU blocks." << std::endl;
		std::cout << std::endl;
		exit(0);
	}

	for ( int j=1; j<argc; j++ )
	{
		std::string argstr = std::string(argv[j]);
		if (
			 argstr != "-h" && argstr != "--help" &&
			 argstr != "-v" && argstr != "--verbose" &&
			 argstr != "-c" && argstr != "--cuda" &&
			 argstr != "-u" && argstr != "--unified-memory" &&
			 argstr != "-r" && argstr != "--resolution" &&
			 argstr != "-W" && argstr != "--width" &&
			 argstr != "-H" && argstr != "--height" &&
			 argstr != "-C" && argstr != "--channels" &&
			 argstr != "-B" && argstr != "--blocks"
		)
		{
			std::cerr << "ERROR: " << argstr << " is not an option." << std::endl;
			exit(1);
		}

		if ( argstr == "-h" || argstr == "--help" )
		{
			std::cout << std::endl;
			std::cout << "options:" << std::endl;
			std::cout << "    -h | --help           : Help options." << std::endl;
			std::cout << "    -v | --verbose        : Debug logs." << std::endl;
			std::cout << "    -c | --cuda           : (on/off) Use CUDA computing." << std::endl;
			std::cout << "    -u | --unified-memory : (on/off) Use unified memory system." << std::endl;
			std::cout << "    -r | --resolution     : (QQVGA/QVGA/VGA/XGA/HD/FHD/2K/3K/4K/5K/6K/8K/10K/16K)" << std::endl;
			std::cout << "    -W | --width          : (int value) Image width." << std::endl;
			std::cout << "    -H | --height         : (int value) Image height." << std::endl;
			std::cout << "    -C | --channels       : (int value) Number of image color channels." << std::endl;
			std::cout << "    -B | --blocks         : (int value) Number of GPU blocks." << std::endl;
			std::cout << std::endl;
			exit(0);
		}
		if ( argstr == "-v" || argstr == "--verbose" )
			*mode_debug = true;
		if ( argstr == "-c" || argstr == "--cuda" )
		{
			if ( j+1 == argc )
			{
				std::cerr << "ERROR: No command found." << std::endl;
				exit(1);
			}
			std::string argval = std::string(argv[j+1]);
			if ( argval == "on" ) *mode_cuda = true;
			else if ( argval == "off" ) *mode_cuda = false;
			else
			{
				std::cerr << "ERROR: Unknown command found." << std::endl;
				exit(1);
			}
			j ++;
		}
		if ( argstr == "-u" || argstr == "--unified-memory" )
		{
			if ( j+1 == argc )
			{
				std::cerr << "ERROR: No command found." << std::endl;
				exit(1);
			}
			std::string argval = std::string(argv[j+1]);
			if ( argval == "on" ) *mode_cuda_um = true;
			else if ( argval == "off" ) *mode_cuda_um = false;
			else
			{
				std::cerr << "ERROR: Unknown command found." << std::endl;
				exit(1);
			}
			j ++;
		}
		if ( argstr == "-r" || argstr == "--resolution" )
		{
			if ( j+1 == argc )
			{
				std::cerr << "ERROR: No command found." << std::endl;
				exit(1);
			}
			std::string argval = std::string(argv[j+1]);
			if ( argval == "QQVGA" )
			{
				*width = 160; *height = 120;
			}
			else if ( argval == "QVGA" )
			{
				*width = 320; *height = 240;
			}
			else if ( argval == "VGA" )
			{
				*width = 640; *height = 480;
			}
			else if ( argval == "XGA" )
			{
				*width = 1024; *height = 768;
			}
			else if ( argval == "HD" )
			{
				*width = 1280; *height = 720;
			}
			else if ( argval == "FHD" || argval == "2K" )
			{
				*width = 1920; *height = 1080;
			}
			else if ( argval == "3K" )
			{
				*width = 2880; *height = 1620;
			}
			else if ( argval == "4K" )
			{
				*width = 3840; *height = 2160;
			}
			else if ( argval == "5K" )
			{
				*width = 5120; *height = 2880;
			}
			else if ( argval == "6K" )
			{
				*width = 6016; *height = 3384;
			}
			else if ( argval == "8K" )
			{
				*width = 7680; *height = 4320;
			}
			else if ( argval == "10K" )
			{
				*width = 10240; *height = 4320;
			}
			else if ( argval == "16K" )
			{
				*width = 15360; *height = 4320;
			}
			else
			{
				std::cerr << "ERROR: Unknown command found." << std::endl;
				exit(1);
			}
			j ++;
		}
		if ( argstr == "-W" || argstr == "--width" )
		{
			if ( j+1 == argc )
			{
				std::cerr << "ERROR: No command found." << std::endl;
				exit(1);
			}
			char *endptr;
			int num = strtol(argv[j+1], &endptr, 10);
			if (*endptr != '\0' || (num == INT_MAX && ERANGE == 0))
			{
				std::cerr << "ERROR: Irregal value has inputted" << std::endl;
				exit(1);
			}
			*width = num;
			j ++;
		}
		if ( argstr == "-H" || argstr == "--height" )
		{
			if ( j+1 == argc )
			{
				std::cerr << "ERROR: No command found." << std::endl;
				exit(1);
			}
			char *endptr;
			int num = strtol(argv[j+1], &endptr, 10);
			if (*endptr != '\0' || (num == INT_MAX && ERANGE == 0))
			{
				std::cerr << "ERROR: Irregal value has inputted" << std::endl;
				exit(1);
			}
			*height = num;
			j ++;
		}
		if ( argstr == "-C" || argstr == "--channels" )
		{
			if ( j+1 == argc )
			{
				std::cerr << "ERROR: No command found." << std::endl;
				exit(1);
			}
			char *endptr;
			int num = strtol(argv[j+1], &endptr, 10);
			if (*endptr != '\0' || (num == INT_MAX && ERANGE == 0))
			{
				std::cerr << "ERROR: Irregal value has inputted" << std::endl;
				exit(1);
			}
			*channels = num;
			j ++;
		}
		if ( argstr == "-B" || argstr == "--blocks" )
		{
			if ( j+1 == argc )
			{
				std::cerr << "ERROR: No command found." << std::endl;
				exit(1);
			}
			char *endptr;
			int num = strtol(argv[j+1], &endptr, 10);
			if (*endptr != '\0' || (num == INT_MAX && ERANGE == 0))
			{
				std::cerr << "ERROR: Irregal value has inputted" << std::endl;
				exit(1);
			}
			*dim_block = num;
			j ++;
		}
	}
}

__host__ void disp_args ( const bool mode_debug, const bool mode_cuda, bool mode_cuda_um,
						  const int width, const int height, const int channels,
						  const int dim_block )
{
	std::cout << std::endl;
	std::cout << "===============================" << std::endl;
	std::cout << " parameters : value" << std::endl;
	std::cout << "-------------------------------" << std::endl;
	std::cout << " debug mode : ";
	if ( mode_debug ) std::cout << "on";
	else std::cout << "off";
	std::cout << std::endl;
	std::cout << " cuda mode  : ";
	if ( mode_cuda )
	{
		std::cout << "on";
		if ( mode_cuda_um ) std::cout << " ( unified memory )";
		std::cout << std::endl;
		std::cout << " GPU blocks : " << dim_block;
	}
	else std::cout << "off";
	std::cout << std::endl;
	std::cout << " width      : " << width << std::endl;
	std::cout << " height     : " << height << std::endl;
	std::cout << " channels   : " << channels << std::endl;
	std::cout << "===============================" << std::endl;
	std::cout << std::endl;
}

int main ( const int argc, const char **argv )
{

	// define logging params
	timer t;
	int logcount = 0;
	char buffer[256];

	// mode params
	bool mode_debug = false;
	bool mode_cuda = false;
	bool mode_cuda_um = false;

	// define image params
	int width = 4, height = 4, channels = 3;
	int dim_block = 8;

	// define frame instances @ host
	double *frame_host, *pcd_host;

	logging("Start execution.", &logcount, mode_debug);
	t.start();

	get_args(&mode_debug, &mode_cuda, &mode_cuda_um, &width, &height, &channels, &dim_block, argc, argv);
	t.push();
	snprintf(buffer, sizeof(buffer), "Got args. / %.6f s ( %.6f s )", t.local, t.total);
	logging(std::string(buffer), &logcount, mode_debug);

	// define frame params
	int size = width*height*channels;
	int nbytes = sizeof(double)*size;
	disp_args(mode_debug, mode_cuda, mode_cuda_um, width, height, channels, dim_block);

	// allocation
	if ( mode_cuda && mode_cuda_um )
	{
		cudaMallocManaged(&frame_host, nbytes, cudaMemAttachGlobal);
		t.push();
		snprintf(buffer, sizeof(buffer), "Global input memories allocated. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);
	}
	else
	{
		// frame_host = new double [size];
		frame_host = new double [size]();  // and zero-initilization
		t.push();
		snprintf(buffer, sizeof(buffer), "Host input memories allocated. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);
	}
	// get image frame on host
	create_imageframe(&frame_host, width, height, channels);
	t.push();
	snprintf(buffer, sizeof(buffer), "Got image. / %.6f s ( %.6f s )", t.local, t.total);
	logging(std::string(buffer), &logcount, mode_debug);

#if defined(_DISP_IMAGE_ARRAY)
	disp_image(frame_host, width, height, channels, mode_debug);
#endif

	if ( mode_cuda && !mode_cuda_um )
	{

		// define CUDA params
		dim3 dB(dim_block), dG((int)(size/8));

		// define frame instances @ device
		double *frame_device, *pcd_device;

		// allocate device memory
		cudaMalloc(&frame_device, nbytes);
		t.push();
		snprintf(buffer, sizeof(buffer), "Device input memories allocated. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);
		cudaMalloc(&pcd_device, nbytes);
		t.push();
		snprintf(buffer, sizeof(buffer), "Device output memories allocated. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);
	
		// copy host's image frame to device memory
		cudaMemcpy(frame_device, frame_host, nbytes, cudaMemcpyHostToDevice);
		t.push();
		snprintf(buffer, sizeof(buffer), "Host memory has copied to GPU device. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);

		t.push();
		snprintf(buffer, sizeof(buffer), "Start GPU routine. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);
		// CUDA proc
		cvtImg_cuda <<< dG, dB >>> (pcd_device, frame_device, size);

		// barrier
		cudaDeviceSynchronize();
		t.push();
		snprintf(buffer, sizeof(buffer), "Finish GPU routine. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);

		snprintf(buffer, sizeof(buffer), "Memory usage: %d B", nbytes);
		logging(std::string(buffer), &logcount, mode_debug);

		// copy device's frame to host memory
		pcd_host = new double [width*height*3];  // width*height*[xyz]
		cudaMemcpy(pcd_host, pcd_device, nbytes, cudaMemcpyDeviceToHost);
		t.push();
		snprintf(buffer, sizeof(buffer), "Device memory has copied to CPU device. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);

		// free device memory
		cudaFree(frame_device);
		cudaFree(pcd_device);
		t.push();
		snprintf(buffer, sizeof(buffer), "Device memory has been free. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);

		// disp_image(pcd_host, width, height, channels, mode_debug);

	}
	else if ( mode_cuda && mode_cuda_um )
	{

		// define CUDA params
		dim3 dB(dim_block), dG((int)(size/8));

		// no define frame instances @ device
		// no allocate device memory
		// no copy host's image frame to device memory

		// allocate GLOBAL memory
		cudaMallocManaged(&pcd_host, nbytes, cudaMemAttachGlobal);
		t.push();
		snprintf(buffer, sizeof(buffer), "Global output memories allocated. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);

		t.push();
		snprintf(buffer, sizeof(buffer), "Start GPU routine. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);

		// CUDA proc on unified memory
		cvtImg_cuda <<< dG, dB >>> (pcd_host, frame_host, size);

		// barrier
		cudaDeviceSynchronize();
		t.push();
		snprintf(buffer, sizeof(buffer), "Finish GPU routine. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);

		snprintf(buffer, sizeof(buffer), "Memory usage: %d B", nbytes);
		logging(std::string(buffer), &logcount, mode_debug);

		// no copy device's frame to host memory
		// no free device memory
#if defined(_DISP_IMAGE_ARRAY)
		disp_image(pcd_host, width, height, channels, mode_debug);
#endif

	}
	else
	{

		// allocate host's memory
		// pcd_host = new double [size];
		pcd_host = new double [size](); // and zero-initilization
		t.push();
		snprintf(buffer, sizeof(buffer), "Host output memories allocated. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);

		t.push();
		snprintf(buffer, sizeof(buffer), "Start CPU routine. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);

		cvtImg(pcd_host, frame_host, size);

		t.push();
		snprintf(buffer, sizeof(buffer), "Finish CPU routine. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);

		snprintf(buffer, sizeof(buffer), "Memory usage: %d B", nbytes);
		logging(std::string(buffer), &logcount, mode_debug);

#if defined(_DISP_IMAGE_ARRAY)
		disp_image(pcd_host, width, height, channels, mode_debug);
#endif

	}

	// free host and device memories
	if ( mode_cuda && mode_cuda_um )
	{
		cudaFree(frame_host);
		t.stop();
		snprintf(buffer, sizeof(buffer), "Global input memory has been free. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);	
		cudaFree(pcd_host);
		t.stop();
		snprintf(buffer, sizeof(buffer), "Global output memory has been free. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);	
	}
	else
	{
		delete [] frame_host;
		t.stop();
		snprintf(buffer, sizeof(buffer), "Host input memory has been free. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);	
		delete [] pcd_host;
		t.stop();
		snprintf(buffer, sizeof(buffer), "Host output memory has been free. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);	
	}

	return 0;
}

