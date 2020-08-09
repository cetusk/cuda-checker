
// For the CUDA runtime routines ( prefixed with "cuda_" )
#include <cuda_runtime.h>

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <string>

#include <timer.hpp>
#include <argtools.hpp>

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
	if ( idx < size )
	{
        pcd[idx] = image[idx] + 1.0;
	}
}

__host__ void cvtImg ( double *pcd, const double *image, const int size )
{
	for ( int idx=0; idx<size; idx++ )
	{
        pcd[idx] = image[idx] + 1.0;
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


int main ( const int argc, const char **argv )
{

	// define logging params
	timer t, looptime;
	int logcount = 0;
	char buffer[256];

	// mode params
	bool mode_debug, mode_pagelock, mode_cuda, mode_cuda_um;
	// define image params
    int numloops, width, height, channels;
	int dim_block[3], dim_grid[2];

    // initialization
    argtools::init_params(&mode_debug, &mode_cuda, &mode_pagelock, &mode_cuda_um, &numloops, &width, &height, &channels, &dim_block, &dim_grid);

	// define frame instances
	double *frame_host, *pcd_host; // @ host
    double *frame_device, *pcd_device; // @ device ( use only non-unified memory of GPU calc )

    // GPU grid/block params
    dim3 dB, dG;

	logging("Start execution.", &logcount, mode_debug);
	t.start();

	argtools::get_args(&mode_debug, &mode_cuda, &mode_pagelock, &mode_cuda_um, &numloops, &width, &height, &channels, &dim_block, &dim_grid, argc, argv);
    argtools::adjust_cudaparams(&dim_block, &dim_grid, &mode_pagelock, &mode_cuda_um, mode_cuda, width, height, channels);
	t.push();
	snprintf(buffer, sizeof(buffer), "Got args. / %.6f s ( %.6f s )", t.local, t.total);
	logging(std::string(buffer), &logcount, mode_debug);
	argtools::disp_args(mode_debug, mode_cuda, mode_pagelock, mode_cuda_um, numloops, width, height, channels, dim_block, dim_grid);

	// define frame params
	int size = width*height*channels;
	int nbytes = sizeof(double)*size;
    int out_size = width*height*3; // width*height*[xyz]
    int out_nbytes = sizeof(double)*out_size;

	// allocation of input memory
    if ( mode_cuda && !mode_cuda_um ) // GPU
    {
		frame_host = new double [size]();  // and zero-initilization
        if ( mode_pagelock )
            cudaHostRegister(frame_host, nbytes, cudaHostRegisterDefault);
        // allocate device memory
        cudaMalloc(&frame_device, nbytes);
        t.push();
        snprintf(buffer, sizeof(buffer), "Device input memories allocated. / %.6f s ( %.6f s )", t.local, t.total);
        logging(std::string(buffer), &logcount, mode_debug);
    }
	else if ( mode_cuda && mode_cuda_um ) // GPU ( unified memory )
	{
		cudaMallocManaged(&frame_host, nbytes, cudaMemAttachGlobal);
		t.push();
		snprintf(buffer, sizeof(buffer), "Global input memories allocated. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);
	}
	else // CPU
	{
        frame_host = new double [size]();  // and zero-initilization
		t.push();
		snprintf(buffer, sizeof(buffer), "Host input memories allocated. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);
	}

    // allocation of output memory
    if ( mode_cuda && !mode_cuda_um ) // GPU
    {
        // define CUDA params
        dB.x = dim_block[0]; dB.y = dim_block[1]; dB.z = dim_block[2];
        dG.x = dim_grid[0]; dG.y = dim_grid[1];
        // allocate device memory
        cudaMalloc(&pcd_device, nbytes);
        t.push();
        snprintf(buffer, sizeof(buffer), "Device output memories allocated. / %.6f s ( %.6f s )", t.local, t.total);
        logging(std::string(buffer), &logcount, mode_debug);
        // allocate host memory
        pcd_host = new double [out_size]();
        if ( mode_pagelock )
            cudaHostRegister(pcd_host, out_nbytes, cudaHostRegisterDefault);
    }
    else if ( mode_cuda && mode_cuda_um ) // GPU ( unified memory )
    {
        // define CUDA params
        dB.x = dim_block[0]; dB.y = dim_block[1]; dB.z = dim_block[2];
        dG.x = dim_grid[0]; dG.y = dim_grid[1];
        // allocate GLOBAL memory
        cudaMallocManaged(&pcd_host, out_nbytes, cudaMemAttachGlobal);
        t.push();
        snprintf(buffer, sizeof(buffer), "Global output memories allocated. / %.6f s ( %.6f s )", t.local, t.total);
        logging(std::string(buffer), &logcount, mode_debug);
    }
    else
    {
        // allocate host's memory
        pcd_host = new double [out_size](); // and zero-initilization
        t.push();
        snprintf(buffer, sizeof(buffer), "Host output memories allocated. / %.6f s ( %.6f s )", t.local, t.total);
        logging(std::string(buffer), &logcount, mode_debug);
    }

    /*--->>> loop start <<<---*/
    looptime.start();
    for ( int n=0; n<numloops; n++ )
    {
        snprintf(buffer, sizeof(buffer), "Loop %d", n+1);
        logging(std::string(buffer), &logcount, mode_debug);

        // get image frame on host
        create_imageframe(&frame_host, width, height, channels);
        t.push();
        snprintf(buffer, sizeof(buffer), "Got image. / %.6f s ( %.6f s )", t.local, t.total);
        logging(std::string(buffer), &logcount, mode_debug);

#if defined(_DISP_IMAGE_ARRAY)
        disp_image(frame_host, width, height, channels, mode_debug);
#endif

        // calculation
        if ( mode_cuda && !mode_cuda_um ) // GPU
        {
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
            cudaMemcpy(pcd_host, pcd_device, nbytes, cudaMemcpyDeviceToHost);
            t.push();
            snprintf(buffer, sizeof(buffer), "Device memory has copied to CPU device. / %.6f s ( %.6f s )", t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);

#if defined(_DISP_IMAGE_ARRAY)
            disp_image(pcd_host, width, height, channels, mode_debug);
#endif

        }
        else if ( mode_cuda && mode_cuda_um ) // GPU ( unified memory )
        {
            // define CUDA params
            dB.x = dim_block[0]; dB.y = dim_block[1]; dB.z = dim_block[2];
            dG.x = dim_grid[0]; dG.y = dim_grid[1];

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
        else // CPU
        {
            t.push();
            snprintf(buffer, sizeof(buffer), "Start CPU routine. / %.6f s ( %.6f s )", t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);

            // CPU proc
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

    }
    looptime.stop();
    snprintf(buffer, sizeof(buffer), "Loop time.: %.6f s", looptime.total);
    logging(std::string(buffer), &logcount, true);
    /*--->>> loop end <<<---*/

	// free host and device memories
    if ( mode_cuda && !mode_cuda_um ) // GPU
    {
        if ( mode_pagelock )
            cudaHostUnregister(frame_host);
		delete [] frame_host;
		t.stop();
		snprintf(buffer, sizeof(buffer), "Host input memory has been free. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);

        if ( mode_pagelock )
            cudaHostUnregister(pcd_host);
		delete [] pcd_host;
		t.stop();
		snprintf(buffer, sizeof(buffer), "Host output memory has been free. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);

        cudaFree(frame_device);
        t.push();
        snprintf(buffer, sizeof(buffer), "Device input memory has been free. / %.6f s ( %.6f s )", t.local, t.total);
        logging(std::string(buffer), &logcount, mode_debug);

        cudaFree(pcd_device);
        t.push();
        snprintf(buffer, sizeof(buffer), "Device output memory has been free. / %.6f s ( %.6f s )", t.local, t.total);
        logging(std::string(buffer), &logcount, mode_debug);

        cudaDeviceReset();
    }
	else if ( mode_cuda && mode_cuda_um ) // GPU ( unified memory )
	{
		cudaFree(frame_host);
		t.stop();
		snprintf(buffer, sizeof(buffer), "Global input memory has been free. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);	

		cudaFree(pcd_host);
		t.stop();
		snprintf(buffer, sizeof(buffer), "Global output memory has been free. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);	

        cudaDeviceReset();
	}
	else // CPU
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

