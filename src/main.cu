
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
__host__ void create_imageframe ( double *(*frame), const int width, const int height )
{
	// generate frame
	for ( int idy=0; idy<height; idy++ ) for ( int idx=0; idx<width; idx++ )
	{
		int pos = idy*width + idx;
		(*frame)[pos] = (double)( pos );
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
__global__ void cvtImg_cuda_2d ( double *pcd, const double *image, const int width, const int height, const int pitch )
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int idy = blockDim.y*blockIdx.y + threadIdx.y;
	if ( idx < width && idy < height )
	{
        int pos = idy*pitch + idx;
        // debug
        // printf("idx=%d, idy=%d, pitch=%d, pos=%d, image=%.3f\n", idx, idy, pitch, pos, image[pos]);
        pcd[pos] = image[pos] + 1.0;
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
__host__ void disp_image ( const double *image, const int width, const int height, const bool mode_debug=true )
{
	if ( !mode_debug ) return;
	for ( int idy=0; idy<height; idy++ )
	{
		for ( int idx=0; idx<width; idx++ )
		{
			int pos = idy*width + idx;
			std::cout << " (" << image[pos] << ")";
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
    int numloops, width, height;
	int dim_block[3], dim_grid[2];

    // initialization
    argtools::init_params(&mode_debug, &mode_cuda, &mode_pagelock, &mode_cuda_um, &numloops, &width, &height, &dim_block, &dim_grid);

	// define frame instances
	double *frame_host, *pcd_host; // @ host
    double *frame_device, *pcd_device; // @ device ( use only non-unified memory of GPU calc )

    // GPU grid/block params
    dim3 dB, dG;

	logging("Start execution.", &logcount, mode_debug);
	t.start();

	argtools::get_args(&mode_debug, &mode_cuda, &mode_pagelock, &mode_cuda_um, &numloops, &width, &height, &dim_block, &dim_grid, argc, argv);
    int cudaDim = argtools::adjust_cudaparams(&dim_block, &dim_grid, &mode_pagelock, &mode_cuda_um, mode_cuda, width, height);
	t.push();
	snprintf(buffer, sizeof(buffer), "Got args. / %.6f s ( %.6f s )", t.local, t.total);
	logging(std::string(buffer), &logcount, mode_debug);
    argtools::disp_args(mode_debug, mode_cuda, mode_pagelock, mode_cuda_um, numloops, width, height, cudaDim, dim_block, dim_grid);

	// define frame params
	int size_host_in    = width*height;
	int size_host_out   = width*height;
	int size_device_in  = width*height;
	int size_device_out = width*height;
	size_t size_bytes_host_in    = sizeof(double)*size_host_in;
	size_t size_bytes_host_out   = sizeof(double)*size_host_out;
	size_t size_bytes_device_in  = sizeof(double)*size_device_in;
	size_t size_bytes_device_out = sizeof(double)*size_device_out;

    if ( mode_cuda )
    {
        if ( cudaDim < 1 && cudaDim > 3 )
        {
            std::cerr << "ERROR: CUDA dimension error." << std::endl;
            exit(1);
        }
        int numthread = dim_block[0]*dim_block[1]*dim_block[2]*dim_grid[0]*dim_grid[1];
        if ( numthread < size_host_in )
            std::cout << "WARNING: CUDA thread size ( " << numthread << " ) is smaller than input array size ( " << size_host_in << " ). Some array datum will be missed." << std::endl;
    }

    // for 2D
    // int pitch_host_in = width; // not used
    // int pitch_host_out = width; // not used
    int pitch_device_in;
    int pitch_device_out;
    size_t pitch_bytes_host_in = sizeof(double)*width;
    size_t pitch_bytes_host_out = sizeof(double)*width;
    size_t pitch_bytes_device_in;
    size_t pitch_bytes_device_out;

	// allocation of input memory
    if ( mode_cuda && !mode_cuda_um ) // GPU
    {
        if ( !mode_pagelock )
        {
            frame_host = new double [size_host_in]();  // and zero-initilization
            t.push();
            snprintf(buffer, sizeof(buffer), "Host input memories ( %d array, %d B ) allocated. / %.6f s ( %.6f s )", size_host_in, (int)size_bytes_host_in, t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);
        }
        else if ( mode_pagelock )
        {
            // cudaHostRegister(frame_host, size_bytes_host_in, cudaHostRegisterDefault);
            // snprintf(buffer, sizeof(buffer), "Host input memory pages locked. / %.6f s ( %.6f s )", t.local, t.total);
            cudaMallocHost(&frame_host, size_bytes_host_in);
            t.push();
            snprintf(buffer, sizeof(buffer), "Host input page-locked memories ( %d array, %d B ) allocated. / %.6f s ( %.6f s )", size_host_in, (int)size_bytes_host_in, t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);
        }
        // allocate device memory
        if ( cudaDim == 1 )
            cudaMalloc(&frame_device, size_bytes_device_in);
        else if ( cudaDim == 2 )
        {
            cudaMallocPitch(&frame_device, &pitch_bytes_device_in, width*sizeof(double), height);
            pitch_device_in = (int)( pitch_bytes_device_in/sizeof(double) );
        }
        t.push();
        // logging
        if ( cudaDim == 1 )
            snprintf(buffer, sizeof(buffer), "Device input 1D memories ( %d array, %d B ) allocated. / %.6f s ( %.6f s )", size_device_in, (int)size_bytes_device_in, t.local, t.total);
        else if ( cudaDim == 2 )
            snprintf(buffer, sizeof(buffer), "Device input 2D memories ( %d x %d array, %d B ) allocated. / %.6f s ( %.6f s )", pitch_device_in, height, (int)pitch_bytes_device_in*height, t.local, t.total);
        logging(std::string(buffer), &logcount, mode_debug);
    }
	else if ( mode_cuda && mode_cuda_um ) // GPU ( unified memory )
	{
		cudaMallocManaged(&frame_host, size_bytes_host_in, cudaMemAttachGlobal);
		t.push();
		snprintf(buffer, sizeof(buffer), "Shared input memories ( %d array, %d B ) allocated. / %.6f s ( %.6f s )", size_host_in, (int)size_bytes_host_in, t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);
	}
	else // CPU
	{
        frame_host = new double [size_host_in]();  // and zero-initilization
		t.push();
		snprintf(buffer, sizeof(buffer), "Host input memories ( %d array, %d B ) allocated. / %.6f s ( %.6f s )", size_host_in, (int)size_bytes_host_in, t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);
	}

    // allocation of output memory
    if ( mode_cuda && !mode_cuda_um ) // GPU
    {
        // define CUDA params
        dB.x = dim_block[0]; dB.y = dim_block[1]; dB.z = dim_block[2];
        dG.x = dim_grid[0]; dG.y = dim_grid[1];
        // allocate device memory
        if ( cudaDim == 1 )
            cudaMalloc(&pcd_device, size_bytes_device_out);
        else if ( cudaDim == 2 )
        {
            cudaMallocPitch(&pcd_device, &pitch_bytes_device_out, width*sizeof(double), height);
            pitch_device_out = (int)( pitch_bytes_device_out/sizeof(double) );
        }
        t.push();
        // logging
        if ( cudaDim == 1 )
            snprintf(buffer, sizeof(buffer), "Device output 1D memories ( %d array, %d B ) allocated. / %.6f s ( %.6f s )", size_device_out, (int)size_bytes_device_out, t.local, t.total);
        else if ( cudaDim == 2 )
            snprintf(buffer, sizeof(buffer), "Device output 2D memories ( %d x %d array, %d B ) allocated. / %.6f s ( %.6f s )", pitch_device_out, height, (int)pitch_bytes_device_out*height, t.local, t.total);
        logging(std::string(buffer), &logcount, mode_debug);
        // allocate host memory
        if ( !mode_pagelock )
        {
            pcd_host = new double [size_host_out]();  // and zero-initilization
            t.push();
            snprintf(buffer, sizeof(buffer), "Host output memories ( %d array, %d B ) allocated. / %.6f s ( %.6f s )", size_host_out, (int)size_bytes_host_out, t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);
        }
        else if ( mode_pagelock )
        {
            // cudaHostRegister(pcd_host, size_bytes_host_out, cudaHostRegisterDefault);
            // snprintf(buffer, sizeof(buffer), "Host output memory pages locked. / %.6f s ( %.6f s )", t.local, t.total);
            cudaMallocHost(&pcd_host, size_bytes_host_out);
            t.push();
            snprintf(buffer, sizeof(buffer), "Host output page-locked memories ( %d array, %d B ) allocated. / %.6f s ( %.6f s )", size_host_out, (int)size_bytes_host_out, t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);
        }
    }
    else if ( mode_cuda && mode_cuda_um ) // GPU ( unified memory )
    {
        // define CUDA params
        dB.x = dim_block[0]; dB.y = dim_block[1]; dB.z = dim_block[2];
        dG.x = dim_grid[0]; dG.y = dim_grid[1];
        // allocate GLOBAL memory
        cudaMallocManaged(&pcd_host, size_bytes_host_out, cudaMemAttachGlobal);
        t.push();
        snprintf(buffer, sizeof(buffer), "Shared output memories ( %d array, %d B ) allocated. / %.6f s ( %.6f s )", size_host_out, (int)size_bytes_host_out, t.local, t.total);
        logging(std::string(buffer), &logcount, mode_debug);
    }
    else // CPU
    {
        // allocate host's memory
        pcd_host = new double [size_host_out](); // and zero-initilization
        t.push();
        snprintf(buffer, sizeof(buffer), "Host output memories ( %d array, %d B ) allocated. / %.6f s ( %.6f s )", size_host_out, (int)size_bytes_host_out, t.local, t.total);
        logging(std::string(buffer), &logcount, mode_debug);
    }

    /*--->>> loop start <<<---*/
    looptime.start();
    for ( int n=0; n<numloops; n++ )
    {
        snprintf(buffer, sizeof(buffer), "Loop %d", n+1);
        logging(std::string(buffer), &logcount, mode_debug);

        // get image frame on host
        create_imageframe(&frame_host, width, height);
        t.push();
        snprintf(buffer, sizeof(buffer), "Got image. / %.6f s ( %.6f s )", t.local, t.total);
        logging(std::string(buffer), &logcount, mode_debug);

#if defined(_DISP_IMAGE_ARRAY)
            disp_image(frame_host, width, height, mode_debug);
#endif

        // calculation
        if ( mode_cuda && !mode_cuda_um ) // GPU
        {
            // copy host's image frame to device memory
            if ( cudaDim == 1 )
                cudaMemcpy(frame_device, frame_host, size_bytes_host_in, cudaMemcpyHostToDevice);
            else if ( cudaDim == 2 )
                cudaMemcpy2D(frame_device, pitch_bytes_device_in, frame_host, pitch_bytes_host_in, width*sizeof(double), height, cudaMemcpyHostToDevice);
            t.push();
            // logging
            if ( cudaDim == 1 )
                snprintf(buffer, sizeof(buffer), "Host 1D memory has copied to GPU device. / %.6f s ( %.6f s )", t.local, t.total);
            else if ( cudaDim == 2 )
                snprintf(buffer, sizeof(buffer), "Host 2D memory has copied to GPU device. / %.6f s ( %.6f s )", t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);

            t.push();
            snprintf(buffer, sizeof(buffer), "Start GPU routine. / %.6f s ( %.6f s )", t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);

            // CUDA proc
            if ( cudaDim == 1 )
                cvtImg_cuda <<< dG, dB >>> (pcd_device, frame_device, size_device_in);
            else if ( cudaDim == 2 )
                cvtImg_cuda_2d <<< dG, dB >>> (pcd_device, frame_device, width, height, pitch_device_in);

            // barrier
            cudaDeviceSynchronize();
            t.push();
            snprintf(buffer, sizeof(buffer), "Finish GPU routine. / %.6f s ( %.6f s )", t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);

            // copy device's frame to host memory
            if ( cudaDim == 1 )
                cudaMemcpy(pcd_host, pcd_device, size_bytes_device_out, cudaMemcpyDeviceToHost);
            else if ( cudaDim == 2 )
                cudaMemcpy2D(pcd_host, pitch_bytes_host_out, pcd_device, pitch_bytes_device_out, width*sizeof(double), height, cudaMemcpyDeviceToHost);
            t.push();
            // logging
            if ( cudaDim == 1 )
                snprintf(buffer, sizeof(buffer), "Device 1D memory has copied to CPU device. / %.6f s ( %.6f s )", t.local, t.total);
            else if ( cudaDim == 2 )
                snprintf(buffer, sizeof(buffer), "Device 2D memory has copied to CPU device. / %.6f s ( %.6f s )", t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);

#if defined(_DISP_IMAGE_ARRAY)
            disp_image(pcd_host, width, height, mode_debug);
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
            cvtImg_cuda <<< dG, dB >>> (pcd_host, frame_host, size_host_in);

            // barrier
            cudaDeviceSynchronize();
            t.push();
            snprintf(buffer, sizeof(buffer), "Finish GPU routine. / %.6f s ( %.6f s )", t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);

            // no copy device's frame to host memory
            // no free device memory
#if defined(_DISP_IMAGE_ARRAY)
            disp_image(pcd_host, width, height, mode_debug);
#endif

        }
        else // CPU
        {
            t.push();
            snprintf(buffer, sizeof(buffer), "Start CPU routine. / %.6f s ( %.6f s )", t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);

            // CPU proc
            cvtImg(pcd_host, frame_host, size_host_in);

            t.push();
            snprintf(buffer, sizeof(buffer), "Finish CPU routine. / %.6f s ( %.6f s )", t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);

#if defined(_DISP_IMAGE_ARRAY)
            disp_image(pcd_host, width, height, mode_debug);
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
        if ( !mode_pagelock )
        {
            delete [] frame_host;
            t.push();
            snprintf(buffer, sizeof(buffer), "Host input memories has been free. / %.6f s ( %.6f s )", t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);

            delete [] pcd_host;
            t.push();
            snprintf(buffer, sizeof(buffer), "Host output memories has been free. / %.6f s ( %.6f s )", t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);

        }
        else if ( mode_pagelock )
        {
            // cudaHostUnregister(frame_host);
            // snprintf(buffer, sizeof(buffer), "Host input memories pages unlocked. / %.6f s ( %.6f s )", t.local, t.total);
            cudaFreeHost(frame_host);
            t.push();
            snprintf(buffer, sizeof(buffer), "Host input page-locked memories has been free. / %.6f s ( %.6f s )", t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);

            // cudaHostUnregister(pcd_host);
            // snprintf(buffer, sizeof(buffer), "Host output memories pages unlocked. / %.6f s ( %.6f s )", t.local, t.total);
            cudaFreeHost(pcd_host);
            t.push();
            snprintf(buffer, sizeof(buffer), "Host output page-locked memories has been free. / %.6f s ( %.6f s )", t.local, t.total);
            logging(std::string(buffer), &logcount, mode_debug);


        }

        cudaFree(frame_device);
        t.push();
        snprintf(buffer, sizeof(buffer), "Device input memories has been free. / %.6f s ( %.6f s )", t.local, t.total);
        logging(std::string(buffer), &logcount, mode_debug);

        cudaFree(pcd_device);
        t.stop();
        snprintf(buffer, sizeof(buffer), "Device output memories has been free. / %.6f s ( %.6f s )", t.local, t.total);
        logging(std::string(buffer), &logcount, mode_debug);

        cudaDeviceReset();
    }
	else if ( mode_cuda && mode_cuda_um ) // GPU ( unified memories )
	{
		cudaFree(frame_host);
		t.push();
		snprintf(buffer, sizeof(buffer), "Shared input memories has been free. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);	

		cudaFree(pcd_host);
		t.stop();
		snprintf(buffer, sizeof(buffer), "Shared output memories has been free. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);	

        cudaDeviceReset();
	}
	else // CPU
	{
		delete [] frame_host;
		t.push();
		snprintf(buffer, sizeof(buffer), "Host input memories has been free. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);

		delete [] pcd_host;
		t.stop();
		snprintf(buffer, sizeof(buffer), "Host output memories has been free. / %.6f s ( %.6f s )", t.local, t.total);
		logging(std::string(buffer), &logcount, mode_debug);	
	}

	return 0;
}

