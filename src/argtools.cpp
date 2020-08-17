#include <argtools.hpp>

#include <stdio.h>
#include <iostream>
#include <string>
#include <climits>

namespace argtools {

int _roundup(const int value, const int radix)
{
    return (int)( (value+radix-1)/radix );
}

void init_params ( bool *mode_debug, bool *mode_cuda, bool *mode_pagelock, bool *mode_cuda_um,
                   int *numloops, int *width, int *height,
                   int (*dim_block)[3], int (*dim_grid)[2] )
{
    *mode_debug = false;
    *mode_cuda = false;
    *mode_pagelock = false;
    *mode_cuda_um = false;

    *numloops = 1;
    *width = 1;
    *height = 1;

    (*dim_block)[0] = 0; // for differentiating whether specificated ( > 0 ) or not
    (*dim_block)[1] = 0;
    (*dim_block)[2] = 0;
    (*dim_grid)[0] = 0; // for differentiating whether specificated ( > 0 ) or not
    (*dim_grid)[1] = 0;

}

int adjust_cudaparams ( int (*dim_block)[3], int (*dim_grid)[2], bool *mode_pagelock, bool *mode_cuda_um,
                        const bool mode_cuda, const int width, const int height )
{
    if ( !mode_cuda ){
        *mode_pagelock = false;
        *mode_cuda_um = false;
        return 0;
    }

    // Page-lock or Unified memory mode
    if ( *mode_pagelock && *mode_cuda_um )
    {
        std::cout << "WARNING: Page-locking and unified memory was specified. Page-locking will be ignored." << std::endl;
        *mode_pagelock = false;
    }

    bool bdbx = ( (*dim_block)[0] == 0 ), bdby = ( (*dim_block)[1] == 0 ), bdbz = ( (*dim_block)[2] == 0 );
    bool bdgx = ( (*dim_grid)[0] == 0 ), bdgy = ( (*dim_grid)[1] == 0 );

    int cudaDim = 0;
    if ( !bdbx || !bdgx ) cudaDim = 1;
    if ( !bdby || !bdgy ) cudaDim = 2;
    if ( !bdbz ) cudaDim = 3;

    if ( *mode_cuda_um && cudaDim > 1 )
    {
        std::cout << "WARNING: CUDA multi dimension is not supported on an unified memory system. 1D dimension will be used." << std::endl;

        if ( cudaDim == 2 )
        {
            if ( !bdbx && !bdby )
            {
                (*dim_block)[0] = (*dim_block)[0]*(*dim_block)[1];
                (*dim_block)[1] = 1;
            }
            if ( !bdgx && !bdgy )
            {
                (*dim_grid)[0] = (*dim_grid)[0]*(*dim_grid)[1];
                (*dim_grid)[1] = 1;
            }
        }
        else if ( cudaDim == 3 )
        {
            if ( !bdbx && !bdby && !bdbz )
            {
                (*dim_block)[0] = (*dim_block)[0]*(*dim_block)[1]*(*dim_block)[2];
                (*dim_block)[1] = (*dim_block)[2] = 1;
            }
            if ( !bdgx && !bdgy )
            {
                (*dim_grid)[0] = (*dim_grid)[0]*(*dim_grid)[1];
                (*dim_grid)[1] = 1;
            }
        }

        cudaDim = 1;
    }

    // not supported now
    if ( cudaDim == 3 )
    {
        std::cout << "WARNING: CUDA 3D dimension is not supported. 2D dimension will be used." << std::endl;
        cudaDim = 2;
    }

    int size = width*height;
    int dbx = (*dim_block)[0], dby = (*dim_block)[1], dbz = (*dim_block)[2];
    int dgx = (*dim_grid)[0], dgy = (*dim_grid)[1];

    if ( cudaDim == 0 ) // bdbx and bdgx
    {
        (*dim_block)[0] = 128;
        (*dim_grid)[0] = _roundup(size, 128);
        cudaDim = 1;

        (*dim_block)[1] = (*dim_block)[2] = 1;
        (*dim_grid)[1] = 1;

    }
    else if ( cudaDim == 1 ) // !bdbx or !bdgx
    {
        if ( !bdbx && bdgx ) (*dim_grid)[0] = _roundup(size, dbx);
        else if ( bdbx && !bdgx ) (*dim_block)[0] = _roundup(size, dgx);

        (*dim_block)[1] = (*dim_block)[2] = 1;
        (*dim_grid)[1] = 1;

    }
    else if ( cudaDim == 2 ) // !bdby or !bdgy
    {
        if ( !bdbx && !bdby && bdgx && bdgy )
        {
            (*dim_grid)[0] = _roundup(width, dbx);
            (*dim_grid)[1] = _roundup(height, dby);
        }
        else if ( !bdbx && !bdby && !bdgx && bdgy )
        {
            (*dim_grid)[1] = _roundup(height, dby);
        }
        else if ( bdbx && bdby && !bdgx && !bdgy )
        {
            (*dim_block)[0] = _roundup(width, dgx);
            (*dim_block)[1] = _roundup(height, dgy);
        }
        else if ( !bdbx && bdby && !bdgx && !bdgy )
        {
            (*dim_block)[1] = _roundup(height, dgy);
        }

        (*dim_block)[2] = 1;

    }
    else if ( cudaDim == 3 ) // !bdbz
    {
        if ( !bdbx && !bdby && !bdbz && bdgx && bdgy )
        {
            (*dim_grid)[0] = _roundup(width, dbx);
            (*dim_grid)[1] = _roundup(height, dby);
        }
        else if ( !bdbx && !bdby && !bdbz && !bdgx && bdgy )
        {
            (*dim_grid)[1] = _roundup(height, dby);
        }
        else if ( bdbx && bdby && bdbz && !bdgx && !bdgy )
        {
            (*dim_block)[0] = _roundup(width, dgx);
            (*dim_block)[1] = _roundup(height, dgy);
        }
        else if ( !bdbx && bdby && bdbz && !bdgx && !bdgy )
        {
            (*dim_block)[1] = _roundup(height, dgy);
        }
    }

    return cudaDim;

}

void get_args ( bool *mode_debug, bool *mode_cuda, bool *mode_pagelock, bool *mode_cuda_um,
                int *numloops, int *width, int *height,
                int (*dim_block)[3], int (*dim_grid)[2],
                const int argc, const char **argv )
{
	if ( argc == 1 )
	{
		std::cout << std::endl;
		std::cout << "options:" << std::endl;
		std::cout << "    -h | --help           : Help options." << std::endl;
		std::cout << "    -v | --verbose        : Debug logs." << std::endl;
		std::cout << "    -c | --cuda           : Use CUDA computing." << std::endl;
        std::cout << "    -p | --page-lock      : Use page-locked memory." << std::endl;
		std::cout << "    -u | --unified-memory : Use unified memory system." << std::endl;
		std::cout << "    -r | --resolution     : (QQVGA/QVGA/VGA/XGA/HD/FHD/2K/3K/4K/5K/6K/8K/10K/16K)" << std::endl;
        std::cout << "    -N | --num-loops      : (int value) Number of image processing loops." << std::endl;
		std::cout << "    -W | --width          : (int value) Image width." << std::endl;
		std::cout << "    -H | --height         : (int value) Image height." << std::endl;
		std::cout << "    -B | --blocks         : (int value) Number of GPU blocks." << std::endl;
		std::cout << "    -G | --grids          : (int value) Number of GPU grids." << std::endl;
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
             argstr != "-p" && argstr != "--page-lock" &&
			 argstr != "-u" && argstr != "--unified-memory" &&
			 argstr != "-r" && argstr != "--resolution" &&
             argstr != "-N" && argstr != "--num-loops" &&
			 argstr != "-W" && argstr != "--width" &&
			 argstr != "-H" && argstr != "--height" &&
			 argstr != "-B" && argstr != "--blocks" &&
			 argstr != "-G" && argstr != "--grids"
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
			std::cout << "    -c | --cuda           : Use CUDA computing." << std::endl;
            std::cout << "    -p | --page-lock      : Use page-locked memory." << std::endl;
			std::cout << "    -u | --unified-memory : Use unified memory system." << std::endl;
			std::cout << "    -r | --resolution     : (QQVGA/QVGA/VGA/XGA/HD/FHD/2K/3K/4K/5K/6K/8K/10K/16K)" << std::endl;
			std::cout << "    -N | --num-loops      : (int value) Number of image processing loops." << std::endl;
			std::cout << "    -W | --width          : (int value) Image width." << std::endl;
			std::cout << "    -H | --height         : (int value) Image height." << std::endl;
			std::cout << "    -B | --blocks         : (int value) Number of GPU blocks." << std::endl;
            std::cout << "    -G | --grids          : (int value) Number of GPU grids." << std::endl;
			std::cout << std::endl;
			exit(0);
		}
		if ( argstr == "-v" || argstr == "--verbose" )
			*mode_debug = true;
		if ( argstr == "-c" || argstr == "--cuda" )
		{
			*mode_cuda = true;
		}
		if ( argstr == "-p" || argstr == "--page-lock" )
		{
            *mode_pagelock = true;
		}
		if ( argstr == "-u" || argstr == "--unified-memory" )
		{
			*mode_cuda_um = true;
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
		if ( argstr == "-N" || argstr == "--num-loops" )
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
			*numloops = num;
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
			(*dim_block)[0] = num;
            if ( j+2 < argc )
            {
                num = strtol(argv[j+2], &endptr, 10);
                if (*endptr != '\0' || (num == INT_MAX && ERANGE == 0))
                {
                    (*dim_block)[1] = (*dim_block)[2] = 0;
                    j ++;
                    continue;
                }
                else
                {
                    (*dim_block)[1] = num;
                    if ( j+3 < argc )
                    {
                        num = strtol(argv[j+3], &endptr, 10);
                        if (*endptr != '\0' || (num == INT_MAX && ERANGE == 0))
                        {
                            (*dim_block)[2] = 0;
                            j += 2;
                            continue;
                        }
                        else
                        {
                            (*dim_block)[2] = num;
                            j += 3;
                        }
                    }
                    else
                    {
                            (*dim_block)[2] = 0;
                            j += 2;
                    }
                }
            }
            else
            {
                (*dim_block)[1] = (*dim_block)[2] = 0;
                j ++;
            }
		}
		if ( argstr == "-G" || argstr == "--grids" )
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
			(*dim_grid)[0] = num;
            if ( j+2 < argc )
            {
                num = strtol(argv[j+2], &endptr, 10);
                if (*endptr != '\0' || (num == INT_MAX && ERANGE == 0))
                {
                    (*dim_grid)[1] = 0;
                    j ++;
                    continue;
                }
                else
                {
                    (*dim_grid)[1] = num;
                    j += 2;
                }
            }
            else
            {
                (*dim_grid)[1] = 0;
                j ++;
            }
		}
	}
}

void disp_args ( const bool mode_debug, const bool mode_cuda, const bool mode_pagelock, const bool mode_cuda_um,
                 const int numloops, const int width, const int height,
                 const int cudaDim, const int dim_block[3], const int dim_grid[2] )
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
        else if ( mode_pagelock ) std::cout << " ( page-locked )";
		std::cout << std::endl;

		std::cout << " GPU blocks : " << "x/" << dim_block[0];
        if ( cudaDim > 1 ) std::cout << ", y/" << dim_block[1];
        if ( cudaDim > 2 ) std::cout << ", z/" << dim_block[2];
        std::cout << std::endl;
		std::cout << " GPU grids  : " << "x/" << dim_grid[0];
        if ( cudaDim > 1 ) std::cout << ", y/" << dim_grid[1];
	}
	else std::cout << "off";
	std::cout << std::endl;
    std::cout << " numloops   : " << numloops << std::endl;
	std::cout << " width      : " << width << std::endl;
	std::cout << " height     : " << height << std::endl;
	std::cout << "===============================" << std::endl;
	std::cout << std::endl;
}

}