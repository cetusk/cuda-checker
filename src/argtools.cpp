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
                   int *numloops, int *width, int *height, int *channels,
                   int (*dim_block)[3], int (*dim_grid)[2] )
{
    *mode_debug = false;
    *mode_cuda = false;
    *mode_pagelock = false;
    *mode_cuda_um = false;

    *numloops = 1;
    *width = 1;
    *height = 1;
    *channels = 1;

    (*dim_block)[0] = 0; // for differentiating whether specificated ( > 0 ) or not
    (*dim_block)[1] = 1;
    (*dim_block)[2] = 1;
    (*dim_grid)[0] = 0; // for differentiating whether specificated ( > 0 ) or not
    (*dim_grid)[1] = 1;

}

void adjust_cudaparams ( int (*dim_block)[3], int (*dim_grid)[2], bool *mode_pagelock, bool *mode_cuda_um,
                         const bool mode_cuda, const int width, const int height, const int channels )
{
    if ( !mode_cuda ){
        *mode_pagelock = false;
        *mode_cuda_um = false;
        return;
    }

    // Page-lock or Unified memory mode
    if ( *mode_cuda_um ) *mode_pagelock = false;

    // block/grid params
    if ( (*dim_block)[0] > 0 && (*dim_grid)[0] > 0 ) return; // specificated

    bool bdb = ( (*dim_block)[0] == 0 );
    bool bdg = ( (*dim_grid)[0] == 0 );
    int db = (*dim_block)[0];
    int dg = (*dim_grid)[0];

    int size = width*height*channels;
    if ( size == 1e0 )
    {
        if ( bdb && bdg )
        {
            (*dim_block)[0] = 1;
            (*dim_grid)[0] = 1;
        }
        else if ( bdb && !bdg ) (*dim_block)[0] = 1;
        else if ( !bdb && bdg ) (*dim_grid)[0] = 1;
    }
    else if ( size < 1e1 )
    {
        if ( bdb && bdg )
        {
            (*dim_block)[0] = 4;
            (*dim_grid)[0] = 4;
        }
        else if ( bdb && !bdg ) (*dim_block)[0] = _roundup(size, dg);
        else if ( !bdb && bdg ) (*dim_grid)[0] = _roundup(size, db);
    }
    else if ( size < 1e2 )
    {
        if ( bdb && bdg )
        {
            (*dim_block)[0] = 8;
            (*dim_grid)[0] = _roundup(size, 8);
        }
        else if ( bdb && !bdg ) (*dim_block)[0] = _roundup(size, dg);
        else if ( !bdb && bdg ) (*dim_grid)[0] = _roundup(size, db);
    }
    else if ( size < 1e3 )
    {
        if ( bdb && bdg )
        {
            (*dim_block)[0] = 16;
            (*dim_grid)[0] = _roundup(size, 16);
        }
        else if ( bdb && !bdg ) (*dim_block)[0] = _roundup(size, dg);
        else if ( !bdb && bdg ) (*dim_grid)[0] = _roundup(size, db);
    }
    else if ( size < 1e4 )
    {
        if ( bdb && bdg )
        {
            (*dim_block)[0] = 32;
            (*dim_grid)[0] = _roundup(size, 32);
        }
        else if ( bdb && !bdg ) (*dim_block)[0] = _roundup(size, dg);
        else if ( !bdb && bdg ) (*dim_grid)[0] = _roundup(size, db);
    }
    else if ( size < 1e5 )
    {
        if ( bdb && bdg )
        {
            (*dim_block)[0] = 64;
            (*dim_grid)[0] = _roundup(size, 64);
        }
        else if ( bdb && !bdg ) (*dim_block)[0] = _roundup(size, dg);
        else if ( !bdb && bdg ) (*dim_grid)[0] = _roundup(size, db);
    }
    else
    {
        if ( bdb && bdg )
        {
            (*dim_block)[0] = 128;
            (*dim_grid)[0] = _roundup(size, 128);
        }
        else if ( bdb && !bdg ) (*dim_block)[0] = _roundup(size, dg);
        else if ( !bdb && bdg ) (*dim_grid)[0] = _roundup(size, db);
    }
}

void get_args ( bool *mode_debug, bool *mode_cuda, bool *mode_pagelock, bool *mode_cuda_um,
                int *numloops, int *width, int *height, int *channels,
                int (*dim_block)[3], int (*dim_grid)[2],
                const int argc, const char **argv )
{
	if ( argc == 1 )
	{
		std::cout << std::endl;
		std::cout << "options:" << std::endl;
		std::cout << "    -h | --help           : Help options." << std::endl;
		std::cout << "    -v | --verbose        : Debug logs." << std::endl;
		std::cout << "    -c | --cuda           : (on/off) Use CUDA computing." << std::endl;
        std::cout << "    -p | --page-lock      : (on/off) Use page-locked memory." << std::endl;
		std::cout << "    -u | --unified-memory : (on/off) Use unified memory system." << std::endl;
		std::cout << "    -r | --resolution     : (QQVGA/QVGA/VGA/XGA/HD/FHD/2K/3K/4K/5K/6K/8K/10K/16K)" << std::endl;
        std::cout << "    -N | --num-loops      : (int value) Number of image processing loops." << std::endl;
		std::cout << "    -W | --width          : (int value) Image width." << std::endl;
		std::cout << "    -H | --height         : (int value) Image height." << std::endl;
		std::cout << "    -C | --channels       : (int value) Number of image color channels." << std::endl;
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
			 argstr != "-C" && argstr != "--channels" &&
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
			std::cout << "    -c | --cuda           : (on/off) Use CUDA computing." << std::endl;
            std::cout << "    -p | --page-lock      : (on/off) Use page-locked memory." << std::endl;
			std::cout << "    -u | --unified-memory : (on/off) Use unified memory system." << std::endl;
			std::cout << "    -r | --resolution     : (QQVGA/QVGA/VGA/XGA/HD/FHD/2K/3K/4K/5K/6K/8K/10K/16K)" << std::endl;
			std::cout << "    -N | --num-loops      : (int value) Number of image processing loops." << std::endl;
			std::cout << "    -W | --width          : (int value) Image width." << std::endl;
			std::cout << "    -H | --height         : (int value) Image height." << std::endl;
			std::cout << "    -C | --channels       : (int value) Number of image color channels." << std::endl;
			std::cout << "    -B | --blocks         : (int value) Number of GPU blocks." << std::endl;
            std::cout << "    -G | --grids          : (int value) Number of GPU grids." << std::endl;
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
		if ( argstr == "-p" || argstr == "--page-lock" )
		{
			if ( j+1 == argc )
			{
				std::cerr << "ERROR: No command found." << std::endl;
				exit(1);
			}
			std::string argval = std::string(argv[j+1]);
			if ( argval == "on" ) *mode_pagelock = true;
			else if ( argval == "off" ) *mode_pagelock = false;
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
			(*dim_block)[0] = num;
            if ( j+2 < argc )
            {
                num = strtol(argv[j+2], &endptr, 10);
                if (*endptr != '\0' || (num == INT_MAX && ERANGE == 0))
                {
                    (*dim_block)[1] = (*dim_block)[2] = 1;
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
                            (*dim_block)[2] = 1;
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
                            (*dim_block)[2] = 1;
                            j += 2;
                    }
                }
            }
            else
            {
                (*dim_block)[1] = (*dim_block)[2] = 1;
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
                    (*dim_grid)[1] = 1;
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
                (*dim_grid)[1] = 1;
                j ++;
            }
		}
	}
}

void disp_args ( const bool mode_debug, const bool mode_cuda, const bool mode_pagelock, const bool mode_cuda_um,
                 const int numloops, const int width, const int height, const int channels,
                 const int dim_block[3], const int dim_grid[2] )
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
		if ( mode_cuda_um && !mode_pagelock ) std::cout << " ( unified memory )";
        else if ( !mode_cuda_um && mode_pagelock ) std::cout << " ( page-locked )";
		std::cout << std::endl;

		std::cout << " GPU blocks : " << "x/" << dim_block[0] << ", y/" << dim_block[1] << ", z/" << dim_block[2] << std::endl;
		std::cout << " GPU grids  : " << "x/" << dim_grid[0] << ", y/" << dim_grid[1];
	}
	else std::cout << "off";
	std::cout << std::endl;
    std::cout << " numloops   : " << numloops << std::endl;
	std::cout << " width      : " << width << std::endl;
	std::cout << " height     : " << height << std::endl;
	std::cout << " channels   : " << channels << std::endl;
	std::cout << "===============================" << std::endl;
	std::cout << std::endl;
}

}