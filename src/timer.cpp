
#include <timer.hpp>

timer::timer ()
{
	is_run = false;
	offset = clock();
}
timer::~timer ()
{
	;
}

void timer::start ( void )
{
	offset = clock();
	is_run = true;
	total = 0.0;
}

void timer::push ( void )
{
	clock_t buffer = clock(); 
	local = (double)(buffer - offset) / CLOCKS_PER_SEC;
	total += local;
	offset = buffer;
}

void timer::stop ( void )
{
	clock_t buffer = clock(); 
	local = (double)(buffer - offset) / CLOCKS_PER_SEC;
	total += local;
	offset = buffer;
	is_run = false;
}
