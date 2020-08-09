
#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <ctime>

class timer {

	private:
		clock_t offset;
		bool is_run;

	public:

		double local;
		double total;
		
		timer ();
		~timer ();

		void start ( void );
		void push ( void );
		void stop ( void );

};

#endif
