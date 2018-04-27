#include<windows.h>
#include <iostream>

double PCFreq = 0.0;
__int64 CounterStart = 0;

void startCounter()
{
	LARGE_INTEGER li;
	if(!QueryPerformanceFrequency(&li))
		std::cout<<"Query performance failed"<<std::endl;
	PCFreq = double(li.QuadPart)/1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}

double getCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart-CounterStart)/PCFreq;
}