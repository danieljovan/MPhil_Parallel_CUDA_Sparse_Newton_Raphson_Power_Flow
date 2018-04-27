#include <iostream>
#include "gpuFunctions.h"

using namespace std;


int main () {
	//command line program. May want to look into using Qt or C# WPF for GUI-based program
	//User selects from the menu which NPF option they would like
	int option; 
	cout<<"-------M.Phil Project - Parallel Newton Power Flow Solver-------"<<endl;
	cout<<"Please select one of the following options for simulation:"<<endl;
	//cout<<"1. IEEE 14 bus system\n2. IEEE 118 bus system\n3. IEEE 300 bus system\n4. Polish Winter Peak 2383 Bus System\n5. Polish Summer Peak 3120 bus system\n6. PEGASE 9241 bus system\n7. PEGASE 13659 bus system"<<endl;
	cout<<"1. IEEE 14 bus system\n2. IEEE 118 bus system\n3. IEEE 300 bus system\n4. Polish Winter Peak 2383 Bus System\n5. Polish Summer Peak 3120 bus system\n6. PEGASE 6515 bus system\n7. PEGASE 9241 bus system\n8. 6 bus system"<<endl;
	cout<<"Your option: \n";
	cin>>option;
	optionSelect(option);
	cudaDeviceReset();
	return 0;

}

