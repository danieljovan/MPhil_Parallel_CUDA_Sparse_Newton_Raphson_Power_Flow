//#include <iostream>
#include <string>
#include <vector>

#include "gpuFunctions.h"
#include "TimingFunctions.h"
using namespace std;

#define threads 16
#define Tol ((double) 0.0001)
#define max_it 2500
__device__ int counter; 

typedef thrust::tuple<int, int, float> tuple; 
typedef thrust::tuple<int, int, double> jacTuple;

struct isZero
{
	__host__ __device__ bool operator() (const tuple& a)
	{
		const double x = thrust::get<2>(a);
		return x == 0;
	};
};

struct is_PVbusRow {
	int id;
	is_PVbusRow(int num) : id(num) {};
	__host__ __device__ 
		bool operator () (const jacTuple& tup) {
			const int row = thrust::get<0> (tup);
			//const int col = thrust::get<1> (tup);
			return ((row == id));
	}
};

struct is_PVbusCol {
	int id;
	is_PVbusCol(int num) : id(num) {};
	__host__ __device__ 
		bool operator () (const jacTuple& tup) {
			const int col = thrust::get<1> (tup);
			return ((col == id));
	}
};

//Main Menu function - allows user to select power system for simulation
void optionSelect(int option) {
	ifstream busdata, linedata;
	int numLines=0;
	string line;
	if ((option<1) || (option>8)) {
		cout<<"Not a valid option"<<endl;
		answerSelect();
	}
	else if (option==1) {
		cout<<"----IEEE 14 bus system----\n"<<endl;
		linedata.open("linedata.txt");
		while (getline (linedata, line)) {
			++numLines;
		}
		linedata.close(); //pointer is at EOF. Need to close and reopen to stream data into variables
		busdata.open("busdata.txt");
		linedata.open("linedata.txt");
		IEEEStandardBusSystems(14, busdata, linedata, numLines); //calls function to execute data intialization for 14 bus system - N.B. read from text file (for now)
		answerSelect();
	}
	else if (option==2){
		cout<<"----IEEE 118 bus system----\n"<<endl;
		linedata.open("118Bus_LineData.txt");
		while (getline (linedata, line)) {
			++numLines;
		}
		linedata.close();
		busdata.open("118BusData.txt");
		linedata.open("118Bus_LineData.txt");
		IEEEStandardBusSystems(118, busdata, linedata, numLines);
		answerSelect();
	}
		
	else if (option==3) {
		cout<<"----IEEE 300 bus system----"<<endl;
		linedata.open("300Bus_LineData.txt");
		while (getline(linedata, line)) {
			++numLines;
		}
		linedata.close();
		busdata.open("300BusData.txt");
		linedata.open("300Bus_LineData.txt");
		IEEEStandardBusSystems(300, busdata, linedata, numLines);
		answerSelect();
	}
	else if (option==4) {
		cout<<"----Polish Winter Peak 2383 bus system----"<<endl;
		linedata.open("2383LineData.txt");
		while (getline(linedata, line)) {
			++numLines;
		}
		linedata.close();
		busdata.open("2383BusData.txt");
		linedata.open("2383LineData.txt");
		IEEEStandardBusSystems(2383, busdata, linedata, numLines);
		answerSelect();
	}
	else if (option==5) {
		cout<<"----Polish Summer Peak 3120 bus system----"<<endl;
		linedata.open("3120LineData.txt");
		while (getline(linedata, line)) {
			++numLines;
		}
		linedata.close();
		busdata.open("3120BusData.txt");
		linedata.open("3120LineData.txt");
		IEEEStandardBusSystems(3120, busdata, linedata, numLines);
		answerSelect();
	}
	else if (option==6) {
		cout<<"----PEGASE 6515 bus system----"<<endl;
		linedata.open("6515LineData.txt");
		while (getline(linedata, line)) {
			++numLines;
		}
		linedata.close();
		busdata.open("6515BusData.txt");
		linedata.open("6515LineData.txt");
		IEEEStandardBusSystems(6515, busdata, linedata, numLines);
		answerSelect();
	}
	else if (option==7) {
		cout<<"----PEGASE 9241 bus system----"<<endl;
		linedata.open("9241linedata.txt");
		while (getline(linedata, line)) {
			++numLines;
		}
		linedata.close();
		busdata.open("9241busdata.txt");
		linedata.open("9241linedata.txt");
		IEEEStandardBusSystems(9241, busdata, linedata, numLines);
		answerSelect();
	}
	/*else if (option==7) {
		cout<<"----PEGASE 13659 bus system----"<<endl;
		linedata.open("13659linedata.txt");
		while (getline(linedata, line)) {
			++numLines;
		}
		linedata.close();
		busdata.open("13659busdata.txt");
		linedata.open("13659linedata.txt");
		IEEEStandardBusSystems(13659, busdata, linedata, numLines);
		answerSelect();
	}*/
	else if (option==8) {
		cout<<"----Case 6 bus system----"<<endl;
		linedata.open("6BusLineData.txt");
		while (getline(linedata, line)) {
			++numLines;
		}
		linedata.close();
		busdata.open("6BusData.txt");
		linedata.open("6BusLineData.txt");
		IEEEStandardBusSystems(6, busdata, linedata, numLines);
		answerSelect();
	}
}

//Allows user to enter a valid option from the Main Menu
void answerSelect() {
	char answer;
	int option;
	cout<<"\nDo you want to perform another simulation (y/n)?"<<endl;
	cin>>answer;
	if (cin.fail()) {
		cin.clear();
		cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		cout<<"Invalid option"<<endl;
	}
	if ((answer=='y') || (answer=='Y')) {
		system("CLS"); //clears text on command line interface 
		cout<<"\nPlease select one of the following options for simulation:"<<endl;
		//cout<<"1. IEEE 14 bus system\n2. IEEE 118 bus system\n3. IEEE 300 bus system\n4. Polish Winter Peak 2383 Bus System\n5. Polish Summer Peak 3120 bus system\n6. PEGASE 9241 bus system\n7. PEGASE 13659 bus system"<<endl;
		cout<<"1. IEEE 14 bus system\n2. IEEE 118 bus system\n3. IEEE 300 bus system\n4. Polish Winter Peak 2383 Bus System\n5. Polish Summer Peak 3120 bus system\n6. PEGASE 6515 bus system\n7. PEGASE 9241 bus system\n8. 6 bus system"<<endl;
		cout<<"Your option: ";
		cin>>option;
		optionSelect(option);
	}
	else if((answer=='n') || (answer=='N')) {
		cout<<"Thank you for using this program"<<endl;
		exit(0); //exits program 
	}
	else {
		cout<<"Invalid response"<<endl;
		answerSelect();
	}
}

//This is the "main" function in gpuFunctions.cu where the NR load flow solution of standard power systems occurs
int IEEEStandardBusSystems(int numberOfBuses, ifstream &busData, ifstream &lineData, int numLines) 
{
	//cudaProfilerStart();
//-------------------------------------------------------------VARIABLE DECLARATION SECTION---------------------------------------------------------------------------
	//bus data ifstream variables
	int bus_i, bustype, busDataIdx=0, lineDataIdx=0, N_g=0, N_p=0, jacSize=0, slackBus, numSlackLines = 0;
	double P,Q, Vmag, Vang, VMax, VMin;

	//line data ifstream variables
	int fromBus, toBus;
	double r, x, b;

	//dynamic arrays to hold bus data
	int *busNum = new int[numberOfBuses], *busType = new int[numberOfBuses], *tempBusNum = new int[numberOfBuses];
	double *Pd = new double[numberOfBuses], *Qd = new double[numberOfBuses], *Vm=new double[numberOfBuses],
		*Va=new double[numberOfBuses], *Vmax=new double[numberOfBuses], *Vmin=new double[numberOfBuses],
		*P_eq = new double[numberOfBuses], *Q_eq = new double[numberOfBuses], *theta = new double[numberOfBuses];
	
	//dynamic arrays to hold line data
	int *fromBusArr = new int[numLines], *toBusArr = new int[numLines]; 
	int *PQindex;
	double *R = new double[numLines], *X = new double[numLines], *Bact = new double[numLines];
	complex<double> *B1 = new complex<double> [numLines];
	cuDoubleComplex *B = new cuDoubleComplex[numLines]; //N.B.: cuDoubleComplex data type can be used on host once CUDA library is included.
	cuDoubleComplex *Z = new cuDoubleComplex[numLines];

	//Vectors needed for push_back() operations to build PQindex[] and PQspec[]
	vector<double> Pval_spec, Qval_spec, Pval_calc, Qval_calc;
	vector<int> Pindex, Qindex;
	vector<double> V_mag, V_ang; //for constructing stateVector[] from hot start
	double *PQspec, *PQcalc;

	//Device variables - to be allocated on and copied to the GPU
	int *dev_fromBus, *dev_toBus, *dev_PQindex, *dev_PQbuses, *dev_PVbuses;
	double *dev_Pd, *dev_Qd, *dev_Peq, *dev_Qeq, *dev_Vmag, *dev_theta, *dev_powerMismatch, *dev_stateVector, *dev_PQspec, *dev_PQcalc, *dev_xi;
	cuDoubleComplex *dev_z, *dev_B;
	
	//-----------------------sparse---------------------------
	int ynnz = (2*numLines)+numberOfBuses;	//createYBus()
	//int ynnz = numLines + numberOfBuses;		//createYBusConcise()
	cuDoubleComplex *yHost = new cuDoubleComplex[ynnz];
	int *yRowHost = new int[ynnz], *yColHost = new int[ynnz];

	cuDoubleComplex *yDev;
	int *yRowDev, *yColDev;

	//arrays holding indices of off-diagonal elements in either upper or lower triangle
	int* dev_yUpperTriangleRow, *dev_yUpperTriangleCol;
	int *yUpperTriangleRow = new int[numLines], *yUpperTriangleCol = new int[numLines];
	cudaMalloc((void**)&dev_yUpperTriangleRow, numLines*sizeof(int));
	cudaMalloc((void**)&dev_yUpperTriangleCol, numLines*sizeof(int));

	//--------------------------------------------------------


	//In the linear system Ax=b, the vectors are of the same degree as the Jacobian matrix
	double *powerMismatch;	//b in Ax = b
	double *stateVector;		//x in Ax = b 

	//----------------------------------------------Prompt user for flat start or hot start-----------------------------------------
	/*int startType;
	cout<<"What type of simulation would you like to perform:\n1)Flat start - V = 1+j0\n2)\"Hot\" start - V magnitude and angle come from previous solution"<<endl;
	cin>>startType;*/

	//Error variables for mem copies and mem allocation on GPU
	cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4, cudaStat5, cudaStat6, cudaStat7, cudaStat8, cudaStat9, cudaStat10, 
		cudaStat11, cudaStat12, cudaStat13, cudaStat14, cudaStat15, cudaStat16, cudaStat17, cudaStat18;
	
	//Variables for timing on GPU
	cudaEvent_t start, stop;
	float elapsedTime;

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

	//Reading from busdata.txt (tab delimited)
	if (!busData) {
		cout<<"There was a problem reading the 'Bus Data' file."<<endl;
		return 1;
	}

	while (busData>>bus_i>>bustype>>P>>Q>>Vmag>>Vang>>VMax>>VMin) {
		busNum[busDataIdx]=bus_i;
		busType[busDataIdx]=bustype;
		Pd[busDataIdx]=P/100;
		Qd[busDataIdx]=Q/100;
		Vm[busDataIdx]=Vmag;
		Va[busDataIdx]=0.01745329252*Vang;
		Vmax[busDataIdx]=VMax;
		Vmin[busDataIdx]=VMin;
		busDataIdx++;
	}

	//For flat start, not using previous solution. V = 1+j0. Vmag = 1pu for PQ and slack buses. Vmag is known at PV buses.
	/*if (startType == 1) {
		for (int i=0; i<numberOfBuses; i++) {
			if (busType[i]!=2)
				Vm[i] = 1;
			Va[i] = 0;
		}
	}*/
	vector<int> PVbusesVec;
	//Constructing PQindex vector which holds indices of PV and PQ buses 
	for (int i=0; i<numberOfBuses; i++) {
		if (busType[i] == 1) {
			N_p++; //increment PQ bus counter since PQ is represented by 1
			Pindex.push_back(i);
			Qindex.push_back(i);
		}
		if (busType[i] == 2) {
			N_g++; //increment PV bus counter since PV is represented by 2 
			Pindex.push_back(i);
			PVbusesVec.push_back(i);
			//For a PV bus there is no initial value of Q 
		}
		if (busType[i] == 3) {
			slackBus = i;
		}
	}

	jacSize = numberOfBuses+N_p-1; //Degree of the Jacobian matrix and length of vectors in linear system
	int *PQbuses = &Qindex[0];
	int *PVbuses = &PVbusesVec[0];
	cudaMalloc((void**)&dev_PQbuses, Qindex.size()*sizeof(int));
	cudaMalloc((void**)&dev_PVbuses, PVbusesVec.size()*sizeof(int));
	cudaMemcpy(dev_PQbuses, PQbuses, Qindex.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_PVbuses, PVbuses, PVbusesVec.size(), cudaMemcpyHostToDevice);

	//int jacSizeFull = (2*numberOfBuses) - 2;
	Pindex.insert(Pindex.end(), Qindex.begin(), Qindex.end()); //joins Pindex and Qindex to get a vector which holds indices of both PV and PQ buses. Store in Pindex
	PQindex = &Pindex[0]; //store vector Pindex as an array PQindex - compatible with GPU

	/*ofstream check("PQindex.txt");
	for (unsigned int i=0; i<Pindex.size(); i++)
		check<<PQindex[i]<<endl;
	check.close();

	check.open("PQbuses.txt");
	for (unsigned int i=0; i<Qindex.size(); i++)
		check<<PQbuses[i]<<endl;
	check.close();

	cout<<N_g<<endl;
	cout<<PVbusesVec.size()<<endl;

	check.open("PVbuses.txt");
	for (unsigned int i=0; i<PVbusesVec.size(); i++)
		check<<PVbuses[i]<<endl;
	check.close();*/

	//In the linear system Ax=b, the vectors are of the same degree as the Jacobian matrix
	stateVector = new double[jacSize];			//x in Ax = b
	powerMismatch = new double[jacSize];			//b in Ax = b

	//stateVector = new double[jacSizeFull];			//x in Ax = b
	//powerMismatch = new double[jacSizeFull];			//b in Ax = b
	

	//Allocation of GPU memory
	cudaStat1 = cudaMalloc((void**)&dev_fromBus, numLines*sizeof(int));
	cudaStat2 = cudaMalloc((void**)&dev_toBus, numLines*sizeof(int));
	cudaStat3 = cudaMalloc((void**)&dev_z, numLines*sizeof(cuDoubleComplex));
	cudaStat4 = cudaMalloc((void**)&dev_B, numLines*sizeof(cuDoubleComplex));
	cudaStat6 = cudaMalloc((void**)&dev_Pd, numberOfBuses*sizeof(double));
	cudaStat7 = cudaMalloc((void**)&dev_Qd, numberOfBuses*sizeof(double));
	cudaStat8 = cudaMalloc((void**)&dev_Vmag, numberOfBuses*sizeof(double));
	cudaStat9 = cudaMalloc((void**)&dev_theta, numberOfBuses*sizeof(double));
	cudaStat10 = cudaMalloc((void**)&dev_Peq, numberOfBuses*sizeof(double));
	cudaStat11 = cudaMalloc((void**)&dev_Qeq, numberOfBuses*sizeof(double));
	
	cudaStat12 = cudaMalloc((void**)&dev_powerMismatch, jacSize*sizeof(double));
	cudaStat13 = cudaMalloc((void**)&dev_stateVector, jacSize*sizeof(double));
	cudaStat14 = cudaMalloc((void**)&dev_PQindex, jacSize*sizeof(int));
	cudaStat15 = cudaMalloc((void**)&dev_PQspec, jacSize*sizeof(double));
	cudaStat16 = cudaMalloc((void**)&dev_PQcalc, jacSize*sizeof(double));
	cudaStat17 = cudaMalloc((void**)&dev_xi, jacSize*sizeof(double));
	
	/*cudaStat12 = cudaMalloc((void**)&dev_powerMismatch, jacSizeFull*sizeof(double));
	cudaStat13 = cudaMalloc((void**)&dev_stateVector, jacSizeFull*sizeof(double));
	cudaStat14 = cudaMalloc((void**)&dev_PQindex, jacSizeFull*sizeof(int));
	cudaStat15 = cudaMalloc((void**)&dev_PQspec, jacSizeFull*sizeof(double));
	cudaStat16 = cudaMalloc((void**)&dev_PQcalc, jacSizeFull*sizeof(double));
	cudaStat17 = cudaMalloc((void**)&dev_xi, jacSizeFull*sizeof(double));*/

	cudaStat18 = cudaMalloc((void**)&yDev, ynnz*sizeof(cuDoubleComplex));
	cudaMalloc((void**)&yRowDev, ynnz*sizeof(int));
	cudaMalloc((void**)&yColDev, ynnz*sizeof(int));

	if (cudaStat1 != cudaSuccess ||
		cudaStat2 != cudaSuccess ||
		cudaStat3 != cudaSuccess ||
		cudaStat4 != cudaSuccess ||
		cudaStat6 != cudaSuccess ||
		cudaStat7 != cudaSuccess ||
		cudaStat8 != cudaSuccess ||
		cudaStat9 != cudaSuccess ||
		cudaStat10 != cudaSuccess ||
		cudaStat11 != cudaSuccess ||
		cudaStat12 != cudaSuccess ||
		cudaStat13 != cudaSuccess ||
		cudaStat14 != cudaSuccess ||
		cudaStat15 != cudaSuccess ||
		cudaStat16 != cudaSuccess ||
		cudaStat17 != cudaSuccess ||
		cudaStat18 != cudaSuccess) {
		cout<<"Device memory allocation failed"<<endl;
		return 1;
	}

//-------------------------------------------------------------------BUS ADMITTANCE MATRIX CONSTRUCTION--------------------------------------------------------------

	//Reading from linedata.txt (tab delimited)
	if(!lineData) {
		cout<<"There was a problem reading the 'Line Data' file"<<endl;
		return 1;
	}

	while (lineData>>fromBus>>toBus>>r>>x>>b) {
		fromBusArr[lineDataIdx] = fromBus;
		toBusArr[lineDataIdx] = toBus;
		R[lineDataIdx] = r;
		X[lineDataIdx] = x; //r+jX
		Bact[lineDataIdx] = b;
		lineDataIdx++;
	}

	for (int i=0; i<numLines; i++) {
		B[i] = make_cuDoubleComplex(0, Bact[i]/2);
		Z[i] = make_cuDoubleComplex(R[i], X[i]);
	}

	//This is used to number buses consecutively from 0-299 for 300 bus case
	for (int i=0; i<numberOfBuses; i++) {
		tempBusNum[i] = i;
	}

	//Arranging bus numbering 
	for (int i=0; i<numLines; i++) {
		for (int j=0; j<numberOfBuses; j++) {
			if (fromBusArr[i] == busNum[j]) 
				fromBusArr[i] = tempBusNum[j];
			if (toBusArr[i] == busNum[j])
				toBusArr[i] = tempBusNum[j];
		}
		if (fromBusArr[i] == slackBus) {
			numSlackLines+=2;
			//numSlackLines++; //concise
		}
		if (toBusArr[i] == slackBus) {
			numSlackLines+=2;
			//numSlackLines++; //concise
		}
	}
	numSlackLines++; //for element y(0,0)

	for (int i=0; i<ynnz; i++) 
		yHost[i] = make_cuDoubleComplex(0,0);

	//copy memory from host to device - parameters needed for createYbus()
	cudaStat1 = cudaMemcpy(dev_fromBus, fromBusArr, numLines*sizeof(int), cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(dev_toBus, toBusArr, numLines*sizeof(int), cudaMemcpyHostToDevice);
	cudaStat3 = cudaMemcpy(dev_z, Z, numLines*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaStat4 = cudaMemcpy(dev_B, B, numLines*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaStat5 = cudaMemcpy(yDev, yHost, ynnz*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	if (cudaStat1 != cudaSuccess ||
		cudaStat2 != cudaSuccess ||
		cudaStat3 != cudaSuccess ||
		cudaStat4 != cudaSuccess ||
		cudaStat5 != cudaSuccess) {
		cout<<"Device memory copy failed"<<endl;
		return 1;
	}

	//grid and block dimensions - user defined
	dim3 dimBlock(threads, threads); //number of threads 
	dim3 dimGrid((numberOfBuses+(threads-1))/threads, (numberOfBuses+(threads-1))/threads);

	dim3 ythreads(threads);
	dim3 yblocks((numLines+(threads-1))/threads);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//launch kernel once data has been copied to GPU
	createYBusSparse<<<yblocks, ythreads>>>(numLines, numberOfBuses, dev_fromBus, dev_toBus, dev_z, dev_B, yDev, yRowDev, yColDev);
	//createYBusSparseConcise<<<yblocks, ythreads>>>(numLines, numberOfBuses, dev_fromBus, dev_toBus, dev_z, dev_B, yDev, yRowDev, yColDev);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"Y-bus Sparse: "<<elapsedTime<<" ms"<<endl;

	//------------------------------------Sorting yrow and ycol-----------------------------------------------------

	/*int *dev_yUpperTriangleRow2, *dev_yUpperTriangleCol2; //second array for sort_by_key()
	//int* dev_yRowTemp, *dev_yColTemp; //third set of arrays to sort row index using column index as key

	int *yUpperTriangleRow2, *yUpperTriangleCol2;
	yUpperTriangleRow2 = new int[numLines];
	yUpperTriangleCol2 = new int[numLines];

	cudaMalloc((void**)&dev_yUpperTriangleRow2, numLines*sizeof(int));
	cudaMalloc((void**)&dev_yUpperTriangleCol2, numLines*sizeof(int));

	//Copy row and column indices to duplicate arrays
	cudaMemcpy(dev_yUpperTriangleRow2, dev_yUpperTriangleRow, numLines*sizeof(int), cudaMemcpyDeviceToDevice);	
	cudaMemcpy(dev_yUpperTriangleCol2, dev_yUpperTriangleCol, numLines*sizeof(int), cudaMemcpyDeviceToDevice);

	//wrapping device pointers to arrays in device memory to treat as thrust vectors and use thrust sort function
	thrust::device_ptr<int> yrowPtr(dev_yUpperTriangleRow);
	thrust::device_ptr<int> ycolPtr(dev_yUpperTriangleCol);				//pointers to the original sparse Y
	thrust::device_ptr<int> ycolPtr2(dev_yUpperTriangleCol2);
	thrust::device_ptr<int> yrowPtr2(dev_yUpperTriangleRow2); //create wrapper to copy of yrow

	thrust::stable_sort_by_key(ycolPtr, ycolPtr+numLines, yrowPtr, thrust::less<int>());	//sort original yRow by original yCol using wrappers (both are sorted)
	thrust::stable_sort_by_key(yrowPtr2, yrowPtr2+numLines, ycolPtr2, thrust::less<int>());	//sort copy of yCol by copy of yRow (both are sorted) 

	//Copy row and column indices (sorted by column) to duplicate yRow and yCol index host arrays
	cudaMemcpy(yUpperTriangleRow, dev_yUpperTriangleRow, numLines*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(yUpperTriangleCol, dev_yUpperTriangleCol, numLines*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(yUpperTriangleRow2, dev_yUpperTriangleRow2, numLines*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(yUpperTriangleCol2, dev_yUpperTriangleCol2, numLines*sizeof(int), cudaMemcpyDeviceToHost);*/

	//------------------------------------------------------------------------------------------------------------------

	cudaMemcpy(yHost, yDev, ynnz*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(yRowHost, yRowDev, ynnz*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(yColHost, yColDev, ynnz*sizeof(int), cudaMemcpyDeviceToHost);

	ofstream output;
	output.open("SparseYbus.txt");
	for (int i=0; i<ynnz; i++) {
		output<<yRowHost[i]<<"\t"<<yColHost[i]<<"\t"<<"("<<cuCreal(yHost[i])<<","<<cuCimag(yHost[i])<<")"<<endl;
	}
	output.close();

//--------------------------------------------------------------------------Power Equations--------------------------------------------------------------------------

	for (int i=0; i<numberOfBuses; i++) {
		theta[i] = 0;
		P_eq[i] = 0;
		Q_eq[i] = 0;
	}

	cudaStat1 = cudaMemcpy(dev_Vmag, Vm, numberOfBuses*sizeof(double), cudaMemcpyHostToDevice);
	//cudaStat2 = cudaMemcpy(dev_theta, Va, numberOfBuses*sizeof(double), cudaMemcpyHostToDevice); //FOR HOT START
	cudaStat2 = cudaMemcpy(dev_theta, theta, numberOfBuses*sizeof(double), cudaMemcpyHostToDevice); //FOR FLAT START
	cudaStat3 = cudaMemcpy(dev_Peq, P_eq, numberOfBuses*sizeof(double), cudaMemcpyHostToDevice);
	cudaStat4 = cudaMemcpy(dev_Qeq, Q_eq, numberOfBuses*sizeof(double), cudaMemcpyHostToDevice);

	if (cudaStat1 != cudaSuccess ||
		cudaStat2 != cudaSuccess ||
		cudaStat3 != cudaSuccess ||
		cudaStat4 != cudaSuccess){
		cout<<"Device memory copy failed"<<endl;
		return 1;
	}

	dim3 pthreads(threads);
	dim3 pblock((ynnz+(threads-1))/threads);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	powerEqnSparse<<<pblock, pthreads>>>(dev_Peq, dev_Qeq, yDev, yRowDev, yColDev, dev_Vmag, dev_theta, ynnz);
	//powerEqnSparseConcise<<<pblock, pthreads>>>(dev_Peq, dev_Qeq, yDev, yRowDev, yColDev, dev_Vmag, dev_theta, ynnz, numLines);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"Sparse PQ eqns: "<<elapsedTime<<" ms"<<endl;

	cudaMemcpy(P_eq, dev_Peq, numberOfBuses*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Q_eq, dev_Qeq, numberOfBuses*sizeof(double), cudaMemcpyDeviceToHost);

	output.open("power equations sparse.txt");
	for (int i=0; i<numberOfBuses; i++) {
		output<<P_eq[i]<<endl;
	}
	output<<endl;
	for (int i=0; i<numberOfBuses; i++) {
		output<<Q_eq[i]<<endl;
	}
	cout<<endl;
	output.close();


	//To construct the power mismatch vector
	//SPECIFIED values of P and Q read from text file into Pd and Qd. Place these values in separate vectors for SPECIFIED value of P and Q
	for (int i=0; i<numberOfBuses; i++) {
		if (busType[i]!=3) { //if bus is not a slack bus
			Pval_spec.push_back(Pd[i]);
			Pval_calc.push_back(P_eq[i]);
			V_ang.push_back(theta[i]);
		}
		if (busType[i]==1) { //if bus is a PQ bus
			Qval_spec.push_back(Qd[i]);
			Qval_calc.push_back(Q_eq[i]);
			V_mag.push_back(Vm[i]);
		}
	}

	//power mismatch vector for Full Jacobian matrix (minus slack bus)
	/*for (int i=0; i<numberOfBuses; i++) {
		if (busType[i]!=3) {
			Pval_spec.push_back(Pd[i]);
			Pval_calc.push_back(P_eq[i]);
			V_ang.push_back(theta[i]);
			Qval_spec.push_back(Qd[i]);
			Qval_calc.push_back(Q_eq[i]);
			V_mag.push_back(Vm[i]);
		}
	}*/

	Pval_spec.insert(Pval_spec.end(), Qval_spec.begin(), Qval_spec.end()); //Append vectors yield vector of SPECIFIED real and rxve power
	PQspec = &Pval_spec[0]; //Create an array and store the appended vector in it to use on the GPU
	//cudaMemcpy(dev_PQspec, PQspec, jacSizeFull*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_PQspec, PQspec, jacSize*sizeof(double), cudaMemcpyHostToDevice);

	Pval_calc.insert(Pval_calc.end(), Qval_calc.begin(), Qval_calc.end()); //Appended vectors yield vector of CALCULATED real and rxve power
	PQcalc = &Pval_calc[0]; //Assign the appended vector to an array for use on the GPU

	//Append these vectors to get stateVector (Vang and Vmag)
	V_ang.insert(V_ang.end(), V_mag.begin(), V_mag.end());
	stateVector = &V_ang[0];
	output.open("stateVector.txt");
	for (unsigned int i=0; i<V_ang.size(); i++)
		output<<stateVector[i]<<"; "<<endl;
	output.close();

	//Power mismatch vector, b in Ax=b, is found by subtracting calculated from specified.
	output.open("powerMismatch.txt");
	for (unsigned int i=0; i<Pval_spec.size(); i++) {
		powerMismatch[i] = PQspec[i] - PQcalc[i];
		output<<powerMismatch[i]<<";"<<endl;
	}
	output.close();
//------------------------------------------------Creation of Jacobian Matrix----------------------------------------------------------
	cout<<"Y nnz: "<<ynnz<<endl;
	//int nnzJac= (ynnz + numLines - numSlackLines)*4; //concise
	int nnzJac= (ynnz - numSlackLines)*4;
	cout<<"NNZ Jac before: "<<nnzJac<<endl;	
	
	bool *dev_boolRow, *dev_boolCol;
	int *dev_J22row, *dev_J22col;
	cudaMalloc((void**)&dev_boolRow, ynnz*sizeof(bool));
	cudaMalloc((void**)&dev_boolCol, ynnz*sizeof(bool));
	cudaMalloc((void**)&dev_J22row, ynnz*sizeof(int));
	cudaMalloc((void**)&dev_J22col, ynnz*sizeof(int));

	//Initial algorithm to count nnzJac
	bool *boolCheck = new bool[numLines];
	for (int i=0; i<numLines; i++)
		boolCheck[i] = 0;
	bool *dev_boolCheck;
	int J12count=0; int J22count = 0;
	int* dev_J12count, *dev_J22count;
	cudaMalloc((void**)&dev_J12count, sizeof(int));
	cudaMalloc((void**)&dev_J22count, sizeof(int));
	cudaMalloc((void**)&dev_boolCheck, numLines*sizeof(bool));
	cudaMemcpy(dev_boolCheck, boolCheck, numLines*sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_J12count, &J12count, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_J22count, &J22count, sizeof(int), cudaMemcpyHostToDevice);

	countNnzJac<<<yblocks,ythreads>>>(dev_boolCheck, yRowDev, yColDev, yDev, dev_PVbuses, N_g, slackBus, dev_J12count, dev_J22count);

	cudaMemcpy(&J12count, dev_J12count, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&J22count, dev_J22count, sizeof(int), cudaMemcpyDeviceToHost);

	cout<<J12count<<endl;
	cout<<J22count<<endl;

	/*startCounter();
	for (int i=0; i<numLines; i++) {
		for (int j=0; j<N_g; j++) {
			if (yRowHost[i] != slackBus && yColHost[i] == PVbuses[j]) {
				J12count++;
				boolCheck[i] = true;
			}
			if (yColHost[i] != slackBus && yRowHost[i] == PVbuses[j]) {
				J12count++;
				boolCheck[i] = true;
			}
		}
		if (boolCheck[i] == true)
			J22count++;
	}*/
	
	J12count+=N_g;
	J22count*=2;
	J22count+=N_g;
	
	nnzJac = nnzJac - (J12count*2) - J22count;
	//cout<<"Time taken to find nnzJac: "<<getCounter()<<endl;
	cout<<"nnzJac: "<<nnzJac<<endl;
	//--------------------------------------------------------------------------------------------

	//int jacCount = ynnz - numSlackLines;
	//int nnzJac = jacCount*4; //for full jacobian (not reduced based on PQ buses)

	int h_counter = 0;
	cudaMemcpyToSymbol(counter, &h_counter, sizeof(int), 0, cudaMemcpyHostToDevice);

	//Jacobian variables
	int *jacRow, *jacCol;
	double* jac;

	jacRow = new int[nnzJac];
	jacCol = new int[nnzJac];
	jac = new double[nnzJac];

	int *dev_jacRow, *dev_jacCol;
	double* dev_jac;

	cudaMalloc((void**)&dev_jacRow, nnzJac*sizeof(int));
	cudaMalloc((void**)&dev_jacCol, nnzJac*sizeof(int));
	cudaMalloc((void**)&dev_jac, nnzJac*sizeof(double));
	cudaStat2 = cudaMemcpy(dev_PQindex, PQindex, jacSize*sizeof(int), cudaMemcpyHostToDevice);

	//dim3 dimGridJac2((ynnz+(threads-1))/threads, (jacSize+(threads-1))/threads, (jacSize+3)/4);
	//dim3 dimBlockJac2(threads, threads, 4);

	//dim3 dimGridJac((ynnz+(threads-1))/threads, (N_p+(threads-1)/threads));
	dim3 dimBlockJac(threads, threads);
	dim3 dimGridJ22((ynnz+(threads-1))/threads);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//createJ11<<<dimGridJ22, threads>>>(ynnz, numLines, numSlackLines, slackBus, dev_Peq, dev_Qeq, dev_Vmag, dev_theta, yDev, yRowDev, yColDev, dev_jac, dev_jacRow, dev_jacCol);
	createJ11Copy<<<dimGridJ22, threads>>>(ynnz, slackBus, dev_Peq, dev_Qeq, dev_Vmag, dev_theta, yDev, yRowDev, yColDev, dev_jac, dev_jacRow, dev_jacCol);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"GPU elapsed time - Jacobian (sparse): "<<elapsedTime<<" ms"<<endl;
	

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//createJacobianSparse3<<<dimGridJac, threads>>>(ynnz, slackBus, numberOfBuses, dev_Peq, dev_Qeq, dev_Vmag, dev_theta, yDev, yRowDev, yColDev, dev_jac, dev_jacRow, dev_jacCol, dev_PQbuses, N_p, dev_boolRow, dev_boolCol, dev_J22row, dev_J22col);
	createJ12_J21<<<dimGridJ22, threads>>>(ynnz, slackBus, numberOfBuses, dev_Peq, dev_Qeq, dev_Vmag, dev_theta, yDev, yRowDev, yColDev, dev_jac, dev_jacRow, dev_jacCol, dev_PQbuses, N_p, dev_boolRow, dev_boolCol, dev_J22row, dev_J22col);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"GPU elapsed time - Jacobian (sparse): "<<elapsedTime<<" ms"<<endl;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	createJ22<<<dimGridJ22, threads>>>(ynnz, numberOfBuses, dev_Peq, dev_Qeq, dev_Vmag, dev_theta, yDev, yRowDev, yColDev, dev_jac, dev_jacRow, dev_jacCol, dev_boolRow, dev_boolCol, dev_J22row, dev_J22col);
	//createJacobianSparse3<<<dimGridJac, threads>>>(ynnz, slackBus, numberOfBuses, dev_Peq, dev_Qeq, dev_Vmag, dev_theta, yDev, yRowDev, yColDev, dev_jac, dev_jacRow, dev_jacCol, dev_PQbuses, N_p, dev_boolRow, dev_boolCol, dev_J22row, dev_J22col);
	//createJacobianSparse2<<<dimGridJac2, dimBlockJac2>>>(ynnz, jacCount, slackBus, numberOfBuses, dev_Peq, dev_Qeq, dev_Vmag, dev_theta, yDev, yRowDev, yColDev, dev_jac, dev_jacRow, dev_jacCol, dev_PQindex, N_g, N_p, jacSize);
	//createJacobianSparse<<<dimGridJac, threads>>>(ynnz, jacCount, slackBus, numberOfBuses, dev_Peq, dev_Qeq, dev_Vmag, dev_theta, yDev, yRowDev, yColDev, dev_jac, dev_jacRow, dev_jacCol);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"GPU elapsed time - Jacobian (sparse): "<<elapsedTime<<" ms"<<endl;

	/*cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	createJacobianSparse<<<dimGridJ22, threads>>>(ynnz, jacCount, slackBus, numberOfBuses, dev_Peq, dev_Qeq, dev_Vmag, dev_theta, yDev, yRowDev, yColDev, dev_jac, dev_jacRow, dev_jacCol);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"GPU elapsed time - Jacobian (sparse): "<<elapsedTime<<" ms"<<endl;*/

	//----------------------------------------Thrust remove_if() trial---------------------------------------
	/*thrust::device_ptr<int> rowVecPtr(dev_jacRow);
	thrust::device_ptr<int> colVecPtr(dev_jacCol);
	thrust::device_ptr<double> valVecPtr(dev_jac);

	thrust::device_vector<int> rowVec(rowVecPtr, rowVecPtr+nnzJac);
	thrust::device_vector<int> colVec(colVecPtr, colVecPtr+nnzJac);
	thrust::device_vector<double> valVec(valVecPtr, valVecPtr+nnzJac);

	typedef thrust::device_vector<int>::iterator intDiter; //typedef iterator for row and col
	typedef thrust::device_vector<double>::iterator doubleDiter; //typedef iterator for double values
	typedef thrust::tuple<intDiter, intDiter, doubleDiter> iteratorTuple; //iterator tuple
	typedef thrust::zip_iterator<iteratorTuple> zipIt; //zip vectors together using tuple
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for (int i=0; i<N_g; i++) {
		zipIt zipBegin = thrust::make_zip_iterator(thrust::make_tuple(rowVec.begin(), colVec.begin(), valVec.begin()));
		zipIt zipEnd = zipBegin+rowVec.size();
		int eraseJacIdx = (numberOfBuses - 2) + PVbuses[i];
		zipIt newEnd = thrust::remove_if(zipBegin, zipEnd, is_PVbusRow(eraseJacIdx));
		iteratorTuple endTuple = newEnd.get_iterator_tuple();

		rowVec.erase(thrust::get<0>(endTuple), rowVec.end());
		colVec.erase(thrust::get<1>(endTuple), colVec.end());
		valVec.erase(thrust::get<2>(endTuple), valVec.end());
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"Remove jac row indices: "<<elapsedTime<<" ms"<<endl;

	cout<<rowVec.size()<<endl;
	cout<<colVec.size()<<endl;
	cout<<valVec.size()<<endl;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (int i=0; i<N_g; i++) {
		zipIt zipBegin = thrust::make_zip_iterator(thrust::make_tuple(rowVec.begin(), colVec.begin(), valVec.begin()));
		zipIt zipEnd = zipBegin+rowVec.size();
		int eraseJacIdx = (numberOfBuses - 2) + PVbuses[i];
		zipIt newEnd = thrust::remove_if(zipBegin, zipEnd, is_PVbusCol(eraseJacIdx));
		iteratorTuple endTuple = newEnd.get_iterator_tuple();

		rowVec.erase(thrust::get<0>(endTuple), rowVec.end());
		colVec.erase(thrust::get<1>(endTuple), colVec.end());
		valVec.erase(thrust::get<2>(endTuple), valVec.end());
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"Remove jac col indices: "<<elapsedTime<<" ms"<<endl;

	cout<<rowVec.size()<<endl;
	cout<<colVec.size()<<endl;
	cout<<valVec.size()<<endl;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	zipIt zipBegin = thrust::make_zip_iterator(thrust::make_tuple(rowVec.begin(), colVec.begin(), valVec.begin()));
	zipIt zipEnd = zipBegin+rowVec.size();
	zipIt newEnd = thrust::remove_if(zipBegin, zipEnd, isZero());
	iteratorTuple endTuple = newEnd.get_iterator_tuple();

	rowVec.erase(thrust::get<0>(endTuple), rowVec.end());
	colVec.erase(thrust::get<1>(endTuple), colVec.end());
	valVec.erase(thrust::get<2>(endTuple), valVec.end());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"Remove jac values: "<<elapsedTime<<" ms"<<endl;

	cout<<rowVec.size()<<endl;
	cout<<colVec.size()<<endl;
	cout<<valVec.size()<<endl;

	int* dev_jacRowNew = thrust::raw_pointer_cast(&rowVec[0]);
	int* dev_jacColNew = thrust::raw_pointer_cast(&colVec[0]);
	double* dev_jacNew = thrust::raw_pointer_cast(&valVec[0]);*/

	//-------------------------------------------------------------------------------------------------------

	cudaMemcpy(jacRow, dev_jacRow, nnzJac*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(jacCol, dev_jacCol, nnzJac*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(jac, dev_jac, nnzJac*sizeof(double), cudaMemcpyDeviceToHost);

	output.open("GPUJac.txt");
	for (int i=0; i<nnzJac; i++) 
		output<<jacRow[i]<<"\t"<<jacCol[i]<<"\t"<<jac[i]<<endl;
	output.close();

//--------------------------------------------------------------SOLUTION OF LINEAR SYSTEM-----------------------------------------------------------------
	//Removing zeros from dev_jac after calculation

	//Create wrapper around device pointer
	thrust::device_ptr<int> rowVecPtr(dev_jacRow);
	thrust::device_ptr<int> colVecPtr(dev_jacCol);
	thrust::device_ptr<double> valuesPtr(dev_jac);

	//Copy to device_vector for functionality
	thrust::device_vector<int> rowVec(rowVecPtr, rowVecPtr+nnzJac);
	thrust::device_vector<int> colVec(colVecPtr, colVecPtr+nnzJac);
	thrust::device_vector<double> valVec(valuesPtr, valuesPtr+nnzJac);

	//typedef thrust::tuple<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator, thrust::device_vector<float>::iterator> iteratorTuple;
	typedef thrust::device_vector<int>::iterator intDiter; //typedef iterator for row and col
	typedef thrust::device_vector<double>::iterator doubleDiter; //typedef iterator for double values
	typedef thrust::tuple<intDiter, intDiter, doubleDiter> iteratorTuple; //iterator tuple
	typedef thrust::zip_iterator<iteratorTuple> zipIt; //zip vectors together using tuple
	
	zipIt zipBegin = thrust::make_zip_iterator(thrust::make_tuple(rowVec.begin(), colVec.begin(), valVec.begin()));
	zipIt zipEnd = zipBegin+nnzJac;

	//Timing remove_if() and erase() operations
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	zipIt newEnd = thrust::remove_if(zipBegin, zipEnd, isZero());
	iteratorTuple endTuple = newEnd.get_iterator_tuple();

	rowVec.erase(thrust::get<0>(endTuple), rowVec.end());
	colVec.erase(thrust::get<1>(endTuple), colVec.end());
	valVec.erase(thrust::get<2>(endTuple), valVec.end());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"Thrust remove_if() for Jacobian: "<<elapsedTime<<" ms"<<endl;

	cout<<"nnzJac after removing zeros: "<<rowVec.size()<<endl;

	int* dev_jacRowNew = thrust::raw_pointer_cast(&rowVec[0]);
	int* dev_jacColNew = thrust::raw_pointer_cast(&colVec[0]);
	double* dev_jacNew = thrust::raw_pointer_cast(&valVec[0]);

	nnzJac = rowVec.size();
	
	//need to sort COO format Jacobian matrix in row-major order to get CSR format
	int* dev_jacRow2, *dev_jacCol2; //second array for sort_by_key()
	cudaMalloc((void**)&dev_jacRow2, nnzJac*sizeof(int));
	cudaMalloc((void**)&dev_jacCol2, nnzJac*sizeof(int));
	cudaMemcpy(dev_jacCol2, dev_jacColNew, nnzJac*sizeof(int), cudaMemcpyDeviceToDevice);

	//wrapping device pointers to arrays in device memory to treat as thrust vectors and use thrust sort function
	thrust::device_ptr<int> rowVecPtr2(dev_jacRowNew);
	thrust::device_ptr<int> colVecPtr2(dev_jacColNew);
	thrust::device_ptr<double> valuesPtr2(dev_jacNew);
	thrust::device_ptr<int> colVec2(dev_jacCol2);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//Perform sorting (sort column indices and values) based on rows (corresponding array sorting)
	thrust::stable_sort_by_key(colVecPtr2, colVecPtr2+nnzJac, rowVecPtr2, thrust::less<int>());
	thrust::stable_sort_by_key(colVec2, colVec2+nnzJac, valuesPtr2, thrust::less<int>());

	cudaMemcpy(dev_jacRow2, dev_jacRowNew, nnzJac*sizeof(int), cudaMemcpyDeviceToDevice);
	thrust::device_ptr<int> rowVec2(dev_jacRow2);

	thrust::stable_sort_by_key(rowVecPtr2, rowVecPtr2+nnzJac, colVecPtr2, thrust::less<int>());
	thrust::stable_sort_by_key(rowVec2, rowVec2+nnzJac, valuesPtr2, thrust::less<int>());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"Sorting Jacobian to convert to CSR: "<<elapsedTime<<" ms"<<endl;

	//Required to represent coefficient matrix (Jacobian) in sparse (CSR) format
	int *csrRowPtrJac;

	//Setup cuSPARSE
	cusparseStatus_t status;
	cusparseHandle_t handle = 0;
	cusparseMatDescr_t descr_A = 0;
	
	//Initialize cuSPARSE
	status = cusparseCreate(&handle);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		cout<<"CUSPARSE Library Initialization failed"<<endl;
		return 1;
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//Create and setup matrix descriptor for Coefficient Matrix
	cusparseCreateMatDescr(&descr_A);
	cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"CUPARSE Initialization: "<<elapsedTime<<" ms"<<endl;

	//cudaStat1 = cudaMalloc((void**)&csrRowPtrJac, (jacSizeFull+1)*sizeof(int));
	cudaStat1 = cudaMalloc((void**)&csrRowPtrJac, (jacSize+1)*sizeof(int));

	if (cudaStat1 != cudaSuccess) {
			cout<<"Device memory allocation failed."<<endl;
			return 1;
	}


	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//Convert matrix to sparse format (CSR)
	cusparseXcoo2csr(handle, dev_jacRowNew, nnzJac, jacSize, csrRowPtrJac, CUSPARSE_INDEX_BASE_ZERO);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"Converting COO to CSR: "<<elapsedTime<<" ms"<<endl;

	/*double eigen = 0.0;
	double *initVec = new double[jacSize];
	double *dev_initVec, *dev_c;

	for (int i=0; i<jacSize; i++)
		initVec[i] = 1;
	
	cudaStat1 = cudaMalloc((void**)&dev_initVec, jacSize*sizeof(double));
	cudaStat2 = cudaMalloc((void**)&dev_c, jacSize*sizeof(double));
	if (cudaStat1 != cudaSuccess ||
		cudaStat2 != cudaSuccess) {
			cout<<"Device memory allocation failed - BiCGStab variables."<<endl;
			return 1;
	}
	cudaMemcpy(dev_initVec, initVec, jacSize*sizeof(double), cudaMemcpyHostToDevice);

	powerMethod(csrRowPtrJac, dev_jacColNew, dev_jacNew, descr_A, nnzJac, initVec, dev_initVec, dev_c, handle, &eigen, jacSize);
	cout<<"Max eigenvalue of Jacobian matrix is: "<<eigen<<endl;*/

	int* h_jacRowNew, *h_jacColNew, *h_csrJacRowPtr;
	double* h_jacNew;

	h_jacRowNew = new int[nnzJac];
	h_jacColNew = new int[nnzJac];
	h_jacNew = new double[nnzJac];
	h_csrJacRowPtr = new int[jacSize+1];

	cudaMemcpy(h_jacRowNew, dev_jacRowNew, nnzJac*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_jacColNew, dev_jacColNew, nnzJac*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_jacNew, dev_jacNew, nnzJac*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_csrJacRowPtr, csrRowPtrJac, (jacSize+1)*sizeof(int), cudaMemcpyDeviceToHost);

	output.open("new jacobian after elimination and sorting.txt");
	for (int i=0; i<nnzJac; i++)
		output<<h_jacRowNew[i]<<"\t"<<h_jacColNew[i]<<"\t"<<h_jacNew[i]<<endl;
	for (int i=0; i<jacSize+1; i++)
		output<<h_csrJacRowPtr[i]<<endl;
	output.close();

	//Setup cuBLAS
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cublasStatus cublas_status;
	cublas_status = cublasInit();
	if (cublas_status!=CUBLAS_STATUS_SUCCESS) {
		cout<<"cuBLAS Initialization Error!"<<endl;
		return 1;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"CUBLAS Initialization: "<<elapsedTime<<" ms"<<endl;

	//Vectors required in BiCGStab
	double *res, *r_tld, *p, *s, *t, *v, *p_hat, *s_hat;

	/*cudaStat1 = cudaMalloc((void**)&res, jacSizeFull*sizeof(double));
	cudaStat2 = cudaMalloc((void**)&r_tld, jacSizeFull*sizeof(double));
	cudaStat3 = cudaMalloc((void**)&p, jacSizeFull*sizeof(double));
	cudaStat4 = cudaMalloc((void**)&p_hat, jacSizeFull*sizeof(double));
	cudaStat5 = cudaMalloc((void**)&s, jacSizeFull*sizeof(double));
	cudaStat6 = cudaMalloc((void**)&s_hat, jacSizeFull*sizeof(double));
	cudaStat7 = cudaMalloc((void**)&v, jacSizeFull*sizeof(double));
	cudaStat8 = cudaMalloc((void**)&t, jacSizeFull*sizeof(double));*/

	cudaStat1 = cudaMalloc((void**)&res, jacSize*sizeof(double));
	cudaStat2 = cudaMalloc((void**)&r_tld, jacSize*sizeof(double));
	cudaStat3 = cudaMalloc((void**)&p, jacSize*sizeof(double));
	cudaStat4 = cudaMalloc((void**)&p_hat, jacSize*sizeof(double));
	cudaStat5 = cudaMalloc((void**)&s, jacSize*sizeof(double));
	cudaStat6 = cudaMalloc((void**)&s_hat, jacSize*sizeof(double));
	cudaStat7 = cudaMalloc((void**)&v, jacSize*sizeof(double));
	cudaStat8 = cudaMalloc((void**)&t, jacSize*sizeof(double));

	if (cudaStat1 != cudaSuccess ||
		cudaStat2 != cudaSuccess ||
		cudaStat3 != cudaSuccess ||
		cudaStat4 != cudaSuccess ||
		cudaStat5 != cudaSuccess ||
		cudaStat6 != cudaSuccess ||
		cudaStat7 != cudaSuccess ||
		cudaStat8 != cudaSuccess) {
			cout<<"Device memory allocation failed - BiCGStab variables."<<endl;
			return 1;
	}

	cudaMemcpy(dev_stateVector, stateVector, jacSize*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_xi, stateVector, jacSize*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_powerMismatch, powerMismatch, jacSize*sizeof(double), cudaMemcpyHostToDevice);

	/*cudaMemcpy(dev_stateVector, stateVector, jacSizeFull*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_xi, stateVector, jacSizeFull*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_powerMismatch, powerMismatch, jacSizeFull*sizeof(double), cudaMemcpyHostToDevice);*/
	
	;

	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);
	biCGStab2(status, handle, descr_A, jacSize, jacSize, nnzJac, dev_jacColNew, csrRowPtrJac, dev_jacNew, dev_stateVector, res, r_tld, p, p_hat, s, s_hat, v, t, dev_powerMismatch, jacSize);
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&elapsedTime, start, stop);
	//cout<<"PBiCG-Stab: "<<elapsedTime<<" ms"<<endl;

	cudaMemcpy(stateVector, dev_stateVector, jacSize*sizeof(double), cudaMemcpyDeviceToHost);
	output.open("output.txt");
	for(int i=0; i<jacSize; i++)
		output<<stateVector[i]<<endl;
	output.close();
	//updateX<<<((jacSize+(threads-1))/threads), threads>>>(jacSize, numberOfBuses, dev_PQindex, dev_Vmag, dev_theta, dev_stateVector, dev_xi);
	//powerEqn <<<dimGrid, dimBlock>>>(dev_Peq1, dev_Qeq1, dev_y, dev_Vmag, dev_theta, numberOfBuses);
	//updateMismatch<<<((jacSize+(threads-1))/threads), threads>>>(numberOfBuses, jacSize, dev_Peq1, dev_Qeq1, dev_PQindex, dev_PQcalc, dev_PQspec, dev_powerMismatch);

	//createJacobian<<<dimGrid2, dimBlock>>>(numberOfBuses, jacSize, dev_Peq1, dev_Qeq1, dev_Vmag, dev_theta, dev_y, dev_jacobian, dev_PQindex);


	//check convergence
	//call Jacobian
	//loop

	cublas_status = cublasShutdown();
	if (cublas_status != CUBLAS_STATUS_SUCCESS) {
		cout<<"Shut down error"<<endl;
		return 1;
	}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------
	cout<<"\nThere will be "<< numberOfBuses-1<<" P equations and "<<N_p<<" Q equations to solve."<<endl;
	cout<<"There will be a total of "<<numberOfBuses+N_p-1<<" equations for the system and "<< numberOfBuses+N_p-1<< " unknowns (V and delta) to be solved."<<endl;
	cout<<"The Jacobian matrix will be of size "<<2*(numberOfBuses-1)-N_g<<" x "<<2*(numberOfBuses-1)-N_g<<endl;

	//free CUDA memory
	cudaFree(dev_fromBus);
	cudaFree(dev_toBus);
	cudaFree(dev_z);
	cudaFree(dev_B);
	cudaFree(dev_Pd);
	cudaFree(dev_Qd);
	cudaFree(yDev);
	cudaFree(yRowDev);
	cudaFree(yColDev);
	cudaFree(dev_jac);
	cudaFree(dev_jacRow);
//	cudaFree(dev_jacRow2);
	cudaFree(dev_jacCol);
	cudaFree(dev_Vmag);
	cudaFree(dev_theta);
	cudaFree(dev_Peq);
	cudaFree(dev_Qeq);
	cudaFree(dev_powerMismatch);
	cudaFree(dev_stateVector);
	cudaFree(dev_PQindex);
//	cudaFree(csrRowPtrJac);


	//delete all dynamic arrays
	delete[] busNum;
	delete[] busType;
	delete[] Pd;
	delete[] Qd;
	delete[] Vm;
	delete[] Va;
	delete[] Vmax;
	delete[] Vmin;
	delete[] fromBusArr;
	delete[] toBusArr;
	delete[] R;
	delete[] X;

	delete[] Z;

	delete[] B;
	delete[] P_eq;
	delete[] Q_eq;
	delete[] theta;
	
	return 0;
}

__global__ void createYBusSparse(int numLines, int numberOfBuses, int *fromBus, int* toBus, cuDoubleComplex *Z, cuDoubleComplex *B, cuDoubleComplex *y, int *yrow, int *ycol)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index<numLines)
	{
		int i = fromBus[index];
		int j = toBus[index];
		yrow[index] = i;
		ycol[index] = j;
		yrow[index+numLines] = j;
		ycol[index+numLines] = i;		

		y[index] = cuCsub(make_cuDoubleComplex(0,0), cuCdiv(make_cuDoubleComplex(1,0),Z[index]));
		y[index+numLines] = cuCsub(make_cuDoubleComplex(0,0), cuCdiv(make_cuDoubleComplex(1,0),Z[index]));
		
		cuDoubleComplex temp = cuCadd(cuCdiv(make_cuDoubleComplex(1,0),Z[index]),B[index]);
		atomicAddComplex(&y[i+(2*numLines)], temp);
		atomicAddComplex(&y[j+(2*numLines)], temp);

		if (index<numberOfBuses) {
			yrow[2*numLines+index] = index;
			ycol[2*numLines+index] = index;
		}
	}
}

__global__ void createYBusSparseConcise(int numLines, int numberOfBuses, int *fromBus, int* toBus, cuDoubleComplex *Z, cuDoubleComplex *B, cuDoubleComplex *y, int *yrow, int *ycol)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index<numLines)
	{
		int i = fromBus[index];
		int j = toBus[index];
		yrow[index] = i;
		ycol[index] = j;

		y[index] = cuCsub(make_cuDoubleComplex(0,0), cuCdiv(make_cuDoubleComplex(1,0),Z[index]));
		
		cuDoubleComplex temp = cuCadd(cuCdiv(make_cuDoubleComplex(1,0),Z[index]),B[index]);
		atomicAddComplex(&y[i+numLines], temp);
		atomicAddComplex(&y[j+numLines], temp);

		if (index<numberOfBuses) {
			yrow[numLines+index] = index;
			ycol[numLines+index] = index;
		}
	}
}

__global__ void powerEqnSparse(double *P, double *Q, cuDoubleComplex* y, int* yrow, int* ycol, double *Vm, double *theta, int ynnz)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < ynnz) 
	{
		atomicAdd2(&P[yrow[i]], (Vm[yrow[i]]*(Vm[ycol[i]]*((cuCreal(y[i])*cos(theta[yrow[i]] - theta[ycol[i]])) + cuCimag(y[i])*sin(theta[yrow[i]] - theta[ycol[i]])))));
		//atomicAdd2(&P[ycol[i]], (Vm[ycol[i]]*(Vm[yrow[i]]*((cuCreal(y[i])*cos(theta[ycol[i]] - theta[yrow[i]])) + cuCimag(y[i])*sin(theta[ycol[i]] - theta[yrow[i]])))));
		atomicAdd2(&Q[yrow[i]], (Vm[yrow[i]]*(Vm[ycol[i]]*((cuCreal(y[i])*sin(theta[yrow[i]] - theta[ycol[i]])) - cuCimag(y[i])*cos(theta[yrow[i]] - theta[ycol[i]])))));
		//atomicAdd2(&Q[ycol[i]], (Vm[ycol[i]]*(Vm[yrow[i]]*((cuCreal(y[i])*sin(theta[ycol[i]] - theta[yrow[i]])) - cuCimag(y[i])*cos(theta[ycol[i]] - theta[yrow[i]])))));
	}
}

__global__ void powerEqnSparseConcise(double *P, double *Q, cuDoubleComplex* y, int* yrow, int* ycol, double *Vm, double *theta, int ynnz, int numLines)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < numLines) 
	{
		atomicAdd2(&P[yrow[i]], (Vm[yrow[i]]*(Vm[ycol[i]]*((cuCreal(y[i])*cos(theta[yrow[i]] - theta[ycol[i]])) + cuCimag(y[i])*sin(theta[yrow[i]] - theta[ycol[i]])))));
		atomicAdd2(&P[ycol[i]], (Vm[ycol[i]]*(Vm[yrow[i]]*((cuCreal(y[i])*cos(theta[ycol[i]] - theta[yrow[i]])) + cuCimag(y[i])*sin(theta[ycol[i]] - theta[yrow[i]])))));
		atomicAdd2(&Q[yrow[i]], (Vm[yrow[i]]*(Vm[ycol[i]]*((cuCreal(y[i])*sin(theta[yrow[i]] - theta[ycol[i]])) - cuCimag(y[i])*cos(theta[yrow[i]] - theta[ycol[i]])))));
		atomicAdd2(&Q[ycol[i]], (Vm[ycol[i]]*(Vm[yrow[i]]*((cuCreal(y[i])*sin(theta[ycol[i]] - theta[yrow[i]])) - cuCimag(y[i])*cos(theta[ycol[i]] - theta[yrow[i]])))));
	}
	if (i >= numLines && i < ynnz)
	{
		atomicAdd2(&P[yrow[i]], (Vm[yrow[i]]*(Vm[ycol[i]]*((cuCreal(y[i])*cos(theta[yrow[i]] - theta[ycol[i]])) + cuCimag(y[i])*sin(theta[yrow[i]] - theta[ycol[i]])))));
		atomicAdd2(&Q[yrow[i]], (Vm[yrow[i]]*(Vm[ycol[i]]*((cuCreal(y[i])*sin(theta[yrow[i]] - theta[ycol[i]])) - cuCimag(y[i])*cos(theta[yrow[i]] - theta[ycol[i]])))));
	}
}

__global__ void countNnzJac(bool* boolCheck, int *yrow, int* ycol, cuDoubleComplex *y, int *PVbuses, int N_g, int slackBus, int* dev_J12count, int* dev_J22count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (int j=0; j<N_g; j++) {
		if (yrow[i] != slackBus && ycol[i] == PVbuses[j]) {
			atomicAdd(dev_J12count, 1); //J12count++
			boolCheck[i] = true;
		}
		if (ycol[i] != slackBus && yrow[i] == PVbuses[j]) {
			atomicAdd(dev_J12count, 1); //J12count++
			boolCheck[i] = true;
		}
	}
	if (boolCheck[i] == true)
		atomicAdd(dev_J22count, 1); //J22count++
}

//full jacobian
__global__ void createJacobianSparse(int ynnz, int jacCount, int slackBus, int numBus, double* P, double *Q, double *Vmag, double *Vang, cuDoubleComplex *y, int *yrow, int *ycol, double *jac, int *jacRow, int *jacCol)
{
	int yIdx	= blockIdx.x * blockDim.x + threadIdx.x; 
	if (yIdx < ynnz) {
		if (yrow[yIdx] != slackBus && ycol[yIdx] !=slackBus) {
			int i = atomicAdd(&counter, 1);
			//J11
			jacRow[i] = yrow[yIdx] - 1;
			jacCol[i] = ycol[yIdx] - 1;
			//J12 
			jacRow[i + jacCount] = yrow[yIdx] - 1;
			jacCol[i + jacCount] = ycol[yIdx] + numBus - 2;
			//J21 
			jacRow[i + (2*jacCount)] = yrow[yIdx] + numBus - 2;
			jacCol[i + (2*jacCount)] = ycol[yIdx] - 1;
			//J22 
			jacRow[i + (3*jacCount)] = yrow[yIdx] + numBus - 2;
			jacCol[i + (3*jacCount)] = ycol[yIdx] + numBus - 2;

			if (yrow[yIdx] == ycol[yIdx]) {
				//J11 diagonal calculations
				jac[i] = -Q[yrow[yIdx]] - (Vmag[yrow[yIdx]]*Vmag[yrow[yIdx]]*cuCimag(y[yIdx]));
				//J12 diagonal calculations
				jac[i + jacCount] = Vmag[yrow[yIdx]]*Vmag[yrow[yIdx]]*cuCreal(y[yIdx]) + P[yrow[yIdx]];
				//J21 diagonal calculations
				jac[i + (2*jacCount)] = P[yrow[yIdx]] -Vmag[yrow[yIdx]]*Vmag[yrow[yIdx]]*cuCreal(y[yIdx]);
				//J22 diagonal calculations
				jac[i + (3*jacCount)] = Q[yrow[yIdx]] -Vmag[yrow[yIdx]]*Vmag[yrow[yIdx]]*cuCimag(y[yIdx]);
			} else {
				//J11 off-diagonal calculations
				jac[i] = Vmag[yrow[yIdx]]*Vmag[ycol[yIdx]] *((cuCreal(y[yIdx])*sin(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])) - (cuCimag(y[yIdx]) * cos(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])));
				//J12 off-diagonal calculations
				jac[i + jacCount] = Vmag[yrow[yIdx]]*Vmag[ycol[yIdx]] *((cuCreal(y[yIdx])*cos(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])) + (cuCimag(y[yIdx]) * sin(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])));
				//J21 off-diagonal calculations
				jac[i + (2*jacCount)] = -Vmag[yrow[yIdx]]*Vmag[ycol[yIdx]] *((cuCreal(y[yIdx])*cos(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])) + (cuCimag(y[yIdx]) * sin(Vang[yrow[yIdx]] - Vang[ycol[yIdx]]))); 
				//J22 off-diagonal calculations
				jac[i + (3*jacCount)] = Vmag[yrow[yIdx]]*Vmag[ycol[yIdx]] *((cuCreal(y[yIdx])*sin(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])) - (cuCimag(y[yIdx]) * cos(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])));
			}
		}
	}
}

__global__ void createJacobianSparse3(int ynnz, int slackBus, int numBus, double* P, double *Q, double *Vmag, double *Vang, cuDoubleComplex *y, int *yrow, int *ycol, 
	double *jac, int *jacRow, int *jacCol, int* PQbuses, int N_p, bool* boolRow, bool* boolCol, int* J22row, int* J22col)
{
	int yIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	int PQidx = blockIdx.y * blockDim.y + threadIdx.y;

	if (yIdx < ynnz) {
		if (yrow[yIdx] != slackBus && ycol[yIdx] !=slackBus) {
			/*if (PQidx==0) {
				int i = atomicAdd(&counter, 1);
				int j = atomicAdd(&counter2, 1);
				//J11
				jacRow[i] = yrow[yIdx] - 1;
				jacCol[i] = ycol[yIdx] - 1;
				if (yrow[yIdx] == ycol[yIdx]) {
					//J11 diagonal calculations
					jac[i] = -Q[yrow[yIdx]] - (Vmag[yrow[yIdx]]*Vmag[yrow[yIdx]]*cuCimag(y[yIdx]));
				} else {
					//J11 off-diagonal calculations
					jac[i] = Vmag[yrow[yIdx]]*Vmag[ycol[yIdx]] *((cuCreal(y[yIdx])*sin(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])) - (cuCimag(y[yIdx]) * cos(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])));
				}
			}*/

			if (PQidx<N_p) {
				if (ycol[yIdx] == PQbuses[PQidx]) {
					boolCol[yIdx] = true;
					int i = atomicAdd(&counter, 1);
					if (yrow[yIdx] < slackBus)
						jacRow[i] = yrow[yIdx];
					else
						jacRow[i] = yrow[yIdx] - 1;
					jacCol[i] = numBus + PQidx -1;	  //J12
					J22col[yIdx] = numBus + PQidx - 1;

					if (yrow[yIdx] == ycol[yIdx]) {
						//J12 diagonal calculations
						jac[i] = Vmag[yrow[yIdx]]*Vmag[yrow[yIdx]]*cuCreal(y[yIdx]) + P[yrow[yIdx]];
					} else {
						//J12 off-diagonal calculations
						jac[i] = Vmag[yrow[yIdx]]*Vmag[ycol[yIdx]] *((cuCreal(y[yIdx])*cos(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])) + (cuCimag(y[yIdx]) * sin(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])));
					}
				}

				if (yrow[yIdx] == PQbuses[PQidx]) {
					boolRow[yIdx] = true;
					int i = atomicAdd(&counter, 1);
					//int j = atomicAdd(&counter2, 1);
					jacRow[i] = numBus + PQidx -1; //J21
					if (ycol[yIdx] < slackBus)
						jacCol[i] = ycol[yIdx];
					else
						jacCol[i] = ycol[yIdx] - 1;
					J22row[yIdx] = numBus + PQidx - 1;

					if (yrow[yIdx] == ycol[yIdx]) {
						//J21 diagonal calculations
						jac[i] = P[yrow[yIdx]] -Vmag[yrow[yIdx]]*Vmag[yrow[yIdx]]*cuCreal(y[yIdx]);
					} else {
						//J21 off-diagonal calculations
						jac[i] = -Vmag[yrow[yIdx]]*Vmag[ycol[yIdx]] *((cuCreal(y[yIdx])*cos(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])) + (cuCimag(y[yIdx]) * sin(Vang[yrow[yIdx]] - Vang[ycol[yIdx]]))); 
					}
				}

				/*if (boolRow[yIdx] == true && boolCol[yIdx] == true) {
					int i = atomicAdd(&counter, 1);
					if (yrow[yIdx] == ycol[yIdx]) {
						jac[i] = Q[yrow[yIdx]] -Vmag[yrow[yIdx]]*Vmag[yrow[yIdx]]*cuCimag(y[yIdx]);
					} else {
						jac[i] = Vmag[yrow[yIdx]]*Vmag[ycol[yIdx]] *((cuCreal(y[yIdx])*sin(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])) - (cuCimag(y[yIdx]) * cos(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])));
					}
				}*/
			}
		}
	}
}

__global__ void createJ12_J21(int ynnz, int slackBus, int numBus, double* P, double *Q, double *Vmag, double *Vang, cuDoubleComplex *y, int *yrow, int *ycol, 
	double *jac, int *jacRow, int *jacCol, int* PQbuses, int N_p, bool* boolRow, bool* boolCol, int* J22row, int* J22col)
{
	int yIdx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (yIdx < ynnz) {
		if (yrow[yIdx] != slackBus && ycol[yIdx] !=slackBus) {
			for (int PQidx=0; PQidx<N_p; PQidx++) {
				if (ycol[yIdx] == PQbuses[PQidx]) {
					boolCol[yIdx] = true;
					
					int i = atomicAdd(&counter, 1);
					jacCol[i] = numBus + PQidx -1;	  //J12
					if (yrow[yIdx] < slackBus)
						jacRow[i] = yrow[yIdx];
					else
						jacRow[i] = yrow[yIdx] - 1;
					J22col[yIdx] = numBus + PQidx - 1;
					if (yrow[yIdx] == ycol[yIdx]) {
						//J12 diagonal calculations
						jac[i] = Vmag[yrow[yIdx]]*Vmag[yrow[yIdx]]*cuCreal(y[yIdx]) + P[yrow[yIdx]];				
					} else {
						//J12 off-diagonal calculations
						jac[i] = Vmag[yrow[yIdx]]*Vmag[ycol[yIdx]] *((cuCreal(y[yIdx])*cos(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])) + (cuCimag(y[yIdx]) * sin(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])));
					}
				}

				if (yrow[yIdx] == PQbuses[PQidx]) {
					boolRow[yIdx] = true;
					int i = atomicAdd(&counter, 1);
					jacRow[i] = numBus + PQidx -1; //J21
					if (ycol[yIdx] < slackBus)
						jacCol[i] = ycol[yIdx];
					else
						jacCol[i] = ycol[yIdx] - 1;
					J22row[yIdx] = numBus + PQidx - 1;

					if (yrow[yIdx] == ycol[yIdx]) {
						//J21 diagonal calculations
						jac[i] = P[yrow[yIdx]] -Vmag[yrow[yIdx]]*Vmag[yrow[yIdx]]*cuCreal(y[yIdx]);
					} else {
						//J21 off-diagonal calculations
						jac[i] = -Vmag[yrow[yIdx]]*Vmag[ycol[yIdx]] *((cuCreal(y[yIdx])*cos(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])) + (cuCimag(y[yIdx]) * sin(Vang[yrow[yIdx]] - Vang[ycol[yIdx]]))); 
					}
				}
			}
		}
	}
}

__global__ void createJ22(int ynnz, int numBus, double* P, double *Q, double *Vmag, double *Vang, cuDoubleComplex *y, int *yrow, int *ycol, double *jac, int *jacRow, 
	int *jacCol, bool* boolRow, bool* boolCol, int* J22row, int* J22col)
{
	int yIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (yIdx<ynnz) {
		if (boolRow[yIdx] == true && boolCol[yIdx] == true) {
			int i = atomicAdd(&counter, 1);
			jacRow[i] = J22row[yIdx];
			jacCol[i] = J22col[yIdx];
			if (yrow[yIdx] == ycol[yIdx]) {
				jac[i] = Q[yrow[yIdx]] -Vmag[yrow[yIdx]]*Vmag[yrow[yIdx]]*cuCimag(y[yIdx]);
			} else {
				jac[i] = Vmag[yrow[yIdx]]*Vmag[ycol[yIdx]] *((cuCreal(y[yIdx])*sin(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])) - (cuCimag(y[yIdx]) * cos(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])));
			}
		}
	}
}

__global__ void createJ11(int ynnz, int numLines, int numSlackLines, int slackBus, double* P, double *Q, double *Vmag, double *Vang, cuDoubleComplex *y, int *yrow, int *ycol, double *jac, int *jacRow, 
	int *jacCol)
{
	int yIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = numLines - ((numSlackLines - 1)/2);
	if (yIdx < numLines) {
		if (yrow[yIdx] != slackBus && ycol[yIdx] !=slackBus) {
			int i = atomicAdd(&counter, 1);
			if (yrow[yIdx] < slackBus) {
				jacRow[i] = yrow[yIdx];
				jacCol[i+offset] = yrow[yIdx];
			}
			else {
				jacRow[i] = yrow[yIdx] - 1;
				jacCol[i+offset] = yrow[yIdx] - 1;
			}
			
			if (ycol[yIdx] < slackBus) {
				jacCol[i] = ycol[yIdx];
				jacRow[i+offset] = ycol[yIdx];
			}
			else {
				jacCol[i] = ycol[yIdx] - 1;
				jacRow[i+offset] = ycol[yIdx] - 1;
			}
			jac[i] = Vmag[yrow[yIdx]]*Vmag[ycol[yIdx]] *((cuCreal(y[yIdx])*sin(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])) - (cuCimag(y[yIdx]) * cos(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])));
			jac[i+offset] = Vmag[yrow[yIdx]]*Vmag[ycol[yIdx]] *((cuCreal(y[yIdx])*sin(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])) - (cuCimag(y[yIdx]) * cos(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])));
		}
	}
	if (yIdx >= numLines && yIdx < ynnz) {
		if (yrow[yIdx] != slackBus && ycol[yIdx] !=slackBus) {
			int i = atomicAdd(&counter, 1);

			if (yrow[yIdx] < slackBus) 
				jacRow[i+offset] = yrow[yIdx];
			else 
				jacRow[i+offset] = yrow[yIdx] - 1;
			
			
			if (ycol[yIdx] < slackBus) 
				jacCol[i+offset] = ycol[yIdx];
			else 
				jacCol[i+offset] = ycol[yIdx] - 1;
			
			jac[i+offset] = -Q[yrow[yIdx]] - (Vmag[yrow[yIdx]]*Vmag[yrow[yIdx]]*cuCimag(y[yIdx]));
		}
	}
}


__global__ void createJ11Copy(int ynnz, int slackBus, double* P, double *Q, double *Vmag, double *Vang, cuDoubleComplex *y, int *yrow, int *ycol, double *jac, int *jacRow, 
	int *jacCol)
{
	int yIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (yIdx < ynnz) {
		if (yrow[yIdx] != slackBus && ycol[yIdx] !=slackBus) {
			int i = atomicAdd(&counter, 1);
			if (yrow[yIdx] < slackBus)
				jacRow[i] = yrow[yIdx];
			else
				jacRow[i] = yrow[yIdx] - 1;
			
			if (ycol[yIdx] < slackBus)
				jacCol[i] = ycol[yIdx];
			else
				jacCol[i] = ycol[yIdx] - 1;

			if (yrow[yIdx] == ycol[yIdx]) //J11 diagonal calculations
				jac[i] = -Q[yrow[yIdx]] - (Vmag[yrow[yIdx]]*Vmag[yrow[yIdx]]*cuCimag(y[yIdx]));
			else //J11 off-diagonal calculations
				jac[i] = Vmag[yrow[yIdx]]*Vmag[ycol[yIdx]] *((cuCreal(y[yIdx])*sin(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])) - (cuCimag(y[yIdx]) * cos(Vang[yrow[yIdx]] - Vang[ycol[yIdx]])));
		}
	}
}


/*void biCGStab(cusparseDtatus_t status, cusparseHandle_t handle, cusparseMatDescr_t descr_A, int M, int N, int nnz, int* csrColIndAdev, int* csrRowPtrAdev, 
	double* csrValAdev, double* x, double* r, double* r_tld, double* p, double *p_hat, double* s, double *s_hat, double* v, double* t, double* b)*/
/*void biCGStab(cusparseDtatus_t status, cusparseHandle_t handle, cusparseMatDescr_t descr_A, cusparseMatDescr_t descr_M, int M, int N, int nnz, int nnzJacPre, int* csrColIndAdev, int* csrRowPtrAdev, 
	double* csrValAdev, int* csrColIndPre, int* csrRowPtrPre, double* csrValPre, double* x, double* r, double* r_tld, double* p, double *p_hat, double* s, double *s_hat, double* v, double* t, double* b)*/
void biCGStab(cusparseStatus_t status, cusparseHandle_t handle, cusparseMatDescr_t descr_A, cusparseMatDescr_t descr_L, cusparseMatDescr_t descr_U, 
	cusparseSolveAnalysisInfo_t info_L, cusparseSolveAnalysisInfo_t info_U, int M, int N, int nnz, int* csrColIndAdev, int* csrRowPtrAdev, 	
	double* csrValAdev, int* csrColIndPre, int* csrRowPtrPre, double* csrValPre, 	double* x, double* r, double* r_tld, double* p, double *p_hat, 
	double* s, double *s_hat, double* v, double* t, double* b)
{
	double bnorm, snorm, err, alpha=1.0, beta, omega=1.0, rho=1.0, rho_1, resid=0; //BiCG scalars - previously global variables
	int flag, iter;
	
	//For cusparse csrmv function
	double d_one = 1.0;
	double dzero = 0.0;
	double temp=0, temp2=0;

	//Setup cuBLAS
	cublasStatus cublas_status;
	cublas_status = cublasInit();
	if (cublas_status!=CUBLAS_STATUS_SUCCESS) {
		cout<<"cuBLAS Initialization Error!"<<endl;
		return;
	}

	cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nnz, &d_one, descr_A, csrValAdev, csrRowPtrAdev, csrColIndAdev, x, &dzero, r);
	cublasDscal(M, -1.0, r, 1);
	cublasDaxpy(M, 1.0, b, 1, r, 1);

	cublasDcopy(N, r, 1, p, 1);
	cublasDcopy(N, r, 1, r_tld, 1);
	bnorm = cublasDnrm2(N, b, 1);

	//To find Error: error = ||r||/||b||
	err = cublasDnrm2(N, r, 1)/bnorm;
	if (err<Tol) {
		cout<<"Solution has already converged"<<endl;
		return;
	}

	for (iter=0; iter<max_it; iter++)
	{
		rho_1 = rho;
		rho = cublasDdot(N, r_tld, 1, r, 1);
		//cout<<"Rho: "<<rho<<endl;
		if (rho == 0) //check for breakdown
			break;

		//For every iteration after the first
		if (iter>0) {
			beta = (rho/rho_1)*(alpha/omega);
			//p = r+ beta(p-omega*v)
			cublasDaxpy(N, -omega, v, 1, p, 1);
			cublasDscal(N, beta, p, 1);
			cublasDaxpy(N, 1.0, r, 1, p, 1);
		}

		//Preconditioner to find v
		//If M=I, this implies p_hat = p => can use cudaMemcpyDeviceToDevice to transfer p to p_hat
		//p_hat = M\p
		cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, &d_one, descr_L, csrValPre, csrRowPtrPre, csrColIndPre, info_L, p, t);
		//Here we are using t as a temporary vector - it is not needed in the algorithm at this point and saves memory of creating another vector

		cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, &d_one, descr_U, csrValPre, csrRowPtrPre, csrColIndPre, info_U, t, p_hat);

		//p_hat = inv(M)*p
		//cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nnzJacPre, &d_one, descr_M, csrValPre, csrRowPtrPre, csrColIndPre, p, &dzero, p_hat);

		//v = A*p_hat
		//cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nnz, &d_one, descr_A, csrValAdev, csrRowPtrAdev, csrColIndAdev, p, &dzero, v);
		cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nnz, &d_one, descr_A, csrValAdev, csrRowPtrAdev, csrColIndAdev, p_hat, &dzero, v);

		alpha = rho/cublasDdot(N, r_tld, 1, v, 1); //alpha = rho/(r_tld,v)

		cublasDaxpy(N, -alpha, v, 1, r, 1); //s=r - alpha*v
		cublasDcopy(N, r, 1, s, 1);
		//cublasDaxpy(N, alpha, p, 1, x, 1); 
		cublasDaxpy(N, alpha, p_hat, 1, x, 1); //x = x+ alpha*p

		//Check for convergence
		snorm = cublasDnrm2(N, s, 1);
		if (snorm/bnorm < Tol) {
			resid = snorm/bnorm;
			break;
		}

		//Preconditioner to find t
		//M=I implies s = s_hat => t=As
		//s_hat = M\s
		cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, &d_one, descr_L, csrValPre, csrRowPtrPre, csrColIndPre, info_L, r, t);
		cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, &d_one, descr_U, csrValPre, csrRowPtrPre, csrColIndPre, info_U, t, s_hat);

		//s_hat = inv(M)*s
		//cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nnzJacPre, &d_one, descr_M, csrValPre, csrRowPtrPre, csrColIndPre, s, &dzero, s_hat);

		//t=A*s_hat
		//cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nnz, &d_one, descr_A, csrValAdev, csrRowPtrAdev, csrColIndAdev, s, &dzero, t);
		cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nnz, &d_one, descr_A, csrValAdev, csrRowPtrAdev, csrColIndAdev, s_hat, &dzero, t);

		temp = cublasDdot(N, t, 1, r, 1);
		temp2 = cublasDdot(N, t, 1, t, 1);
		omega = temp/temp2;

		//x = x+ omega*s
		//cublasDaxpy(N, omega, s, 1, x, 1);
		cublasDaxpy(N, omega, s_hat, 1, x, 1);

		//r = s-omega*t
		cublasDaxpy(N, -omega, t, 1, r, 1);

		err = cublasDnrm2(N, r, 1)/bnorm;
		if (err<=Tol) {
			resid = cublasDnrm2(N, s, 1)/bnorm;
			break;
		}
		if (omega == 0.0)
			break;
		//rho_1 = rho;
	}

	if (err <= Tol || snorm/bnorm < Tol)
		flag = 0;
	else if (omega == 0.0)
		flag = -2;
	else if (rho == 0)
		flag = -1;
	else 
		flag = 1;

	if (!flag) 
		cout<<"The solution converged with residual "<<resid<<" in "<<iter<<" iterations."<<endl;
	else
		cout<<"BiCGStab produced error "<<flag<<" after "<<iter<<" iterations."<<endl;

	//Shutdown cuBLAS
	cublas_status = cublasShutdown();
	if (cublas_status != CUBLAS_STATUS_SUCCESS) {
		cout<<"Shut down error"<<endl;
		return;
	}
}

//----------------------------------------------------------------------------------------------------------------------------------

__global__ void updateX(int jacSize, int N, int *PQindex, double *Vmag, double *theta, double *stateVector, double *x)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index<(N-1)) {
		theta[PQindex[index]] = theta[PQindex[index]] + stateVector[index]; 
		stateVector[index] = x[index] + stateVector[index];
	}
	else if(index>=(N-1) && index<jacSize) {
		Vmag[PQindex[index]] = Vmag[PQindex[index]] + stateVector[index];
		stateVector[index] = x[index] + stateVector[index];
	}
}

__global__ void updateMismatch(int N, int jacSize, double *P_eq, double *Q_eq, int *PQindex, double* PQcalc, double* PQspec, double *powerMismatch)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index<(N-1)) {
		PQcalc[index] = P_eq[PQindex[index]];
	}
	else if(index>=(N-1) && index<jacSize) {
		PQcalc[index] = Q_eq[PQindex[index]];
	}
	powerMismatch[index] = PQspec[index] - PQcalc[index];
}

__global__ void jacobiPrecond(int jacSize, double *jacobian, double *precInv)
{
	int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;
	int colIdx = blockDim.x*blockIdx.x + threadIdx.x;
	int index = rowIdx*jacSize + colIdx; //row major order - modify for column major
	precInv[index] = 0;
	if (rowIdx == colIdx)
		precInv[index] = 1/jacobian[index]; //inverse of strictly diagonal matrix is 1/a_ii
}

__device__ double radToDeg(double a)
{
	return 57.29577951*a;
}

__device__ void atomicAddComplex(cuDoubleComplex *a, cuDoubleComplex b)
{
	double *x = (double*)a;
	double *y = x+1;
	atomicAdd2(x, cuCreal(b));
	atomicAdd2(y, cuCimag(b));
}

__device__ double atomicAdd2(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

void biCGStab2(cusparseStatus_t status, cusparseHandle_t handle, cusparseMatDescr_t descr_A, int M, int N, int nnz, int* csrColIndAdev, int* csrRowPtrAdev, double* csrValAdev, 
	double* x, double* r, double* r_tld, double* p, double *p_hat, double* s, double *s_hat, double* v, double* t, double* b, int jacSize)
{
	double* csrValPre;
	cudaEvent_t start, stop;
	float elapsedTime;
	ofstream triSolve1("Tri solve 1.txt");
	ofstream triSolve2("Tri solve 2.txt");
	ofstream spMV1("SpMV 1.txt");
	ofstream spMV2("SpMV 2.txt");
	ofstream ilu("ILU matrix.txt");

	cudaMalloc((void**)&csrValPre, nnz*sizeof(double));
	cudaMemcpy(csrValPre, csrValAdev, nnz*sizeof(double), cudaMemcpyDeviceToDevice);

	cusparseMatDescr_t descr_M = 0;
	cusparseMatDescr_t descr_L = 0;
	cusparseMatDescr_t descr_U = 0;
	cusparseSolveAnalysisInfo_t info_M;
	cusparseSolveAnalysisInfo_t info_L;
	cusparseSolveAnalysisInfo_t info_U;

	//Create and setup matrix descriptors for ILU preconditioner
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cusparseCreateMatDescr(&descr_M);
	cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);

	cusparseCreateMatDescr(&descr_L);
	cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

	cusparseCreateMatDescr(&descr_U);
	cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

	cusparseCreateSolveAnalysisInfo(&info_M);
	cusparseCreateSolveAnalysisInfo(&info_L);
	cusparseCreateSolveAnalysisInfo(&info_U);

	//Perform Analysis before calling ilu0(); NB analysis can be done on L and U at this point since sparsity pattern is the same 
	cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, nnz, descr_M, csrValPre, csrRowPtrAdev, csrColIndAdev, info_M);
	cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, nnz, descr_L, csrValPre, csrRowPtrAdev, csrColIndAdev, info_L);
	cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, nnz, descr_U, csrValPre, csrRowPtrAdev, csrColIndAdev, info_U);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"ILU setup: "<<elapsedTime<<" ms"<<endl;

	//Perform ILU0 factorization
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cusparseDcsrilu0(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, descr_M, csrValPre, csrRowPtrAdev, csrColIndAdev, info_M);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"ILU formation: "<<elapsedTime<<" ms"<<endl;

	//double *csrValPreHost = new double[nnz];
	//cudaMemcpy(csrValPreHost, csrValPre, nnz*sizeof(double), cudaMemcpyDeviceToHost);
	//for (int i=0; i<nnz; i++) {
	//	ilu<<csrValPreHost[i]<<endl;
	//}
	//ilu.close();

	double bnorm, snorm=0, err, alpha=1.0, beta, omega=1.0, rho=1.0, rho_1, resid=0; //BiCG scalars - previously global variables
	int flag, iter;
	
	//For cusparse csrmv function
	double d_one = 1.0;
	double dzero = 0.0;
	double temp=0, temp2=0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nnz, &d_one, descr_A, csrValAdev, csrRowPtrAdev, csrColIndAdev, x, &dzero, r);
	cublasDscal(M, -1.0, r, 1);
	cublasDaxpy(M, 1.0, b, 1, r, 1);

	cublasDcopy(N, r, 1, p, 1);
	cublasDcopy(N, r, 1, r_tld, 1);
	bnorm = cublasDnrm2(N, b, 1);

	//To find Error: error = ||r||/||b||
	err = cublasDnrm2(N, r, 1)/bnorm;
	if (err<Tol) {
		cout<<"Solution has already converged"<<endl;
		return;
	}

	for (iter=0; iter<max_it; iter++)
	{
		rho_1 = rho;
		rho = cublasDdot(N, r_tld, 1, r, 1);
		//cout<<"Rho: "<<rho<<endl;
		if (rho == 0) //check for breakdown
			break;

		//For every iteration after the first
		if (iter>0) {
			beta = (rho/rho_1)*(alpha/omega);
			//p = r+ beta(p-omega*v)
			cublasDaxpy(N, -omega, v, 1, p, 1);
			cublasDscal(N, beta, p, 1);
			cublasDaxpy(N, 1.0, r, 1, p, 1);
		}

		cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, &d_one, descr_L, csrValPre, csrRowPtrAdev, csrColIndAdev, info_L, p, t);
		cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, &d_one, descr_U, csrValPre, csrRowPtrAdev, csrColIndAdev, info_U, t, p_hat);

		//v = A*p_hat
		//cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nnz, &d_one, descr_A, csrValAdev, csrRowPtrAdev, csrColIndAdev, p, &dzero, v);
		cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nnz, &d_one, descr_A, csrValAdev, csrRowPtrAdev, csrColIndAdev, p_hat, &dzero, v);

		/*double *vhost = new double[jacSize];
		cudaMemcpy(vhost, v, jacSize*sizeof(double), cudaMemcpyDeviceToHost);
		ofstream phat("v.txt");
		for (int i=0; i<jacSize; i++)
			phat<<vhost[i]<<endl;
		phat.close();*/

		alpha = rho/cublasDdot(N, r_tld, 1, v, 1); //alpha = rho/(r_tld,v)

		cublasDaxpy(N, -alpha, v, 1, r, 1); //s=r - alpha*v
		cublasDcopy(N, r, 1, s, 1);
		//cublasDaxpy(N, alpha, p, 1, x, 1); 
		cublasDaxpy(N, alpha, p_hat, 1, x, 1); //x = x+ alpha*p

		//Check for convergence
		snorm = cublasDnrm2(N, s, 1);
		if (snorm/bnorm < Tol) {
			resid = snorm/bnorm;
			//break;
		}

		//Preconditioner to find t
		//M=I implies s = s_hat => t=As
		//s_hat = M\s
		cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, &d_one, descr_L, csrValPre, csrRowPtrAdev, csrColIndAdev, info_L, r, t);
		cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, &d_one, descr_U, csrValPre, csrRowPtrAdev, csrColIndAdev, info_U, t, s_hat);

		//t=A*s_hat
		//cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nnz, &d_one, descr_A, csrValAdev, csrRowPtrAdev, csrColIndAdev, s, &dzero, t);
		cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nnz, &d_one, descr_A, csrValAdev, csrRowPtrAdev, csrColIndAdev, s_hat, &dzero, t);

		temp = cublasDdot(N, t, 1, r, 1);
		temp2 = cublasDdot(N, t, 1, t, 1);
		omega = temp/temp2;

		//x = x+ omega*s
		//cublasSaxpy(N, omega, s, 1, x, 1);
		cublasDaxpy(N, omega, s_hat, 1, x, 1);

		//r = s-omega*t
		cublasDaxpy(N, -omega, t, 1, r, 1);

		err = cublasDnrm2(N, r, 1)/bnorm;
		if (err<=Tol) {
			resid = cublasDnrm2(N, s, 1)/bnorm;
			break;
		}
		if (omega == 0.0)
			break;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"BiCG Stab: "<<elapsedTime<<" ms"<<endl;

	if (err <= Tol || snorm/bnorm < Tol)
		flag = 0;
	else if (omega == 0.0)
		flag = -2;
	else if (rho == 0)
		flag = -1;
	else 
		flag = 1;

	if (!flag) 
		cout<<"The solution converged with residual "<<resid<<" in "<<iter<<" iterations."<<endl;
	else
		cout<<"BiCGStab produced error "<<flag<<" after "<<iter<<" iterations."<<endl;
}

void powerMethod(int *csrRowPtrA, int *csrColIndA, double *csrValA, cusparseMatDescr_t descr_A, int nnz, double *x, double *dev_x, double *dev_c, cusparseHandle_t handle, double* eigen, int n)
{
	double temp, invEigen;
	int d, i=0;
	double numOne = 1.0;
	double numZero = 0.0;
	do {
		i++;
		//c = Ax ... MV mult
		//cublasSgemv('n', n, n, 1, dev_a, n, dev_x, 1, 0, dev_c, 1);
		cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &numOne, descr_A, csrValA, csrRowPtrA, csrColIndA, dev_x, &numZero, dev_c);
		
		//copy c[] to x[]
		cublasDcopy(n, dev_c, 1, dev_x, 1);
		temp = *eigen;

		//get max value in result x[]
		d = cublasIdamax(n, dev_x, 1); 
		cudaMemcpy(x, dev_x, n*sizeof(double), cudaMemcpyDeviceToHost);
		*eigen = x[d-1];
		//factorize largest value out, obtain next vector
		invEigen = 1.0/(*eigen);
		cublasDscal(n, invEigen, dev_x, 1);
	} while (fabs((*eigen)-temp)>0.00001);
}
