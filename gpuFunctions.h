#include <fstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <complex>
#include <cusparse.h>
#include <cublas.h>
#include <cuda_profiler_api.h>
#include <thrust\sort.h>
#include <thrust\device_ptr.h>
#include <thrust\gather.h>
#include <thrust\iterator\counting_iterator.h>
#include <thrust\device_vector.h>

void optionSelect(int option);
void answerSelect();
int IEEEStandardBusSystems(int numberOfBuses, std::ifstream &infile, std::ifstream &lineData, int numLines);

//Setup Functions
__global__ void createYBusSparse(int numLines, int numberOfBuses, int *fromBus, int* toBus, cuDoubleComplex *Z, cuDoubleComplex *B, cuDoubleComplex *y, int *yrow, int *ycol);
__global__ void powerEqnSparse(double *P, double *Q, cuDoubleComplex* y, int* yrow, int* ycol, double *Vm, double *theta, int ynnz);

__global__ void createYBusSparseConcise(int numLines, int numberOfBuses, int *fromBus, int* toBus, cuDoubleComplex *Z, cuDoubleComplex *B, cuDoubleComplex *y, int *yrow, int *ycol);
__global__ void powerEqnSparseConcise(double *P, double *Q, cuDoubleComplex* y, int* yrow, int* ycol, double *Vm, double *theta, int ynnz, int numLines);

__global__ void createJ11(int ynnz, int numLines, int numSlackLines, int slackBus, double* P, double *Q, double *Vmag, double *Vang, cuDoubleComplex *y, int *yrow, int *ycol, double *jac, int *jacRow, 
	int *jacCol);

__global__ void createJ11Copy(int ynnz, int slackBus, double* P, double *Q, double *Vmag, double *Vang, cuDoubleComplex *y, int *yrow, int *ycol, double *jac, int *jacRow, 
	int *jacCol);
__global__ void createJ12_J21(int ynnz, int slackBus, int numBus, double* P, double *Q, double *Vmag, double *Vang, cuDoubleComplex *y, int *yrow, int *ycol, 
	double *jac, int *jacRow, int *jacCol, int* PQbuses, int N_p, bool* boolRow, bool* boolCol, int* J22row, int* J22col);
__global__ void createJ22(int ynnz, int numBus, double* P, double *Q, double *Vmag, double *Vang, cuDoubleComplex *y, int *yrow, int *ycol, double *jac, int *jacRow, 
	int *jacCol, bool* boolRow, bool* boolCol, int* J22row, int* J22col);

__global__ void countNnzJac(bool* boolCheck, int *yrow, int* ycol, cuDoubleComplex *y, int *PVbuses, int N_g, int slackBus, int* dev_J12count, int* dev_J22count);

//Solution of Linear System
void biCGStab(cusparseStatus_t status, cusparseHandle_t handle, cusparseMatDescr_t descr_A, cusparseMatDescr_t descr_L, cusparseMatDescr_t descr_U, 
	cusparseSolveAnalysisInfo_t info_L, cusparseSolveAnalysisInfo_t info_U,	int M, int N, int nnz, int* csrColIndAdev, int* csrRowPtrAdev, 
	double* csrValAdev, int* csrColIndPre, int* csrRowPtrPre, double* csrValPre, 	double* x, double* r, double* r_tld, double* p, double *p_hat, 
	double* s, double *s_hat, double* v, double* t, double* b);

void biCGStab2(cusparseStatus_t status, cusparseHandle_t handle, cusparseMatDescr_t descr_A, int M, int N, int nnz, int* csrColIndAdev, int* csrRowPtrAdev, double* csrValAdev, 
	double* x, double* r, double* r_tld, double* p, double *p_hat, double* s, double *s_hat, double* v, double* t, double* b, int jacSize);

//Auxiliary Functions
__global__ void updateX(int jacSize, int N, int *PQindex, double *Vmag, double *theta, double *stateVector, double *x);
__global__ void updateMismatch(int N, int jacSize, double *P_eq, double *Q_eq, int *PQindex, double* PQcalc, double* PQspec, double *powerMismatch);
__global__ void jacobiPrecond(int jacSize, double *jacobian, double *preconditioner);
__device__ double radToDeg(double a);
__device__ void atomicAddComplex(cuDoubleComplex *a, cuDoubleComplex b);
__device__ double atomicAdd2(double* address, double val);


//Chebyshev
void powerMethod(int *csrRowPtrA, int *csrColIndA, double *csrValA, cusparseMatDescr_t descr_A, int nnz, double *x, double *dev_x, double *dev_c, cusparseHandle_t handle, double* eigen, int n);



__global__ void createJ11Copy(int ynnz, int slackBus, double* P, double *Q, double *Vmag, double *Vang, cuDoubleComplex *y, int *yrow, int *ycol, double *jac, int *jacRow, 
	int *jacCol);


__global__ void createJacobianSparse2(int ynnz, int jacCount, int slackBus, int numBus, double* P, double *Q, double *Vmag, double *Vang, cuDoubleComplex *y, int *yrow, int *ycol, double *jac, int *jacRow, int *jacCol,
	int* PQindex, int N_g, int N_p, int jacSize);

__global__ void createJacobianSparse(int ynnz, int jacCount, int slackBus, int numBus, double* P, double *Q, double *Vmag, double *Vang, cuDoubleComplex *y, int *yrow, int *ycol, double *jac, int *jacRow, int *jacCol);

__global__ void createJacobianSparse3(int ynnz, int slackBus, int numBus, double* P, double *Q, double *Vmag, double *Vang, cuDoubleComplex *y, int *yrow, int *ycol, 
	double *jac, int *jacRow, int *jacCol, int* PQbuses, int N_p, bool* boolRow, bool* boolCol,int* J22row, int* J22col);