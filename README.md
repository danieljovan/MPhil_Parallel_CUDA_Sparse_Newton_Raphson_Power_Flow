# MPhil_Parallel_CUDA_Sparse_Newton_Raphson_Power_Flow
This code shows a parallel (GPU-computing based) implementation of the NR load flow method, including: 
  - parallel formation of sparse admittance and Jacobian matrices 
  - computation of real and reactive power equations 
  - formation of the linear system of equations and 
  - solution of the resultant sparse linear system using a parallel implementation of an iterative Krylov subspace method (Preconditioned       Stabilized Bi-conjugate Gradient method)
