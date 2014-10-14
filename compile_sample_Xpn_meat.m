
mex CFLAGS="-std=c99 -fPIC" -O -largeArrayDims sample_Xpn_meat.c -lmwblas
mex CFLAGS="-std=c99 -fPIC" -O -largeArrayDims sample_Xpn_kernel_meat.c -lmwblas
