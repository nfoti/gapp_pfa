
mex CFLAGS="-std=c99 -fPIC" -O -largeArrayDims sample_Znk_meat.c -lmwblas
%mex CFLAGS="-std=c99 -fPIC" -g -largeArrayDims sample_Znk_meat.c -lmwblas
