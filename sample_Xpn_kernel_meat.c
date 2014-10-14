
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "matrix.h"
#include "mex.h"
#include "blas.h"

#define NINPUTS 8
#define NOUTPUTS 3

#ifndef max
  #define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

double sum(const double *v, unsigned int nn);
void cumsum(double *cc, const double *pp, unsigned int nn);
unsigned int discrnd(const double *pp, unsigned int nn, double *cump, double eps);
void multirnd(double *out, unsigned int N, const double *pp, unsigned int nn, double *cump, double eps);
double normcdf(double *x, unsigned int n);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  size_t NN, X_sum, P, D, Lp1;
  mwSize K;
  double *X_inds, *X_vals;
  double *lPhi, *lTheta, *lPsi, *W;
  mxArray *Km;
  double *inds, *vals, *num_nz;
  double eps;

  if(nlhs != NOUTPUTS)
    mexErrMsgTxt("Usage: see \"help sample_Xpn_kernel_meat\"");
  if(nrhs != NINPUTS)
    mexErrMsgTxt("Usage: see \"help sample_Xpn_kernel_meat\"");

  // Grab inputs
  X_inds = mxGetPr(prhs[0]);
  X_vals = mxGetPr(prhs[1]);
  lPhi = mxGetPr(prhs[2]);
  lTheta = mxGetPr(prhs[3]);
  lPsi = mxGetPr(prhs[4]);
  W = mxGetPr(prhs[6]);
  X_sum = mxGetScalar(prhs[7]);
  Km = (mxArray*)prhs[5]; // Cell array

  NN = mxGetM(prhs[0]);
  P = mxGetM(prhs[2]);
  K = mxGetN(prhs[2]);
  D = mxGetN(prhs[4]);
  Lp1 = mxGetM(prhs[6]);
  
  eps = mxGetEps();
  
  // Allocate output
  plhs[0] = mxCreateDoubleMatrix(X_sum, 3, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(X_sum, 1, mxREAL);
  plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
  inds = mxGetPr(plhs[0]);
  vals = mxGetPr(plhs[1]);
  num_nz = mxGetPr(plhs[2]);
  
  // Pre-allocate scratch arrays
  unsigned int topic_sz = K*sizeof(double);
  double *zeta = (double*)mxMalloc(topic_sz);
  double *cump_scr = (double*)mxMalloc(topic_sz);
  double *mnr = (double*)mxMalloc(topic_sz);
  
  // log-thinning activation function, not Phi as in the topics
  // Change this to just a double*
  //mxArray *lphi = mxCreateDoubleMatrix(D, K, mxREAL);
  //mxArray *res;
  double *A, *lphi;
  char *chn = "N";  // don't transpose for dgemv
  double onef = 1.0, zerof = 0.0; // alpha, beta in dgemv
  size_t onei = 1; // incx in dgemv
  
  lphi = mxMalloc(D*K*sizeof(double));
  //lphi_ptr = mxGetPr(lphi);
  
  //printf("Computing lphi...\n");
  //printf(" K: %d\n", K);
  
  // Compute phi{k} = p(Znk = 1) for all k here (Each one is a matrix)
  for (int k = 0; k < K; k++)
  {
    //printf("k: %d ", k);
    A = mxGetPr(mxGetCell(Km, k));
    //printf("Before dgemv..");
    dgemv(chn, &D, &Lp1, &onef, A, &D, &W[k*Lp1], &onei, &zerof, &lphi[k*D], &onei); 
    //dgemv(chn, &D, &Lp1, &onef, A, &D, &W[k*Lp1], &onei, &zerof, &lphi_ptr[k*D], &onei); 
    //printf("after dgemv\n");
  }
  
  //printf("Sampling...\n");
  // Probit each entry and take log
  normcdf(lphi, D*K);
  for (int ii = 0; ii < D*K; ii++)              
    lphi[ii] = log(lphi[ii] + eps);
  
  // main loop
  unsigned int cur_idx = 0;
  unsigned int noff = X_sum;
  unsigned int koff = 2*X_sum;
  for(int ii = 0; ii < NN; ii++)
  {

    // Get indices and data (convert to 0-based)
    unsigned int p = X_inds[ii] - 1;
    unsigned int n = X_inds[ii + NN] - 1; // These arrays have NN rows
    unsigned int xx = X_vals[ii];

    // Construct probability
    double zeta_max = DBL_MIN;
    memset(zeta,0,topic_sz);
    for(int k = 0; k < K; k++)
    {
      // When we say N we mean D
      // lZnk is NxK and lPsi is KxN so we index them backwards
      // lphi is NxK and contains the log-thinning probabilities
      //zeta[k] = lPhi[p + k*P] + lTheta[k] + lZnk[n + k*D] + lPsi[k + n*K];
      zeta[k] = lPhi[p + k*P] + lTheta[k] + lphi[n + k*D] + lPsi[k + n*K];
      if(zeta[k] > zeta_max)
        zeta_max = zeta[k];
    }
    for(int k = 0; k < K; k++)
    {
      zeta[k] -= zeta_max;
      zeta[k] = exp(zeta[k]);
    }
    double zeta_sum = sum(zeta, K);
    for(int k = 0; k < K; k++)
      zeta[k] /= zeta_sum;

//     for(int k = 0; k < K; k++)
//       printf("%.2f,", zeta[k]);
//     printf("\n");
    
    // Split up observed counts into topics and add to output
    if(xx > 1)
    {
      multirnd(mnr, xx, zeta, K, cump_scr, eps);
      for(int k = 0; k < K; k++)
      {
        if(mnr[k] > 0)
        {
          inds[cur_idx] = p + 1;
          inds[cur_idx + noff] = n + 1;
          inds[cur_idx + koff] = k + 1; // +1 b/c k is index
          vals[cur_idx] = mnr[k];
          cur_idx++;
        }
      }
    }
    else
    {
      // Result of below is 1-based, so don't add 1 to kk below
      double kk = discrnd(zeta, K, cump_scr, eps);
      inds[cur_idx] = p + 1;
      inds[cur_idx + noff] = n + 1;
      inds[cur_idx + koff] = kk;
      vals[cur_idx] = 1;
      cur_idx++;
    }

  }

  // Free scratch memory
  mxFree(zeta);
  mxFree(cump_scr);
  mxFree(mnr);
  mxFree(lphi);

  *num_nz = cur_idx; // cur_idx has been incr. 1 beyond already

}

// Return sum of entries of a vector with nn entries
inline double sum(const double *v, unsigned int nn)
{
  double ss = 0;
  for(int i = 0; i < nn; i++)
    ss += v[i];
  return ss;
}

// Return cumsum of entries of pp in array cc both with nn entries
inline void cumsum(double *cc, const double *pp, unsigned int nn)
{
  cc[0] = pp[0];
  for(int i = 1; i < nn; i++)
    cc[i] = cc[i-1] + pp[i];
}

// Draw random variable from discrete distribution given by pp with nn entries.
// !! Labels are 1:nn, NOT 0:NN-1 since we don't use them as indices !!
// pp : array with probabilities
// nn : number of possible outcomes
// cump : array to store cumsum in (pre-allocated)
// eps : machine epsilon
inline unsigned int discrnd(const double *pp, unsigned int nn, double *cump, double eps)
{
  double u;

  cumsum(cump,pp,nn);
  cump[nn-1] = 1+eps;
  u = rand() / (double)RAND_MAX;
  u *= 1-eps;
  for(int i = 0; i < nn; i++)
  {
    if(u < cump[i])
      return (i+1);
  }
  mexErrMsgTxt("Did not sample discrte random variable!!\n");
  return -1;
}

// Draw multinomial variable with distribution pp with nn entries.
// This is just a port of the meat of Matlab's mnrnd function.
// out : array of uints to store result (pre-allocated).  Since returned in
//       array results will need to be made 1-based.  
// N : number of trials
// pp : array with probabilities
// nn : number of possible outcomes
// cump : array to store cumsum in (pre-allocated)
// eps : machine epsilon
inline void multirnd(double *out, unsigned int N, const double *pp, 
                     unsigned int nn, double *cump, double eps)
{
  double u;

  memset(out,0,nn*sizeof(double));
  cumsum(cump,pp,nn);
  cump[nn-1] = 1+eps;
  for(int n = 0; n < N; n++)
  {
    // Draw discrete r.v. according to pp and accumlate in entries of out
    u = rand() / (double)RAND_MAX;
    u *= 1-eps;
    for(int i = 0; i < nn; i++)
    {
      if(u < cump[i])
      {
        out[i] += 1;
        break;
      }
    }
  }
}

// Modifies the array in place
double normcdf(double* x, unsigned int n)
{
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;

    for (int i = 0; i < n; i++)
    {
      // Save the sign of x
      double xx = x[i];
      int sign = 1;
      if (xx < 0)
          sign = -1;
      xx = fabs(xx)/sqrt(2.0);

      // A&S (Handbook of mathematical functions) formula 7.1.26
      double t = 1.0/(1.0 + p*xx);
      double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-xx*xx);

      x[i] = 0.5*(1.0 + sign*y);
    }
}
