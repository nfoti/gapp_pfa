
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "matrix.h"
#include "mex.h"
#include "blas.h"

#define NINPUTS 10
#define NOUTPUTS 1

#define ZIN 0
#define THETAIN 1
#define PHIIN 2
#define PSIIN 3
#define KMIN 4
#define WIN 5
#define INDSIN 6
#define VALSIN 7
#define KINDSIN 8
#define RPIN 9

// Helper functions
double sum(const double *v, unsigned int nn);
void cumsum(double *cc, const double *pp, unsigned int nn);
unsigned int discrnd(const double *pp, unsigned int nn, double *cump, double eps);
double normcdf(double *x, unsigned int n);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    size_t P, N, Lp1, nrow_inds;
    size_t K;
    double *Znk, *Theta, *Phi, *Psi, *W, *inds, *vals, *rp;
    double *Zout;
    mxArray *Km, *kinds;
    double eps;

    if(nlhs != NOUTPUTS)
        mexErrMsgTxt("Usage: see \"help sample_Znk\"");
    if(nrhs != NINPUTS)
      mexErrMsgTxt("Usage: see \"help sample_Znk\"");

    // Grab inputs
    Znk = mxGetPr(prhs[ZIN]);  // Actually don't neet this
    Theta = mxGetPr(prhs[THETAIN]);
    Phi = mxGetPr(prhs[PHIIN]);
    Psi = mxGetPr(prhs[PSIIN]);
    Km = (mxArray*)prhs[KMIN];  // Cell array
    W = mxGetPr(prhs[WIN]);
    inds = mxGetPr(prhs[INDSIN]);
    vals = mxGetPr(prhs[VALSIN]);
    kinds = (mxArray*)prhs[KINDSIN];
    rp = mxGetPr(prhs[RPIN]);

    P = mxGetM(prhs[PHIIN]);
    N = mxGetN(prhs[PSIIN]);
    K = mxGetN(prhs[PHIIN]);
    Lp1 = mxGetM(prhs[WIN]);
    // Total number of rows in inds so that we can get the 2nd column
    nrow_inds = mxGetM(prhs[INDSIN]);

    //for(int k = 0; k < K; k++)
    //    mexPrintf("\tkinds_mat: %p\n", mxGetCell(kinds, k));
    
    //mexPrintf("P: %d, N: %d, K: %d\n", P, N, K);
    
    // Allocate output matrix Zout
    plhs[0] = mxCreateDoubleMatrix(N, K, mxREAL); 
    Zout = mxGetPr(plhs[0]);
    //memset(Zout, 0, N*K*sizeof(double));
    
    // Utilities
    eps = mxGetEps();
    char *chn = "N";  // don't transpose for dgemv
    char *cht = "T";  // do transpose for dgemv
    double onef = 1.0, zerof = 0.0; // alpha, beta in dgemm
    size_t onei = 1;  // incx in dgemv

    double zcode[] = {0.0, 0.0, 1.0};

    double *phi = mxMalloc(N*sizeof(double));
    
    //plhs[0] = mxCreateNumericMatrix(N, 1, mxINT32_CLASS, mxREAL);
    //unsigned int *Xpnk_sum = (unsigned int*)mxGetPr(plhs[0]);
    unsigned int *Xpnk_sum = mxMalloc(N*sizeof(unsigned int));
    
    //plhs[0] = mxCreateDoubleMatrix(P, N, mxREAL);
    //double *rate = mxGetPr(plhs[0]);
    double *rate = mxMalloc(P*N*sizeof(double));
    
    //plhs[0] = mxCreateDoubleMatrix(1, N, mxREAL);
    //double *ratesum = mxGetPr(plhs[0]);
    double *ratesum = mxMalloc(N*sizeof(double));

    double *onesP = mxMalloc(P*sizeof(double));
    for(int ii = 0; ii < P; ii++)
        onesP[ii] = 1.0;
    
    for(int k = 0; k < K; k++)
    {
        // Set topic counts to 0
        memset(Xpnk_sum, 0, N*sizeof(unsigned int));
        // Get indices into vals where inds(:,3) == k 
        mxArray *kinds_mat = mxGetCell(kinds, k);
        //mexPrintf("kinds_mat: %p\n", kinds_mat);
        if(kinds_mat != NULL)
        {
            double *xpnk_indsk = mxGetPr(kinds_mat);  // Will cast when used
            int num_indsk = mxGetM(kinds_mat);
        
            //mexPrintf("xpnk_indsk[0]: %f\n", xpnk_indsk[0]);
        
            // Xpnk_sum to determine if we set Zout to 1 by default
            //memset(Xpnk_sum, 0, N*sizeof(int));
            for(int c = 0; c < num_indsk; c++)
            {
                // Entries of kinds_mat and inds are 1-based, so fix them for C
                unsigned int xpnk_i = (unsigned int)xpnk_indsk[c] - 1;
                unsigned int nind = inds[xpnk_i + nrow_inds] - 1;
                //unsigned int pind = inds[xpnk_i] - 1;
                Xpnk_sum[nind] += vals[xpnk_i];
            }
        }
        
        // Activation function
        double *Km_k = mxGetPr(mxGetCell(Km, k));
        dgemv(chn, &N, &Lp1, &onef, Km_k, &N, &W[k*Lp1], &onei, &zerof, phi, &onei); 
        normcdf(phi, N);
        
        // Compute rate matrix for this topic
        memset(rate, 0, P*N*sizeof(double));
        dger(&P, &N, &Theta[k], &Phi[P*k], &onei, &Psi[k], &K, rate, &P);
        
        // sum(rate)
        dgemv(cht, &P, &N, &onef, rate, &P, onesP, &onei, &zerof, ratesum, &onei);

        for(int ri = 0; ri < N; ri++)
        {
            unsigned int n = (unsigned int)rp[ri] - 1;
            
            if(Xpnk_sum[n] > 0)
            {
                Zout[N*k + n] = 1;
                continue;
            }

            double p[3];
            double phi_n = phi[n];
            double ratesum_n = ratesum[n];
            double lPhix = log(phi_n + eps);
            double lPhix_c = log(1 - phi_n + eps);

            double denom = log(exp(-ratesum_n)*phi_n + (1-phi_n) + eps);
            p[0] = -ratesum_n + lPhix_c - denom;
            p[1] = log(1-exp(-ratesum_n)+eps) + lPhix_c - denom;
            p[2] = -ratesum_n + lPhix - denom;

            p[0] = exp(p[0]);
            p[1] = exp(p[1]);
            p[2] = exp(p[2]);
            
            //mexPrintf("p[0] = %.3f, p[1] = %.3f, p[2] = %.3f\n", p[0], p[1], p[2]);

            double cump_scr[] = {0.0, 0.0, 0.0};
            //mexPrintf("BEFORE: c[0]: %.3f, c[1]: %.3f, c[2]: %.3f\n", cump_scr[0], cump_scr[1], cump_scr[2]);
            unsigned int r = discrnd(p, 3, &cump_scr[0], eps);
            //mexPrintf("AFTER: c[0]: %.3f, c[1]: %.3f, c[2]: %.3f\n", cump_scr[0], cump_scr[1], cump_scr[2]);
            
            //mexPrintf("r: %d, zcode[r]: %f\n", r, zcode[r]);
                    
            Zout[N*k + n] = zcode[r];

        }

    }
    
    // Clean up
    if(phi)
        mxFree(phi);
    if(Xpnk_sum)
        mxFree(Xpnk_sum);
    if(rate)
        mxFree(rate);
    if(ratesum)
        mxFree(ratesum);
    if(onesP)
        mxFree(onesP);

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
// !! Labels are 0:(nn-1) as they're used as indices.  THIS IS DIFFERENT FROM
// THE DISCRND IN SAMPLE_XPNK_MEAT!!
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
      return i;
  }
  mexErrMsgTxt("Did not sample discrte random variable!!\n");
  return -1;
}
//
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
