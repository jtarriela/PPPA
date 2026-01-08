// Linux-compatible version of mexPPPA.cpp
#include <iostream>
#include <mex.h>
#include <ctime>
#include "global.h"
#include "hv.cpp"

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
	double *Output;
	double *bounds;
	double *Points;
	int Samp;

	if (nrhs >= 2)
	{
        Points = mxGetPr(prhs[0]);
		bounds = mxGetPr(prhs[1]);
        if(nrhs >= 3)
        {
            Samp = (int)*mxGetPr(prhs[2]);
        }
        else
        {
            Samp = 10000000;
        }

        if(nrhs >= 4)
        {
            numK = (int)*mxGetPr(prhs[3]);
        }
        else
        {
            numK = 6;
        }

		pop = mxGetM(prhs[0]);
		dim = mxGetN(prhs[0]);

		plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
		Output = mxGetPr(plhs[0]);
        if (pop == 0) { *Output = 0; return; }

		double ** Point_ = new double*[pop];

        int i, j;
		for (i = 0; i < pop; i++)
		{
			Point_[i] = new double[dim];
			for (j = 0; j < dim; j++)
			{
				Point_[i][j] = bounds[j] - Points[j * pop + i];
				if (Point_[i][j] < 0) {
					Point_[i][j] = 0;
				}
			}
		}

		*Output = hv(Point_, pop, dim, Samp, 1);

		for (int i = 0; i < pop; i++)
		{
			delete[] Point_[i];
		}
		delete[] Point_;
	}
	else
	{
		mexPrintf("Error: Not enough input arguments\n");
	}
}
