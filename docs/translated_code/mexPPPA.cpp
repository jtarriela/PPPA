#include <WINDOWS.H>      
#include<iostream>
#include <mex.h>
#include "time.h"
#include"global.h"

//#include"HV_selection.h"

#include "hv.cpp"

//#include"HV_selection.cpp"
//#include"global.cpp"



void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
	//int Dim, popsize, RefSize;
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
            Samp = *mxGetPr(prhs[2]);
        }
        else
        {
            Samp = 10000000;
        }
        
        if(nrhs >= 4)
        {
            numK = *mxGetPr(prhs[3]);
        }
        else
        {
            numK = 6;
        }

		pop = mxGetM(prhs[0]);		//行数
		dim = mxGetN(prhs[0]);

		plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
		Output = mxGetPr(plhs[0]);
        if (pop == 0) { *Output = 0; return; }

		double ** Point_ = new double*[pop];		//矩阵，知道行列的
		
		
        int i,j;
		for (i = 0; i < pop; i++)
		{
			Point_[i] = new double[dim];
			for (j = 0; j < dim; j++)
			{
				Point_[i][j] = bounds[j] - Points[j * pop + i];
				if (Point_[i][j] < 0) {
					//mexPrintf("参数超过边界\n");
					Point_[i][j] = 0;
				}
			}
		}
        

		*Output = hv(Point_, pop,dim, Samp,1);
		//*Output = Approximate(Point_, pop, dim, Samp,numK);
		
		for (int i = 0; i < pop; i++)
		{
			delete[] Point_[i];
		}
		delete[] Point_;

	}
	else
	{
		//plhs[0] = mxCreateDoubleMatrix(20, 1, mxREAL);
		mexPrintf("抱歉，输入参数有误\n");
	}

}
