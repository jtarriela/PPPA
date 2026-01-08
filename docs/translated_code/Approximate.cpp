// Compute HV for an n-by-d matrix
// Already normalized
// Note: this modifies the original points
// TODO: improve reference point selection: currently uses min/max; could use a minimum split point to reduce sorting issues

#include "Utils.h"
#include "math.h"
#include "LinkList.h"

bool dom(double* a, double* b, int dim);

double Approximate(double ** p, int nrP, int dim, int Samp, int numK)
{
	LinkList<double *> Subp_ind;
	LinkList<double *> *Aa = &Subp_ind;
	int len = 0;

	double * volume = new double[nrP];
	for (int i = 0; i < nrP; i++)
	{
		volume[i] = 1;
		for (int j = 0; j < dim; j++)
		{
			volume[i] *= p[i][j];
		}
	}
	double maxV = Max_Vector(volume, nrP);


	for (int i = 0; i < nrP; i++)
	{
		int tmplen = len;
		Aa = &Subp_ind;
		for (int j = 0; j < tmplen; j++)
		{
			// Remove sampled points
			if (dom(p[i], Aa->next->data, dim))		// If the individual dominates the sample point, delete it (maximization)
			{
				LinkList<double *> *e = Aa->next;
				Aa->next = Aa->next->next;
				delete[] e->data;
				delete e;
				len--;
			}
			else
			{
				Aa = Aa->next;
			}
		}
		// Compute sample size
		//double SampSiz = Samp;		// Method 1
		//for (int j = 0; j < dim; j++)
		//{
		//	SampSiz *= p[i][j];
		//}

		// Method 2
		double SampSiz = ((double)Samp)*(volume[i] / maxV);


		// Add sample points
		for (int j = 0; j < SampSiz; j++)
		{
			double *tmpInd = new double[dim];
			for (int k = 0; k < dim; k++)
			{
				tmpInd[k] = drand(0, p[i][k]);
			}
			Insert_ele(Aa, tmpInd);
			len++;
		}
	}

	Aa = &Subp_ind;
	for (int j = 0; j < len; j++)
	{
		LinkList<double *> *e = Aa->next;
		Aa->next = Aa->next->next;
		delete[] e->data;
		delete e;
	}
	//delete[] Aa;

	delete[] volume;

	return ((double)len) / ((double)Samp/ maxV);
	//return ((double)len) / ((double)Samp);
}


bool dom(double* a, double* b, int dim)
{
	bool dom = true;
	for (int i = 0; i < dim; i++)
	{
		if (a[i] < b[i])
		{
			dom = false;
		}
	}
	return dom;
}
