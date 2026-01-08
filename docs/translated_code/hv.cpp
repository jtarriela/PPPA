// Compute HV for an n-by-d matrix
// Already normalized
// Note: this modifies the original points
// TODO: improve reference point selection: currently uses min/max; could use a minimum split point to reduce sorting issues

#include "Utils.h"
#include "math.h"
#include "LinkList.h"
#include "global.h"
#include "Approximate.cpp"

// Test-only; remove in real use (debug)
//#include<iostream>
//using namespace std;


double hv(double ** p, int nrP,int dim, int Samp, int layer)
{
	if (nrP == 0)
	{
		return 0;
	}
	//if (nrP <= 2)


	if (layer >= numK)
	{
		return Approximate(p, nrP, dim, Samp);
	}


	if (nrP <= 2)
	{
		if (nrP == 2)
		{
			//double * Min = new double[dim];
			double Min = 1; double F = 1; double G = 1;
			for (int i = 0; i < dim;i++ )
			{
				Min *= min( p[0][i] , p[1][i] );
				F *= p[0][i]; G *= p[1][i];
			}
			return F + G - Min;
		}
		//else
		else if (nrP == 1)
		{
			double F = 1;
			for (int i = 0; i < dim; i++)
			{
				F *= p[0][i];
			}
			return F;
		}
	}
	else
	{

		double *tempP = new double[nrP];
		for (int i = 0; i < nrP; i++)
		{
			tempP[i] = 1;
			for (int j = 0; j < dim; j++)
			{
				tempP[i] *= p[i][j];
			}
		}
		int MIndex = Max_Index(tempP, nrP);  //cout << MIndex << endl;




		//double *RP = new double[dim];
		double *RP = p[MIndex];
		double F = 1;
		for (int i = 0; i < dim; i++)
		{
			F *= RP[i];
			//cout << RP[i] <<" ";
		}

		for (int i = 0; i < dim; i++)
		{
			LinkList<double *> Subp_ind;
			LinkList<double *> *Aa = &Subp_ind;
			//LinkList<double *> *Subp_ind = new LinkList<double *>;
			//LinkList<double *> *Aa = Subp_ind;
			int len = 0;
			// Add split point and partition
			for (int j = 0; j < nrP; j++)
			{
				if (p[j][i] > RP[i])
				{
					p[j][i] = p[j][i] - RP[i];
					double *Subp = new double[dim];
					for (int k = 0; k < dim; k++)
					{
						Subp[k] = p[j][k];
					}
					p[j][i] = RP[i];
					Insert_ele(Aa,Subp);
					len++;
				}
			}
			if (len != 0)
			{
				//Aa = Subp_ind -> next;
				Aa = Subp_ind.next;
				double ** Sub_Region = new double*[len];
				for (int j = 0; j < len; j++, Aa = Aa->next)
				{
					Sub_Region[j] = Aa->data;
				}
				F += hv(Sub_Region, len, dim,Samp,layer+1);
				//delete[] Sub_Region;


				// Free memory
				Aa = &Subp_ind;
				for (int j = 0; j < len; j++)
				{
					LinkList<double *> *e = Aa->next;
					Aa->next = Aa->next->next;
					delete[] e->data;
					delete e;
				}
				delete[] Sub_Region, Aa;

			}
		}

		delete[] tempP, RP;

		return F;
	}

}




// Select reference point using the minimum-split method; not used because sorting equal points is tricky and the complexity is high
//int *Num_DD = new int[nrP];
//int *index = new int[nrP];
//double *tempP = new double[nrP];
//
//
//
//for (int j = 0; j < nrP; j++)
//{
//	index[j] = j; tempP[j] = p[j][0];
//}
//
////test
//for (int j = 0; j < nrP; j++)
//{
//	cout << tempP[j] << endl;
//}
//
//
//quickSort(tempP, index, 0, nrP - 1);
//
//for (int j = 0; j < nrP; j++)
//{
//	Num_DD[index[j]] = nrP - 1 - j;
//}
//for (int i = 1; i < dim; i++)
//{
//	// Quick sort
//	for (int j = 0; j < nrP; j++)
//	{
//		index[j] = j; tempP[j] = p[j][i];		// index starts at 0
//	}
//
//	//test
//	//for (int j = 0; j < nrP; j++)
//	//{
//	//	cout << tempP[j] << endl;
//	//}
//
//
//	quickSort(tempP, index, 0, nrP - 1);
//
//
//	//test
//	for (int j = 0; j < nrP; j++)
//	{
//		cout << tempP[j] << endl;
//	}
//
//
//
//
//	for (int j = 0; j < nrP; j++)
//	{
//		Num_DD[index[j]] += nrP - 1 - j;
//	}
//}
//
//
//int MIndex = Min_Index(Num_DD, nrP);
//
//
////test
//for (int j = 0; j < nrP; j++)
//{
//	cout << Num_DD[j] << endl;
//}

// Test subregion points (debug)
//for (int i = 0; i < nrP; i++)
//{
//	for (int j = 0; j < dim; j++)
//	{
//		cout << p[i][j] << " ";
//	}
//	cout << endl;
//}
//cout << endl;
