#pragma once
#ifndef UTILS
#define UTILS
#define max(a,b)  (a>b)?a:b
#define min(a,b)  (a<b)?a:b
#define min3(a,b,c)  (a<b)? ((a<c)?a:c) : ((b<c)?b:c)

#include "LinkList.h"


double drand(double from, double to)
{
	double j;
	j = from + (double)((to - from) * rand() / (RAND_MAX + 1.0));
	return j;
}

//int Max_Index(double *P, int len);
//int Max_Index(int *P, int len);

double Max_Vector(double *P, int len)
{
	double Max = P[0];
	for (int i = 1; i < len; i++)
	{
		if (P[i] > Max)
		{
			Max = P[i];
		}
	}
	return Max;
}

double Min_Vector(double *P, int len)
{
	double Min = P[0];
	for (int i = 1; i < len; i++)
	{
		if (P[i] < Min)
		{
			Min = P[i];
		}
	}
	return Min;
}



template <typename T>
int Min_Index(T *P, int len)
{
	int MinI = 0;
	for (int i = 1; i < len; i++)
	{
		if (P[i] < P[MinI])
		{
			MinI = i;
		}
	}
	return MinI;
}

template <typename T>
int Max_Index(T *P, int len)
{
	int MaxI = 0;
	for (int i = 1; i < len; i++)
	{
		if (P[i] > P[MaxI])
		{
			MaxI = i;
		}
	}
	return MaxI;
}


//void Sort(double *, int *, int);		// sort index starts at 0; third param is array length; second param only needs declaration
void quickSort(double[], int [] , int , int);		// includes sorted indices, but requires a sequential index array up front


// Quick sort
template <typename T>
void quickSort(T s[], int l, int r)
{
	if (l < r)
	{
		int i = l, j = r, x = s[l];// , y = index[l];
		while (i < j)
		{
			while (i < j && s[j] >= x) // find the first number < x from right to left  
				j--;
			if (i < j)
			{
				s[i++] = s[j];		// Note: assign first, then run i++
				//index[i++] = index[j];
			}

			while (i < j && s[i]< x) // find the first number >= x from left to right  
				i++;
			if (i < j)
			{
				s[j--] = s[i]; //index[j--] = index[i];
			}
		}

		s[i] = x; //index[i] = y;
		quickSort(s, l, i - 1); // recursive call  
		quickSort(s, i + 1, r);
	}
}



// Modified insert function: insert an element, then return p as the address of the last node in the list
// Usage: LinkList<double *> *p = &RP; declare a linear list, then assign its address to p
// Here p changes, but the list itself does not
// NOTE: ensure the template type is not a reference; if it's an array, the inserted data is only a pointer, so list operations affect the original data
// Best practice: allocate a new array and copy the values before inserting; see hv.cpp lines 82-87
template <typename ElemType>
Status Insert_ele(LinkList<ElemType> *(&p), ElemType Col_ele)
{
	LinkList<ElemType> *s = new LinkList<ElemType>;
	s->data = Col_ele;
	s->next = p->next;
	p->next = s;
	p = s;
	return TRUE;
}


#endif
