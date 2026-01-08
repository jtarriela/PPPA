#include<iostream>


//产生的是一个在to和from之间的随机数
//double drand(double from, double to)
//{
//	double j;
//	j = from + (double)((to - from) * rand() / (RAND_MAX + 1.0));		
//	return j;
//}


//double Max_Vector(double *P, int len)
//{
//	double Max = P[0];
//	for (int i = 1; i < len; i++)
//	{
//		if (P[i] > Max)
//		{
//			Max = P[i];
//		}
//	}
//	return Max;
//}

//double Min_Vector(double *P, int len)
//{
//	double Min = P[0];
//	for (int i = 1; i < len; i++)
//	{
//		if (P[i] < Min)
//		{
//			Min = P[i];
//		}
//	}
//	return Min;
//}



//快速排序，包含排序的序号
void quickSort(double s[], int index[], int l, int r)
{
	if (l < r)
	{
		int i = l, j = r, x = s[l], y = index[l];
		while (i < j)
		{
			while (i < j && s[j] >= x) // 从右向左找第一个小于x的数  
				j--;
			if (i < j)
			{
				s[i] = s[j];		//这里注意，应该是先赋值了，再运行i++
				index[i++] = index[j];
			}

			while (i < j && s[i]< x) // 从左向右找第一个大于等于x的数  
				i++;
			if (i < j)
			{
				s[j] = s[i]; index[j--] = index[i];
			}
				
		}

		s[i] = x; index[i] = y;
		quickSort(s, index, l, i - 1); // 递归调用  
		quickSort(s, index, i + 1, r);
	}
}







