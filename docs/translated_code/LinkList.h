#ifndef LINKLIST_H
#define LINKLIST_H


#include<stdlib.h>
#define TRUE 1
#define FALSE 0
#define OK 1
#define ERROR 0
#define INFEASIBLE -1
#define OVERFLOW -2
typedef int Status;

template <typename ElemType>
struct LinkList {
	ElemType data;
	LinkList * next;

};		//如果我是声明LNode，那么就是一个变量，但是声明LinkList，那么就是一个地址指针


//从线性表L中获取第i个元素，这个元素是e

//Status Init_L(LinkList &L);								//初始化一个线性链表
//Status GetElem_L(LinkList L,int i,ElemType &e);			//获取元素
//Status ListInsert_L(LinkList &L,int i,ElemType e);		//插入元素，注意，插入和获取的位置指的是实际位置，跟matlab对应，而不是从0开始
template <typename ElemType>
Status Init_L(LinkList<ElemType> &L)			//可有可无
{
	LinkList<ElemType> *s = new LinkList<ElemType>;
	L = *s;
	if (!(&L)) exit(OVERFLOW);
	return OK;
}
template <typename ElemType>
Status GetElem_L(LinkList<ElemType> L, int i, ElemType &e)
{
	//LinkList<ElemType> * p = (&L)->next;
	LinkList<ElemType> * p = L.next;
	int j = 1;
	while (p && j < i)
	{
		p = p->next;
		++j;
	}
	if (!p || j > i) return ERROR;
	e = p->data;
	return OK;
}
template <typename ElemType>
Status ListInsert_L(LinkList<ElemType> &L, int i, ElemType e)
{
	//LNode * p = L;
	LinkList<ElemType> * p = &L;
	int j = 0;
	while (p && j < i - 1)
	{
		p = p->next;
		++j;
	}
	if (!p || j > i) return ERROR;
	//LinkList s = ( LinkList )malloc(sizeof(LNode) );
	LinkList<ElemType> *s = new LinkList<ElemType>;

	s->data = e; s->next = p->next;
	p->next = s;
	return OK;
}

//删除某index元素，返回该元素,此处的返回的元素e主要是用于构析函数等用来释放元素e的空间的
template <typename ElemType>
Status ListDelete_L(LinkList<ElemType> &L, int i, ElemType &e)		
{
	LinkList<ElemType> * p = &L; 
	LinkList<ElemType> * q;

	int j = 0;
	while (p && j < i - 1)
	{
		p = p->next;
		++j;
	}
	if (!p || j > i) return ERROR;

	q = p->next;
	e = q->data;
	p->next = q->next;
	delete q;		//注意，此处delete q，会把原来的属性里面的指针都delete掉了

	return OK;
}

#endif
