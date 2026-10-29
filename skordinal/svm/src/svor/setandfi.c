/*******************************************************************************\

	setandfi.c in Sequential Minimal Optimization ver2.0
		
	calculates Fi and assign Set Name according to alphas. 

	Chu Wei Copyright(C) National Univeristy of Singapore
	Create on Jan. 16 2000 at Control Lab of Mechanical Engineering 
	Update on Aug. 23 2001 

\*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#ifndef __MACH__
    #include <malloc.h>
#endif
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/types.h> 
#include <sys/timeb.h>
#include "smo.h"


#define PI				(3.141592654)

/*******************************************************************************\

	double Calculate_Fi ( long unsigned int i, smo_Settings * settings )
	
	calculate Fi for input index i, which is defined as Fi=yi-fi
	input:  index i in Data_List Pairs, and the pointer to smo_Settings 
	output: the value of Fi

\*******************************************************************************/

double Calculate_Fi ( long unsigned int i, smo_Settings * settings )/*/ i is index here*/
{
	Alphas * ai ;
	Alphas * aj ;
	Data_Node * Pi ;
	Data_Node * Pj ;
    double Fi = 0 ;
	long unsigned int j = 0 ;	

	
	if ( NULL == settings || i <= 0 )
	{
		printf ("\r\nFATAL ERROR : input pointer is NULL in Calc_Fi.\r\n") ;	
		return 0 ;
	}

	if ( i > settings->pairs->count )
	{
		printf ("\r\nFATAL ERROR : input index exceed the count of Pairs in Calc_Fi.\r\n") ;	
		return 0 ;
	}

	ai = ALPHA + i - 1 ;
	Pi = ai->pair ;
	Pj = settings->pairs->front ;
	
	while ( Pj != NULL )
	{		
		aj = ALPHA + j ;
		if ( aj->alpha != 0 )
			Fi = Fi + (aj->alpha) * Calc_Kernel( aj, ai, settings ) ;
		Pj = Pj->next ;
		j++ ;
	}

	/*/ai->pair->guess = Fi ;*/

#ifdef SMO_DEBUG
	if ( j != settings->pairs->count ) 
		printf ( "Error in Calculate Fi \n" ) ;
#endif

	Fi = Pi->target - Fi ;

	return Fi ;

} /*/ end of Caculate_Fi*/


double Calculate_Ordinal_Fi ( long unsigned int i, smo_Settings * settings )/*/ i is index here*/
{
	Alphas * ai ;
	Alphas * aj ;
	Data_Node * Pj ;
    double Fi = 0 ;
	double alpha ;
	long unsigned int j = 0 ;	
	long unsigned int k ;

	if ( NULL == settings || i <= 0 )
	{
		printf ("\r\nFATAL ERROR : input pointer is NULL in Calc_Fi.\r\n") ;	
		return 0 ;
	}

	if ( i > settings->pairs->count )
	{
		printf ("\r\nFATAL ERROR : input index exceed the count of Pairs in Calc_Fi.\r\n") ;	
		return 0 ;
	}

	ai = ALPHA + i - 1 ;
	Pj = settings->pairs->front ;
	
	while ( Pj != NULL )
	{		
		aj = ALPHA + j ;
		if (IMPLICIT_CONSTRAINTS == CONSTRAINTS)
		{
			/*/ there is no usable cached scalar here: alpha_j keeps changing
			   through the SMO loop, so the signed sum is recomputed*/
			alpha = 0 ;
			for (k=0;k<settings->pairs->classes-1;k++)
			{
				if (aj->pair->target<=k+1)
					alpha -= aj->alpha_j[k] ;
				else
					alpha += aj->alpha_j[k] ;
			}
			if ( alpha != 0 )
				Fi = Fi + alpha * Calc_Kernel( aj, ai, settings ) ;
		}
		else if ( aj->alpha != 0 )
			Fi = Fi + (-aj->alpha_up+aj->alpha_dw) * Calc_Kernel( aj, ai, settings ) ;
		Pj = Pj->next ;
		j++ ;
	}

#ifdef _ORDINAL_DEBUG
	if ( j != settings->pairs->count ) 
		printf ( "Error in Calculate Fi \n" ) ;
#endif
	return Fi ;

} /*/ end of Caculate_Ordinal_Fi*/

Set_Name Get_DW_Label ( Alphas * alpha, smo_Settings * settings)
{
	double a ;

	if ( NULL == alpha || NULL == settings )
	{
		printf("\r\nFATAL ERROR: input is NULL in Get_Label.\r\n");
		return I_o ;
	}

	if ( alpha->alpha_dw > settings->vc ) 
	{		
		if (alpha->alpha_dw > settings->vc+EPS)
			printf("\r\nWarning: alpha %f is greater than u=%f in Get_DW_Label.\r\n", alpha->alpha_dw,settings->vc);
		alpha->alpha_dw = settings->vc ;
	}
	if ( alpha->alpha_dw < 0 )
	{
		if (alpha->alpha_dw < -EPS)		
			printf("\r\nWarning: alpha %f is less than l=%d in Get_DW_Label.\r\n", alpha->alpha_dw,0);
		alpha->alpha_dw = 0 ;
	}

/*       if ( l*u > 0 )
{
if (l<u)
if (u>l)
{alpha->alpha_up=u-l;
alpha->alpha_dw=0;}
else
{alpha->alpha_dw=l-u;
alpha->alpha_up=0;}

//              printf("Warning: alpha_up * alpha_dw  > 0 ---- %d.\n",alpha-ALPHA+1) ;^M
}*/

	a = alpha->alpha_dw ; 

	if (1 == alpha->pair->target)
		return I_One ;
	
	/*if ( fabs(a - u)<EPS*EPS )				return I_Two ;
	else if ( fabs(l - a)<EPS*EPS )		return I_Thr ;*/
	if ( fabs(settings->vc - a)<EPS*EPS )				return I_Fou ;
	else if ( fabs(a)<EPS*EPS )						return I_One ;
	else if ( a > 0 && a < settings->vc )	return Io_b ;
	else
	{
		printf ( "\r\nFATAL ERROR : wrong alpha in Get_Label. %d \r\n", (int)(alpha-ALPHA) ) ;		
	    return I_o ;		
	}

} /*/ end of Get_Setname */


Set_Name Get_UP_Label ( Alphas * alpha, smo_Settings * settings)
{
	double a ;

	if ( NULL == alpha || NULL == settings )
	{
		printf("\r\nFATAL ERROR: input is NULL in Get_Label.\r\n");
		return I_o ;
	}
	
	if ( alpha->alpha_up > settings->vc ) 
	{
		if (alpha->alpha_up > settings->vc+EPS)		
			printf("\r\nWarning: alpha %f is greater than u=%f in Get_UP_Label.\r\n", alpha->alpha_up,settings->vc);
		alpha->alpha_up = settings->vc ;
	}
	if ( alpha->alpha_up < 0 )
	{		
		if (alpha->alpha_up < -EPS)
			printf("\r\nWarning: alpha %f is less than l=%d in Get_UP_Label.\r\n", alpha->alpha_up,0);
		alpha->alpha_up = 0 ;
	}
	a = alpha->alpha_up ; 

	if (alpha->pair->target == settings->pairs->classes)
		return I_Two ;	
	
	/*if ( fabs(a - u)<EPS*EPS )				return I_Two ;
	else if ( fabs(l - a)<EPS*EPS )		return I_Thr ;*/
	if ( fabs(settings->vc - a)<EPS*EPS )				return I_Thr ;
	else if ( fabs(a)<EPS*EPS )						return I_Two ;
	else if ( a > 0 && a < settings->vc )	return Io_a ;
	else
	{
		printf ( "\r\nFATAL ERROR : wrong alpha in Get_Label. %u \r\n", (int)(alpha-ALPHA) ) ;		
	    return I_o ;		
	}
} 
/*******************************************************************************\

	Set_Name Get_Ordinal_Label ( Alphas * alpha, unsigned int j, smo_Settings * settings)

	assign a Set_Name associated with j-th threshold for the input alpha
	input: the pointer to alpha structure, the threshold index and the pointer to smo_Settings
	output: Set_Name is assigned

\*******************************************************************************/

Set_Name Get_Ordinal_Label ( Alphas * alpha, unsigned int j, smo_Settings * settings)
{
	if ( NULL == alpha || NULL == settings )
	{
		printf("\r\nFATAL ERROR: input is NULL in Get_Ordinal_Label.\r\n") ;
		return I_o ;
	}
	if (j>=settings->pairs->classes||j<=0)
	{
		printf("\r\nFATAL ERROR: threshold index is out of region in Get_Ordinal_Label.\r\n") ;
		return I_o ;
	}

	if (alpha->alpha_j[j-1]>settings->vc)
	{
		if (alpha->alpha_j[j-1]>settings->vc+EPS)
			printf("\r\nWarning : alpha %f is greater than C.\r\n", alpha->alpha_j[j-1]) ;
		alpha->alpha_j[j-1]=settings->vc ;
	}
	else if (alpha->alpha_j[j-1]<0)
	{
		if (alpha->alpha_j[j-1]<-EPS)
			printf("\r\nWarning : alpha %f is less than 0.\r\n", alpha->alpha_j[j-1]) ;
		alpha->alpha_j[j-1]=0 ;
	}

	if ( alpha->pair->target > j )
	{

		if ( fabs(settings->vc - alpha->alpha_j[j-1])<EPS*EPS*EPS )	return I_Fou ;
		else if ( fabs(alpha->alpha_j[j-1])<EPS*EPS*EPS )				return I_One ;
		else return Io_b ;
	}
	else
	{

		if ( fabs(settings->vc - alpha->alpha_j[j-1])<EPS*EPS*EPS )	return I_Thr ;
		else if ( fabs(alpha->alpha_j[j-1])<EPS*EPS*EPS )				return I_Two ;
		else return Io_a ;
	}
} /*/ end of Get_Ordinal_Label*/

BOOL Is_Io ( Alphas * alpha, smo_Settings * settings )
{
	unsigned int i ;
	if (NULL == alpha || NULL == settings)
	{
		printf("\r\nFATAL ERROR : input pointer is NULL.\r\n") ;
		return FALSE ;
	}
	for (i=0;i<settings->pairs->classes-1;i++)
	{
		if (Io_a == alpha->setname[i] || Io_b == alpha->setname[i])
			return TRUE ;
	}
	return FALSE ;
} /*/ end of Is_Io*/

/* end of Get_Setname
 end of file setandfi.c */
