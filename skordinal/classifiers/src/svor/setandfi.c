/*******************************************************************************\

	setandfi.c in Sequential Minimal Optimization ver2.0
		
	calculates Fi and assign Set Name according to alphas. 

	Chu Wei Copyright(C) National Univeristy of Singapore
	Create on Jan. 16 2000 at Control Lab of Mechanical Engineering 
	Update on Aug. 23 2001 

\*******************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "smo.h"


/*******************************************************************************\

    double Calculate_Ordinal_Fi ( long unsigned int i, smo_Settings * settings )
    
    purpose: calculate Fi for input index i, which is defined as Fi=f(x_i), 
             handling both SVOREX and SVORIM model types.
    input:   index i in Data_List Pairs, and the pointer to smo_Settings 
    output:  the calculated value of Fi (double)

\*******************************************************************************/

double Calculate_Ordinal_Fi ( long unsigned int i, smo_Settings * settings ) {
	Alphas * ai ; Alphas * aj ;
	Data_Node * Pj ; Data_Node * Pi ;
	double alpha, Fi = 0 ;
	long unsigned int j = 0, k ;

	if ( NULL == settings || i <= 0 )
	{
		printf ("\nFATAL ERROR : input pointer is NULL in Calc_Fi.\n") ;	
		exit(1) ;
	}

	if ( i > settings->pairs->count )
	{
		printf ("\r\nFATAL ERROR : input index exceed the count of Pairs in Calc_Fi.\r\n") ;		
		exit(1) ;
	}

	ai = settings->alpha + i - 1 ;
	Pi = ai->pair ;
	Pj = settings->pairs->front ;
	
	while ( Pj != NULL ) {		
		aj = settings->alpha + j ;
		
		if (settings->model_type == 0) { // SVOREX
			if ( aj->alpha_up != 0 || aj->alpha_dw != 0) 
				Fi += (-aj->alpha_up+aj->alpha_dw) * Calc_Kernel( aj, ai, settings ) ;
		} else { // SVORIM
			alpha = 0 ;
			for (k=0;k<settings->pairs->classes-1;k++) {
				if (aj->pair->target<=k+1) alpha -= aj->alpha_ptr[k] ;
				else alpha += aj->alpha_ptr[k] ;
			}
			if ( alpha != 0 ) Fi += alpha * Calc_Kernel( aj, ai, settings ) ;
		}
		Pj = Pj->next ; j++ ;
	}
	return Fi ;
} /*/ end of Calculate_Ordinal_Fi */


/*******************************************************************************\

    Set_Name Get_Ordinal_Label ( Alphas * alpha, unsigned int j, smo_Settings * settings)
    
    purpose: assign a Set_Name associated with the j-th threshold for the input 
             alpha, validating its bounds against the C penalty parameter (SVORIM).
    input:   the pointer to alpha structure, the threshold index j, and the pointer 
             to smo_Settings 
    output:  the appropriate Set_Name is assigned and returned (Io_a, Io_b, I_One, etc.)

\*******************************************************************************/

Set_Name Get_Ordinal_Label ( Alphas * alpha, unsigned int j, smo_Settings * settings) {

	FILE * fid ;
	if ( NULL == alpha || NULL == settings )
	{
		printf("\r\nFATAL ERROR: input is NULL in Get_Label.\r\n") ;
		exit(1) ;
	}
	if (j>=settings->pairs->classes||j<=0)
	{
		printf("\r\nFATAL ERROR: threshold index is out of region in Get_Label.\r\n") ;
		exit(1) ;
	}

	if (alpha->alpha_ptr[j-1] > settings->vc) 
	{
		if (alpha->alpha_ptr[j-1]>settings->vc+EPS)
		{
		fid = fopen ("error_message.txt","a+t") ;
		if (NULL != fid)
		{
			fprintf(fid,"\nWarning : alpha %f is greater than C.\n", alpha->alpha_ptr[j-1]) ;
			fclose(fid) ;
		}
		printf("\nWarning : alpha %f is greater than C.\n", alpha->alpha_ptr[j-1]) ;
		}
		alpha->alpha_ptr[j-1] = settings->vc ;
		exit(1) ;
	}
	else if (alpha->alpha_ptr[j-1] < 0) 
	{
		if (alpha->alpha_ptr[j-1]<-EPS)
		{
			fid = fopen ("error_message.txt","a+t") ;
			if (NULL != fid)
			{
					fprintf(fid,"\nWarning : alpha %f is less than 0.\n", alpha->alpha_ptr[j-1]) ;
					fclose(fid) ;
			}
			printf("\nWarning : alpha %f is less than 0.\n", alpha->alpha_ptr[j-1]) ;
		}
		alpha->alpha_ptr[j-1] = 0 ;
		exit(1) ;
	}

	if ( alpha->pair->target > j ) {
		if ( fabs(settings->vc - alpha->alpha_ptr[j-1])<EPS*EPS*EPS ) return I_Fou ;
		else if ( fabs(alpha->alpha_ptr[j-1])<EPS*EPS*EPS ) return I_One ;
		else return Io_b ;		
	} else {
		if ( fabs(settings->vc - alpha->alpha_ptr[j-1])<EPS*EPS*EPS ) return I_Thr ;
		else if ( fabs(alpha->alpha_ptr[j-1])<EPS*EPS*EPS ) return I_Two ;
		else return Io_a ;
	}
} /*/ end of Get_Ordinal_Label */


// SVORIM Methods


/*******************************************************************************\

    BOOL Is_Io ( Alphas * alpha, smo_Settings * settings )
    
    purpose: determine if a given alpha belongs to the intermediate, unbound sets 
             (Io_a or Io_b) across any threshold classes.
    input:   the pointer to alpha structure and the pointer to smo_Settings 
    output:  returns TRUE if the alpha is in Io_a or Io_b, FALSE otherwise.

\*******************************************************************************/

BOOL Is_Io ( Alphas * alpha, smo_Settings * settings ) {
	unsigned int i ;
	if (NULL == alpha || NULL == settings)
	{
		printf("\nFATAL ERROR : input pointer is NULL.\n") ;
		exit(1) ;
	}
	for (i=0;i<settings->pairs->classes-1;i++) {
		if (Io_a == alpha->setname_ptr[i] || Io_b == alpha->setname_ptr[i]) return TRUE ;
	}
	return FALSE ;
} /*/ end of Is_Io */


/*******************************************************************************\

    Set_Name Get_DW_Label ( Alphas * alpha, smo_Settings * settings)
    
    purpose: evaluate and assign the downward (lower bound) Set_Name for a given 
             alpha based on its alpha_dw value (used primarily in SVOREX methods).
    input:   the pointer to alpha structure and the pointer to smo_Settings 
    output:  the appropriate downward Set_Name (Io_b, I_One, I_Fou, or I_o on error)

\*******************************************************************************/

Set_Name Get_DW_Label ( Alphas * alpha, smo_Settings * settings) {
	double a ;
	double u ;
	double l ;

	if ( NULL == alpha || NULL == settings )
	{
		printf("\r\nFATAL ERROR: input is NULL in Get_Label.\r\n");
		return I_o ;
	}

	u = alpha->alpha_up ;
	l = alpha->alpha_dw ;

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

	a = alpha->alpha_dw ; 
	if (1 == alpha->pair->target) return I_One ;
	if ( fabs(settings->vc - a)<EPS*EPS ) return I_Fou ;
	else if ( fabs(a)<EPS*EPS )	return I_One ;
	else if ( a > 0 && a < settings->vc ) return Io_b ;
	else
	{
		printf ( "\r\nFATAL ERROR : wrong alpha in Get_Label. %d \r\n", (int)(alpha-ALPHA) ) ;		
	    return I_o ;		
	}		
} /*/ end of Get_Setname */


/*******************************************************************************\

    Set_Name Get_UP_Label ( Alphas * alpha, smo_Settings * settings)
    
    purpose: evaluate and assign the upward (upper bound) Set_Name for a given 
             alpha based on its alpha_up value (used primarily in SVOREX methods).
    input:   the pointer to alpha structure and the pointer to smo_Settings 
    output:  the appropriate upward Set_Name (Io_a, I_Two, I_Thr, or I_o on error)

\*******************************************************************************/

Set_Name Get_UP_Label ( Alphas * alpha, smo_Settings * settings) {
	double a ;
	double u ;
	double l ;

	if ( NULL == alpha || NULL == settings )
	{
		printf("\r\nFATAL ERROR: input is NULL in Get_Label.\r\n");
		return I_o ;
	}

	u = alpha->alpha_up ;
	l = alpha->alpha_dw ;

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
	if (alpha->pair->target == settings->pairs->classes) return I_Two ;	
	if ( fabs(settings->vc - a)<EPS*EPS ) return I_Thr ;
	else if ( fabs(a)<EPS*EPS )	return I_Two ;
	else if ( a > 0 && a < settings->vc ) return Io_a ;
	else
	{
		printf ( "\r\nFATAL ERROR : wrong alpha in Get_Label. %u \r\n", (int)(alpha-ALPHA) ) ;		
	    return I_o ;		
	}	
}
/* end of Get_Setname
 
end of file setandfi.c */