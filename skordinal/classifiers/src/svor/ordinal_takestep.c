/*******************************************************************************\

	smoc_takestep.c in Sequential Minimal Optimization ver2.0
	
	implements the takestep function of SMO for Classification.
			
	Chu Wei Copyright(C) National Univeristy of Singapore
	Create on Jan. 16 2000 at Control Lab of Mechanical Engineering 
	Update on Aug. 24 2001 	

\*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "smo.h"


/*******************************************************************************\

    BOOL Decide_Boundary (double gamma, int s1, int s2, smo_Settings * settings, double * H, double * L)
    
    purpose: calculate and set the High (H) and Low (L) boundaries for the SMO step 
             based on gamma, direction signs, and model settings.
    input:   gamma (double), s1 and s2 (integers for direction), settings (pointer 
             to smo_Settings), H and L (pointers to double for the output bounds).
    output:  returns TRUE if bounds are successfully decided, FALSE otherwise.

\*******************************************************************************/

BOOL Decide_Boundary (double gamma, int s1, int s2, smo_Settings * settings, double * H, double * L) {

	if (NULL == settings)	{ printf("pointer is NULL.\n"); return FALSE ; }

	if (s1*s2<0) {
		if (gamma>=0&&gamma<=settings->vc) { *H = settings->vc; *L = gamma; }
		else if (gamma<0&&gamma>=-settings->vc) { *H = settings->vc + gamma; *L = 0; }
		else return FALSE ;
	} else {
		if (gamma>=0&&gamma<=settings->vc) { *H = gamma; *L = 0; }
		else if (gamma>settings->vc&&gamma<=(settings->vc+settings->vc)) { *H = settings->vc; *L = gamma-settings->vc; }
		else return FALSE ;
	}
	return TRUE ;
} /*/ end of Decide_Boundary */


/*******************************************************************************\

    BOOL ordinal_takestep_SVORIM ( Alphas * alpha1, Alphas * alpha2, unsigned int threshold, smo_Settings * settings )
    
    purpose: execute a single Sequential Minimal Optimization (SMO) step for two 
             alpha variables under the SVORIM (Implicit Margin) model.
    input:   alpha1 and alpha2 (pointers to Alphas), threshold (active threshold 
             index), settings (pointer to smo_Settings).
    output:  returns TRUE if the optimization step updated the alphas successfully, 
             FALSE otherwise.

\*******************************************************************************/

BOOL ordinal_takestep_SVORIM ( Alphas * alpha1, Alphas * alpha2, unsigned int threshold, smo_Settings * settings ) {
	double  n1 = 0, n2 = 0, a1 = 0, a2 = 0;	
	BOOL nset1, nset2, set1, set2 ; 
	double F1 = 0, F2 = 0 ;
	double s1 = 1, s2 = 1 ;
	double K11 = 0, K12 = 0, K22 = 0 ;
	double ueta = 0, gamma = 0, delphi = 0 ;
	double H = 0, L = 0 ;
	double ObjH = 0, ObjL = 0 ;
	Set_Name name1, name2 ;
	Alphas * alpha3 = NULL ;
	Cache_Node * cache = NULL ;
	int * index ;
#ifdef _ORDINAL_DEBUG
	double temp = 0 ; 
	double temp1, temp2 ;
#endif
	long unsigned int i1 = 0 ;
	long unsigned int i2 = 0 ;
	unsigned int t1, t2, loop ;

	if ( NULL == alpha1 || NULL == alpha2 || NULL == settings )
	{
		printf( " Alpha list error. \r\n" ) ;
		return FALSE ;
	}
	if (threshold<=0)
	{
		printf( " Active threshold %u is zero.\n", threshold) ;
		return FALSE ;
	}

	if (threshold >= settings->pairs->classes)
	{
		printf( " Active threshold %u is greater than %u.\n", threshold, settings->pairs->classes-1) ;
		return FALSE ;
	}

	i1 = alpha1-ALPHA+1 ;
	i2 = alpha2-ALPHA+1 ;

	if ( i1 == i2 ) 
		return FALSE ;

	t1 = alpha1->pair->target ; t2 = alpha2->pair->target ;

	name1 = alpha1->setname_ptr[threshold-1] ;
	name2 = alpha2->setname_ptr[threshold-1] ;

	set1 = Is_Io(alpha1,settings) ; set2 = Is_Io(alpha2,settings) ;
	a1 = n1 = alpha1->alpha_ptr[threshold-1] ; a2 = n2 = alpha2->alpha_ptr[threshold-1];		
	
	if (t1<=(threshold)&&t2>=(threshold+1)) { s1 = +1; s2 = -1; }
	else if (t1>=(threshold+1)&&t2<=(threshold)) { s1 = -1; s2 = +1; }
	else if (t1<=(threshold)&&t2<=(threshold)) { s1 = +1; s2 = +1; }
	else if (t1>=(threshold+1)&&t2>=(threshold+1)) { s1 = -1; s2 = -1; }
	else {
		printf("\nWarning : fail to specify the case.\n") ;
		exit(1) ;
	}

	F1 = alpha1->f_cache ; F2 = alpha2->f_cache ;		
	K11 = Calc_Kernel( alpha1, alpha1, settings ) ; K12 = Calc_Kernel( alpha1, alpha2, settings ) ;
	K22 = Calc_Kernel( alpha2, alpha2, settings ) ; 
	ueta = K11 + K22 - K12 - K12 ;
	
	if ( 0 >= ueta )
	{ 

		printf("\n Warning: Negative Definite Matrix.\n") ;
		ObjH=0 ;
		ObjL=0 ;
		return FALSE ;
	}
	else {
		if (s1*s2<0) {
			gamma = a1 - a2 ;
			if (gamma>=0&&gamma<=settings->vc) { H = settings->vc ; L = gamma ; }
			else if (gamma<0&&gamma>=-settings->vc) { H = settings->vc + gamma ; L = 0 ; }
			else
			{
				printf("beyond corner 1.\n");
				return FALSE ;
			}
		} else {
			gamma = a1 + a2 ;
			if (gamma>=0&&gamma<=settings->vc) { H = gamma ; L = 0 ; }
			else if (gamma>settings->vc&&gamma<=(settings->vc+settings->vc)) { H = settings->vc ; L = gamma-settings->vc ; }
			else
			{
				printf("beyond corner 3.\n");
				return FALSE ;
			}
		}		
		delphi = - F1 + F2 - s1 + s2 ;		
		n1 = a1 - s1*delphi/ueta ; n2 = a2 + s2*delphi/ueta ;
		if (s1*s2<0) {
			if (n1>H) { n1 = H ; n2 = (gamma>=0) ? settings->vc - gamma : settings->vc ; }
			else if (n1<L) { n1 = L ; n2 = (gamma>=0) ? 0 : - gamma ; }
			if (n2<0) { n2 = 0 ; n1 = gamma ; }
			else if (n2>settings->vc&&gamma<0) { n2 = settings->vc ; n1 = gamma + settings->vc ; }
		} else {
			if (n1>H) { n1 = H ; n2 = (gamma<=settings->vc) ? 0 : gamma - settings->vc ; }
			else if (n1<L) { n1 = L ; n2 = (gamma<=settings->vc) ? gamma : settings->vc ; }
			if (n2<0) { n2 = 0 ; n1 = gamma ; }
			else if (n2>settings->vc&&gamma>settings->vc) { n2 = settings->vc ; n1 = gamma - settings->vc ; }
		}
	}

	if ( fabs(n2 - a2) > 0 ) {
		alpha1->alpha_ptr[threshold-1] = n1 ; alpha2->alpha_ptr[threshold-1] = n2 ;
		alpha1->setname_ptr[threshold-1] = Get_Ordinal_Label(alpha1,threshold,settings) ;
		alpha2->setname_ptr[threshold-1] = Get_Ordinal_Label(alpha2,threshold,settings) ;

		nset1 = Is_Io(alpha1,settings) ; nset2 = Is_Io(alpha2,settings) ;
		if ( nset1 != set1 ) { if(nset1 && !set1) Add_Cache_Node(&settings->io_cache, alpha1); if(!nset1 && set1) Del_Cache_Node(&settings->io_cache, alpha1); }		
		if ( nset2 != set2 ) { if(nset2 && !set2) Add_Cache_Node(&settings->io_cache, alpha2); if(!nset2 && set2) Del_Cache_Node(&settings->io_cache, alpha2); }

		index = (int *)calloc(settings->pairs->count,sizeof(int)) ;
		for (loop = 1 ; loop < settings->pairs->classes ; loop ++) {
			alpha3 = settings->alpha + settings->ij_up[loop-1] - 1 ;
			if (alpha3!=alpha1&&alpha3!=alpha2&&FALSE==Is_Io(alpha3,settings)) {
				settings->bj_up[loop-1] += - s1*(n1 - a1)*Calc_Kernel( alpha1, alpha3, settings ) - s2*(n2 - a2)*Calc_Kernel( alpha2, alpha3, settings ) ;
				if (0==index[alpha3-settings->alpha]) {
					alpha3->f_cache += - s1*(n1 - a1)*Calc_Kernel( alpha1, alpha3, settings ) - s2*(n2 - a2)*Calc_Kernel( alpha2, alpha3, settings ) ;
					index[alpha3-settings->alpha] = 1 ;
				}
			} else { settings->bj_up[loop-1] = INT_MAX ; settings->ij_up[loop-1] = 0 ; }

			alpha3 = settings->alpha + settings->ij_low[loop-1] - 1 ;
			if (alpha3!=alpha1&&alpha2!=alpha3&&FALSE==Is_Io(alpha3,settings)) {	
				settings->bj_low[loop-1] += - s1*(n1 - a1)*Calc_Kernel( alpha1, alpha3, settings ) - s2*(n2 - a2)*Calc_Kernel( alpha2, alpha3, settings ) ;
				if (0==index[alpha3-settings->alpha]) {
					alpha3->f_cache += - s1*(n1 - a1)*Calc_Kernel( alpha1, alpha3, settings ) - s2*(n2 - a2)*Calc_Kernel( alpha2, alpha3, settings ) ;
					index[alpha3-settings->alpha] = 1 ;
				}
			} else { settings->bj_low[loop-1] = INT_MIN ; settings->ij_low[loop-1] = 0 ; }
		}

		if ( FALSE==Is_Io(alpha1,settings) ) {
			if (0==index[alpha1-settings->alpha]) {
				alpha1->f_cache = alpha1->f_cache - s1*(n1 - a1)*K11 - s2*(n2 - a2)*K12 ;
				index[alpha1-settings->alpha] = 1 ;
			}
			alpha3 = alpha1 ;
			for (loop = 0 ; loop < settings->pairs->classes-1 ; loop ++) {
				if (alpha3->pair->target > (loop+1) ) {
					if (alpha3->setname_ptr[loop]==Io_b || alpha3->setname_ptr[loop]==I_One) {
						if (alpha3->f_cache-1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache-1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
					}
					if (alpha3->setname_ptr[loop]==Io_b || alpha3->setname_ptr[loop]==I_Fou) {
						if (alpha3->f_cache-1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache-1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
					}
				} else {
					if (alpha3->setname_ptr[loop]==Io_a || alpha3->setname_ptr[loop]==I_Thr) {
						if (alpha3->f_cache+1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache+1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
					}
					if (alpha3->setname_ptr[loop]==Io_a || alpha3->setname_ptr[loop]==I_Two) {
						if (alpha3->f_cache+1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache+1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
					}
				}
			}		
		}
		if ( FALSE==Is_Io(alpha2,settings) ) {
			if (0==index[alpha2-settings->alpha]) {
				alpha2->f_cache = alpha2->f_cache - s1*(n1 - a1)*K12 - s2*(n2 - a2)*K22 ;
				index[alpha2-settings->alpha] = 1 ;
			}			
			alpha3 = alpha2 ;
			for (loop = 0 ; loop < settings->pairs->classes-1 ; loop ++) {
				if (alpha3->pair->target > (loop+1) ) {
					if (alpha3->setname_ptr[loop]==Io_b || alpha3->setname_ptr[loop]==I_One) {
						if (alpha3->f_cache-1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache-1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
					}
					if (alpha3->setname_ptr[loop]==Io_b || alpha3->setname_ptr[loop]==I_Fou) {
						if (alpha3->f_cache-1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache-1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
					}
				} else {
					if (alpha3->setname_ptr[loop]==Io_a || alpha3->setname_ptr[loop]==I_Thr) {
						if (alpha3->f_cache+1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache+1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
					}
					if (alpha3->setname_ptr[loop]==Io_a || alpha3->setname_ptr[loop]==I_Two) {
						if (alpha3->f_cache+1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache+1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
					}
				}
			}		
		}

		cache = Io_CACHE.front ;
		while ( NULL != cache ) {
			alpha3 = cache->alpha ;
			if (0==index[alpha3-settings->alpha]) {
				alpha3->f_cache = alpha3->f_cache - s1*(n1 - a1)*Calc_Kernel( alpha1, alpha3, settings ) - s2*(n2 - a2)*Calc_Kernel( alpha2, alpha3, settings ) ;
				index[alpha3-settings->alpha] = 1 ;
			}
			for (loop = 0 ; loop < settings->pairs->classes-1 ; loop ++) {
				if (cache->alpha->pair->target > (loop+1) ) {
					if (cache->alpha->setname_ptr[loop]==Io_b || cache->alpha->setname_ptr[loop]==I_One) {
						if (cache->alpha->f_cache-1<settings->bj_up[loop]) { settings->bj_up[loop] = cache->alpha->f_cache-1 ; settings->ij_up[loop] = cache->alpha - settings->alpha + 1 ; }
					}
					if (cache->alpha->setname_ptr[loop]==Io_b || cache->alpha->setname_ptr[loop]==I_Fou) {
						if (cache->alpha->f_cache-1>settings->bj_low[loop]) { settings->bj_low[loop] = cache->alpha->f_cache-1 ; settings->ij_low[loop] = cache->alpha - settings->alpha + 1 ; }
					}
				} else {
					if (cache->alpha->setname_ptr[loop]==Io_a || cache->alpha->setname_ptr[loop]==I_Thr) {
						if (cache->alpha->f_cache+1<settings->bj_up[loop]) { settings->bj_up[loop] = cache->alpha->f_cache+1 ; settings->ij_up[loop] = cache->alpha - settings->alpha + 1 ; }
					}
					if (cache->alpha->setname_ptr[loop]==Io_a || cache->alpha->setname_ptr[loop]==I_Two) {
						if (cache->alpha->f_cache+1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache+1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
					}
				}
			}
			cache = cache->next ;
		}
		
		free(index) ;

#ifdef _ORDINAL_DEBUG
		if (TRUE == SMO_DISPLAY)
		{
			for (loop=1;loop<settings->pairs->classes;loop++)
				printf("threshold %u : up=%f(%u), low=%f(%u)\n", loop, settings->bj_up[loop-1], 
				settings->ij_up[loop-1], settings->bj_low[loop-1],settings->ij_low[loop-1]) ;
			for ( loop = 1; loop <= settings->pairs->count; loop ++ )
			{	
				alpha3 = ALPHA + loop - 1 ;
				printf("%u-target %u---func %f: ",loop, alpha3->pair->target, Calculate_Ordinal_Fi(alpha3-ALPHA+1,settings)) ;
				for (t1=0;t1<settings->pairs->classes-1;t1++)
					printf("a%d %.3f  ",t1+1,alpha3->alpha[t1]) ;
				printf("\n") ;
			}
		}
		temp = 0 ;
		for (t1=0;t1<settings->pairs->count;t1++)
		{
			alpha1 = ALPHA+t1 ;
			temp1 = 0 ;
			for (loop=0;loop<settings->pairs->classes-1;loop++)
			{
				if (alpha1->pair->target<=loop+1)
					temp1 -= alpha1->alpha[loop] ;
				else
					temp1 += alpha1->alpha[loop] ;
				temp -= alpha1->alpha[loop] ;
			}
			for (t2=0;t2<t1;t2++)
			{
				alpha2 = ALPHA+t2 ;				
				temp2 = 0 ;
				for (loop=0;loop<settings->pairs->classes-1;loop++)
				{
					if (alpha2->pair->target<=loop+1)
						temp2 -= alpha2->alpha[loop] ;
					else
						temp2 += alpha2->alpha[loop] ;
				}
				temp += temp1*temp2*Calc_Kernel( alpha1, alpha2, settings ) ;
			}
			temp += 0.5*temp1*temp1*Calc_Kernel( alpha1, alpha1, settings ) ;
		}
		printf("objective functional %f\n",temp) ;
#endif

		for (loop = 1 ; loop < settings->pairs->classes ; loop ++)
		{
			if (0==settings->ij_up[loop-1]||0==settings->ij_low[loop-1])
				return Check_Alphas ( ALPHA, settings ) ;
		}
		
		return TRUE ;
	}
	else
	{
		return FALSE ;
	}
} /*/ end of ordinal_takestep_SVORIM */


/*******************************************************************************\

    BOOL ordinal_takestep_SVOREX ( Alphas * alpha1, Alphas * alpha2, unsigned int threshold, smo_Settings * settings )
    
    purpose: execute a single Sequential Minimal Optimization (SMO) step for two 
             alpha variables under the SVOREX (Explicit Margin) model.
    input:   alpha1 and alpha2 (pointers to Alphas), threshold (active threshold 
             index), settings (pointer to smo_Settings).
    output:  returns TRUE if the optimization step updated the alphas successfully, 
             FALSE otherwise.

\*******************************************************************************/

BOOL ordinal_takestep_SVOREX ( Alphas * alpha1, Alphas * alpha2, unsigned int threshold, smo_Settings * settings ) {
	double a1 = 0, a1a = 0, a2 = 0, a2a = 0 ;	/*/old alpha */
	double n1 = 0, n1a = 0, n2 = 0, n2a = 0 ;	/*/new alpha */
	double F1 = 0, F2 = 0 ;
	BOOL case1 = FALSE, case2 = FALSE, case3 = FALSE, case4 = FALSE ;
	double K11 = 0, K12 = 0, K22 = 0 ;
	double ueta = 0, gamma = 0, delphi = 0 ;
	double H = 0, L = 0 ;
	double ObjH = 0, ObjL = 0 ;
	Set_Name name1_up, name1_dw, name2_up, name2_dw ;
	Alphas * alpha3 = NULL ;
	Cache_Node * cache = NULL ;

	long unsigned int i1 = 0 ;
	long unsigned int i2 = 0 ; 
	int * index ;
	unsigned int t1, t2, loop ;
	
#ifdef _ORDINAL_DEBUG
	double temp ;
#endif
	if ( NULL == alpha1 || NULL == alpha2 || NULL == settings )
	{
		printf( " Alpha list error. \r\n" ) ;
		return FALSE ;
	}

	if (threshold > settings->pairs->classes-1 || threshold < 1)
	{
		printf( " Active threshold %u is greater than %u.\n", threshold, settings->pairs->classes-1) ;
		return FALSE ;
	}
    
	i1 = alpha1-ALPHA+1 ;
	i2 = alpha2-ALPHA+1 ;

	if ( i1 == i2 ) 
		return FALSE ;
#ifdef _ORDINAL_DEBUG
	/*/printf("%u and %u in takestep.\n",i1,i2) ;*/
#endif

	t1 = alpha1->pair->target ;
	t2 = alpha2->pair->target ;

	name1_up = alpha1->setname_up ; name1_dw = alpha1->setname_dw ;
	name2_up = alpha2->setname_up ; name2_dw = alpha2->setname_dw ;

	if (t1==(threshold)&&t2==(threshold+1)) case1 = TRUE ;
	else if (t1==(threshold+1)&&t2==(threshold)) case2 = TRUE ;
	else if (t1==(threshold)&&t2==(threshold)) case3 = TRUE ;
	else if (t1==(threshold+1)&&t2==(threshold+1)) case4 = TRUE ;
	else return FALSE ;
	
	a1 = n1 = alpha1->alpha_up ; a1a = n1a = alpha1->alpha_dw ;		
	a2 = n2 = alpha2->alpha_up ; a2a = n2a = alpha2->alpha_dw ;

	F1 = alpha1->f_cache ; F2 = alpha2->f_cache ;		
	K11 = Calc_Kernel( alpha1, alpha1, settings ) ; K12 = Calc_Kernel( alpha1, alpha2, settings ) ; K22 = Calc_Kernel( alpha2, alpha2, settings ) ; 
	ueta = K11 + K22 - K12 - K12 ;
	
	if ( 0 >= ueta )
	{
		printf(" Negative Definite Matrix.\n") ; 
		/*/ calculate objective function at H or L, choose the smaller one*/
		ObjH=0 ;
		ObjL=0 ;
		return FALSE ;
	}
	else {
		if (TRUE==case1) {
			gamma = a1 - a2a ;
			if (gamma>0&&gamma<=settings->vc) { H = settings->vc ; L = gamma ; }
			else if (gamma<=0&&gamma>=-settings->vc) { H = settings->vc + gamma ; L = 0 ; }
			else return FALSE ;
			delphi = - F1 + F2 - 2 ;
			n1 = a1 - delphi/ueta ; n2a = a2a - delphi/ueta ;
			if (n1>H) { n1 = H ; n2a = (gamma>=0) ? settings->vc - gamma : settings->vc ; }
			else if (n1<L) { n1 = L ; n2a = (gamma>=0) ? 0 : - gamma ; }
		} else if (TRUE==case2) {
			gamma = a1a - a2 ;
			if (gamma>0&&gamma<=settings->vc) { H = settings->vc ; L = gamma ; }
			else if (gamma<=0&&gamma>=-settings->vc) { H = settings->vc + gamma ; L = 0 ; }
			else return FALSE ;
			delphi = F1 - F2 - 2 ;
			n1a = a1a - delphi/ueta ; n2 = a2 - delphi/ueta ;
			if (n1a>H) { n1a = H ; n2 = (gamma>=0) ? settings->vc - gamma : settings->vc ; }
			else if (n1a<L) { n1a = L ; n2 = (gamma>=0) ? 0 : - gamma ; }
		} else if (TRUE==case3) {
			/*/ alpha1_up alpha2_up*/
			gamma = a1 + a2 ;
			if (gamma>=0&&gamma<settings->vc) { H = gamma ; L = 0 ; }
			else if (gamma>=settings->vc&&gamma<=(settings->vc+settings->vc)) { H = settings->vc ; L = gamma-settings->vc ; }
			else return FALSE ;
			delphi = - F1 + F2 ;
			n1 = a1 - delphi/ueta ; n2 = a2 + delphi/ueta ;
			if (n1>H) { n1 = H ; n2 = (gamma>0&&gamma<settings->vc) ? 0 : gamma - settings->vc ; }
			else if (n1<L) { n1 = L ; n2 = (gamma>0&&gamma<settings->vc) ? gamma : settings->vc ; }
		} else if (TRUE==case4) {
			gamma = a1a + a2a ;
			if (gamma>=0&&gamma<settings->vc) { H = gamma ; L = 0 ; }
			else if (gamma>=settings->vc&&gamma<=(settings->vc+settings->vc)) { H = settings->vc ; L = gamma-settings->vc ; }
			else return FALSE ;
			delphi = F1 - F2 ;
			n1a = a1a - delphi/ueta ; n2a = a2a + delphi/ueta ;
			if (n1a>H) { n1a = H ; n2a = (gamma>0&&gamma<settings->vc) ? 0 : gamma - settings->vc ; }
			else if (n1a<L) { n1a = L ; n2a = (gamma>0&&gamma<settings->vc) ? gamma : settings->vc ; }
		}
	} /*/end of if ueta */

	/*/ update Alpha List if necessary, then update Io_Cache, and vote B_LOW & B_UP*/
	if ( fabs((n2 - n2a) - (alpha2->alpha_up - alpha2->alpha_dw)) > 0 ) {

		/*/ store alphas in Alpha List*/
		a1 = alpha1->alpha_up ;	a1a = alpha1->alpha_dw ;
		a2 = alpha2->alpha_up ;	a2a = alpha2->alpha_dw ;
		alpha1->alpha_up = n1 ;	alpha1->alpha_dw = n1a ;
		alpha2->alpha_up = n2 ;	alpha2->alpha_dw = n2a ;
		alpha1->alpha = - alpha1->alpha_up + alpha1->alpha_dw ;		
		alpha2->alpha = - alpha2->alpha_up + alpha2->alpha_dw ;

		/*/ update Set & Cache_List  */

		if ( TRUE == case1 ) { name1_up = Get_UP_Label(alpha1,settings) ; name2_dw = Get_DW_Label(alpha2,settings) ; }
		else if ( TRUE == case2 ) { name1_dw = Get_DW_Label(alpha1,settings) ; name2_up = Get_UP_Label(alpha2,settings) ; }
		else if ( TRUE == case3 ) { name1_up = Get_UP_Label(alpha1,settings) ; name2_up = Get_UP_Label(alpha2,settings) ; }
		else if ( TRUE == case4 ) { name1_dw = Get_DW_Label(alpha1,settings) ; name2_dw = Get_DW_Label(alpha2,settings) ; }
		
		if ( alpha1->setname_up != name1_up || alpha1->setname_dw != name1_dw ) {			
			if ( (Io_a == name1_up || Io_b == name1_dw) && (alpha1->setname_up != Io_a && alpha1->setname_dw != Io_b) )	Add_Cache_Node( &settings->io_cache, alpha1 ) ; 
			if ( (alpha1->setname_up == Io_a || alpha1->setname_dw == Io_b) && name1_up != Io_a && name1_dw != Io_b ) Del_Cache_Node( &settings->io_cache, alpha1 ) ;
			if (TRUE == case1||TRUE == case3) alpha1->setname_up = name1_up ;
			if (TRUE == case2||TRUE == case4) alpha1->setname_dw = name1_dw ;
		}		
		if ( alpha2->setname_up != name2_up || alpha2->setname_dw != name2_dw  ) {						
			if ( (Io_a == name2_up || Io_b == name2_dw) && (alpha2->setname_up != Io_a && alpha2->setname_dw != Io_b) ) Add_Cache_Node( &settings->io_cache, alpha2 ) ; 				
			if ( (Io_a == alpha2->setname_up || Io_b == alpha2->setname_dw) && name2_up != Io_a && name2_dw != Io_b ) Del_Cache_Node( &settings->io_cache, alpha2 ) ;
			if (TRUE == case2||TRUE == case3) alpha2->setname_up = name2_up ;
			if (TRUE == case1||TRUE == case4) alpha2->setname_dw = name2_dw ;
		}

		/* initialize b_up b_low */
		index = (int *)calloc(settings->pairs->count,sizeof(int)) ;
		if (NULL == index)
		{
			printf("\n FATAL ERROR : fail to malloc index.\n") ;
			exit(1) ;
		}

		for (loop = 1 ; loop < settings->pairs->classes ; loop ++) {
			if (settings->ij_up[loop-1]!=0) {
				alpha3 = settings->alpha + settings->ij_up[loop-1] - 1 ;
				if (alpha3!=alpha1 && alpha3!=alpha2 && Io_a!=alpha3->setname_up && Io_b!=alpha3->setname_dw) {
					settings->bj_up[loop-1] += - ((alpha1->alpha_up - alpha1->alpha_dw) - (a1 - a1a)) * Calc_Kernel( alpha1, alpha3, settings ) - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * Calc_Kernel( alpha2, alpha3, settings ) ;
					if (0==index[alpha3-settings->alpha]) {
						alpha3->f_cache += - ((alpha1->alpha_up - alpha1->alpha_dw) - (a1 - a1a)) * Calc_Kernel( alpha1, alpha3, settings ) - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * Calc_Kernel( alpha2, alpha3, settings ) ;
                        index[alpha3-settings->alpha] = 1 ;
					}
				} else { settings->bj_up[loop-1] = INT_MAX ; settings->ij_up[loop-1] = 0 ; }
			}
			if (settings->ij_low[loop-1]!=0) {
				alpha3 = settings->alpha + settings->ij_low[loop-1] - 1 ;
				if (alpha3!=alpha1 && alpha2!=alpha3 && Io_a!=alpha3->setname_up && Io_b!=alpha3->setname_dw) {	
					settings->bj_low[loop-1] += - ((alpha1->alpha_up - alpha1->alpha_dw) - (a1 - a1a)) * Calc_Kernel( alpha1, alpha3, settings ) - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * Calc_Kernel( alpha2, alpha3, settings ) ;
					if (0==index[alpha3-settings->alpha]) {
						alpha3->f_cache += - ((alpha1->alpha_up - alpha1->alpha_dw) - (a1 - a1a)) * Calc_Kernel( alpha1, alpha3, settings ) - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * Calc_Kernel( alpha2, alpha3, settings ) ;
						index[alpha3-settings->alpha] = 1 ;
					}
				} else { settings->bj_low[loop-1] = INT_MIN ; settings->ij_low[loop-1] = 0 ; }
			}
		}

		/* update f-cache of i1 & i2 if not in Io_Cache*/
		if (alpha1->setname_up != Io_a && alpha1->setname_dw != Io_b) {
			if (0==index[alpha1-settings->alpha]) {
				alpha1->f_cache = alpha1->f_cache - ((alpha1->alpha_up - alpha1->alpha_dw) - (a1 - a1a)) * K11 - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * K12 ;
				index[alpha1-settings->alpha] = 1 ;
			}
			alpha3=alpha1 ;
			if (alpha3->pair->target > 1 ) {
				loop = alpha3->pair->target - 2 ;
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_One) {
					if (alpha3->f_cache-1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache-1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_Fou) {
					if (alpha3->f_cache-1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache-1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
			if ( alpha3->pair->target < settings->pairs->classes ) {
				/*/upper*/
				loop = alpha3->pair->target - 1 ;
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Thr) {
					if (alpha3->f_cache+1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache+1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Two) {
					if (alpha3->f_cache+1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache+1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
		}
		if (alpha2->setname_up != Io_a && alpha2->setname_dw != Io_b) {
			if (0==index[alpha2-settings->alpha]) {
				alpha2->f_cache = alpha2->f_cache - ((alpha1->alpha_up - alpha1->alpha_dw) - (a1 - a1a)) * K12 - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * K22 ;
				index[alpha2-settings->alpha] = 1 ;
			}
			alpha3=alpha2 ;			
			if (alpha3->pair->target > 1 ) {
				/*/lower */
				loop = alpha3->pair->target - 2 ;
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_One) {
					if (alpha3->f_cache-1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache-1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_Fou) {
					if (alpha3->f_cache-1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache-1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
			if ( alpha3->pair->target < settings->pairs->classes ) {
				loop = alpha3->pair->target - 1 ;
				/*/upper*/
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Thr) {
					if (alpha3->f_cache+1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache+1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Two) {
					if (alpha3->f_cache+1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache+1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
		}			

		/*/ update Fi in Io_Cache and vote B_LOW & B_UP if possible*/
		cache = Io_CACHE.front ;
		while ( NULL != cache ) {	
			alpha3 = cache->alpha ;	
			if ( 0==index[alpha3-settings->alpha]) {
				alpha3->f_cache = alpha3->f_cache - ((alpha1->alpha_up - alpha1->alpha_dw) - (a1 - a1a)) * Calc_Kernel( alpha1, alpha3, settings ) - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * Calc_Kernel( alpha2, alpha3, settings ) ;
				index[alpha3-settings->alpha] = 1 ;
			}
			if (alpha3->pair->target > 1 ) {
				loop = alpha3->pair->target - 2 ;
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_One) {
					if (alpha3->f_cache-1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache-1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_Fou) {
					if (alpha3->f_cache-1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache-1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
			if ( alpha3->pair->target < settings->pairs->classes ) {
				loop = alpha3->pair->target - 1 ;
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Thr) {
					if (alpha3->f_cache+1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache+1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Two) {
					if (alpha3->f_cache+1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache+1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
			cache = cache->next ;				
		} /*/ end of while*/
		
		free(index) ;

		for (loop = 1 ; loop < settings->pairs->classes ; loop ++) {
			if (0==settings->ij_up[loop-1]||0==settings->ij_low[loop-1]) { 
				Check_Alphas ( settings->alpha, settings ) ;
				loop = settings->pairs->classes ;
			}
		}

		for (loop = 1; loop < settings->pairs->classes; loop ++) {
			settings->bmu_low[loop-1]=settings->bj_low[loop-1] ; settings->imu_low[loop-1]=loop ;
			if (loop>1 && settings->bmu_low[loop-2]>settings->bmu_low[loop-1]) { settings->bmu_low[loop-1]=settings->bmu_low[loop-2] ; settings->imu_low[loop-1]=settings->imu_low[loop-2] ; }
		}
		

		for (loop = settings->pairs->classes-1; loop > 0; loop --) {
			settings->bmu_up[loop-1]=settings->bj_up[loop-1] ; settings->imu_up[loop-1]=loop ;
			if (loop<settings->pairs->classes-1 && settings->bmu_up[loop-1]>settings->bmu_up[loop]) { settings->bmu_up[loop-1]=settings->bmu_up[loop] ; settings->imu_up[loop-1]=settings->imu_up[loop] ; }           
		}
		
		
		for (loop = 2; loop < settings->pairs->classes; loop ++) {
			if (settings->mu[loop-1]>EPS*EPS) {
				if (settings->bmu_up[loop-1]>settings->bmu_up[loop-2]) { settings->bmu_up[loop-1]=settings->bmu_up[loop-2] ; settings->imu_up[loop-1]=settings->imu_up[loop-2] ; }
				if (settings->bmu_low[loop-2]<settings->bmu_low[loop-1]) { settings->bmu_low[loop-2]=settings->bmu_low[loop-1] ; settings->imu_low[loop-2]=settings->imu_low[loop-1] ; }
			}
		}
#ifdef _ORDINAL_DEBUG
		for (t1=1;t1<settings->pairs->classes;t1++)
			printf("threshold %u : upper=%f(%u), lower=%f(%u), mu=%f\n", t1, settings->bj_up[t1-1],
				settings->ij_up[t1-1], settings->bj_low[t1-1], settings->ij_low[t1-1], settings->mu[t1-1]) ;
		temp = 0 ;
		for (t1=0;t1<settings->pairs->count;t1++)
		{
			alpha1 = ALPHA+t1 ;
			for (t2=0;t2<t1;t2++)
			{
				alpha2 = ALPHA+t2 ;
				temp += (-alpha1->alpha_up+alpha1->alpha_dw)
					*(-alpha2->alpha_up+alpha2->alpha_dw)*Calc_Kernel( alpha1, alpha2, settings ) ;
			}
			temp += 0.5*(-alpha1->alpha_up+alpha1->alpha_dw)
					*(-alpha1->alpha_up+alpha1->alpha_dw)*Calc_Kernel( alpha1, alpha1, settings ) ;
			temp -= (alpha1->alpha_up+alpha1->alpha_dw) ;
		}
		printf("objective functional %f\n",temp) ;
#endif
		return TRUE ;
	} /*/ end of update */
	else
	{
		{
			/*printf("fail to update pairs %u and %u\n",i1,i2) ;*/
			return FALSE ;
		}
	}
} /*/ end of ordinal_takestep_SVOREX */


/*******************************************************************************\

    BOOL ordinal_cross_identical ( Alphas * alpha1, Alphas * alpha2, unsigned int threshold, smo_Settings * settings )
    
    purpose: handle the optimization step for edge cases where the cross-threshold 
             update involves identical alpha pointers or overlapping targets.
    input:   alpha1 and alpha2 (pointers to Alphas), threshold (active threshold 
             index), settings (pointer to smo_Settings).
    output:  returns TRUE if the step is successful, FALSE if identical or invalid.

\*******************************************************************************/

BOOL ordinal_cross_identical ( Alphas * alpha1, Alphas * alpha2, unsigned int threshold, smo_Settings * settings ) {
	double a1a = 0, a2 = 0, a2a = 0 ;	
	double n1 = 0, n1a = 0, n2 = 0, n2a = 0 ;	
	BOOL case4 = FALSE ;
	double K12 = 0 ;
	double gamma = 0, delphi = 0 ;
	double H = 0, L = 0 ;
	Set_Name name1_up, name1_dw;
	Alphas * alpha3 = NULL ;
	Cache_Node * cache = NULL ;
	unsigned int t1, t2;
	int * index ;
	unsigned int loop ;
	int s1=0, s2=0, mu, mu1=0, mu2=0 ;

	if ( alpha1 == alpha2 ) return FALSE ;

	t1 = alpha1->pair->target ; t2 = alpha2->pair->target ;
	if (!(t1<=threshold&&threshold<=t2)) threshold = t1 ;
	if (threshold<=1||threshold>=settings->pairs->classes) return FALSE ;

	name1_up = alpha1->setname_up ; name1_dw = alpha1->setname_dw ;
	n1 = alpha1->alpha_up ; a1a = n1a = alpha1->alpha_dw ;		
	a2 = n2 = alpha2->alpha_up ; a2a = n2a = alpha2->alpha_dw ;

	K12 = Calc_Kernel( alpha1, alpha2, settings ) ; 

	s1 = +1 ; mu1 = threshold ; s2 = -1 ; mu2 = threshold ; case4 = TRUE ;
	if (TRUE==case4) {		
		gamma = a1a + s1*s2*a2 ;
		Decide_Boundary (gamma, s1, s2, settings, &H, &L) ;
		delphi = s1*(H-a1a) ;
		for (mu=mu1;mu<=mu2;mu++) { if (settings->mu[mu-1]<delphi) delphi = settings->mu[mu-1] ; }
		n1a=a1a+s1*delphi ; n2=a2-s2*delphi ;
		if (n1a>H) { n1a = H ; n2 = (gamma>=0) ? settings->vc - gamma : settings->vc ; delphi = s1*(H-a1a) ; }
		else if (n1a<L) { n1a = L ; n2 = (gamma>=0) ? 0 : - gamma ; delphi = s1*(L-a1a) ; }
		n1=n2 ; n2a=n1a ;
	}

	if ( fabs(delphi) > 0 ) {
		a1a = alpha1->alpha_dw ;
		a2 = alpha2->alpha_up ; a2a = alpha2->alpha_dw ;
		alpha1->alpha_up = n1 ; alpha1->alpha_dw = n1a ;
		alpha2->alpha_up = n2 ; alpha2->alpha_dw = n2a ;
		alpha1->alpha = - alpha1->alpha_up + alpha1->alpha_dw ;		
		alpha2->alpha = - alpha2->alpha_up + alpha2->alpha_dw ;

		for (mu=mu1;mu<=mu2;mu++) settings->mu[mu-1] -= delphi ;

		name1_up = Get_UP_Label(alpha1,settings) ; name1_dw = Get_DW_Label(alpha1,settings) ;
		
		if ( alpha1->setname_up != name1_up || alpha1->setname_dw != name1_dw ) {			
			if ( (Io_a == name1_up || Io_b == name1_dw) && (alpha1->setname_up != Io_a && alpha1->setname_dw != Io_b) ) Add_Cache_Node( &settings->io_cache, alpha1 ) ;  
			if ( (alpha1->setname_up == Io_a || alpha1->setname_dw == Io_b) && name1_up != Io_a && name1_dw != Io_b ) Del_Cache_Node( &settings->io_cache, alpha1 ) ;
			alpha1->setname_up = name1_up ; alpha1->setname_dw = name1_dw ;
		}

		index = (int *)calloc(settings->pairs->count,sizeof(int)) ;
		for (loop = 1 ; loop < settings->pairs->classes ; loop ++) {
			if (settings->ij_up[loop-1]!=0) {
				alpha3 = settings->alpha + settings->ij_up[loop-1] - 1 ;
				if (alpha3!=alpha1 && alpha3!=alpha2 && Io_a!=alpha3->setname_up && Io_b!=alpha3->setname_dw) {
					settings->bj_up[loop-1] += - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * Calc_Kernel( alpha2, alpha3, settings ) ;
					if (0==index[alpha3-settings->alpha]) {
						alpha3->f_cache += - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * Calc_Kernel( alpha2, alpha3, settings ) ;
						index[alpha3-settings->alpha] = 1 ;
					}
				} else { settings->bj_up[loop-1] = INT_MAX ; settings->ij_up[loop-1] = 0 ; }
			}
			if (settings->ij_low[loop-1]!=0) {
				alpha3 = settings->alpha + settings->ij_low[loop-1] - 1 ;
				if (alpha3!=alpha1 && alpha2!=alpha3 && Io_a!=alpha3->setname_up && Io_b!=alpha3->setname_dw) {   
					settings->bj_low[loop-1] += - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * Calc_Kernel( alpha2, alpha3, settings ) ;
					if (0==index[alpha3-settings->alpha]) {
						alpha3->f_cache += - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * Calc_Kernel( alpha2, alpha3, settings ) ;
						index[alpha3-settings->alpha] = 1 ;
					}
				} else { settings->bj_low[loop-1] = INT_MIN ; settings->ij_low[loop-1] = 0 ; }
			}
		}

		if (alpha1->setname_up != Io_a && alpha1->setname_dw != Io_b) {
			if (0==index[alpha1-settings->alpha]) {
				alpha1->f_cache = alpha1->f_cache - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * K12 ;
				index[alpha1-settings->alpha] = 1 ;
			}
			alpha3=alpha1 ;
			if (alpha3->pair->target > 1 ) {
				loop = alpha3->pair->target - 2 ;
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_One) {
					if (alpha3->f_cache-1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache-1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_Fou) {
					if (alpha3->f_cache-1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache-1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
			if ( alpha3->pair->target < settings->pairs->classes ) {
				loop = alpha3->pair->target - 1 ;
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Thr) {
					if (alpha3->f_cache+1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache+1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Two) {
					if (alpha3->f_cache+1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache+1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
		}

		cache = settings->io_cache.front ;
		while ( NULL != cache ) {   
			alpha3 = cache->alpha ; 
			if ( 0==index[alpha3-settings->alpha]) {
				alpha3->f_cache = alpha3->f_cache - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * Calc_Kernel( alpha2, alpha3, settings ) ;
				index[alpha3-settings->alpha] = 1 ;
			}
			if (alpha3->pair->target > 1 ) {
				loop = alpha3->pair->target - 2 ;
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_One) {
					if (alpha3->f_cache-1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache-1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_Fou) {
					if (alpha3->f_cache-1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache-1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
			if ( alpha3->pair->target < settings->pairs->classes ) {
				loop = alpha3->pair->target - 1 ;
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Thr) {
					if (alpha3->f_cache+1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache+1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Two) {
					if (alpha3->f_cache+1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache+1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
			cache = cache->next ;               
		}
		free(index) ;

		for (loop = 1 ; loop < settings->pairs->classes ; loop ++) {
			if (0==settings->ij_up[loop-1]||0==settings->ij_low[loop-1]) { 
				Check_Alphas ( settings->alpha, settings ) ;
				loop = settings->pairs->classes ;
			}
		}

		for (loop = 1; loop < settings->pairs->classes; loop ++) {
			settings->bmu_low[loop-1]=settings->bj_low[loop-1] ; settings->imu_low[loop-1]=loop ;
			if (loop>1 && settings->bmu_low[loop-2]>settings->bmu_low[loop-1]) { settings->bmu_low[loop-1]=settings->bmu_low[loop-2] ; settings->imu_low[loop-1]=settings->imu_low[loop-2] ; }
		}
		for (loop = settings->pairs->classes-1; loop > 0; loop --) {
			settings->bmu_up[loop-1]=settings->bj_up[loop-1] ; settings->imu_up[loop-1]=loop ;
			if (loop<settings->pairs->classes-1 && settings->bmu_up[loop-1]>settings->bmu_up[loop]) { settings->bmu_up[loop-1]=settings->bmu_up[loop] ; settings->imu_up[loop-1]=settings->imu_up[loop] ; }
		}
		for (loop = 2; loop < settings->pairs->classes; loop ++) {
			if (settings->mu[loop-1]>EPS*EPS) {
				if (settings->bmu_up[loop-1]>settings->bmu_up[loop-2]) { settings->bmu_up[loop-1]=settings->bmu_up[loop-2] ; settings->imu_up[loop-1]=settings->imu_up[loop-2] ; }
				if (settings->bmu_low[loop-2]<settings->bmu_low[loop-1]) { settings->bmu_low[loop-2]=settings->bmu_low[loop-1] ; settings->imu_low[loop-2]=settings->imu_low[loop-1] ; }
			}
		}
		return TRUE ;
	}
	return FALSE ;
} /*/ end of ordinal_cross_identical */


/*******************************************************************************\

    BOOL ordinal_cross_takestep ( Alphas * Bup, unsigned int b1, Alphas * Blow, unsigned int b2, smo_Settings * settings )
    
    purpose: execute an SMO cross-step involving alpha variables from different 
             thresholds (upper and lower bounds) to explicitly optimize the margin.
    input:   Bup (pointer to upper Alphas), b1 (upper threshold index), Blow 
             (pointer to lower Alphas), b2 (lower threshold index), settings 
             (pointer to smo_Settings).
    output:  returns TRUE if the cross-step updated the alphas and margins 
             successfully, FALSE otherwise.

\*******************************************************************************/

BOOL ordinal_cross_takestep ( Alphas * Bup, unsigned int b1, Alphas * Blow, unsigned int b2, smo_Settings * settings ) {
	double a1 = 0, a1a = 0, a2 = 0, a2a = 0 ;	
	double n1 = 0, n1a = 0, n2 = 0, n2a = 0 ;	
	double F1 = 0, F2 = 0 ;
	BOOL case1 = FALSE, case2 = FALSE, case3 = FALSE, case4 = FALSE ;   
	double K11 = 0, K12 = 0, K22 = 0 ;
	double gamma = 0, delphi = 0 ;
	double H = 0, L = 0 ;
	double ObjH = 0, ObjL = 0 ;
	Set_Name name1_up, name1_dw, name2_up, name2_dw ;
	Alphas * alpha1 = NULL ; Alphas * alpha2 = NULL ; Alphas * alpha3 = NULL ;
	Cache_Node * cache = NULL ;
	unsigned int t1, t2, loop ;
	int s1=0, s2=0, mu, mu1=0, mu2=0 ;
	double deltamu = -1 ;
	int * index = NULL ;

	if ( NULL == Bup || NULL == Blow || NULL == settings )
	{
		printf( " Alpha list error. \r\n" ) ;
		return FALSE ;
	}

	if (b1 > settings->pairs->classes-1 || b1 < 1 || b2 > settings->pairs->classes-1 || b2 < 1)
	{
		printf( " Active threshold is greater than %u.\n", settings->pairs->classes-1) ;
		return FALSE ;
	}

	if ( Bup == Blow ) return ordinal_cross_identical (Bup, Blow, b1, settings) ;
	if (b1==b2) return ordinal_takestep(Bup, Blow, b1, settings) ;

	alpha1 = Bup ; alpha2 = Blow ;
	t1 = alpha1->pair->target ; t2 = alpha2->pair->target ;
	name1_up = alpha1->setname_up ; name1_dw = alpha1->setname_dw ;
	name2_up = alpha2->setname_up ; name2_dw = alpha2->setname_dw ;

	if (t1==b1) { 
		if (name1_up==Io_a||name1_up==I_Thr) { s1 = -1 ; if (b1<b2) mu1 = t1 + 1 ; else mu1 = t1 ; }
	} else if (t1==b1+1) { 
		if (name1_dw==Io_b||name1_dw==I_One) { s1 = +1 ; if (b1<b2) mu1 = t1 ; else mu1 = t1 - 1 ; }
	}
	if (t2==b2) { 
		if (name2_up==Io_a||name2_up==I_Two) { s2 = -1 ; if (b1<b2) mu2 = t2 ; else mu2 = t2 + 1 ; }
	} else if (t2==b2+1) { 
		if (name2_dw==Io_b||name2_dw==I_Fou) { s2 = +1 ; if (b1<b2) mu2 = t2 - 1 ; else mu2 = t2 ; }
	}

	if (b1>b2) { alpha3=alpha1; alpha1=alpha2; alpha2=alpha3; mu=s1; s1=s2; s2=mu; }
	b1=min(mu1,mu2); b2=max(mu1,mu2); mu1=b1; mu2=b2;
	
	name1_up = alpha1->setname_up ; name1_dw = alpha1->setname_dw ;
	name2_up = alpha2->setname_up ; name2_dw = alpha2->setname_dw ;
	
	a1 = n1 = alpha1->alpha_up ; a1a = n1a = alpha1->alpha_dw ;		
	a2 = n2 = alpha2->alpha_up ; a2a = n2a = alpha2->alpha_dw ;

	F1 = alpha1->f_cache ; F2 = alpha2->f_cache ;		
	K11 = Calc_Kernel( alpha1, alpha1, settings ) ; K12 = Calc_Kernel( alpha1, alpha2, settings ) ; K22 = Calc_Kernel( alpha2, alpha2, settings ) ; 
	double ueta = K11 + K22 - K12 - K12 ;

	/*/ case 1*/
	if ( (s1==-1) && (s2==+1) )
	{
		/*/ - a_{k} + a_{k+2}* = c.*/
		case1 = TRUE ;
	}
	/*/ case 2*/
	else if ( (s1==-1) && (s2==-1) )
	{
		/*/ - a_{k-1} - a_{k} = c.*/
		case2 = TRUE ;
	}
	/*/ case 3*/
	else if ( (s1==+1) && (s2==+1) )
	{
		/*/ a_{k+1}* + a_{k+2}* = c.*/
		case3 = TRUE ;
	}
	/*/ case 4*/
	else if ( (s1==+1) && (s2==-1) )
	{
		/*/ a_{k-1}* - a_{k+1} = c.*/
		case4 = TRUE ;
	}
	else
	{
		printf("\nWarning : fail to specify the case.\n") ;
		return FALSE ;
	}

		if ( 0 >= ueta )
	{
		printf(" Negative Definite Matrix cross.\n") ;
		/*/ calculate objective function at H or L, choose the smaller one*/
		ObjH=0 ;
		ObjL=0 ;
		return FALSE ;
	}
	else /*/ normal condition*/
	{
		if (TRUE==case1)
		{
			/*/ - a_{k} + a_{k+2}* = c.	*/	
			gamma = a1 + s1*s2*a2a ;
			Decide_Boundary (gamma, s1, s2, settings, &H, &L) ;
			if (ueta>0)
				delphi = (- F1 + F2 + s1 - s2)/ueta ;/*/ n1=a1+s1*adlphi ;*/
			else
				delphi = (- F1 + F2 + s1 - s2) ;
			for (mu=mu1;mu<=mu2;mu++)
			{
				if (settings->mu[mu-1]<delphi)
				{
					delphi = settings->mu[mu-1] ;
					deltamu = 0 ;
				}
			}
			n1=a1+s1*delphi ;				
			n2a=a2a-s2*delphi ;
			if (n1>H)
			{
				n1 = H ;
				if (gamma>=0)
					n2a = VC - gamma ;
				else 
					n2a = VC ;
				delphi = s1*(H-a1) ;
				deltamu = delphi ;
			}
			else if (n1<L)
			{
				n1 = L ;
				if (gamma>=0)
					n2a = 0 ;
				else 
					n2a = - gamma ;
				delphi = s1*(L-a1) ;
				deltamu = delphi ;
			}
		}
		else if (TRUE==case2)
		{			
			/*/ - a_{k-1} - a_{k} = c.		*/
			gamma = a1 + s1*s2*a2 ;
			Decide_Boundary (gamma, s1, s2, settings, &H, &L) ;
			if (ueta>0)
				delphi = (- F1 + F2 + s1 - s2)/ueta ;/*/ n1=a1+s1*adlphi ;*/
			else
				delphi = (- F1 + F2 + s1 - s2) ;
			for (mu=mu1;mu<=mu2;mu++)
			{
				if (settings->mu[mu-1]<delphi)
				{
					delphi = settings->mu[mu-1] ;
					deltamu = 0 ;
				}
			}
			n1=a1+s1*delphi ;				
			n2=a2-s2*delphi ;
			if (n1>H)
			{
				n1 = H ;
				if (gamma>0&&gamma<VC)
					n2 = 0 ;
				else 
					n2 = gamma - VC ;
				delphi = s1*(H-a1) ;
				deltamu = delphi ;
			}
			else if (n1<L)
			{
				n1 = L ;
				if (gamma>0&&gamma<VC)
					n2 = gamma ;
				else 
					n2 = VC ;
				delphi = s1*(L-a1) ;
				deltamu = delphi ;
			}
		}
		else if (TRUE==case3)
		{

			gamma = a1a + s1*s2*a2a ;
			Decide_Boundary (gamma, s1, s2, settings, &H, &L) ;
			if (ueta>0)
				delphi = (- F1 + F2 + s1 - s2)/ueta ;
			else
				delphi = (- F1 + F2 + s1 - s2) ;
			for (mu=mu1;mu<=mu2;mu++)
			{
				if (settings->mu[mu-1]<delphi)
				{
					delphi = settings->mu[mu-1] ;
					deltamu = 0 ;
				}
			}
			n1a=a1a+s1*delphi ;				
			n2a=a2a-s2*delphi ;
			if (n1a>H)
			{
				n1a = H ;
				if (gamma>0&&gamma<VC)
					n2a = 0 ;
				else 
					n2a = gamma - VC ;
				delphi = s1*(H-a1a) ;
				deltamu = delphi ;
			}
			else if (n1a<L)
			{
				n1a = L ;
				if (gamma>0&&gamma<VC)
					n2a = gamma ;
				else 
					n2a = VC ;
				delphi = s1*(L-a1a) ;
				deltamu = delphi ;
			}
		}
		else if (TRUE==case4)
		{

			gamma = a1a + s1*s2*a2 ;
			Decide_Boundary (gamma, s1, s2, settings, &H, &L) ;
			if (ueta>0)
				delphi = (- F1 + F2 + s1 - s2)/ueta ;
			else
				delphi = (- F1 + F2 + s1 - s2) ;
			for (mu=mu1;mu<=mu2;mu++)
			{
				if (settings->mu[mu-1]<delphi)
				{
					delphi = settings->mu[mu-1] ;
					deltamu = 0 ;
				}
			}
			n1a=a1a+s1*delphi ;
			n2=a2-s2*delphi ;
			if (n1a>H)
			{
				n1a = H ;
				if (gamma>=0)
					n2 = VC - gamma ;
				else 
					n2 = VC ;
				delphi = s1*(H-a1a) ;
				deltamu = delphi ;
			}
			else if (n1a<L)
			{
				n1a = L ;
				if (gamma>=0)
					n2 = 0 ;
				else 
					n2 = - gamma ;
				delphi = s1*(L-a1a) ;
				deltamu = delphi ;
			}
		}
		else
		{
			printf(" Unknown case.\n") ;
		}
	} /*/end of if ueta */			


	/*/ update Alpha List if necessary, then update Io_Cache, and vote B_LOW & B_UP*/
	if ( fabs((n2 - n2a) - (alpha2->alpha_up - alpha2->alpha_dw)) > 0 ) {
		/*/ store alphas in Alpha List*/
		a1 = alpha1->alpha_up ;	a1a = alpha1->alpha_dw ;
		a2 = alpha2->alpha_up ;	a2a = alpha2->alpha_dw ;
		alpha1->alpha_up = n1 ;	alpha1->alpha_dw = n1a ;
		alpha2->alpha_up = n2 ;	alpha2->alpha_dw = n2a ;
		alpha1->alpha = - alpha1->alpha_up + alpha1->alpha_dw ;		
		alpha2->alpha = - alpha2->alpha_up + alpha2->alpha_dw ;

		/*/ update mu*/
		for (mu=mu1;mu<=mu2;mu++) settings->mu[mu-1] -= delphi ;

		/*/ update Set & Cache_List  */
		if ( TRUE == case1 ) { name1_up = Get_UP_Label(alpha1,settings) ; name2_dw = Get_DW_Label(alpha2,settings) ; }
		else if ( TRUE == case2 ) { name1_up = Get_UP_Label(alpha1,settings) ; name2_up = Get_UP_Label(alpha2,settings) ; }
		else if ( TRUE == case3 ) { name1_dw = Get_DW_Label(alpha1,settings) ; name2_dw = Get_DW_Label(alpha2,settings) ; }
		else if ( TRUE == case4 ) { name1_dw = Get_DW_Label(alpha1,settings) ; name2_up = Get_UP_Label(alpha2,settings) ; }
		
		if ( alpha1->setname_up != name1_up || alpha1->setname_dw != name1_dw ) {			
			if ( (Io_a == name1_up || Io_b == name1_dw) && (alpha1->setname_up != Io_a && alpha1->setname_dw != Io_b) ) Add_Cache_Node( &settings->io_cache, alpha1 ) ;	
			if ( (alpha1->setname_up == Io_a || alpha1->setname_dw == Io_b) && name1_up != Io_a && name1_dw != Io_b ) Del_Cache_Node( &settings->io_cache, alpha1 ) ;
			if (TRUE == case1||TRUE == case2) alpha1->setname_up = name1_up ;
			if (TRUE == case3||TRUE == case4) alpha1->setname_dw = name1_dw ;
		}		
		if ( alpha2->setname_up != name2_up || alpha2->setname_dw != name2_dw ) {						
			if ( (Io_a == name2_up || Io_b == name2_dw) && (alpha2->setname_up != Io_a && alpha2->setname_dw != Io_b) ) Add_Cache_Node( &settings->io_cache, alpha2 ) ; 						
			if ( (Io_a == alpha2->setname_up || Io_b == alpha2->setname_dw) && name2_up != Io_a && name2_dw != Io_b ) Del_Cache_Node( &settings->io_cache, alpha2 ) ;
			if (TRUE == case2||TRUE == case4) alpha2->setname_up = name2_up ;
			if (TRUE == case1||TRUE == case3) alpha2->setname_dw = name2_dw ;
		}

		/*/ initialize b_up b_low*/
		index = (int *)calloc(settings->pairs->count,sizeof(int)) ;
		for (loop = 1 ; loop < settings->pairs->classes ; loop ++) {
			if (settings->ij_up[loop-1]!=0) {
				alpha3 = settings->alpha + settings->ij_up[loop-1] - 1 ;
				if (alpha3!=alpha1 && alpha3!=alpha2 && Io_a!=alpha3->setname_up && Io_b!=alpha3->setname_dw) {
					settings->bj_up[loop-1] += - ((alpha1->alpha_up - alpha1->alpha_dw) - (a1 - a1a)) * Calc_Kernel( alpha1, alpha3, settings ) - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * Calc_Kernel( alpha2, alpha3, settings ) ;
					if (0==index[alpha3-settings->alpha]) {
						alpha3->f_cache += - ((alpha1->alpha_up - alpha1->alpha_dw) - (a1 - a1a)) * Calc_Kernel( alpha1, alpha3, settings ) - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * Calc_Kernel( alpha2, alpha3, settings ) ;
						index[alpha3-settings->alpha] = 1 ;
					}
				} else { settings->bj_up[loop-1] = INT_MAX ; settings->ij_up[loop-1] = 0 ; }
			}
			if (settings->ij_low[loop-1]!=0) {
				alpha3 = settings->alpha + settings->ij_low[loop-1] - 1 ;
				if (alpha3!=alpha1 && alpha2!=alpha3 && Io_a!=alpha3->setname_up && Io_b!=alpha3->setname_dw) {	
					settings->bj_low[loop-1] += - ((alpha1->alpha_up - alpha1->alpha_dw) - (a1 - a1a)) * Calc_Kernel( alpha1, alpha3, settings ) - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * Calc_Kernel( alpha2, alpha3, settings ) ;
					if (0==index[alpha3-settings->alpha]) {
						alpha3->f_cache += - ((alpha1->alpha_up - alpha1->alpha_dw) - (a1 - a1a)) * Calc_Kernel( alpha1, alpha3, settings ) - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * Calc_Kernel( alpha2, alpha3, settings ) ;
						index[alpha3-settings->alpha] = 1 ;
					}
				} else { settings->bj_low[loop-1] = INT_MIN ; settings->ij_low[loop-1] = 0 ; }
			}
		}

		/*update f-cache of i1 & i2 if not in Io_Cache*/
		if (alpha1->setname_up != Io_a && alpha1->setname_dw != Io_b) {
			if (0==index[alpha1-settings->alpha]) {
				alpha1->f_cache = alpha1->f_cache - ((alpha1->alpha_up - alpha1->alpha_dw) - (a1 - a1a)) * K11 - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * K12 ;
				index[alpha1-settings->alpha] = 1 ;
			}
			alpha3=alpha1 ;
			if (alpha3->pair->target > 1 ) {
				loop = alpha3->pair->target - 2 ;
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_One) {
					if (alpha3->f_cache-1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache-1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_Fou) {
					if (alpha3->f_cache-1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache-1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
			if ( alpha3->pair->target < settings->pairs->classes ) {
				loop = alpha3->pair->target - 1 ;
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Thr) {
					if (alpha3->f_cache+1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache+1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Two) {
					if (alpha3->f_cache+1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache+1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
		}
		if (alpha2->setname_up != Io_a && alpha2->setname_dw != Io_b) {
			if (0==index[alpha2-settings->alpha]) {
				alpha2->f_cache = alpha2->f_cache - ((alpha1->alpha_up - alpha1->alpha_dw) - (a1 - a1a)) * K12 - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * K22 ;
				index[alpha2-settings->alpha] = 1 ;
			}
			alpha3=alpha2 ;			
			if (alpha3->pair->target > 1 ) {
				loop = alpha3->pair->target - 2 ;
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_One) {
					if (alpha3->f_cache-1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache-1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_Fou) {
					if (alpha3->f_cache-1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache-1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
			if ( alpha3->pair->target < settings->pairs->classes ) {
				loop = alpha3->pair->target - 1 ;
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Thr) {
					if (alpha3->f_cache+1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache+1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Two) {
					if (alpha3->f_cache+1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache+1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
		}			

		/*/ update Fi in Io_Cache and vote B_LOW & B_UP if possible*/
		cache = Io_CACHE.front ;
		while ( NULL != cache ) {	
			alpha3 = cache->alpha ;	
			if ( 0==index[alpha3-settings->alpha]) {
				alpha3->f_cache = alpha3->f_cache - ((alpha1->alpha_up - alpha1->alpha_dw) - (a1 - a1a)) * Calc_Kernel( alpha1, alpha3, settings ) - ((alpha2->alpha_up - alpha2->alpha_dw) - (a2 - a2a)) * Calc_Kernel( alpha2, alpha3, settings ) ;
				index[alpha3-settings->alpha] = 1 ;
			}
			if (alpha3->pair->target > 1 ) {
				loop = alpha3->pair->target - 2 ;
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_One) {
					if (alpha3->f_cache-1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache-1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_dw==Io_b || alpha3->setname_dw==I_Fou) {
					if (alpha3->f_cache-1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache-1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
			if ( alpha3->pair->target < settings->pairs->classes ) {
				loop = alpha3->pair->target - 1 ;
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Thr) {
					if (alpha3->f_cache+1<settings->bj_up[loop]) { settings->bj_up[loop] = alpha3->f_cache+1 ; settings->ij_up[loop] = alpha3 - settings->alpha + 1 ; }
				}
				if (alpha3->setname_up==Io_a || alpha3->setname_up==I_Two) {
					if (alpha3->f_cache+1>settings->bj_low[loop]) { settings->bj_low[loop] = alpha3->f_cache+1 ; settings->ij_low[loop] = alpha3 - settings->alpha + 1 ; }
				}
			}
			cache = cache->next ;				
		} /*/ end of while*/
		
		free(index) ;

		for (loop = 1 ; loop < settings->pairs->classes ; loop ++) {
			if (0==settings->ij_up[loop-1]||0==settings->ij_low[loop-1]) { 
				Check_Alphas ( settings->alpha, settings ) ;
				loop = settings->pairs->classes ;
			}
		}

#ifdef _ORDINAL_DEBUG

		for ( t1 = 1; t1 <= settings->pairs->count; t1 ++ )
		{	
			alpha3 = ALPHA + t1 - 1 ;
			printf("%u-target %u---func %f: alpha = %f , alpha* = %f\n",t1, alpha3->pair->target, alpha3->f_cache, alpha3->alpha_up, alpha3->alpha_dw) ;
		}
		for (t1=1;t1<settings->pairs->classes;t1++)
			printf("threshold %u : upper=%f(%u), lower=%f(%u), mu=%f\n", t1, settings->bj_up[t1-1],
				settings->ij_up[t1-1], settings->bj_low[t1-1], settings->ij_low[t1-1], settings->mu[t1-1]) ;
		deltamu = 0 ;
		for (t1=0;t1<settings->pairs->count;t1++)
		{
			alpha1 = ALPHA+t1 ;
			for (t2=0;t2<t1;t2++)
			{
				alpha2 = ALPHA+t2 ;
				deltamu += (-alpha1->alpha_up+alpha1->alpha_dw)
					*(-alpha2->alpha_up+alpha2->alpha_dw)*Calc_Kernel( alpha1, alpha2, settings ) ;
			}
			deltamu += 0.5*(-alpha1->alpha_up+alpha1->alpha_dw)
					*(-alpha1->alpha_up+alpha1->alpha_dw)*Calc_Kernel( alpha1, alpha1, settings ) ;
			deltamu -= (alpha1->alpha_up+alpha1->alpha_dw) ;
		}
		printf("objective functional %f\n",deltamu) ;
#endif

		/*/ update mu_bias*/
		for (loop = 1; loop < settings->pairs->classes; loop ++) {
			settings->bmu_low[loop-1]=settings->bj_low[loop-1] ; settings->imu_low[loop-1]=loop ;
			/*/ b_low^j=max{b_low^j-1,b_low^j}*/
			if (loop>1 && settings->bmu_low[loop-2]>settings->bmu_low[loop-1]) { settings->bmu_low[loop-1]=settings->bmu_low[loop-2] ; settings->imu_low[loop-1]=settings->imu_low[loop-2] ; }
		}
		for (loop = settings->pairs->classes-1; loop > 0; loop --) {
			settings->bmu_up[loop-1]=settings->bj_up[loop-1] ; settings->imu_up[loop-1]=loop ;
			/*/ b_up^j=min{b_up^j,b_up^j+1}*/
			if (loop<settings->pairs->classes-1 && settings->bmu_up[loop-1]>settings->bmu_up[loop]) { settings->bmu_up[loop-1]=settings->bmu_up[loop] ; settings->imu_up[loop-1]=settings->imu_up[loop] ; }           
		}
		for (loop = 2; loop < settings->pairs->classes; loop ++) {
			if (settings->mu[loop-1]>EPS*EPS) {
				if (settings->bmu_up[loop-1]>settings->bmu_up[loop-2]) { settings->bmu_up[loop-1]=settings->bmu_up[loop-2] ; settings->imu_up[loop-1]=settings->imu_up[loop-2] ; }
				if (settings->bmu_low[loop-2]<settings->bmu_low[loop-1]) { settings->bmu_low[loop-2]=settings->bmu_low[loop-1] ; settings->imu_low[loop-2]=settings->imu_low[loop-1] ; }
			}
		}

#ifdef _ORDINAL_DEBUG
		for (t1=1;t1<settings->pairs->classes;t1++)
			printf("threshold %u : mu_up=%f(%u), mu_low=%f(%u), mu=%f\n", t1, settings->bmu_up[t1-1],
				settings->imu_up[t1-1], settings->bmu_low[t1-1], settings->imu_low[t1-1], settings->mu[t1-1]) ;
#endif
		return TRUE ;
	} /*/ end of update */
	else
	{
		/*/printf("fail to update pairs %lu and %lu\n",i1,i2) ; */
		return FALSE ;
	}
} /*/ end of ordinal_cross_takestep */


/*******************************************************************************\

    BOOL ordinal_takestep ( Alphas * alpha1, Alphas * alpha2, unsigned int threshold, smo_Settings * settings )
    
    purpose: act as the main entry point to route the takestep execution to either 
             the SVOREX or SVORIM algorithm depending on the configuration.
    input:   alpha1 and alpha2 (pointers to Alphas), threshold (active threshold 
             index), settings (pointer to smo_Settings).
    output:  returns the boolean result (TRUE/FALSE) of the delegated takestep function.

\*******************************************************************************/

BOOL ordinal_takestep ( Alphas * alpha1, Alphas * alpha2, unsigned int threshold, smo_Settings * settings ) {
	if (settings->model_type == 0) return ordinal_takestep_SVOREX(alpha1, alpha2, threshold, settings);
	else return ordinal_takestep_SVORIM(alpha1, alpha2, threshold, settings);
} /*/ end of ordinal_takestep */

/*/ end of smoc_takestep.cpp*/