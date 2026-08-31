/*******************************************************************************\

	alphas.c in Sequential Minimal Optimization ver2.0
		
	implements initialization for alphas matrix.
		
	Chu Wei Copyright(C) National Univeristy of Singapore
	Create on Jan. 16 2000 at Control Lab of Mechanical Engineering 
	Update on Aug. 23 2001 

\*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "smo.h"


/*******************************************************************************\

	Alphas * Create_Alphas ( smo_Settings * settings )
	
	purpose: create and initialize a structure matrix of Alphas from Data_List 
	input:  the pointer to Data_List / smo_Settings
	output: the pointer to the head of the structure matrix for alphas

\*******************************************************************************/

Alphas * Create_Alphas ( smo_Settings * settings ) {
	Data_Node * pair = NULL ; 	
	Alphas * alpha = NULL ;
	Alphas * alphas = NULL ;
	Data_List * pairs = NULL ;
	unsigned int  i = 0, j ;
	
		if ( NULL == settings )
	{
		printf("\nFATAL ERROR : input is NULL in Create_Alphas.\n") ;
		return NULL ;
	}

	if ( NULL == (pairs = settings->pairs) )
	{
		printf("\nFATAL ERROR : data list is NULL in Create_Alphas.\n") ;
		return NULL ;
	}

	if ( TRUE == Is_Data_Empty( pairs ) || pairs->count < MINNUM )
	{ 
		printf( "\nFATAL ERROR : Data_List have not be initialized.\n") ;
		return  NULL ;
	}

	if ( NULL == (alphas = (Alphas *) malloc( pairs->count*sizeof(Alphas) )) )
	{
		printf( "\nFATAL ERROR : fail to malloc Alphas block.\n") ;
		exit(1) ;		
	}	

	pair = pairs->front ;
	while ( pair != NULL ) {		
		alpha = alphas + i ; i++ ;	
		alpha->f_cache = 0 ;
		alpha->pair = pair ;
		alpha->kernel = (double *) malloc(i*sizeof(double)) ;
		if ( NULL == alpha->kernel )
		{
			printf("Fatal Error : fail to malloc kernel cache.\n") ;
			exit(1) ;
		}
		else
		{
			/* initial the kernel matrix cache*/
			for (j=0 ; j<i ; j++)
				alpha->kernel[j] = Calc_Kernel(alpha, alphas+j, settings) ;	
		}
        
		if (ORDINAL == pairs->datatype) {
			alpha->alpha = 0 ;	// Shared prediction weight
			if (settings->model_type == 0) { // SVOREX
				alpha->alpha_up = 0 ;
				alpha->alpha_dw = 0 ;
				alpha->setname_up = Get_UP_Label (alpha, settings) ;
				alpha->setname_dw = Get_DW_Label (alpha, settings) ;
				alpha->alpha_ptr = NULL;
				alpha->setname_ptr = NULL;
			} else { // SVORIM
				alpha->alpha_up = 0 ;
				alpha->alpha_dw = 0 ;
				alpha->alpha_ptr = (double *) calloc(settings->pairs->classes-1,sizeof(double)) ;
				alpha->setname_ptr = (Set_Name * ) malloc((settings->pairs->classes-1)*sizeof(Set_Name)) ;
				if (NULL == alpha->alpha_ptr || NULL == alpha->setname_ptr)
				{
					printf("\nFatal Error : fail to malloc alpha->setname_ptr.\n") ;
					exit(1) ;
				}
				for (j=0;j<settings->pairs->classes-1;j++)
					alpha->setname_ptr[j] = Get_Ordinal_Label (alpha, j+1, settings) ;
			}
		}
		alpha->cache = NULL ;
		pair = pair->next ;
	}
	return alphas ;
} /*/ end of Create_Alphas*/


/*******************************************************************************\

	BOOL Clear_Alphas ( smo_Settings * settings )
	
	purpose: clear the structure matrix of Alphas from smo_Settings
	input:  the pointer to smo_Settings
	output: TRUE or FALSE

\*******************************************************************************/


BOOL Clear_Alphas ( smo_Settings * settings ) {
	Alphas * alpha ;
	unsigned int  i = 0 ;
	Data_List * pairs = NULL ;	

	if ( NULL == settings )
	{
		printf("\r\nFATAL ERROR : input is NULL in Create_Alphas.\r\n") ;
		return FALSE ;
	}

	if ( NULL == (pairs = settings->pairs) )
	{
		printf("\r\nFATAL ERROR : input is NULL in Create_Alphas.\r\n") ;
		return FALSE ;
	}

	for (i=0;i<settings->pairs->count;i++) {
		alpha = ALPHA + i ;

		if (NULL != alpha->kernel)
			free(alpha->kernel) ;
		if (settings->model_type == 1) { // SVORIM Arrays
			if (NULL != alpha->alpha_ptr)
				free(alpha->alpha_ptr) ;
			if (NULL != alpha->setname_ptr)
				free(alpha->setname_ptr) ;
		}
	}
	free(ALPHA) ;
	return TRUE ;
} /*/ end of Clear_Alphas*/


/*******************************************************************************\

	BOOL Clean_Alphas ( Alphas *, smo_Settings * settings )
	
	purpose: set all the elements in the matrix to be the default values 
	input:  the pointer to the head of Alphas matrix and the pointer to smo_Settings / Data_List
	output: TRUE or FALSE

\*******************************************************************************/

BOOL Clean_Alphas ( Alphas * alphas, smo_Settings * settings ) {
	Alphas * alpha ;
	unsigned int  i = 0, j ;
	Data_Node * node = NULL ;
	Data_List * pairs = NULL ;	
	
	if ( NULL == alphas || NULL == settings )
	{
		printf("\nFATAL ERROR : input is NULL in Create_Alphas.\n") ;
		return FALSE ;
	}

	if ( NULL == (pairs = settings->pairs) )
	{
		printf("\nFATAL ERROR : input is NULL in Create_Alphas.\n") ;
		return FALSE ;
	}

	if(settings->model_type == 0){
		for (i = 1 ; i < settings->pairs->classes ; i ++)
			settings->mu[i-1] = 0 ;
	}

	i=0 ;
	node = settings->pairs->front ;
	while (NULL != node) {		
		alpha = alphas + i ;
		i++ ;
		alpha->f_cache = 0 ;

		if (settings->model_type == 0){
			alpha->alpha = 0 ;
			if ( ORDINAL == settings->pairs->datatype ) {
				alpha->alpha = 0 ;
				alpha->alpha_up = 0 ;
				alpha->alpha_dw = 0 ;
				alpha->setname_up = Get_UP_Label (alpha, settings) ;
				alpha->setname_dw = Get_DW_Label (alpha, settings) ;
			}else{
				printf("Error datatype.\n") ;
				exit(1) ;
			}
		}else{
			for (j=0;j<settings->pairs->classes-1;j++)
			{
				alpha->alpha_ptr[j] = 0 ;	
				alpha->setname_ptr[j] = Get_Ordinal_Label (alpha, j+1, settings) ;
			}
		}
		
		alpha->cache = NULL ;
		alpha->pair = node ;
		node = node->next ;
	}
	return TRUE ;
} /*/ end of Clean_Alphas*/


/*******************************************************************************\

	BOOL Check_Alphas ( Alphas *, smo_Settings * settings )
	
	purpose: check the validation of the Alphas matrix and then itialize the bias terms 
	input:  the pointer to the head of Alphas matrix and the pointer to smo_Settings 
	output: TRUE or FALSE

\*******************************************************************************/

BOOL Check_Alphas ( Alphas * alphas, smo_Settings * settings ) {
	Alphas * alpha ;
	unsigned int loop = 0, j ;
	Data_Node * node = NULL ;
	Data_List * pairs = NULL ;
	long int i = 0 ; 

	if ( NULL == alphas || NULL == settings )
	{
		printf("\r\nFATAL ERROR : input is NULL in Create_Alphas.\r\n") ;
		return FALSE ;
	}

	if ( NULL == (pairs = settings->pairs) )
	{
		printf("\r\nFATAL ERROR : input is NULL in Create_Alphas.\r\n") ;
		return FALSE ;
	}
	Clear_Cache_List( &(Io_CACHE) ) ;
		
	node = pairs->front ;
	while (NULL != node) {		
		alpha = alphas + i ;
		
		if (settings->model_type == 0) { // SVOREX
			if ( ORDINAL == pairs->datatype )
			{	
			if (alpha->alpha_up > settings->vc) alpha->alpha_up = settings->vc ;
			if (alpha->alpha_dw > settings->vc) alpha->alpha_dw = settings->vc ;
			if (alpha->alpha_up < 0) alpha->alpha_up = 0 ;
			if (alpha->alpha_dw < 0) alpha->alpha_dw = 0 ;		
			alpha->alpha = - alpha->alpha_up + alpha->alpha_dw ;			
			alpha->setname_up = Get_UP_Label (alpha, settings) ;
			alpha->setname_dw = Get_DW_Label (alpha, settings) ;
			}
			else
			{
				printf("Error datatype.\n") ;
				exit(1) ;
			}
		} else { // SVORIM
			for (j=0;j<settings->pairs->classes-1;j++) {			
				if (alpha->alpha_ptr[j] > settings->vc)
					alpha->alpha_ptr[j] = settings->vc ;
				else if (alpha->alpha_ptr[j] < 0)
					alpha->alpha_ptr[j] = 0 ;
				alpha->setname_ptr[j] = Get_Ordinal_Label (alpha, j+1, settings) ; 
			}
		}
		alpha->f_cache = Calculate_Ordinal_Fi(i+1, settings) ;
		alpha->cache = NULL ;
		if (alpha->pair != node)
			printf("error in alpha or data list.\n") ;	
		node = node->next ;
		i++ ;
	}

	/*/ initial b_up b_low		*/
	for (loop = 1 ; loop < settings->pairs->classes ; loop ++) {
		settings->bj_up[loop-1] = (double)INT_MAX ;
		settings->bj_low[loop-1] = (double)INT_MIN ;
		settings->ij_up[loop-1] = 0 ; settings->ij_low[loop-1] = 0 ;
	}
	
	i = 0 ;
	node = pairs->front ;
	while (NULL != node) {
		alpha = alphas + i ;
		i++ ;

		if (settings->model_type == 0) { // SVOREX
			if ( alpha->setname_dw==Io_b || alpha->setname_up==Io_a ) Add_Cache_Node(&settings->io_cache, alpha) ;			
			if (alpha->pair->target > 1 ) {
				loop = alpha->pair->target - 2 ;
				/*/lower*/
				if (alpha->setname_dw==Io_b || alpha->setname_dw==I_One) {
					if (alpha->f_cache-1 < settings->bj_up[loop]) { settings->bj_up[loop] = alpha->f_cache-1 ; settings->ij_up[loop] = alpha - ALPHA + 1 ; }
				}
				if (alpha->setname_dw==Io_b || alpha->setname_dw==I_Fou) {
					if (alpha->f_cache-1 > settings->bj_low[loop]) { settings->bj_low[loop] = alpha->f_cache-1 ; settings->ij_low[loop] = alpha - ALPHA + 1 ; }
				}
			}
			if ( alpha->pair->target < settings->pairs->classes ) {
				loop = alpha->pair->target - 1 ;
				/*/upper*/
				if (alpha->setname_up==Io_a || alpha->setname_up==I_Thr) {
					if (alpha->f_cache+1 < settings->bj_up[loop]) { settings->bj_up[loop] = alpha->f_cache+1 ; settings->ij_up[loop] = alpha - ALPHA + 1 ; }
				}
				if (alpha->setname_up==Io_a || alpha->setname_up==I_Two) {
					if (alpha->f_cache+1 > settings->bj_low[loop]) { settings->bj_low[loop] = alpha->f_cache+1 ; settings->ij_low[loop] = alpha - ALPHA + 1 ; }
				}
			}
		} else { // SVORIM
			if (TRUE == Is_Io(alpha, settings)) Add_Cache_Node(&settings->io_cache, alpha) ;			
			for (loop = 0 ; loop < settings->pairs->classes-1 ; loop ++) {
				if (alpha->pair->target > (loop+1) ) {
					/*lower*/
					if (alpha->setname_ptr[loop]==Io_b || alpha->setname_ptr[loop]==I_One) {
						if (alpha->f_cache-1 < settings->bj_up[loop]) { settings->bj_up[loop] = alpha->f_cache-1 ; settings->ij_up[loop] = alpha - ALPHA + 1 ; }
					}
					if (alpha->setname_ptr[loop]==Io_b || alpha->setname_ptr[loop]==I_Fou) {
						if (alpha->f_cache-1 > settings->bj_low[loop]) { settings->bj_low[loop] = alpha->f_cache-1 ; settings->ij_low[loop] = alpha - ALPHA + 1 ; }
					}
				} else {
					if (alpha->setname_ptr[loop]==Io_a || alpha->setname_ptr[loop]==I_Thr) {
						if (alpha->f_cache+1 < settings->bj_up[loop]) { settings->bj_up[loop] = alpha->f_cache+1 ; settings->ij_up[loop] = alpha - ALPHA + 1 ; }
					}
					if (alpha->setname_ptr[loop]==Io_a || alpha->setname_ptr[loop]==I_Two) {
						if (alpha->f_cache+1 > settings->bj_low[loop]) { settings->bj_low[loop] = alpha->f_cache+1 ; settings->ij_low[loop] = alpha - ALPHA + 1 ; }
					}
				}
			}
		}
		node = node->next ;	
	}
	if (settings->model_type == 1){
		return TRUE;
	}else{
		#ifdef _ORDINAL_DEBUG
			for (loop = 1 ; loop < settings->pairs->classes ; loop ++)
			{
					if (0==settings->ij_up[loop-1]||0==settings->ij_low[loop-1])
					{
							printf("FATAL ERROR>\n");
							for (loop=1;loop<settings->pairs->classes;loop++)
								printf("threshold %lu --- %u: up=%f(%lu), low=%f(%lu), mu=%f\n", loop,settings->pairs->labels[loop-1], settings->bj_up[loop-1],
								settings->ij_up[loop-1],settings->bj_low[loop-1],settings->ij_low[loop-1],settings->mu[loop-1]) ;
							printf("\n") ;
							for ( loop = 1; loop <= settings->pairs->count; loop ++ )
							{
								alpha = ALPHA + loop - 1 ;
								printf("%u-target %u---func %f: alpha = %f , alpha* = %f\n",loop, alpha->pair->target, alpha->f_cache, alpha->alpha_up, alpha->alpha_dw) ;
							}								
							loop = settings->pairs->classes ;
					}
			}
		#endif
			/*/ check cross updating*/
			return TRUE ;
		}

} /* end of Check_Alphas */

// the end of alphas.c