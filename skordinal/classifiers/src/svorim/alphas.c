/*******************************************************************************\

	alphas.c in Sequential Minimal Optimization ver2.0

	implements initialization for alphas matrix.

	Chu Wei Copyright(C) National Univeristy of Singapore
	Create on Jan. 16 2000 at Control Lab of Mechanical Engineering
	Update on Aug. 23 2001

\*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "smo.h"


/*******************************************************************************\

	Alphas * Create_Alphas ( smo_Settings * settings )

	create and initialize a structure matrix of Alphas from Data_List
	input:  the pointer to Data_List
	output: the pointer to the head of the structure matrix

\*******************************************************************************/

/*/ release the first count entries of a partially built block and the block
    itself, so a failed Create_Alphas leaves nothing allocated*/
static void Free_Alphas_Block ( Alphas * alphas, unsigned int count )
{
	unsigned int i ;

	if ( NULL == alphas )
		return ;

	for (i=0;i<count;i++)
	{
		if (NULL != alphas[i].setname)
			free(alphas[i].setname) ;
		if (NULL != alphas[i].alpha_j)
			free(alphas[i].alpha_j) ;
		if (NULL != alphas[i].kernel)
			free(alphas[i].kernel) ;
	}
	free(alphas) ;
}

Alphas * Create_Alphas ( smo_Settings * settings )
{
	Data_Node * pair = NULL ;
	Alphas * alpha = NULL ;
	Alphas * alphas = NULL ;
	Data_List * pairs = NULL ;
	unsigned int  i = 0, j ;

	if ( NULL == settings )
	{
		printf("\r\nFATAL ERROR : input is NULL in Create_Alphas.\r\n") ;
		return NULL ;
	}

	if ( NULL == (pairs = settings->pairs) )
	{
		printf("\r\nFATAL ERROR : data list is NULL in Create_Alphas.\r\n") ;
		return NULL ;
	}

	if ( TRUE == Is_Data_Empty( pairs ) || pairs->count < MINNUM )
	{
		printf( "\r\nFATAL ERROR : Data_List have not be initialized.\r\n") ;
		return  NULL ;
	}

	if ( NULL == (alphas = (Alphas *) malloc( pairs->count*sizeof(Alphas) )) )
	{
		printf( "\r\nFATAL ERROR : fail to malloc Alphas block.\r\n") ;
		return NULL ;
	}

	pair = pairs->front ;
	while ( pair != NULL )
	{
		alpha = alphas + i ;
		i++ ;
		alpha->alpha = 0 ;
		alpha->f_cache = 0 ;
		alpha->pair = pair ;
		alpha->kernel = NULL ;
		alpha->alpha_j = NULL ;
		alpha->setname = NULL ;
		alpha->cache = NULL ;
		alpha->kernel = (double *) malloc(i*sizeof(double)) ;
		if ( NULL == alpha->kernel )
			printf("Fatal Error : fail to malloc memory.\r\n") ;
		else
		{
			/*/ initial the kernel matrix cache*/
			for (j=0 ; j<i ; j++)
				alpha->kernel[j] = Calc_Kernel(alpha, alphas+j, settings) ;
		}
		if (ORDINAL == pairs->datatype)
		{
			alpha->alpha_j = (double *) calloc(settings->pairs->classes-1,sizeof(double)) ;
			alpha->setname = (Set_Name * ) malloc((settings->pairs->classes-1)*sizeof(Set_Name)) ;
			if (NULL == alpha->alpha_j || NULL == alpha->setname)
			{
				printf("\r\nFatal Error : fail to malloc alpha->setname.\r\n") ;
				Free_Alphas_Block (alphas, i) ;
				return NULL ;
			}
			for (j=0;j<settings->pairs->classes-1;j++)
				alpha->setname[j] = Get_Ordinal_Label (alpha, j+1, settings) ;
		}
		else
		{
			printf("Error datatype.\n") ;
			Free_Alphas_Block (alphas, i) ;
			return NULL;
		}
		pair = pair->next ;
	}
	return alphas ;
} /*/ end of Create_Alphas*/


BOOL Clear_Alphas ( smo_Settings * settings )
{
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

	/*/ Create_Alphas may have failed, leaving nothing to release*/
	if ( NULL == ALPHA )
		return TRUE ;

	for (i=0;i<settings->pairs->count;i++)
	{
		alpha = ALPHA + i ;
		if (NULL != alpha->setname)
			free(alpha->setname) ;
		if (NULL != alpha->alpha_j)
			free(alpha->alpha_j) ;
		if (NULL != alpha->kernel)
			free(alpha->kernel) ;
	}
	free(ALPHA) ;
	return TRUE ;

} /*/ end of Clear_Alphas*/


/*******************************************************************************\

	BOOL Clean_Alphas ( Alphas *, smo_Settings * settings )

	set all the elements in the matrix to be the default values
	input:  the pointer to the head of Alphas matrix and the pointer to Data_List
	output: TRUE or FALSE

\*******************************************************************************/

BOOL Clean_Alphas ( Alphas * alphas, smo_Settings * settings )
{
	Alphas * alpha ;
	unsigned int  i = 0, j ;
	Data_Node * node = NULL ;
	Data_List * pairs = NULL ;

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

	node = pairs->front ;
	while (NULL != node)
	{
		alpha = alphas + i ;
		i++ ;
		alpha->alpha = 0 ;
		alpha->f_cache = 0 ;

		if ( ORDINAL == pairs->datatype )
		{
			for (j=0;j<settings->pairs->classes-1;j++)
			{
				alpha->alpha_j[j] = 0 ;
				alpha->setname[j] = Get_Ordinal_Label (alpha, j+1, settings) ;
			}
		}
		else
		{
			printf("Error datatype.\n") ;
			return FALSE ;
		}
		alpha->cache = NULL ; /*/ clear the reference to Io_Cache here*/
		alpha->pair = node ;
		node = node->next ;
	}
	return TRUE ;

} /*/ end of Clean_Alphas*/


BOOL Check_Alphas ( Alphas * alphas, smo_Settings * settings )
{
	Alphas * alpha ;
	unsigned int loop = 0 ;
	Data_Node * node = NULL ;
	Data_List * pairs = NULL ;
	long int i = 0 ;
	unsigned int j ;

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
	while (NULL != node)
	{
		alpha = alphas + i ;
		if ( ORDINAL == pairs->datatype )
		{
			alpha->alpha = 0 ;
			for (j=0;j<pairs->classes-1;j++)
			{
				if (alpha->alpha_j[j] > VC)
					alpha->alpha_j[j] = VC ;
				else if (alpha->alpha_j[j] < 0)
					alpha->alpha_j[j] = 0 ;
				alpha->setname[j] = Get_Ordinal_Label (alpha, j+1, settings) ;
				/*/ effective coefficient: signed sum of the per-threshold multipliers*/
				if (alpha->pair->target<=j+1)
					alpha->alpha -= alpha->alpha_j[j] ;
				else
					alpha->alpha += alpha->alpha_j[j] ;
			}
		}
		else
		{
			printf("Error datatype.\n") ;
			return FALSE ;
		}
		alpha->f_cache = Calculate_Ordinal_Fi(i+1,settings) ;
		alpha->cache = NULL ; /*/ clear the reference to Io_Cache here*/
		if (alpha->pair != node)
			printf("error in alpha or data list.\r\n") ;
		node = node->next ;
		i += 1 ;
	}

	/*/ initial b_up b_low*/
	for (loop = 1 ; loop < settings->pairs->classes ; loop ++)
	{
		settings->bj_up[loop-1] = (double)INT_MAX ;
		settings->bj_low[loop-1] = (double)INT_MIN ;
		settings->ij_up[loop-1] = 0 ;
		settings->ij_low[loop-1] = 0 ;
	}
	/*/ create Io_cache*/
	i = 0 ;
	node = pairs->front ;
	while (NULL != node)
	{
		alpha = alphas + i ;
		i += 1 ;
		if (TRUE == Is_Io(alpha,settings))
		{

			Add_Cache_Node(&settings->io_cache, alpha) ;
		}
		for (loop = 0 ; loop < pairs->classes-1 ; loop ++)
		{
			if (alpha->pair->target > (loop+1.5) )
			{
				/*/lower*/
				if (alpha->setname[loop]==Io_b || alpha->setname[loop]==I_One)
				{
					if (alpha->f_cache-1<settings->bj_up[loop])
					{
						settings->bj_up[loop] = alpha->f_cache-1 ;
						settings->ij_up[loop] = alpha - ALPHA + 1 ;
					}
				}
				if (alpha->setname[loop]==Io_b || alpha->setname[loop]==I_Fou)
				{
					if (alpha->f_cache-1>settings->bj_low[loop])
					{
						settings->bj_low[loop] = alpha->f_cache-1 ;
						settings->ij_low[loop] = alpha - ALPHA + 1 ;
					}
				}
			}
			else
			{

				if (alpha->setname[loop]==Io_a || alpha->setname[loop]==I_Thr)
				{
					if (alpha->f_cache+1<settings->bj_up[loop])
					{
						settings->bj_up[loop] = alpha->f_cache+1 ;
						settings->ij_up[loop] = alpha - ALPHA + 1 ;
					}
				}
				if (alpha->setname[loop]==Io_a || alpha->setname[loop]==I_Two)
				{
					if (alpha->f_cache+1>settings->bj_low[loop])
					{
						settings->bj_low[loop] = alpha->f_cache+1 ;
						settings->ij_low[loop] = alpha - ALPHA + 1 ;
					}
				}
			}
		}
		node = node->next ;
	}
	return TRUE ;
} /*/ end of Check_Alphas*/


/*******************************************************************************\

	BOOL Finalize_Alphas ( Alphas *, smo_Settings * settings )

	recompute the scalar effective coefficient (signed sum of the per-threshold
	multipliers) for every alpha once the SMO loop has converged, since
	Check_Alphas only runs before optimization starts and alpha_j keeps
	changing throughout the loop. The predict path reads this scalar, so
	smo_routine_Python gates its success return on this call.
	input:  the pointer to the head of Alphas matrix and the pointer to smo_Settings
	output: TRUE or FALSE

\*******************************************************************************/

BOOL Finalize_Alphas ( Alphas * alphas, smo_Settings * settings )
{
	Alphas * alpha ;
	Data_Node * node = NULL ;
	Data_List * pairs = NULL ;
	long int i = 0 ;
	unsigned int j ;

	if ( NULL == alphas || NULL == settings )
	{
		printf("\r\nFATAL ERROR : input is NULL in Finalize_Alphas.\r\n") ;
		return FALSE ;
	}

	if ( NULL == (pairs = settings->pairs) )
	{
		printf("\r\nFATAL ERROR : input is NULL in Finalize_Alphas.\r\n") ;
		return FALSE ;
	}

	node = pairs->front ;
	while (NULL != node)
	{
		alpha = alphas + i ;
		alpha->alpha = 0 ;
		for (j=0;j<pairs->classes-1;j++)
		{
			if (alpha->pair->target<=j+1)
				alpha->alpha -= alpha->alpha_j[j] ;
			else
				alpha->alpha += alpha->alpha_j[j] ;
		}
		node = node->next ;
		i += 1 ;
	}
	return TRUE ;
} /*/ end of Finalize_Alphas*/

/*/ the end of alphas.c*/
