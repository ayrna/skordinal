#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/types.h>
#include <sys/timeb.h>
#include "smo.h"


unsigned int active_threshold (smo_Settings * settings)
{
	unsigned int i, j = 0 ;
	double active = 0 ;
	double temp = 0 ;

	if (NULL == settings)
	{
		printf("error in the input pointer.\n") ;
		return j ;
	}
	for (i=1;i<settings->pairs->classes;i++)
	{
		temp = settings->bj_low[i-1]-settings->bj_up[i-1] ;
		if (temp>active && temp>TOL)
		{
			active = temp ;
			j = i ;
		}
	}
	return j ; /* optimal j=0 */
}

BOOL ordinal_examine_example ( Alphas * alpha, smo_Settings * settings )
{
	double F2 = 0 ;
	unsigned int j = 0 ;
	unsigned int loop ;
	long unsigned int i1 = 0 ;
	long unsigned int i2 = 0 ;
	BOOL optimal = TRUE ;

	if ( NULL == alpha || NULL == settings )
		return FALSE ;

	if (ORDINAL != settings->pairs->datatype)
		return FALSE ;

	i2 = alpha - ALPHA + 1 ;
#ifdef SMO_DEBUG
	if ( i2 > Pairs.count )
	{
		printf ( "Error input index %d in examineAll\n", i2 ) ;
		return FALSE ;
	}
#endif


	if ( FALSE == Is_Io(alpha,settings) )
	{
		alpha->f_cache = Calculate_Ordinal_Fi(i2, settings) ;

		for (loop = 0 ; loop < settings->pairs->classes-1 ; loop ++)
		{
			if (alpha->pair->target > (loop+1) )
			{

				if (alpha->setname[loop]==Io_b || alpha->setname[loop]==I_One)
				{
					if (alpha->f_cache-1<=settings->bj_up[loop])
					{
						settings->bj_up[loop] = alpha->f_cache-1 ;
						settings->ij_up[loop] = alpha - ALPHA + 1 ;
					}
				}
				if (alpha->setname[loop]==Io_b || alpha->setname[loop]==I_Fou)
				{
					if (alpha->f_cache-1>=settings->bj_low[loop])
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
					if (alpha->f_cache+1<=settings->bj_up[loop])
					{
						settings->bj_up[loop] = alpha->f_cache+1 ;
						settings->ij_up[loop] = alpha - ALPHA + 1 ;
					}
				}
				if (alpha->setname[loop]==Io_a || alpha->setname[loop]==I_Two)
				{
					if (alpha->f_cache+1>=settings->bj_low[loop])
					{
						settings->bj_low[loop] = alpha->f_cache+1 ;
						settings->ij_low[loop] = alpha - ALPHA + 1 ;
					}
				}
			}
		}
	}


	for (loop = 0 ; loop < settings->pairs->classes-1 ; loop ++)
	{
		if (alpha->pair->target > (loop+1) )
		{

			if (alpha->setname[loop]==Io_b || alpha->setname[loop]==I_One)
			{
				if ( settings->bj_low[loop] - (alpha->f_cache-1) > TOL )
				{
					optimal = FALSE ;
					if (settings->bj_low[loop]-(alpha->f_cache-1)>F2)
					{
						i1 = settings->ij_low[loop] ;
						F2 = settings->bj_low[loop]-(alpha->f_cache-1) ;
						j = loop+1 ;
					}
				}
			}
			if (alpha->setname[loop]==Io_b || alpha->setname[loop]==I_Fou)
			{
				if ( (alpha->f_cache-1) - settings->bj_up[loop] > TOL )
				{
					optimal = FALSE ;
					if ((alpha->f_cache-1) - settings->bj_up[loop]>F2)
					{
						i1 = settings->ij_up[loop] ;
						F2 = (alpha->f_cache-1) - settings->bj_up[loop] ;
						j = loop+1 ;
					}
				}
			}
		}
		else
		{

			if (alpha->setname[loop]==Io_a || alpha->setname[loop]==I_Thr)
			{
				if (settings->bj_low[loop]-(alpha->f_cache+1)>TOL)
				{
					optimal = FALSE ;
					if (settings->bj_low[loop]-(alpha->f_cache+1)>F2)
					{
						i1 = settings->ij_low[loop] ;
						F2 = settings->bj_low[loop]-(alpha->f_cache+1) ;
						j = loop+1 ;
					}
				}
			}
			if (alpha->setname[loop]==Io_a || alpha->setname[loop]==I_Two)
			{
				if ((alpha->f_cache+1)-settings->bj_up[loop]>TOL)
				{
					optimal = FALSE ;
					if ((alpha->f_cache+1)-settings->bj_up[loop]>F2)
					{
						i1 = settings->ij_up[loop] ;
						F2 = (alpha->f_cache+1)-settings->bj_up[loop] ;
						j = loop+1 ;
					}
				}
			}
		}
	}

	if (optimal == FALSE)
	{
		if (TRUE ==  ordinal_takestep( ALPHA + i1 - 1, ALPHA + i2 - 1, j , settings) )
			return TRUE ;
		else
		{
			if ( TRUE == SMO_DISPLAY )
			{
				printf("%lu and %lu failed in takestep.\n",i1,i2) ;
			}
			return TRUE ;
		}
	}
	return FALSE ;
}
