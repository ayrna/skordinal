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


unsigned int active_cross_threshold (smo_Settings * settings) 
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
		temp = settings->bmu_low[i-1]-settings->bmu_up[i-1] ;
		if (temp>active && temp>TOL)
		{
			active = temp ;		
			j = i ; 
		}
	}
	return j ;
}


BOOL ordinal_examine_example ( Alphas * alpha, smo_Settings * settings )
{
	double F2 = 0 ;
	unsigned int y2 = 0 ;
	long unsigned int i1 = 0 ;
	long unsigned int i2 = 0 ;
	long unsigned int i3 = 0 ;
	unsigned int b1 = 0 ;
	unsigned int b2 = 0 ;
	BOOL optimal = TRUE ; 
	Set_Name set_up ;
	Set_Name set_dw ;
	unsigned int loop = 0 ;

	if ( NULL == alpha || NULL == settings )
		return FALSE ;

	if (ORDINAL != settings->pairs->datatype)
		return FALSE ;

	i2 = alpha - ALPHA + 1 ;
#ifdef SMO_DEBUG
	if ( fabs((double)i2) > Pairs.count )
	{
		printf ( "Error input index %d in examineAll\n", i2 ) ;
		return FALSE ;
	}
#endif
	set_up = alpha->setname_up ;
	set_dw = alpha->setname_dw ;
	y2 = alpha->pair->target ;

    /*calculate F2 if i2 in I_One or not in I_One*/
	if ( set_up == Io_a || set_dw == Io_b )
		F2 = alpha->f_cache ;
	else
	{
		F2 = Calculate_Ordinal_Fi(i2, settings) ;
		alpha->f_cache = F2 ;		
		if (y2<settings->pairs->classes)
		{
			if ( (I_Thr == set_up || Io_a == set_up) && (F2+1 < settings->bj_up[y2-1]) )
			{
				/* upper */ 
				settings->bj_up[y2-1] = F2+1 ;
				settings->ij_up[y2-1] = i2 ;
			}
			if ( (I_Two == set_up || Io_a == set_up ) && (F2+1 > settings->bj_low[y2-1]) )
			{
				/* lower*/
				settings->bj_low[y2-1] = F2+1 ;
				settings->ij_low[y2-1] = i2 ;
			}
		}		
		if  (y2>1)
		{
			if ( (I_One == set_dw || Io_b == set_dw) && (F2-1 < settings->bj_up[y2-2]) )
			{
				/* upper */
				settings->bj_up[y2-2] = F2-1 ;
				settings->ij_up[y2-2] = i2 ;
			}
			if ( (I_Fou == set_dw || Io_b == set_dw) && (F2-1 > settings->bj_low[y2-2]) )
			{
				/* lower*/
				settings->bj_low[y2-2] = F2-1 ;
				settings->ij_low[y2-2] = i2 ;
			}
		}
	}

	/* update mu_bias*/
	for (loop = 1; loop < settings->pairs->classes; loop ++)
	{
		settings->bmu_low[loop-1]=settings->bj_low[loop-1] ;
		settings->imu_low[loop-1]=loop ;
		if (loop>1)
		{	
			/* b_low^j=max{b_low^j-1,b_low^j}*/
			if (settings->bmu_low[loop-2]>settings->bmu_low[loop-1])
			{
				settings->bmu_low[loop-1]=settings->bmu_low[loop-2] ;
				settings->imu_low[loop-1]=settings->imu_low[loop-2] ;
			}
		}
	}
	for (loop = settings->pairs->classes-1; loop > 0; loop --)
	{
		settings->bmu_up[loop-1]=settings->bj_up[loop-1] ;
		settings->imu_up[loop-1]=loop ;
		if (loop<settings->pairs->classes-1)
		{
			/* b_up^j=min{b_up^j,b_up^j+1}*/
			if (settings->bmu_up[loop-1]>settings->bmu_up[loop])
			{
				settings->bmu_up[loop-1]=settings->bmu_up[loop] ;
				settings->imu_up[loop-1]=settings->imu_up[loop] ;
			}			
		}
	}
	for (loop = 2; loop < settings->pairs->classes; loop ++)
	{
		if (settings->mu[loop-1]>EPS*EPS) 
		{
			if (settings->bmu_up[loop-1]>settings->bmu_up[loop-2])
			{
				settings->bmu_up[loop-1]=settings->bmu_up[loop-2] ;
				settings->imu_up[loop-1]=settings->imu_up[loop-2] ;
			}
			if (settings->bmu_low[loop-2]<settings->bmu_low[loop-1])
			{
				settings->bmu_low[loop-2]=settings->bmu_low[loop-1] ;
				settings->imu_low[loop-2]=settings->imu_low[loop-1] ;
			}
		}
	}
	

	/* find an index in i_low or i_up, to do joint optimization */
	if (y2<settings->pairs->classes) 
	{
		/* check upper part */
		if ( Io_a == set_up || I_Thr == set_up )
		{
			if ( settings->bmu_low[y2-1] - (F2+1) > TOL )
			{
				optimal = FALSE ;
				i1 = i2 ;
				b1 = y2 ;
				i3 = settings->ij_low[settings->imu_low[y2-1]-1] ;
				b2 = settings->imu_low[y2-1] ;
			}
		}
		if ( Io_a == set_up || I_Two == set_up )
		{
			if ( (F2+1) - settings->bmu_up[y2-1] > TOL )
			{
				optimal = FALSE ;
				i1 = settings->ij_up[settings->imu_up[y2-1]-1] ;
				b1 = settings->imu_up[y2-1] ;
				b2 = y2 ;
				i3 = i2 ;
			}
		}
		if (optimal == FALSE)
		{
			if ( set_up == Io_a )
			{
				if ( settings->bmu_low[y2-1] - (F2+1) > (F2+1) - settings->bmu_up[y2-1] )
				{
					i1 = i2 ;
					b1 = y2 ;
					b2 = settings->imu_low[y2-1] ;
					i3 = settings->ij_low[settings->imu_low[y2-1]-1] ;					
				}
				else
				{
					i1 = settings->ij_up[settings->imu_up[y2-1]-1] ;
					b1 = settings->imu_up[y2-1] ;
					b2 = y2 ;
					i3 = i2 ;
				}
			}
			
			if (i1==i3)
			{
				if (TRUE == ordinal_cross_identical( ALPHA + i1 - 1, ALPHA + i3 - 1, y2, settings) )
					return TRUE ;
				else 
					printf("%lu and %lu failed in identical takestep.\n",i1,i3) ;
			}
			if (b1==b2)
			{			
				if (TRUE ==  ordinal_takestep( ALPHA + i1 - 1, ALPHA + i3 - 1, y2 , settings) )
					return TRUE ;
				else 
					printf("%lu and %lu failed in takestep.\n",i1,i3) ;
			}
			else
			{			
				if (TRUE ==  ordinal_cross_takestep( ALPHA + i1 - 1,b1, ALPHA + i3 - 1, b2 , settings) )
					return TRUE ;
				else 
					printf("%lu and %lu failed in cross takestep.\n",i1,i3) ;
			}
		}
		/*else //y2!=settings->pairs->classes
		{
			// check cross updating
			if ( (Io_a == set_up||I_Thr == set_up) && (y2>1) )
			{
				if ( settings->bj_low[y2-2] - (F2+1) > TOL )
				{
					optimal = FALSE ;
					i1 = settings->ij_low[y2-2] ;
				}
			}
			// b_up^j=min{b_up^j,b_up^j+1}
			if ( (Io_a == set_up || I_Two == set_up) && (y2<settings->pairs->classes-1))
			{
				if ( (F2+1) - settings->bj_up[y2] > TOL )
				{
					optimal = FALSE ;
					i1 = settings->ij_up[y2] ;
				}
			}
			if (optimal == FALSE)
			{				
				if (TRUE ==  ordinal_cross_takestep( ALPHA + i1 - 1, ALPHA + i2 - 1, y2 , settings) )
					return TRUE ;
				else 
				{
					//if ( TRUE == SMO_DISPLAY )
					{
						printf("%d and %d failed in cross_takestep.\n",i1,i2) ;
					}
				}
			}
			else
			{
				// check mu
			}
		}*/
	}
	if (y2>1)
	{
		/* check lower part*/
		if ( Io_b == set_dw || I_One == set_dw )
		{		
			if ( settings->bmu_low[y2-2] - (F2-1) > TOL )
			{
				optimal = FALSE ;
				i1 = i2 ;
				b1 = y2-1 ;
				b2 = settings->imu_low[y2-2] ;
				i3 = settings->ij_low[settings->imu_low[y2-2]-1] ;
			}
		}
		if ( Io_b == set_dw || I_Fou == set_dw )
		{
			/* lower */
			if ( (F2-1) - settings->bmu_up[y2-2] > TOL )
			{
				optimal = FALSE ;
				b1 = settings->imu_up[y2-2] ;
				i1 = settings->ij_up[settings->imu_up[y2-2]-1] ;
				b2 = y2-1 ;
				i3 = i2 ;
			}
		}

		if (optimal == FALSE)
		{
			if ( set_dw == Io_b )
			{
				if ( settings->bmu_low[y2-2] - (F2-1) > (F2-1) - settings->bmu_up[y2-2] )
				{					
					i1 = i2 ;
					b1 = y2-1 ;
					b2 = settings->imu_low[y2-2] ;
					i3 = settings->ij_low[settings->imu_low[y2-2]-1] ;
				}
				else
				{
					b1 = settings->imu_up[y2-2] ;
					i1 = settings->ij_up[settings->imu_up[y2-2]-1] ;
					b2 = y2-1 ;
					i3 = i2 ;
				}
			}

			if (i1==i3)
			{
				if (TRUE == ordinal_cross_identical( ALPHA + i1 - 1, ALPHA + i3 - 1, y2-1, settings) )
					return TRUE ;
				else 
					printf("%lu and %lu failed in identical takestep.\n",i1,i3) ;
			}
			else if (b1==b2)
			{	
				if (TRUE == ordinal_takestep( ALPHA + i1 - 1, ALPHA + i3 - 1, y2-1, settings) )
					return TRUE ;
				else 
					printf("%lu and %lu failed in takestep.\n",i1,i3) ;
			}
			else
			{	
				if (TRUE == ordinal_cross_takestep( ALPHA + i1 - 1, b1, ALPHA + i3 - 1, b2, settings) )
					return TRUE ;
				else 
					printf("%lu and %lu failed in cross takestep.\n",i1,i3) ;
			}
		}
		/*else //y2!=1
		{
			// check cross updating
			// check lower part
			if ( (Io_b == set_dw || I_One == set_dw) && (y2>2) )
			{		
				if ( settings->bj_low[y2-3] - (F2-1) > TOL )
				{
					optimal = FALSE ;
					i1 = settings->ij_low[y2-3] ;
				}
			}
			if ( (Io_b == set_dw || I_Fou == set_dw) && (y2<settings->pairs->classes))
			{
				// lower
				if ( (F2-1) - settings->bj_up[y2-1] > TOL )
				{
					optimal = FALSE ;
					i1 = settings->ij_up[y2-1] ;
				}
			}
			if (optimal == FALSE)
			{				
				if (TRUE ==  ordinal_cross_takestep( ALPHA + i1 - 1, ALPHA + i2 - 1, y2 , settings) )
					return TRUE ;
				else 
				{
					//if ( TRUE == SMO_DISPLAY )
					{
						printf("%d and %d failed in cross_takestep.\n",i1,i2) ;
					}
				}
			}
			else
			{
				// check mu
			}
		}*/
	}
	return FALSE ;
}

/*******************************************************************************\

	BOOL ordinal_examine_example_implicit ( Alphas *, smo_Settings * )

	the implicit-constraint counterpart of ordinal_examine_example. It loops over
	every threshold picking the single most violating pair and dispatches one way,
	where the explicit version keeps running mu bounds and dispatches three ways.
	Ported from the implicit-constraint SMO of Chu and Keerthi (2007).

\*******************************************************************************/

BOOL ordinal_examine_example_implicit ( Alphas * alpha, smo_Settings * settings )
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
		if (TRUE ==  ordinal_takestep_implicit( ALPHA + i1 - 1, ALPHA + i2 - 1, j , settings) )
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

/* end of smo_routine.c */
