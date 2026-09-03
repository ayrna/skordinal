#include "Python.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/types.h> 
#include <sys/timeb.h>
#include "smo.h"

BOOL smo_ordinal_Python (smo_Settings * settings)
{
    char buffer[1024];//For error messages
	BOOL examineAll = TRUE ;
	long unsigned int numChanged = 0 ;
	Alphas * alpha = NULL ;   
	long unsigned int loop = 0 ;
	unsigned int j, k ;
	
	if (NULL == settings)
		return FALSE ;
	
	BETA = 1 ;	
	EPSILON = 0 ;
	if ( VC <= 0 || EPSILON < 0 )  
		return FALSE ;
	SMO_WORKING = TRUE ;
	
	if(Clean_Alphas( ALPHA, settings ) == FALSE){
        PyErr_SetString(PyExc_RuntimeError, "Error cleaning Alphas");
		return FALSE;
    }
    if(Check_Alphas( ALPHA, settings ) == FALSE){
        PyErr_SetString(PyExc_RuntimeError, "Error checking Alphas");
		return FALSE;
    }

	if ( TRUE == SMO_DISPLAY )
	{
		printf("SMO for Ordinal Expert...  \r\n") ;			
		printf("C=%f, Kappa=%f, Epsilon=%f, Beta= %f\n", VC, KAPPA, EPSILON, BETA) ;
		for (loop=1;loop<settings->pairs->classes;loop++)
		{
			if (IMPLICIT_CONSTRAINTS == CONSTRAINTS)
				printf("threshold %lu --- %u: up=%f(%lu), low=%f(%lu)\n", loop,settings->pairs->labels[loop-1], settings->bj_up[loop-1],
				settings->ij_up[loop-1],settings->bj_low[loop-1],settings->ij_low[loop-1]) ;
			else
				printf("threshold %lu --- %u: up=%f(%lu), low=%f(%lu), mu=%f\n", loop,settings->pairs->labels[loop-1], settings->bj_up[loop-1], 
				settings->ij_up[loop-1],settings->bj_low[loop-1],settings->ij_low[loop-1],settings->mu[loop-1]) ;
		}
		printf("\n") ;
	}

	tstart() ; /* switch on timer*/

	/* main routine*/ 
	while ( numChanged > 0 || examineAll )
	{
		if ( examineAll )
		{
			/* loop over all pairs*/		
			numChanged = 0 ;
			for ( loop = 1; loop <= settings->pairs->count; loop ++ )
			{
				if (IMPLICIT_CONSTRAINTS == CONSTRAINTS)
					numChanged += ordinal_examine_example_implicit( ALPHA + loop - 1, settings ) ;
				else
					numChanged += ordinal_examine_example( ALPHA + loop - 1, settings ) ; 
			}			
			if (TRUE == SMO_DISPLAY)
			{
				if (IMPLICIT_CONSTRAINTS == CONSTRAINTS)
				{
					for (loop=1;loop<settings->pairs->classes;loop++)
						printf("threshold %lu : up=%f(%lu), low=%f(%lu)\n", loop, settings->bj_up[loop-1],
							settings->ij_up[loop-1], settings->bj_low[loop-1],settings->ij_low[loop-1]) ;
				}
				else
				{
				for (loop=1;loop<settings->pairs->classes;loop++)
					printf("threshold %lu : up=%f(%lu), low=%f(%lu), mu=%f.\n", loop, settings->bj_up[loop-1], 
						settings->ij_up[loop-1], settings->bj_low[loop-1],settings->ij_low[loop-1], settings->mu[loop-1]) ;
				for (loop=1;loop<settings->pairs->classes;loop++)
					printf("threshold %lu : mu_up=%f(%lu), mu_low=%f(%lu), mu=%f.\n", loop, settings->bmu_up[loop-1], 
					settings->imu_up[loop-1], settings->bmu_low[loop-1],settings->imu_low[loop-1], settings->mu[loop-1]) ;
				}
			}
		}
		else
		{
			if (IMPLICIT_CONSTRAINTS == CONSTRAINTS)
			{
				j = active_threshold (settings) ;
				while ( numChanged>0&&j>0 )
				{
					numChanged = ordinal_takestep_implicit (ALPHA + settings->ij_up[j-1] - 1,
						ALPHA + settings->ij_low[j-1] - 1, j, settings) ;
					j = active_threshold (settings) ;
				}
			}
			else
			{
			/* check the worse pair*/
			loop = active_cross_threshold (settings) ;
			while ( loop>0 && numChanged>0 )
			{
				if (TRUE == ordinal_cross_takestep (ALPHA + settings->ij_up[settings->imu_up[loop-1]-1] - 1,settings->imu_up[loop-1],
                                ALPHA + settings->ij_low[settings->imu_low[loop-1]-1] - 1,settings->imu_low[loop-1], settings) )
				{
					numChanged += 1 ;
					loop = active_cross_threshold (settings) ;
				}
				else
				{
                    snprintf(buffer, sizeof(buffer), "fail to update bias %lu", loop);
                    PyErr_SetString(PyExc_RuntimeError, buffer);
					return FALSE;
					numChanged = 0 ;
				}
			}
			}
			numChanged = 0 ;
			if ( TRUE == settings->abort )
			{
				SMO_WORKING = FALSE ;
				return FALSE ;
			}
		} /* end of if-else*/

		if ( TRUE == examineAll )
		{
			examineAll = FALSE ;
		}
		else if ( 0 == numChanged )
		{
			examineAll = TRUE ;
		}

	} /* end of while*/

	tend() ; /* switch off timer*/ 
	settings->smo_timing = tval() ;
	DURATION += settings->smo_timing ;
	if (TRUE == SMO_DISPLAY)
	{
		j = 0 ;
		for ( loop = 1; loop <= settings->pairs->count; loop ++ )
		{	
			alpha = ALPHA + loop - 1 ;

			if (IMPLICIT_CONSTRAINTS == CONSTRAINTS)
			{
				/*/ a sample is a SV when any per-threshold multiplier is active: the
				    signed sum can vanish even when the multipliers do not*/
				for (k=0;k<settings->pairs->classes-1;k++)
				{
					if (fabs(alpha->alpha_j[k])>0)
					{
						j+=1 ;
						break ;
					}
				}
			}
			else if (fabs(alpha->alpha)>0)
				j+=1 ;
			if ( fabs(alpha->f_cache - Calculate_Ordinal_Fi ( loop, settings )) > EPS )
			{
				snprintf(buffer, sizeof(buffer), "index %d, alpha %f, f_cache , whose Fi is different from true value %6.4f to %6.4f", (int)(alpha-ALPHA+1), IMPLICIT_CONSTRAINTS == CONSTRAINTS ? alpha->alpha : alpha->alpha_up, alpha->f_cache, Calculate_Fi ( loop, settings ));
                PyErr_SetString(PyExc_ValueError, buffer);
				return FALSE;
			}
		}
		printf("SMO is done using CPU time %f seconds with %u off-bound SVs.\r\n", settings->smo_timing, j) ;
	}
	for (loop=1;loop<settings->pairs->classes;loop++)
	{
		if (IMPLICIT_CONSTRAINTS == CONSTRAINTS)
		{
			if (settings->bj_low[loop-1] - settings->bj_up[loop-1]>TOL)
			{
				snprintf(buffer, sizeof(buffer), "Warning: KKT conditions are violated on bias!!! %f with C=%.3f K=%.3f", settings->bj_low[loop-1] + settings->bj_up[loop-1], VC, KAPPA);
				PyErr_SetString(PyExc_ValueError, buffer);
				return FALSE;
			}

			settings->biasj[loop-1] = (settings->bj_low[loop-1] + settings->bj_up[loop-1])/2.0 ;
		}
		else
		{
		if (settings->bmu_low[loop-1] - settings->bmu_up[loop-1]>TOL)
		{
			snprintf(buffer, sizeof(buffer), "Warning: KKT conditions are violated on bias!!! %f with C=%.3f K=%.3f", settings->bmu_low[loop-1] + settings->bmu_up[loop-1], VC, KAPPA);
            PyErr_SetString(PyExc_ValueError, buffer);
            return FALSE;
		}

		settings->biasj[loop-1] = (settings->bmu_low[loop-1] + settings->bmu_up[loop-1])/2.0 ;
		}

		if (loop > 1)
		{
			if (settings->biasj[loop-1]+TOL<settings->biasj[loop-2])
			{
				snprintf(buffer, sizeof(buffer), "Warning: thresholds %lu : %f < thresholds %lu : %f", loop, settings->biasj[loop-1], loop-1, settings->biasj[loop-2]);
                PyErr_SetString(PyExc_ValueError, buffer);
                return FALSE;
			}
			if (settings->biasj[loop-1]<settings->biasj[loop-2])
				settings->biasj[loop-1] = settings->biasj[loop-2] ;

		}	
	}

	SMO_WORKING = FALSE ;
	return TRUE ; 
}


BOOL smo_routine_Python (smo_Settings * settings)
{
	if (NULL == settings)
		return FALSE ;

	if (ORDINAL == settings->pairs->datatype)
	{
		if (FALSE == smo_ordinal_Python (settings))
			return FALSE ;
		if (IMPLICIT_CONSTRAINTS == CONSTRAINTS)
		{
			/*/ the SMO loop only maintains the per-threshold multipliers, so the
			    scalar coefficient the predict path reads is stale until now*/
			return Finalize_Alphas ( ALPHA, settings ) ;
		}
		return TRUE ;
	}
	else
	{
		PyErr_SetString(PyExc_ValueError, "SMO can not handle this data type");
        return FALSE;
	}
}
