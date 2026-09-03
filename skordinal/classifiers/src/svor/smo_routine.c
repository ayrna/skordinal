#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "smo.h"


/*******************************************************************************\

    BOOL smo_ordinal (smo_Settings * settings)
    
    purpose: execute the native C Sequential Minimal Optimization (SMO) training 
             loop for ordinal regression. It iteratively examines examples, updates 
             multipliers to reach optimality, and computes the final thresholds (biases).
    input:   settings (pointer to the initialized smo_Settings structure).
    output:  returns TRUE upon successful training convergence, FALSE if settings 
             are invalid, the penalty parameter C is too small, or if aborted.

\*******************************************************************************/

BOOL smo_ordinal (smo_Settings * settings)
{
	BOOL examineAll = TRUE ;
	long unsigned int numChanged = 0 ;
	long unsigned int loop = 0 ;
	unsigned int j ;
	
	if (NULL == settings) return FALSE ;
	
	if ( VC <= EPS*EPS ) return FALSE ;
	SMO_WORKING = TRUE ;
	Clean_Alphas (settings->alpha, settings) ;	
	Check_Alphas ( settings->alpha, settings ) ;

	tstart() ; 

	while ( numChanged > 0 || examineAll )
	{
		if ( examineAll )
		{
			numChanged = 0 ;
			for ( loop = 1; loop <= settings->pairs->count; loop ++ )
			{
				numChanged += ordinal_examine_example( settings->alpha + loop - 1, settings ) ; 
			}			
		}
		else
		{
			j = active_threshold (settings) ;
			while ( numChanged>0&&j>0 )
			{
				numChanged = ordinal_takestep (settings->alpha + settings->ij_up[j-1] - 1, 
					settings->alpha + settings->ij_low[j-1] - 1, j, settings) ;
				j = active_threshold (settings) ;
			}
			numChanged = 0 ;
			if ( TRUE == settings->abort )
			{
				SMO_WORKING = FALSE ;
				return FALSE ;
			}
		} 

		if ( TRUE == examineAll ) examineAll = FALSE ;
		else if ( 0 == numChanged ) examineAll = TRUE ;
	} 

	tend() ;
	settings->smo_timing = tval() ;
	DURATION += settings->smo_timing ;

	for (loop=1;loop<settings->pairs->classes;loop++)
	{
		settings->biasj[loop-1] = (settings->bj_low[loop-1] + settings->bj_up[loop-1])/2.0 ;
		if (loop > 1 && settings->biasj[loop-1]+TOL<settings->biasj[loop-2])
		{
			printf("Warning: thresholds %lu : %f < thresholds %lu : %f.\n",loop, settings->biasj[loop-1], loop-1, settings->biasj[loop-2]) ;
		}
	}
	SMO_WORKING = FALSE ;
	return TRUE ; 
} /* end of smo_ordinal */


/*******************************************************************************\

    BOOL smo_routine (smo_Settings * settings)
    
    purpose: serve as the main native C entry point to start the SMO training 
             routine. It validates that the dataset is configured for ordinal 
             regression before launching the actual optimization process.
    input:   settings (pointer to the initialized smo_Settings structure).
    output:  returns TRUE if the ordinal training completes successfully, FALSE 
             if the settings pointer is NULL or the datatype is unsupported.

\*******************************************************************************/

BOOL smo_routine (smo_Settings * settings)
{
	if (NULL == settings) return FALSE ;
	if (ORDINAL == settings->pairs->datatype) return smo_ordinal (settings) ;
	else return FALSE ;
} /* end of smo_routine */

// the end of smo_routine.c