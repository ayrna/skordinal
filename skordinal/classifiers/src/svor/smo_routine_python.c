#include "Python.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "smo.h"



/*******************************************************************************\

    unsigned int active_threshold (smo_Settings * settings)
    
    purpose: find the active threshold index (primarily for SVORIM) where the 
             difference between the lower bound (bj_low) and upper bound (bj_up) 
             is maximized and exceeds the tolerance.
    input:   settings (pointer to the smo_Settings structure containing bounds).
    output:  returns the index (unsigned int) of the most active threshold, or 
             0 if no threshold violates the tolerance.

\*******************************************************************************/

unsigned int active_threshold (smo_Settings * settings) {
    unsigned int i, j = 0 ; double active = 0, temp = 0 ; 
    for (i=1;i<settings->pairs->classes;i++) {
        temp = settings->bj_low[i-1]-settings->bj_up[i-1] ;
        if (temp>active && temp>TOL) { active = temp ; j = i ; }
    }
    return j ; 
} /* end of active_threshold */


/*******************************************************************************\

    unsigned int active_cross_threshold (smo_Settings * settings)
    
    purpose: find the active cross threshold index (primarily for SVOREX) where 
             the difference between the modified lower (bmu_low) and upper 
             (bmu_up) bounds is maximized and exceeds the tolerance.
    input:   settings (pointer to the smo_Settings structure).
    output:  returns the index (unsigned int) of the most active cross threshold, 
             or 0 if none are found.

\*******************************************************************************/

unsigned int active_cross_threshold (smo_Settings * settings) {
    unsigned int i, j = 0 ; double active = 0, temp = 0 ; 
    for (i=1;i<settings->pairs->classes;i++) {
        temp = settings->bmu_low[i-1]-settings->bmu_up[i-1] ;
        if (temp>active && temp>TOL) { active = temp ; j = i ; }
    }
    return j ;
}  /* end of active_cross_threshold */


/*******************************************************************************\

    BOOL ordinal_examine_example_SVOREX ( Alphas * alpha, smo_Settings * settings )
    
    purpose: evaluate a specific data example under the SVOREX model to check 
             if it violates KKT optimality conditions. If non-optimal, it triggers 
             a normal or cross-step optimization to update the multipliers.
    input:   alpha (pointer to the Alphas structure to examine), settings (pointer 
             to the smo_Settings structure).
    output:  returns TRUE if an optimization step was successfully taken, FALSE 
             if the example is already optimal or no progress can be made.

\*******************************************************************************/

BOOL ordinal_examine_example_SVOREX ( Alphas * alpha, smo_Settings * settings ) {
    double F2 = 0 ; unsigned int y2 = 0, b1 = 0, b2 = 0, loop = 0 ;
    long unsigned int i1 = 0, i2 = 0, i3 = 0 ;
    BOOL optimal = TRUE ; Set_Name set_up, set_dw ;

    i2 = alpha - settings->alpha + 1 ;
    set_up = alpha->setname_up ; set_dw = alpha->setname_dw ; y2 = alpha->pair->target ;

    if ( set_up == Io_a || set_dw == Io_b ) F2 = alpha->f_cache ;
    else {
        F2 = Calculate_Ordinal_Fi(i2, settings) ; alpha->f_cache = F2 ;     
        if (y2<settings->pairs->classes) {
            if ( (I_Thr == set_up || Io_a == set_up) && (F2+1 < settings->bj_up[y2-1]) ) { settings->bj_up[y2-1] = F2+1 ; settings->ij_up[y2-1] = i2 ; }
            if ( (I_Two == set_up || Io_a == set_up ) && (F2+1 > settings->bj_low[y2-1]) ) { settings->bj_low[y2-1] = F2+1 ; settings->ij_low[y2-1] = i2 ; }
        }       
        if  (y2>1) {
            if ( (I_One == set_dw || Io_b == set_dw) && (F2-1 < settings->bj_up[y2-2]) ) { settings->bj_up[y2-2] = F2-1 ; settings->ij_up[y2-2] = i2 ; }
            if ( (I_Fou == set_dw || Io_b == set_dw) && (F2-1 > settings->bj_low[y2-2]) ) { settings->bj_low[y2-2] = F2-1 ; settings->ij_low[y2-2] = i2 ; }
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
    
    if (y2<settings->pairs->classes) {
        if ( Io_a == set_up || I_Thr == set_up ) {
            if ( settings->bmu_low[y2-1] - (F2+1) > TOL ) { optimal = FALSE ; i1 = i2 ; b1 = y2 ; i3 = settings->ij_low[settings->imu_low[y2-1]-1] ; b2 = settings->imu_low[y2-1] ; }
        }
        if ( Io_a == set_up || I_Two == set_up ) {
            if ( (F2+1) - settings->bmu_up[y2-1] > TOL ) { optimal = FALSE ; i1 = settings->ij_up[settings->imu_up[y2-1]-1] ; b1 = settings->imu_up[y2-1] ; b2 = y2 ; i3 = i2 ; }
        }
        if (optimal == FALSE) {
            if ( set_up == Io_a ) {
                if ( settings->bmu_low[y2-1] - (F2+1) > (F2+1) - settings->bmu_up[y2-1] ) { i1 = i2 ; b1 = y2 ; b2 = settings->imu_low[y2-1] ; i3 = settings->ij_low[settings->imu_low[y2-1]-1] ; }
                else { i1 = settings->ij_up[settings->imu_up[y2-1]-1] ; b1 = settings->imu_up[y2-1] ; b2 = y2 ; i3 = i2 ; }
            }
            if (i1==i3) { if (TRUE == ordinal_cross_identical( settings->alpha + i1 - 1, settings->alpha + i3 - 1, y2, settings) ) return TRUE ; }
            else if (b1==b2) { if (TRUE == ordinal_takestep( settings->alpha + i1 - 1, settings->alpha + i3 - 1, y2 , settings) ) return TRUE ; }
            else { if (TRUE == ordinal_cross_takestep( settings->alpha + i1 - 1,b1, settings->alpha + i3 - 1, b2 , settings) ) return TRUE ; }
        }
    }
    if (y2>1) {
        if ( Io_b == set_dw || I_One == set_dw ) {      
            if ( settings->bmu_low[y2-2] - (F2-1) > TOL ) { optimal = FALSE ; i1 = i2 ; b1 = y2-1 ; b2 = settings->imu_low[y2-2] ; i3 = settings->ij_low[settings->imu_low[y2-2]-1] ; }
        }
        if ( Io_b == set_dw || I_Fou == set_dw ) {
            if ( (F2-1) - settings->bmu_up[y2-2] > TOL ) { optimal = FALSE ; b1 = settings->imu_up[y2-2] ; i1 = settings->ij_up[settings->imu_up[y2-2]-1] ; b2 = y2-1 ; i3 = i2 ; }
        }
        if (optimal == FALSE) {
            if ( set_dw == Io_b ) {
                if ( settings->bmu_low[y2-2] - (F2-1) > (F2-1) - settings->bmu_up[y2-2] ) { i1 = i2 ; b1 = y2-1 ; b2 = settings->imu_low[y2-2] ; i3 = settings->ij_low[settings->imu_low[y2-2]-1] ; }
                else { b1 = settings->imu_up[y2-2] ; i1 = settings->ij_up[settings->imu_up[y2-2]-1] ; b2 = y2-1 ; i3 = i2 ; }
            }
            if (i1==i3) { if (TRUE == ordinal_cross_identical( settings->alpha + i1 - 1, settings->alpha + i3 - 1, y2-1, settings) ) return TRUE ; }
            else if (b1==b2) { if (TRUE == ordinal_takestep( settings->alpha + i1 - 1, settings->alpha + i3 - 1, y2-1, settings) ) return TRUE ; }
            else { if (TRUE == ordinal_cross_takestep( settings->alpha + i1 - 1, b1, settings->alpha + i3 - 1, b2, settings) ) return TRUE ; }
        }
    }
    return FALSE ;
}  /* end of ordinal_examine_example_SVORIM */


/*******************************************************************************\

    BOOL ordinal_examine_example_SVORIM ( Alphas * alpha, smo_Settings * settings )
    
    purpose: evaluate a specific data example under the SVORIM model against 
             optimality conditions. If it violates the bounds, it calculates the 
             necessary updates and triggers an SMO step.
    input:   alpha (pointer to the Alphas structure to examine), settings (pointer 
             to the smo_Settings structure).
    output:  returns TRUE if an optimization step was successfully taken, FALSE 
             otherwise.

\*******************************************************************************/

BOOL ordinal_examine_example_SVORIM ( Alphas * alpha, smo_Settings * settings ) {
    double F2 = 0 ; unsigned int j = 0, loop ;
    long unsigned int i1 = 0, i2 = alpha - settings->alpha + 1 ;
    BOOL optimal = TRUE ; 

    if ( FALSE == Is_Io(alpha,settings) ) {
        alpha->f_cache = Calculate_Ordinal_Fi(i2, settings) ;
        for (loop = 0 ; loop < settings->pairs->classes-1 ; loop ++) {
            if (alpha->pair->target > (loop+1) ) {
                if (alpha->setname_ptr[loop]==Io_b || alpha->setname_ptr[loop]==I_One) { if (alpha->f_cache-1<=settings->bj_up[loop]) { settings->bj_up[loop] = alpha->f_cache-1 ; settings->ij_up[loop] = alpha - settings->alpha + 1 ; } }
                if (alpha->setname_ptr[loop]==Io_b || alpha->setname_ptr[loop]==I_Fou) { if (alpha->f_cache-1>=settings->bj_low[loop]) { settings->bj_low[loop] = alpha->f_cache-1 ; settings->ij_low[loop] = alpha - settings->alpha + 1 ; } }
            } else {
                if (alpha->setname_ptr[loop]==Io_a || alpha->setname_ptr[loop]==I_Thr) { if (alpha->f_cache+1<=settings->bj_up[loop]) { settings->bj_up[loop] = alpha->f_cache+1 ; settings->ij_up[loop] = alpha - settings->alpha + 1 ; } }
                if (alpha->setname_ptr[loop]==Io_a || alpha->setname_ptr[loop]==I_Two) { if (alpha->f_cache+1>=settings->bj_low[loop]) { settings->bj_low[loop] = alpha->f_cache+1 ; settings->ij_low[loop] = alpha - settings->alpha + 1 ; } }
            }
        }
    }
    
    for (loop = 0 ; loop < settings->pairs->classes-1 ; loop ++) {
        if (alpha->pair->target > (loop+1) ) {
            if (alpha->setname_ptr[loop]==Io_b || alpha->setname_ptr[loop]==I_One) {
                if ( settings->bj_low[loop] - (alpha->f_cache-1) > TOL ) { optimal = FALSE ; if (settings->bj_low[loop]-(alpha->f_cache-1)>F2) { i1 = settings->ij_low[loop] ; F2 = settings->bj_low[loop]-(alpha->f_cache-1) ; j = loop+1 ; } }
            }
            if (alpha->setname_ptr[loop]==Io_b || alpha->setname_ptr[loop]==I_Fou) {
                if ( (alpha->f_cache-1) - settings->bj_up[loop] > TOL ) { optimal = FALSE ; if ((alpha->f_cache-1) - settings->bj_up[loop]>F2) { i1 = settings->ij_up[loop] ; F2 = (alpha->f_cache-1) - settings->bj_up[loop] ; j = loop+1 ; } }
            }
        } else {
            if (alpha->setname_ptr[loop]==Io_a || alpha->setname_ptr[loop]==I_Thr) {
                if (settings->bj_low[loop]-(alpha->f_cache+1)>TOL) { optimal = FALSE ; if (settings->bj_low[loop]-(alpha->f_cache+1)>F2) { i1 = settings->ij_low[loop] ; F2 = settings->bj_low[loop]-(alpha->f_cache+1) ; j = loop+1 ; } }
            }
            if (alpha->setname_ptr[loop]==Io_a || alpha->setname_ptr[loop]==I_Two) {
                if ((alpha->f_cache+1)-settings->bj_up[loop]>TOL) { optimal = FALSE ; if ((alpha->f_cache+1)-settings->bj_up[loop]>F2) { i1 = settings->ij_up[loop] ; F2 = (alpha->f_cache+1)-settings->bj_up[loop] ; j = loop+1 ; } }
            }
        }
    }

    if (optimal == FALSE) {     
        if (TRUE == ordinal_takestep( settings->alpha + i1 - 1, settings->alpha + i2 - 1, j , settings) ) return TRUE ;
    }
    return FALSE ;
} /* end of ordinal_examine_example_SVORIM */


/*******************************************************************************\

    BOOL ordinal_examine_example ( Alphas * alpha, smo_Settings * settings )
    
    purpose: act as a dispatcher function that delegates the examination of an 
             example to either the SVOREX or SVORIM evaluation functions based 
             on the model configuration.
    input:   alpha (pointer to the Alphas structure), settings (pointer to the 
             smo_Settings structure).
    output:  returns the boolean result (TRUE/FALSE) of the delegated examine 
             function.

\*******************************************************************************/

BOOL ordinal_examine_example ( Alphas * alpha, smo_Settings * settings ) {
    if (settings->model_type == 0) return ordinal_examine_example_SVOREX(alpha, settings);
    else return ordinal_examine_example_SVORIM(alpha, settings);
} /* end of ordinal_examine_example */


/*******************************************************************************\

    BOOL smo_ordinal_Python_SVOREX (smo_Settings * settings)
    
    purpose: execute the main Sequential Minimal Optimization (SMO) outer training 
             loop for the SVOREX algorithm. It iteratively examines examples and 
             updates bounds until convergence or maximum iterations are reached, 
             then computes final biases.
    input:   settings (pointer to the configured smo_Settings structure).
    output:  returns TRUE upon successful completion of the training loop, FALSE 
             if initialization fails.

\*******************************************************************************/

BOOL smo_ordinal_Python_SVOREX (smo_Settings * settings)
{
    BOOL examineAll = TRUE ;
    long unsigned int numChanged = 0 ;
    long unsigned int loop = 0 ;
    
    BETA = 1 ;  
    EPSILON = 0 ;
    
    if ( VC <= 0 || EPSILON < 0 ) return TRUE ;
        
    SMO_WORKING = TRUE ;
    
    if(Clean_Alphas( settings->alpha, settings ) == FALSE) return FALSE;
    if(Check_Alphas( settings->alpha, settings ) == FALSE) return FALSE;

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
            loop = active_cross_threshold (settings) ;
            while ( loop>0 && numChanged>0 )
            {
                if (TRUE == ordinal_cross_takestep (settings->alpha + settings->ij_up[settings->imu_up[loop-1]-1] - 1,settings->imu_up[loop-1],
                                settings->alpha + settings->ij_low[settings->imu_low[loop-1]-1] - 1,settings->imu_low[loop-1], settings) )
                {
                    numChanged += 1 ;
                    loop = active_cross_threshold (settings) ;
                }
                else
                {
                    break;
                }
            }
            numChanged = 0 ;
            if ( TRUE == settings->abort )
            {
                SMO_WORKING = FALSE ;
                return TRUE ;
            }
        }

        if ( TRUE == examineAll )
            examineAll = FALSE ;
        else if ( 0 == numChanged )
            examineAll = TRUE ;
    }

    tend() ; 
    settings->smo_timing = tval() ;
    DURATION += settings->smo_timing ;

    for (loop=1;loop<settings->pairs->classes;loop++)
    {
        settings->biasj[loop-1] = (settings->bmu_low[loop-1] + settings->bmu_up[loop-1])/2.0 ;
        if (loop > 1 && settings->biasj[loop-1]<settings->biasj[loop-2])
        {
            settings->biasj[loop-1] = settings->biasj[loop-2] ;
        }   
    }

    SMO_WORKING = FALSE ;
    return TRUE ; 
} /* end of smo_ordinal_Python_SVOREX */


/*******************************************************************************\

    BOOL smo_ordinal_Python_SVORIM (smo_Settings * settings)
    
    purpose: execute the main SMO outer training loop for the SVORIM algorithm. 
             It iteratively optimizes the variables, computes the final biases 
             across classes, and aggregates the alpha values at the end.
    input:   settings (pointer to the configured smo_Settings structure).
    output:  returns TRUE upon successful completion of the training loop, FALSE 
             if initialization fails.

\*******************************************************************************/

BOOL smo_ordinal_Python_SVORIM (smo_Settings * settings)
{
    BOOL examineAll = TRUE ;
    long unsigned int numChanged = 0 ;
    long unsigned int loop = 0 ;
    unsigned int j ;
    
    BETA = 1 ;  
    EPSILON = 0 ;
    
    if ( VC <= 0 || EPSILON < 0 ) return TRUE ;
        
    SMO_WORKING = TRUE ;
    
    if(Clean_Alphas( settings->alpha, settings ) == FALSE) return FALSE;
    if(Check_Alphas( settings->alpha, settings ) == FALSE) return FALSE;

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
            while ( numChanged>0 && j>0 )
            {
                if (TRUE == ordinal_takestep (settings->alpha + settings->ij_up[j-1] - 1, settings->alpha + settings->ij_low[j-1] - 1, j, settings) )
                {
                    numChanged += 1 ;
                    j = active_threshold (settings) ;
                }
                else
                {
                    break;
                }
            }
            numChanged = 0 ;
            if ( TRUE == settings->abort )
            {
                SMO_WORKING = FALSE ;
                return TRUE ;
            }
        }

        if ( TRUE == examineAll )
            examineAll = FALSE ;
        else if ( 0 == numChanged )
            examineAll = TRUE ;
    }

    tend() ; 
    settings->smo_timing = tval() ;
    DURATION += settings->smo_timing ;

    for (loop=1;loop<settings->pairs->classes;loop++)
    {
        settings->biasj[loop-1] = (settings->bj_low[loop-1] + settings->bj_up[loop-1])/2.0 ;
        if (loop > 1 && settings->biasj[loop-1]<settings->biasj[loop-2])
        {
            settings->biasj[loop-1] = settings->biasj[loop-2] ;
        }   
    }

    for ( loop = 1; loop <= settings->pairs->count; loop ++ ) {
        Alphas *al = settings->alpha + loop - 1;
        double alpha_sum = 0;
        for(unsigned int k=0; k < settings->pairs->classes - 1; k++) {
            if (al->pair->target <= k+1) alpha_sum -= al->alpha_ptr[k];
            else alpha_sum += al->alpha_ptr[k];
        }
        al->alpha = alpha_sum; 
    }

    SMO_WORKING = FALSE ;
    return TRUE ; 
} /* end of smo_ordinal_Python_SVORIM */


/*******************************************************************************\

    BOOL smo_routine_Python (smo_Settings * settings)
    
    purpose: serve as the main C entry point for the Python extension to start 
             the SMO training routine. Validates the data type and routes 
             execution to the appropriate algorithm (SVOREX or SVORIM).
    input:   settings (pointer to the initialized smo_Settings structure).
    output:  returns TRUE if the training routine completes successfully, FALSE 
             if the datatype is incorrect (setting a Python exception) or fails.

\*******************************************************************************/

BOOL smo_routine_Python (smo_Settings * settings)
{
    if (NULL == settings) return FALSE ;
    if (ORDINAL != settings->pairs->datatype) {
        PyErr_SetString(PyExc_ValueError, "SMO can not handle this data type");
        return FALSE;
    }
    if (settings->model_type == 0) return smo_ordinal_Python_SVOREX (settings) ;
    else return smo_ordinal_Python_SVORIM (settings) ;
} /* end of smo_routine_Python */

// the end of smo_routine_python.c