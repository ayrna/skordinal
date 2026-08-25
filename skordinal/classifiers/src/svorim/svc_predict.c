#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "smo.h"


/* predict and save results */

BOOL svm_predict_Python ( Data_List * testlist, smo_Settings * settings )
{	
	Data_List * trainlist ;
	Data_Node * trainnode ;
	Data_Node * testnode ;
	double fx, kernel ;
	unsigned int i, j=0, k ;

	if (testlist == NULL || settings == NULL)
		return FALSE ;
	
	trainlist = settings->pairs ;

	if (TRUE == Is_Data_Empty(testlist))
		return FALSE ;	
	if (trainlist->dimen != testlist->dimen)
		return FALSE ;

	settings->c1p = 0 ;
	settings->c1n = 0 ;
	settings->c2p = 0 ;	
	settings->c2n = 0 ;		
	settings->svs = 0 ;

	i = 0 ;
	testnode = testlist->front ;
	while (testnode!=NULL)
	{ 
		fx = 0 ;
		
		/* transform input point*/
		if (TRUE == trainlist->normalized_input)
		{	
			for (k=0;k<trainlist->dimen;k++)
			{ 
				if ( 0 != trainlist->x_devi[k] )
					testnode->point[k] = (testnode->point[k]-trainlist->x_mean[k])/(trainlist->x_devi[k]) ;
				else
					testnode->point[k] = 0 ;
			}
		}
		trainnode = trainlist->front ;
		j = 0 ;
		while (trainnode!=NULL)
		{		
			/* calculate kernel*/ 
			if ( (ALPHA+j)->alpha != 0 )
			{
				kernel = Calculate_Kernel (trainnode->point, testnode->point, settings) ;				
				fx = fx + (ALPHA+j)->alpha * kernel ;
				if (i==0)
					settings->svs ++ ;
			}
			trainnode = trainnode->next ;
			j++ ;
		}
		
		testnode -> fx = fx ;
		fx = fx + BIAS ;
		testnode->guess = fx ;

		if ( ORDINAL == trainlist->datatype )
		{
			/* save guess of target in datafile	*/
			testnode->guess = 1 ;
			for (k=1;k<settings->pairs->classes;k++)
			{
				if (fx>settings->biasj[k-1])
					testnode->guess = k+1 ;
				else
					k = settings->pairs->classes ;
			}
		}
		testnode = testnode->next ;
		i++ ;
	}
	
	return TRUE ;
}

/* the end of svc_predict.c */
