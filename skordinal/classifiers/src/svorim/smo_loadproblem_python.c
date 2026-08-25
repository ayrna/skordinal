#include <stdlib.h>
#include <stdio.h>

#include "Python.h"
#include "smo_loadproblem_python.h"

BOOL smo_Loadproblem_Python ( Data_List * pairs, PyObject* features, PyObject* labels ){
	char buffer[1024];//For error messages
	int instance_number, label_number, dim = -2 ;
	unsigned long index = 1 ;
	double * point = NULL ;
	unsigned int y, sz ;
	int i = 0, j = 0 ;
	double mean = 0 ;
	double ymax = LONG_MIN ;
	double ymin = LONG_MAX ;
	double * xmean = NULL;
	Data_Node * node = NULL ;
	int t0=0, tr=0 ;

	Data_List label ;

	if ( NULL == pairs ){
		PyErr_SetString(PyExc_MemoryError, "Unable to allocate enough memory");
		return FALSE ;
	}
	
	Clear_Data_List( pairs ) ;
	Create_Data_List( &label ) ;

	// Check the input dimension and instance number here
	
	dim = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(PyList_GetItem(features, 0)))); /*features colums*/ ;
	pairs->dimen = dim ;

	instance_number = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(features))); /*features rows*/
	label_number = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(labels)));

	if(instance_number != label_number){
		PyErr_SetString(PyExc_ValueError, "Number of labels is different to the number of instances");
		return FALSE;
	}
	
	//Initialize the x_mean and x_devi in Data_List pairs

	if ( NULL == (pairs->x_mean = (double *)(malloc(dim*sizeof(double))) ) 
		|| NULL == (pairs->x_devi = (double *)(malloc(dim*sizeof(double))) ) 
		|| NULL == (xmean = (double *)(malloc(dim*sizeof(double))) ) )
	{		
		if (NULL != pairs->x_mean) 
			free(pairs->x_mean) ;
		if (NULL != pairs->x_devi) 
			free(pairs->x_devi) ;
		if (NULL != xmean)
			free(xmean) ;

		PyErr_SetString(PyExc_MemoryError, "Unable to allocate enough memory");
		return FALSE ;
	}
	for ( j = 0; j < dim; j ++ )
		pairs->x_mean[j] = 0 ;
	for ( j = 0; j < dim; j ++ )
		pairs->x_devi[j] = 0 ;
	for ( j = 0; j < dim; j ++ )
		xmean[j] = 0 ;

	pairs->datatype = CLASSIFICATION ; 

	do
	{

		point = (double *) malloc( (dim+1) * sizeof(double) ) ; // Pairs to free them
		if ( NULL == point )
		{
			if (NULL != pairs->x_mean) 
				free(pairs->x_mean) ;
			if (NULL != pairs->x_devi) 
				free(pairs->x_devi) ;
			if (NULL != xmean)
				free(xmean) ;
			Clear_Data_List( pairs ) ;

			PyErr_SetString(PyExc_MemoryError, "Unable to allocate enough memory");
			return FALSE ;
		}
		
		// Load instances features (Instance by instance)
		for (i = 0; i < dim; i++){
			point[i] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(features, index-1), i));
		}
		point[dim] = 0;

		// load y as target (Labels)(One by one)
		y = PyFloat_AsDouble(PyList_GetItem(labels, index-1));
						
		if ( TRUE == Add_Data_List( pairs, Create_Data_Node(index, point, y) ) )
		{
			// update statistics
			pairs->mean = (mean * (((double)(pairs->count)) - 1) + y )/ ((double)(pairs->count))  ;
			pairs->deviation = pairs->deviation + (y-mean)*(y-mean) * ((double)(pairs->count)-1)/((double)(pairs->count));			
			mean = pairs->mean ;	
			for ( j=0; j<dim; j++ )
			{
				pairs->x_mean[j] = (xmean[j] * (((double)(pairs->count)) - 1) + point[j] )/ ((double)(pairs->count))  ;
				pairs->x_devi[j] = pairs->x_devi[j] + (point[j]-xmean[j])*(point[j]-xmean[j]) * ((double)(pairs->count)-1)/((double)(pairs->count));			
				xmean[j] = pairs->x_mean[j] ;
			}
			if (y>ymax)
			{ ymax = y ; pairs->i_ymax = index ;}
			if (y<ymin)
			{ ymin = y ; pairs->i_ymin = index ;}
			
			// check data type 
			Add_Label_Data_List( &label, Create_Data_Node(index, point, y) ) ;
			index ++ ;
		}
		else{
			if (NULL != pairs->x_mean) 
				free(pairs->x_mean) ;
			if (NULL != pairs->x_devi) 
				free(pairs->x_devi) ;
			if (NULL != xmean)
				free(xmean) ;
			Clear_Data_List( pairs ) ;

			PyErr_SetString(PyExc_MemoryError, "Unable to allocate enough memory");
			return FALSE ;
		}
		
	}
	while( (long)index <= instance_number ) ;

	if (label.count>=2)
		pairs->datatype = ORDINAL ;
	else
		printf("Warning : not a ordinal regression.\n") ;

	if (pairs->count < MINNUM || (pairs->datatype == UNKNOWN ) ) 
	{
		Clear_Data_List( pairs ) ;
		if (NULL != pairs->x_mean) 
			free(pairs->x_mean) ;
		if (NULL != pairs->x_devi) 
			free(pairs->x_devi) ;
		if (NULL != xmean)
			free(xmean) ;

		PyErr_SetString(PyExc_ValueError, "Too few input pairs");
		return FALSE ;
	}

	pairs->featuretype = (int *) malloc(pairs->dimen*sizeof(int)) ;
	if (NULL != pairs->featuretype)
	{
		//default 0
		for (sz=0;sz<pairs->dimen;sz++)
			pairs->featuretype[sz] = 0 ;
	}

	pairs->deviation = sqrt( pairs->deviation / ((double)(pairs->count - 1.0)) ) ;
	for ( j=0; j<dim; j++ )
		pairs->x_devi[j] = sqrt( pairs->x_devi[j] / ((double)(pairs->count - 1.0)) ) ;	
	
	// set target value as +1 or -1, if data type is CLASSIFICATION
	if ( UNKNOWN != pairs->datatype )
	{
			pairs->deviation = 1.0 ;
			pairs->mean = 0 ;
			pairs->normalized_output = FALSE ;
	}

	for ( j=0; j<dim; j++ )
	{
		if (pairs->featuretype[j] != 0)
		{
			pairs->x_devi[j] = 1 ;
			pairs->x_mean[j] = 0 ;
		}
	}

	// Normalize the target if needed 
	node = pairs->front ;
	while ( node != NULL )
	{
		if ( TRUE == pairs->normalized_input )
		{
			for ( j=0; j<dim; j++ )
			{				
				if (pairs->x_devi[j]>0)
					node->point[j] = (node->point[j]-pairs->x_mean[j])/(pairs->x_devi[j]) ;
				else
					node->point[j] = 0 ;
			}
		}
		node = node->next ; 
	}

	if ( ORDINAL == pairs->datatype )
	{
		//printf("ORDINAL %lu REGRESSION.\r\n",label.count) ;
		pairs->classes = label.count ;
		if (NULL != pairs->labels)
			free( pairs->labels ) ;
		i=0;
		pairs->labels = (unsigned int*)malloc(pairs->classes*sizeof(unsigned int)) ;
		pairs->labelnum = (unsigned int*)malloc(pairs->classes*sizeof(unsigned int)) ;
		if (NULL != pairs->labels&&NULL != pairs->labelnum)
		{
			node = label.front ;
			//printf("ordinal varibles : ") ;
			while (NULL!=node)
			{
				if (node->target<1 || node->target>pairs->classes)
				{
					snprintf(buffer, sizeof(buffer), "Error : targets should be from 1 to %d", (int)pairs->classes);
					PyErr_SetString(PyExc_ValueError, buffer);
					return FALSE;
				}
				pairs->labels[node->target-1] = node->target ;
				if (node->target-1==0)
					t0 = node->target ;
				if (node->target==pairs->classes)
					tr = node->target ;
				pairs->labelnum[node->target-1] = node->fold ;
				i += node->fold ;
				//printf("%d(%d)  ", node->target, node->fold) ;
				node = node->next ;
			}
			//printf("\n") ;
			if (i!=(int)pairs->count||t0!=1||tr!=(int)pairs->classes)
			{
				PyErr_SetString(PyExc_ValueError, "Error in data list");
				return FALSE;
			}
		}
		else
		{			
			PyErr_SetString(PyExc_MemoryError, "fail to malloc for pairs->labels");
			return FALSE;
		}
	}

	Clear_Label_Data_List (&label) ;
	if ( NULL != xmean )
		free( xmean ) ;

	return TRUE ;
}
