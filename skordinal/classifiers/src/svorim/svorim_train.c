/*******************************************************************************\

	svorim_train.c

	entry function for the python program fit function.

\*******************************************************************************/
#include "Python.h"

#include <stdio.h>
#include <stdlib.h>
#ifndef __MACH__
    #include <malloc.h>
#endif
#include <string.h>

#include "svorim_module_functions.h"
#include "smo.h"
#include "smo_loadproblem_python.h"
#include "smo_model_python.h"

#define CMD_LEN 2048

PyObject* fit(PyObject* self, PyObject* args)
{
   	//Python parameters
   	PyObject* labels = NULL;
   	PyObject* features = NULL;
   	char* options = NULL;

   	/*Options is NULL terminated*/
   	if (!PyArg_ParseTuple(args, "OOs", &labels, &features, &options)){
		PyErr_SetString(PyExc_RuntimeError, "Unable to parse arguments");
   		return NULL;
   	}

	/* Put options in argv[]*/
	int argc = 0;
	char *argv[CMD_LEN/2];
	char options_copy[CMD_LEN];
	strncpy(options_copy, options, CMD_LEN - 1);
	options_copy[CMD_LEN - 1] = '\0';

	if((argv[argc] = strtok(options_copy, " ")) != NULL)
		while((argv[++argc] = strtok(NULL, " ")) != NULL)
			;

	/*Svorim code*/
	def_Settings * defsetting = NULL ;
	smo_Settings * smosetting = NULL ;
	PyObject * py_model = NULL;
	char buf[LENGTH] ;
	char errorBuf[1024]; //For error messages
	unsigned int sz = 0;
	unsigned int index = 0 ;
	double parameter = 0 ;

	if ( NULL == (defsetting = Create_def_Settings_Python()) )
	{
		// display help	
		printf("\nUsage:  model = svorim.fit(training_label_vector, training_instance_matrix, 'svorim_options')\n") ;
		printf("svorim_options:\n") ;
		printf("  -v     activates the verbose mode to display message.\n") ;		
		printf("  -L     use imbalanced Linear kernel (default Gaussian kernel).\n") ;
		printf("  -P  p  use Polynomial kernel with order p (default Gaussian kernel).\n") ;
		printf("  -E  e  set Epsilon at e for regression only (default 0.1). (Not used in orca-python)\n") ;
		printf("  -T  t  set Tolerance at t (default 0.001).\n") ;
		printf("  -K o set kappa value at o (default 1).\n") ;	
		printf("  -C o set C value at o (default  1).\n") ;
		
		if (NULL !=defsetting)
			Clear_def_Settings( defsetting ) ;

		PyErr_SetString(PyExc_MemoryError, "Unable to create the settings structure");
		return NULL;
	}
	else
	{
		//if (argc>1)
		//	printf("Options:\n") ;
		do
		{
			strcpy(buf, argv[--argc]) ;
			sz = strlen(buf) ;

			if ( '-' == buf[0] )
			{				
				for (index = 1 ; index < sz ; index++)
				{
					switch (buf[index])
					{
					case 'v' :
						printf("  - Verbose mode in display.\n") ;
						defsetting->smo_display = TRUE ;
						break ;
					case 'L' :
						//printf("  - choose Linear kernel.\n") ;
						defsetting->kernel = LINEAR ;						
						break ;
					case 'E' :
						if (parameter>0)
						{
							//printf("  - set Epsilon as %.3f.\n", parameter) ;
							defsetting->epsilon = parameter ;
						}
						else	
						{
							parameter = 0;
							PyErr_SetString(PyExc_ValueError, "- E is invalid");
							Clear_def_Settings( defsetting ) ;
							return NULL ;
						}

						break ;					
					case 'T' :
						if (parameter>0)
						{
							//printf("  - set Tol as %.6f.\n", parameter) ;
							defsetting->tol = parameter ;
						}
						else	
						{
							parameter = 0;
							PyErr_SetString(PyExc_ValueError, "- T is invalid");
							Clear_def_Settings( defsetting ) ;
							return NULL ;
						}
						
						break ;
					case 'C' :
						if (parameter > 0)
						{ 
							defsetting->vc = (parameter) ;
							//printf("  - C at %f.\n", parameter) ;
							parameter = 0 ;					
						}
						else
						{
							parameter = 0;
							PyErr_SetString(PyExc_ValueError, "- C is invalid");
							Clear_def_Settings( defsetting ) ;
							return NULL ;
						}
						
						break ;						
					case 'K' :
						if (parameter > 0)
						{ 
							defsetting->kappa = (parameter) ;
							//printf("  - K at %f.\n", parameter) ;
							parameter = 0 ;						
						}
						else
						{
							parameter = 0;
							PyErr_SetString(PyExc_ValueError, "- K is invalid");
							Clear_def_Settings( defsetting ) ;
							return NULL ;
						}
						
						break ;
					case 'P' :						
						if (parameter >= 1)
						{ 
							defsetting->kernel = POLYNOMIAL ;
							defsetting->p = (unsigned int) parameter ;
							//printf("  - choose Polynomial kernel with order %d.\n", defsetting->p) ;
							parameter = 0 ;
						}
						else	
						{
							parameter = 0;
							PyErr_SetString(PyExc_ValueError, "- P is invalid");
							Clear_def_Settings( defsetting ) ;
							return NULL ;
						}	

						break ;	
					default :
						if ('-' != buf[index]){
							snprintf(errorBuf, sizeof(errorBuf), "-%c is invalid", buf[index]);
							PyErr_SetString(PyExc_ValueError, errorBuf);
							Clear_def_Settings( defsetting ) ;
							return NULL ;
						}
					}
				}
			}
			else
				parameter = atof(buf) ;
		}
		while ( argc > 1 ) ;
		//printf("\n") ;
	}

	//Update defsetting
	if (NULL == defsetting){
		PyErr_SetString(PyExc_MemoryError, "Unable to read the settings structure");
		return NULL;
	}
		
	if (defsetting->beta > 1.0)
		defsetting->beta = 1.0;

	if ( FALSE == smo_Loadproblem_Python (&(defsetting->pairs), features, labels) ){
		Clear_def_Settings( defsetting );
		return NULL;
	}

	if ( CLASSIFICATION == defsetting->pairs.datatype )
	{	
		defsetting->beta = 1.0 ;
	}

	//Save validation output		
	defsetting->training.count = defsetting->pairs.count ;		
	defsetting->training.front = defsetting->pairs.front ;		
	defsetting->training.rear = defsetting->pairs.rear ;
	defsetting->training.classes = defsetting->pairs.classes ;	
	defsetting->training.dimen = defsetting->pairs.dimen ;
	defsetting->training.featuretype = defsetting->pairs.featuretype ;
	defsetting->training.datatype = defsetting->pairs.datatype ;

	//Create smosettings
	smosetting = Create_smo_Settings_Python(defsetting) ; 
	if(smosetting == NULL){
		if(defsetting != NULL)
			Clear_def_Settings( defsetting );
		
		PyErr_SetString(PyExc_MemoryError, "Unable to create the model");
		return NULL;
	}

	smosetting->pairs = &defsetting->pairs ;  		
	defsetting->training.count = 0 ;		
	defsetting->training.front = NULL ;		
	defsetting->training.rear = NULL ;
	defsetting->training.featuretype = NULL ;

	//Train process
	if(smo_routine_Python (smosetting) == FALSE){
		Clear_smo_Settings( smosetting ) ;
		Clear_def_Settings( defsetting ) ;

		PyErr_SetString(PyExc_MemoryError, "The train process failed");
		return NULL;
	}

	//Translate the model to python
	py_model = modelToPython(smosetting);
	
	// free memory then exit
	Clear_smo_Settings( smosetting ) ;
	Clear_def_Settings( defsetting ) ;	

	return py_model;
}
//end of svorim_train.c
