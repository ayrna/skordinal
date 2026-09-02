/*******************************************************************************\

	svor_train.c
		
	entry function for the python program fit function.

\*******************************************************************************/

#include "Python.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "svor_module_functions.h"
#include "smo.h"
#include "smo_loadproblem_python.h"
#include "smo_model_python.h"

#define CMD_LEN 2048


/*******************************************************************************\

    PyObject* fit(PyObject* self, PyObject* args)
    
    purpose: serve as the Python C-API entry function for model training (fitting). 
             It parses dataset features, labels, and hyperparameter options from 
             Python, initializes the C SMO configurations, executes the training 
             routine, and converts the resulting model into a Python dictionary.
    input:   self (module reference), args (Python tuple containing the 'labels' 
             list, 'features' list of lists, 'options' string, and 'model_type').
    output:  returns a PyObject pointer representing the trained model as a Python 
             dictionary, or NULL if an error (parsing, memory, or runtime) occurs.

\*******************************************************************************/

PyObject* fit(PyObject* self, PyObject* args) {
   	PyObject* labels = NULL;
   	PyObject* features = NULL;
   	char* options = NULL;
	int model_type = 0; // Default to SVOREX

   	/* "OOs|i" allows options model_type to be passed seamlessly */
   	if (!PyArg_ParseTuple(args, "OOs|i", &labels, &features, &options, &model_type)){
		PyErr_SetString(PyExc_RuntimeError, "Unable to parse arguments");
   		return NULL;
   	}

	int argc = 0; char *argv[CMD_LEN/2]; char options_copy[CMD_LEN];
	strncpy(options_copy, options, CMD_LEN - 1); options_copy[CMD_LEN - 1] = '\0';
	if((argv[argc] = strtok(options_copy, " ")) != NULL) while((argv[++argc] = strtok(NULL, " ")) != NULL);

	def_Settings * defsetting = NULL ;
	smo_Settings * smosetting = NULL ;
	PyObject * py_model = NULL;
	char buf[LENGTH], errorBuf[1024]; 
	unsigned int sz = 0, index = 0 ; double parameter = 0 ;

	if ( NULL == (defsetting = Create_def_Settings_Python()) ) {
		PyErr_SetString(PyExc_MemoryError, "Unable to create the settings structure");
		return NULL;
	}
	
	defsetting->model_type = model_type;

	do {
		strcpy(buf, argv[--argc]) ; sz = strlen(buf) ;
		if ( '-' == buf[0] ) {				
			for (index = 1 ; index < sz ; index++) {
				switch (buf[index]) {
				case 'v' : defsetting->smo_display = TRUE ; break ;
				case 'M' : defsetting->model_type = (int)parameter; parameter = 0; break;
				case 'L' : defsetting->kernel = LINEAR ; break ;
				case 'E' : 
					if (parameter>0) defsetting->epsilon = parameter ;
					else { parameter = 0; PyErr_SetString(PyExc_ValueError, "- E is invalid"); Clear_def_Settings( defsetting ) ; return NULL ; }
					break ;					
				case 'T' :
					if (parameter>0) defsetting->tol = parameter ;
					else { parameter = 0; PyErr_SetString(PyExc_ValueError, "- T is invalid"); Clear_def_Settings( defsetting ) ; return NULL ; }
					break ;
				case 'C' :
					if (parameter > 0) { defsetting->vc = (parameter) ; parameter = 0 ; }
					else { parameter = 0; PyErr_SetString(PyExc_ValueError, "- C is invalid"); Clear_def_Settings( defsetting ) ; return NULL ; }
					break ;						
				case 'K' :
					if (parameter > 0) { defsetting->kappa = (parameter) ; parameter = 0 ; }
					else { parameter = 0; PyErr_SetString(PyExc_ValueError, "- K is invalid"); Clear_def_Settings( defsetting ) ; return NULL ; }
					break ;
				case 'P' :						
					if (parameter >= 1) { defsetting->kernel = POLYNOMIAL ; defsetting->p = (unsigned int) parameter ; parameter = 0 ; }
					else { parameter = 0; PyErr_SetString(PyExc_ValueError, "- P is invalid"); Clear_def_Settings( defsetting ) ; return NULL ; }	
					break ;	
				default :
					if ('-' != buf[index]) { snprintf(errorBuf, sizeof(errorBuf), "-%c is invalid", buf[index]); PyErr_SetString(PyExc_ValueError, errorBuf); Clear_def_Settings( defsetting ) ; return NULL ; }
				}
			}
		} else parameter = atof(buf) ;
	} while ( argc > 1 ) ;

	if (defsetting->beta > 1.0) defsetting->beta = 1.0;

	if ( FALSE == smo_Loadproblem_Python (&(defsetting->pairs), features, labels) ) {
		Clear_def_Settings( defsetting ); return NULL;
	}
	if ( CLASSIFICATION == defsetting->pairs.datatype ) defsetting->beta = 1.0 ;

	defsetting->training.count = defsetting->pairs.count ;		
	defsetting->training.front = defsetting->pairs.front ;		
	defsetting->training.rear = defsetting->pairs.rear ;
	defsetting->training.classes = defsetting->pairs.classes ;	
	defsetting->training.dimen = defsetting->pairs.dimen ;
	defsetting->training.featuretype = defsetting->pairs.featuretype ;
	defsetting->training.datatype = defsetting->pairs.datatype ;

	smosetting = Create_smo_Settings_Python(defsetting) ; 
	if(smosetting == NULL) {
		Clear_def_Settings( defsetting );
		PyErr_SetString(PyExc_MemoryError, "Unable to create the model"); return NULL;
	}

	// Pointer mapping
	smosetting->pairs = (Data_List*)malloc(sizeof(Data_List));
	*(smosetting->pairs) = defsetting->pairs;  		
	
	defsetting->training.count = 0 ;		
	defsetting->training.front = NULL ;		
	defsetting->training.rear = NULL ;
	defsetting->training.featuretype = NULL ;

	// Train process
	if(smo_routine_Python (smosetting) == FALSE){
		Clear_smo_Settings( smosetting ) ;
		Clear_def_Settings( defsetting ) ;

		if (!PyErr_Occurred()) {
			PyErr_SetString(PyExc_RuntimeError, "The train process failed internally.");
		}
		return NULL;
	}
	py_model = modelToPython(smosetting);
	
	// Secure memory cleanup
	Clear_smo_Settings( smosetting ) ;
	defsetting->pairs.front = NULL; defsetting->pairs.rear = NULL; defsetting->pairs.count = 0;
	Clear_def_Settings( defsetting ) ;	

	return py_model;
} /* end of fit() */


// end of svor_train.c