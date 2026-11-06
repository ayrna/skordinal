/*******************************************************************************\

	svor_train.c
		
	entry function for the python program fit function.

\*******************************************************************************/
#include "Python.h"

#ifndef __MACH__
    #include <malloc.h>
#endif

#include "svor_module_functions.h"
#include "smo.h"
#include "smo_loadproblem_python.h"
#include "smo_model_python.h"

PyObject* fit(PyObject* self, PyObject* args, PyObject* kwargs)
{
   	//Python parameters
   	PyObject* labels = NULL;
   	PyObject* features = NULL;
	int kernel = 0;
	int degree = 0;
	double gamma = 0;
	double tol = 0;
	double C = 0;
	int constraints = 0;
	static char *kwlist[] = {"", "", "kernel", "degree", "gamma", "tol", "C",
		"constraints", NULL};

	// Take the hyperparameters as C values, as scikit-learn's libsvm.fit does
   	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOiidddi", kwlist, &labels,
			&features, &kernel, &degree, &gamma, &tol, &C, &constraints))
   		return NULL;

	/*Svor code*/
	def_Settings * defsetting = NULL ;
	smo_Settings * smosetting = NULL ;
	PyObject * py_model = NULL;

	if ( NULL == (defsetting = Create_def_Settings_Python()) )
	{
		PyErr_SetString(PyExc_MemoryError, "Unable to create the settings structure");
		return NULL;
	}
	defsetting->kernel = kernel ;
	// Leave the default power with the other kernels, which ignore it
	if (POLYNOMIAL == kernel)
		defsetting->p = degree ;
	defsetting->kappa = gamma ;
	defsetting->tol = tol ;
	defsetting->vc = C ;
	defsetting->constraints = constraints ;

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

		if (!PyErr_Occurred())
			PyErr_SetString(PyExc_RuntimeError, "The train process failed");
		return NULL;
	}

	//Translate the model to python
	py_model = modelToPython(smosetting);
	
	// free memory then exit
	Clear_smo_Settings( smosetting ) ;
	Clear_def_Settings( defsetting ) ;	

	return py_model;
}
//end of svor_train.c 
