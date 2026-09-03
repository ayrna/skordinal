/*******************************************************************************\

	smo_loadproblem_python.h

	defines all the functions needed to get SMO algorithm working on orca-python.

\*******************************************************************************/
#ifdef  __cplusplus
extern "C" {
#endif

#ifndef _SMO_LOADPROBLEM_PYTHON_H
#define _SMO_LOADPROBLEM_PYTHON_H
#include "Python.h"
#include "smo.h"

//Create the pairs structure from the data sent from python
BOOL smo_Loadproblem_Python ( Data_List * pairs, PyObject* features, PyObject* labels );

#endif

#ifdef  __cplusplus
}
#endif
/*/ the end of smo_loadproblem_python.h*/