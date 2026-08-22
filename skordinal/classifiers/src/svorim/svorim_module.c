#include "Python.h"

#include "svorim_module_functions.h"

/*Python module init*/
#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

static PyMethodDef svorimMethods[] = {
	{ "fit", fit, METH_VARARGS, "Fits a model" },
    { "predict", predict, METH_VARARGS, "Predict labels" },
	{ NULL, NULL, 0, NULL }
};

#ifndef IS_PY3K /*For Python 2*/
	#ifdef __cplusplus
		extern "C" {
	#endif
			DL_EXPORT(void) initsvorim(void)
			{
			  Py_InitModule("svorim", svorimMethods);
			}
	#ifdef __cplusplus
		}
	#endif
#else /*For Python 3*/
	static struct PyModuleDef svorimmodule = {
	    PyModuleDef_HEAD_INIT,
	    "_libsvorim",   /* name of module */
	    NULL, 		 /* module documentation, may be NULL */
	    -1,       	 /* size of per-interpreter state of the module,
	                 or -1 if the module keeps state in global variables. */
	    svorimMethods
	};

	#ifdef __cplusplus
		extern "C" {
	#endif
			PyMODINIT_FUNC
			PyInit__libsvorim(void){
			    return PyModule_Create(&svorimmodule);
			}
	#ifdef __cplusplus
		}
	#endif
#endif
