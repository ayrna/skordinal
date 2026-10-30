#include "Python.h"

#include "svor_module_functions.h"

/*Python module init*/
#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

static PyMethodDef svorMethods[] = {
	{ "fit", fit, METH_VARARGS, "Fits a model" },
    { "predict", predict, METH_VARARGS, "Predict labels" },
	{ NULL, NULL, 0, NULL }
};

#ifndef IS_PY3K /*For Python 2*/
	#ifdef __cplusplus
		extern "C" {
	#endif
			DL_EXPORT(void) initsvor(void)
			{
			  Py_InitModule("svor", svorMethods);
			}
	#ifdef __cplusplus
		}
	#endif
#else /*For Python 3*/
	static struct PyModuleDef svormodule = {
	    PyModuleDef_HEAD_INIT,
	    "_libsvor",   /* name of module */
	    NULL, 		 /* module documentation, may be NULL */
	    -1,       	 /* size of per-interpreter state of the module,
	                 or -1 if the module keeps state in global variables. */
	    svorMethods
	};

	#ifdef __cplusplus
		extern "C" {
	#endif
			PyMODINIT_FUNC
			PyInit__libsvor(void){
			    return PyModule_Create(&svormodule);
			}
	#ifdef __cplusplus
		}
	#endif
#endif
