#include "Python.h"

#include "svm-module-functions.hpp"

/*Python module init*/
#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

static PyMethodDef svmMethods[] = {
	{ "fit", (PyCFunction)(void(*)(void))fit, METH_VARARGS | METH_KEYWORDS,
		"fit($module, labels, features, /, kernel, degree, gamma, coef0, tol, C, "
		"shrinking, cache_size)\n--\n\n"
		"Train the model using libsvm-rank (low-level method)." },
    { "predict", predict, METH_VARARGS, "Predict labels" },
	{ NULL, NULL, 0, NULL }
};

#ifndef IS_PY3K /*For Python 2*/
	#ifdef __cplusplus
		extern "C" {
	#endif
			DL_EXPORT(void) initsvm(void)
			{
			  Py_InitModule("svm", svmMethods);
			}
	#ifdef __cplusplus
		}
	#endif
#else /*For Python 3*/
	static struct PyModuleDef svmmodule = {
	    PyModuleDef_HEAD_INIT,
	    "_libsvmrank",   /* name of module */
	    NULL, 		 /* module documentation, may be NULL */
	    -1,       	 /* size of per-interpreter state of the module,
	                 or -1 if the module keeps state in global variables. */
	    svmMethods
	};

	#ifdef __cplusplus
		extern "C" {
	#endif
			PyMODINIT_FUNC
			PyInit__libsvmrank(void){
			    return PyModule_Create(&svmmodule);
			}
	#ifdef __cplusplus
		}
	#endif
#endif
