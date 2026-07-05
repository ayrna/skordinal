#include "Python.h"

#include "orensemble-module.h"

/*Python module init*/
#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

static PyMethodDef orensembleMethods[] = {
    {"fit", fit, METH_VARARGS, "Fits a model"},
    {"predict", predict, METH_VARARGS, "Predict labels"},
    {NULL, NULL, 0, NULL}
};

#ifndef IS_PY3K /*For Python 2*/
#ifdef __cplusplus
extern "C"
{
#endif
    DL_EXPORT(void)
    initorensemble(void)
    {
        Py_InitModule("_orensemble", orensembleMethods);
    }
#ifdef __cplusplus
}
#endif
#else /*For Python 3*/
static struct PyModuleDef orensemblemodule = {
    PyModuleDef_HEAD_INIT,
    "_orensemble", /* name of module */
    NULL,          /* module documentation, may be NULL */
    -1,            /* size of per-interpreter state of the module,
                   or -1 if the module keeps state in global variables. */
    orensembleMethods
};

#ifdef __cplusplus
extern "C"
{
#endif
    PyMODINIT_FUNC
    PyInit__orensemble(void)
    {
        return PyModule_Create(&orensemblemodule);
    }
#ifdef __cplusplus
}
#endif
#endif
