#ifndef _SMO_MODEL_PYTHON_HPP_
#define _SMO_MODEL_PYTHON_HPP_

#include <Python.h>

#include "smo.h"

PyObject* modelToPython(smo_Settings* model);
smo_Settings* pythonToModel(PyObject* model);

#endif
