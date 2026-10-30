#ifndef _SVM_MODEL_PYTHON_HPP_
#define _SVM_MODEL_PYTHON_HPP_

#include <Python.h>

#include "svm.h"

PyObject * modelToPython(struct svm_model* model);
struct svm_model* pythonToModel(PyObject* model);

#endif
